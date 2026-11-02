#!/usr/bin/env python3
"""Audit local GitHub/Copilot/Hugging Face/Hermes readiness without leaking secrets.

Run from the repo root or anywhere:

    python3 scripts/audit_local_stack_readiness.py
    python3 scripts/audit_local_stack_readiness.py --json
    python3 scripts/audit_local_stack_readiness.py --remote-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HERMES_HOME = Path.home() / ".hermes"
TOP_LEVEL_MODEL_KEYS = {"provider", "default", "base_url", "context_length"}
TOKEN_PATTERNS = (
    re.compile(r"token=[^&\s]+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
)


def _run(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "command": cmd,
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "command": cmd,
    }


def sanitize_text(text: str) -> str:
    sanitized = text
    for pattern in TOKEN_PATTERNS:
        sanitized = pattern.sub(lambda m: m.group(0).split("=", 1)[0] + "=<redacted>" if "=" in m.group(0) else "Bearer <redacted>", sanitized)
    return sanitized


def parse_model_block(config_text: str) -> dict[str, str]:
    model_block: dict[str, str] = {}
    in_model = False
    for raw_line in config_text.splitlines():
        line = raw_line.rstrip()
        if not in_model:
            if line == "model:":
                in_model = True
            continue
        if line and not line.startswith("  "):
            break
        match = re.match(r"  ([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not match:
            continue
        key, value = match.groups()
        if key in TOP_LEVEL_MODEL_KEYS:
            model_block[key] = value.strip().strip("'\"")
    return model_block


def count_occurrences(config_text: str, pattern: str) -> int:
    return len(re.findall(pattern, config_text, flags=re.MULTILINE))


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def audit_git_repo() -> dict[str, Any]:
    branch = _run(["git", "-C", str(REPO_ROOT), "branch", "--show-current"])
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--short", "--branch"])
    remote = _run(["git", "-C", str(REPO_ROOT), "remote", "-v"])
    return {
        "repo_root": str(REPO_ROOT),
        "branch": branch["stdout"] if branch["ok"] else None,
        "status_summary": sanitize_text(status["stdout"] or status["stderr"]),
        "remote_summary": sanitize_text(remote["stdout"] or remote["stderr"]),
    }


def audit_github_cli() -> dict[str, Any]:
    status = _run(["gh", "auth", "status"])
    combined = "\n".join(part for part in (status["stdout"], status["stderr"]) if part).strip()
    text = sanitize_text(combined)
    if status["returncode"] == 127:
        state = "missing"
    elif status["ok"]:
        state = "logged_in"
    elif "not logged into any github hosts" in text.lower():
        state = "not_logged_in"
    else:
        state = "error"
    return {"state": state, "detail": text}


def audit_hf_cli() -> dict[str, Any]:
    whoami = _run(["hf", "auth", "whoami"])
    combined = "\n".join(part for part in (whoami["stdout"], whoami["stderr"]) if part).strip()
    text = sanitize_text(combined)
    if whoami["returncode"] == 127:
        state = "missing"
    elif whoami["ok"]:
        state = "logged_in"
    elif "not logged in" in text.lower():
        state = "not_logged_in"
    else:
        state = "error"
    return {
        "state": state,
        "detail": text,
        "hf_token_env_present": bool(os.environ.get("HF_TOKEN")),
    }


def audit_hermes(hermes_home: Path) -> dict[str, Any]:
    config_path = hermes_home / "config.yaml"
    auth_path = hermes_home / "auth.json"
    jobs_path = hermes_home / "cron" / "jobs.json"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    model_block = parse_model_block(config_text)
    jobs_json = load_json(jobs_path)
    auth_json = load_json(auth_path)

    drift_jobs: list[dict[str, Any]] = []
    if isinstance(jobs_json, list):
        iterable = jobs_json
    elif isinstance(jobs_json, dict):
        iterable = jobs_json.get("jobs", [])
    else:
        iterable = []
    for job in iterable:
        if not isinstance(job, dict):
            continue
        last_error = str(job.get("last_error", ""))
        if "drift_skip" in last_error:
            drift_jobs.append(
                {
                    "id": job.get("id"),
                    "provider_snapshot": job.get("provider_snapshot"),
                    "model_snapshot": job.get("model_snapshot"),
                    "last_error": sanitize_text(last_error),
                }
            )

    auth_sources: dict[str, list[dict[str, str]]] = {}
    if isinstance(auth_json, dict):
        providers = auth_json.get("providers", {})
        if isinstance(providers, dict):
            for provider, entries in providers.items():
                if not isinstance(entries, list):
                    continue
                auth_sources[provider] = []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    auth_sources[provider].append(
                        {
                            "label": str(entry.get("label", "")),
                            "source": str(entry.get("source", "")),
                            "base_url": sanitize_text(str(entry.get("base_url", ""))),
                        }
                    )

    return {
        "hermes_home": str(hermes_home),
        "model": model_block,
        "copilot_provider_references": count_occurrences(config_text, r"^\s*provider:\s*copilot\s*$"),
        "huggingface_reference_present": "huggingface" in config_text.lower(),
        "drift_jobs": drift_jobs,
        "auth_sources": auth_sources,
    }


def collect_audit(hermes_home: Path, remote_only: bool) -> dict[str, Any]:
    return {
        "remote_only": remote_only,
        "git_repo": audit_git_repo(),
        "github_cli": audit_github_cli(),
        "huggingface_cli": audit_hf_cli(),
        "hermes": audit_hermes(hermes_home),
    }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Local stack readiness audit")
    lines.append(f"repo: {report['git_repo']['repo_root']}")
    lines.append(f"branch: {report['git_repo']['branch'] or '<unknown>'}")
    lines.append(f"git status: {report['git_repo']['status_summary'] or '<none>'}")
    lines.append(f"git remote: {report['git_repo']['remote_summary'] or '<none>'}")
    lines.append(f"remote only mode: {report['remote_only']}")
    lines.append("")
    lines.append(f"gh auth: {report['github_cli']['state']}")
    if report["github_cli"]["detail"]:
        lines.append(f"  detail: {report['github_cli']['detail']}")
    lines.append(f"hf auth: {report['huggingface_cli']['state']}")
    lines.append(f"  HF_TOKEN env present: {report['huggingface_cli']['hf_token_env_present']}")
    if report["huggingface_cli"]["detail"]:
        lines.append(f"  detail: {report['huggingface_cli']['detail']}")
    if report["remote_only"]:
        lines.append("  remote-only guidance: prefer headless/device-code/token flows over GUI login")
    lines.append("")
    lines.append(f"Hermes home: {report['hermes']['hermes_home']}")
    model = report["hermes"]["model"]
    if model:
        lines.append("Hermes model block:")
        for key in sorted(model):
            lines.append(f"  {key}: {model[key]}")
    lines.append(f"copilot provider references: {report['hermes']['copilot_provider_references']}")
    lines.append(f"huggingface reference present: {report['hermes']['huggingface_reference_present']}")
    drift_jobs = report["hermes"]["drift_jobs"]
    lines.append(f"drift jobs: {len(drift_jobs)}")
    for job in drift_jobs[:5]:
        lines.append(
            f"  - id={job.get('id')} provider_snapshot={job.get('provider_snapshot')} "
            f"model_snapshot={job.get('model_snapshot')}"
        )
    lines.append("auth sources:")
    for provider in sorted(report["hermes"]["auth_sources"]):
        lines.append(f"  {provider}:")
        for entry in report["hermes"]["auth_sources"][provider]:
            lines.append(
                f"    - label={entry['label'] or '<none>'} "
                f"source={entry['source'] or '<none>'} "
                f"base_url={entry['base_url'] or '<none>'}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--hermes-home",
        default=str(DEFAULT_HERMES_HOME),
        help="Hermes home to inspect (default: ~/.hermes)",
    )
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="annotate the report for remote/headless-only operation",
    )
    args = parser.parse_args()

    report = collect_audit(Path(args.hermes_home).expanduser(), remote_only=args.remote_only)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
