#!/usr/bin/env python3
"""Capy RL readiness and smoke-test helper.

This script is intentionally credential-safe: it reports only whether required
variables are present, never their values.  When credentials are missing it exits
0 with status "not_ready" so health dashboards can display a non-fatal blocker.
When credentials are present, --smoke runs only setup/list checks; it never starts
training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("TINKER_API_KEY", "WANDB_API_KEY")
OPTIONAL_KEYS = ("OPENROUTER_API_KEY",)


def _load_dotenv() -> list[str]:
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.env_loader import load_hermes_dotenv

        project_env = Path(__file__).resolve().parents[1] / ".env"
        return [str(p) for p in load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=project_env)]
    except Exception:
        return []


def _present(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def collect_status() -> dict[str, Any]:
    loaded_env_paths = _load_dotenv()
    missing_required = [name for name in REQUIRED_KEYS if not _present(name)]
    missing_optional = [name for name in OPTIONAL_KEYS if not _present(name)]

    repo = Path(__file__).resolve().parents[1]
    tinker_dir = repo / "tinker-atropos"
    env_dir = tinker_dir / "tinker_atropos" / "environments"

    environments: list[str] = []
    if env_dir.is_dir():
        environments = sorted(p.stem for p in env_dir.glob("*.py") if not p.name.startswith("__"))

    return {
        "status": "ready" if not missing_required else "not_ready",
        "required": {name: _present(name) for name in REQUIRED_KEYS},
        "optional": {name: _present(name) for name in OPTIONAL_KEYS},
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "loaded_env_files": loaded_env_paths,
        "tinker_atropos_dir": str(tinker_dir),
        "tinker_atropos_present": tinker_dir.is_dir(),
        "environment_count": len(environments),
        "environments_sample": environments[:12],
        "notes": [
            "No credential values are printed.",
            "Smoke checks do not start RL training.",
        ],
    }


def _run_checked(cmd: list[str], cwd: Path, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = proc.stdout or ""
        # Defensive redaction in case any dependency logs env-like material.
        for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
            val = os.environ.get(key)
            if val:
                output = output.replace(val, "[REDACTED]")
        return {"cmd": cmd, "exit_code": proc.returncode, "output_tail": output[-4000:]}
    except Exception as exc:
        return {"cmd": cmd, "exit_code": 1, "error": str(exc)}


def run_smoke() -> dict[str, Any]:
    status = collect_status()
    if status["status"] != "ready":
        status["smoke"] = {
            "skipped": True,
            "reason": "missing_required_credentials",
            "missing_required": status["missing_required"],
        }
        return status

    repo = Path(__file__).resolve().parents[1]
    wrapper = Path.home() / ".local" / "bin" / "hermes-rl"
    cmd_base = [str(wrapper)] if wrapper.exists() else [sys.executable, "rl_cli.py"]
    checks = [
        _run_checked([*cmd_base, "--check-server"], repo),
        _run_checked([*cmd_base, "--list-environments"], repo),
    ]
    status["smoke"] = {
        "skipped": False,
        "checks": checks,
        "passed": all(item.get("exit_code") == 0 for item in checks),
    }
    if not status["smoke"]["passed"]:
        status["status"] = "smoke_failed"
    return status


def render_text(status: dict[str, Any]) -> str:
    lines = ["Capy RL readiness", "==================", f"Status: {status['status']}"]
    lines.append("Required credentials:")
    for name, present in status["required"].items():
        lines.append(f"  - {name}: {'present' if present else 'missing'}")
    lines.append("Optional credentials:")
    for name, present in status["optional"].items():
        lines.append(f"  - {name}: {'present' if present else 'missing'}")
    lines.append(f"tinker-atropos: {'present' if status['tinker_atropos_present'] else 'missing'} ({status['tinker_atropos_dir']})")
    lines.append(f"environment_count: {status['environment_count']}")
    if status.get("environments_sample"):
        lines.append("sample: " + ", ".join(status["environments_sample"]))
    if status.get("smoke"):
        smoke = status["smoke"]
        if smoke.get("skipped"):
            lines.append("Smoke: skipped (missing required credentials)")
        else:
            lines.append(f"Smoke: {'passed' if smoke.get('passed') else 'failed'}")
            for check in smoke.get("checks", []):
                lines.append(f"  - {' '.join(check.get('cmd', []))}: exit {check.get('exit_code')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--smoke", action="store_true", help="Run non-training smoke checks when required credentials are present")
    args = parser.parse_args()

    status = run_smoke() if args.smoke else collect_status()
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(render_text(status))
    return 0 if status["status"] in {"ready", "not_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
