"""Stage 01: deterministic repo audit.

Reads git state and writes release/01_audit/audit.json.

Idempotent. No source mutations. No network.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = REPO_ROOT / "release" / "01_audit"
ARTIFACT = STAGE_DIR / "audit.json"


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _changed_paths(base: str) -> list[str]:
    try:
        out = _git(["diff", "--name-only", base])
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.splitlines() if p]


def _classify_risk(paths: list[str]) -> list[str]:
    flags: list[str] = []
    for path in paths:
        if any(path.startswith(prefix) for prefix in ["scripts/release", "pyproject.toml", "setup.py"]):
            flags.append("release_meta")
        if path.endswith("AGENTS.md") or path.endswith("CONTRIBUTING.md"):
            flags.append("policy_doc")
        if path.startswith("gateway/") or path.startswith("run_agent.py"):
            flags.append("core_runtime")
        if path.startswith("tests/"):
            flags.append("test_surface")
    return sorted(set(flags))


def _language_stats(paths: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        suffix = Path(path).suffix.lower()
        lang = {
            ".py": "python",
            ".ts": "typescript",
            ".js": "javascript",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".json": "json",
            ".sh": "shell",
        }.get(suffix, "other")
        counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    base = os.environ.get("ICM_AUDIT_REF") or "HEAD"
    try:
        head_commit = _git(["rev-parse", "HEAD"])
        head_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "detached"
    except subprocess.CalledProcessError as exc:
        payload = {
            "ok": False,
            "stage": "01_audit",
            "error": {"command_failed": str(exc)},
        }
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    changed = _changed_paths(base)
    risk = _classify_risk(changed)

    payload: dict[str, Any] = {
        "ok": True,
        "stage": "01_audit",
        "base_ref": base,
        "head_commit": head_commit,
        "head_branch": head_branch,
        "dirty": len(changed) > 0,
        "changed_count": len(changed),
        "changed_files": changed,
        "changed_langs": _language_stats(changed),
        "risk_flags": risk,
    }

    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
