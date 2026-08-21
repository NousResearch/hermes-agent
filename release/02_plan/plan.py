"""Stage 02: deterministic release plan derived from 01_audit/audit.json.

Idempotent. No source mutations. No network.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = REPO_ROOT / "release" / "02_plan"
UPSTREAM = REPO_ROOT / "release" / "01_audit" / "audit.json"
ARTIFACT = STAGE_DIR / "plan.json"


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _latest_tag() -> str:
    try:
        return _git(["describe", "--tags", "--abbrev=0"])
    except subprocess.CalledProcessError:
        return "0.0.0"


def _version_candidate() -> str:
    explicit = os.environ.get("ICM_PLAN_VERSION")
    if explicit:
        return explicit
    base = _latest_tag()
    return f"{base}-audit"


def main() -> int:
    if not UPSTREAM.exists():
        payload = {
            "ok": False,
            "stage": "02_plan",
            "error": {"missing_upstream": str(UPSTREAM)},
        }
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    audit = json.loads(UPSTREAM.read_text())
    if not audit.get("ok"):
        payload = {
            "ok": False,
            "stage": "02_plan",
            "error": {"upstream_invalid": True},
        }
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    version_candidate = _version_candidate()
    changed = audit.get("changed_files", [])
    risk = audit.get("risk_flags", [])
    scope_summary = " | ".join(sorted(set(changed[:10]))) if changed else "no_changes"

    payload: dict[str, Any] = {
        "ok": True,
        "stage": "02_plan",
        "version_candidate": version_candidate,
        "scope_summary": scope_summary,
        "changed_count": len(changed),
        "risk_flags": risk,
        "validation_gates": [
            "stage_contracts_present",
            "deterministic_scripts_executable",
            "focused_tests_pass",
        ],
    }

    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
