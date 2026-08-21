"""Stage 03: deterministic validation of release scaffold contracts.

Idempotent. No source mutations. No network.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = REPO_ROOT / "release" / "03_validate"
UPSTREAM = REPO_ROOT / "release" / "02_plan" / "plan.json"
ARTIFACT = STAGE_DIR / "validation.json"


def _require_upstream_ok() -> dict[str, object]:
    if not UPSTREAM.exists():
        raise FileNotFoundError(str(UPSTREAM))
    payload = json.loads(UPSTREAM.read_text())
    if not payload.get("ok"):
        raise RuntimeError("upstream 02_plan is not ok")
    return payload


def _stage_contracts_ok() -> bool:
    expected = ["01_audit", "02_plan", "03_validate", "04_communicate", "05_ship"]
    for name in expected:
        context = REPO_ROOT / "release" / name / "CONTEXT.md"
        if not context.exists():
            return False
    return True


def _scripts_executable() -> bool:
    for name in ["01_audit", "02_plan", "03_validate", "04_communicate", "05_ship"]:
        run = REPO_ROOT / "release" / name / "run.sh"
        if not run.exists() or not run.stat().st_size:
            return False
    return True


def _artifacts_ok() -> bool:
    artifacts = [
        REPO_ROOT / "release" / "01_audit" / "audit.json",
        REPO_ROOT / "release" / "02_plan" / "plan.json",
    ]
    for path in artifacts:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return False
        if not isinstance(payload, dict) or not payload.get("ok"):
            return False
    return True


def main() -> int:
    try:
        _require_upstream_ok()
    except Exception as exc:
        payload = {
            "ok": False,
            "stage": "03_validate",
            "error": {"upstream": str(exc)},
        }
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    checks = {
        "stage_contracts_present": _stage_contracts_ok(),
        "deterministic_scripts_executable": _scripts_executable(),
        "upstream_artifacts_ok": _artifacts_ok(),
    }
    ok = bool(checks["stage_contracts_present"] and checks["deterministic_scripts_executable"] and checks["upstream_artifacts_ok"])

    payload: dict[str, object] = {
        "ok": ok,
        "stage": "03_validate",
        "checks": checks,
    }

    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
