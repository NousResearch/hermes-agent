"""Stage 05: deterministic ship manifest assembly.

Idempotent. No source mutations. No network. Does not push or publish.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = REPO_ROOT / "release" / "05_ship"
PLAN = REPO_ROOT / "release" / "02_plan" / "plan.json"
VALIDATION = REPO_ROOT / "release" / "03_validate" / "validation.json"
CHANGELOG = REPO_ROOT / "release" / "04_communicate" / "changelog.md"
NOTES = REPO_ROOT / "release" / "04_communicate" / "notes.md"
ARTIFACT = STAGE_DIR / "ship_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    for path in [PLAN, VALIDATION, CHANGELOG, NOTES]:
        if not path.exists():
            payload = {
                "ok": False,
                "stage": "05_ship",
                "error": {"missing": _rel(path)},
            }
            ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True))
            return 1

    validation = _read_json(VALIDATION)
    if not validation.get("ok"):
        payload = {
            "ok": False,
            "stage": "05_ship",
            "error": {"validation_failed": True},
        }
        ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    plan = _read_json(PLAN)
    payload: dict[str, Any] = {
        "ok": True,
        "stage": "05_ship",
        "version_candidate": plan.get("version_candidate"),
        "artifacts": {
            "plan": _rel(PLAN),
            "validation": _rel(VALIDATION),
            "changelog": _rel(CHANGELOG),
            "notes": _rel(NOTES),
        },
        "warnings": ["publish_gate_external: only proceed after explicit human approval outside this scaffold"],
    }

    ARTIFACT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
