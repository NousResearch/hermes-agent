"""Run the deterministic editor-shape audit and write a comparable JSON scorecard.

Usage: python3 evals/edittool/runner.py --label current-main
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from arms import run_arm
from fixtures import build_workspace
from tasks import TASKS


def _git_revision() -> str:
    """Return HEAD, marking a scorecard made from uncommitted sources as dirty."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return f"{head.stdout.strip()}-dirty" if dirty.stdout else head.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="result directory label")
    parser.add_argument("--output", type=Path, help="explicit output path")
    args = parser.parse_args()

    workspace = build_workspace()
    try:
        arms = {arm: run_arm(arm, workspace, TASKS) for arm in ("str_replace", "hermes_patch")}
    finally:
        workspace.cleanup()

    payload = {
        "schema_version": 2,
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(TASKS),
        "metrics": {
            "status": {"name": "passed", "version": 1},
            "artifact": {"name": "artifact_correct", "version": 1},
        },
        "provenance": {
            "measurement_commit": _git_revision(),
            "fixture_commit": _git_revision(),
            "matcher_commit": _git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "arms": arms,
    }
    output = args.output or EVAL_DIR / "results" / f"{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
