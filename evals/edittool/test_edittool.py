"""Focused tripwire for the edit-tool shape audit.

Run directly so this remains a lightweight eval check rather than a production
test-suite dependency:
    python3 evals/edittool/test_edittool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO_ROOT))

from arms import run_arm  # noqa: E402
from fixtures import build_workspace  # noqa: E402
from tasks import TASKS  # noqa: E402


def test_edit_tool_shape_audit() -> None:
    workspace = build_workspace()
    try:
        strict = run_arm("str_replace", workspace, TASKS)
        hermes = run_arm("hermes_patch", workspace, TASKS)

        assert {record["task_id"] for record in strict} == {
            task.task_id for task in TASKS
        }
        assert strict[0]["outcome"] == "applied"
        assert hermes[0]["outcome"] == "applied"
        assert strict[1]["outcome"] == "rejected"
        assert hermes[1]["outcome"] == "applied"
        assert hermes[-1]["passed"] is False
        assert all("reason" in record for record in hermes)
    finally:
        workspace.cleanup()

    print("edit-tool tripwire: ALL PASS")


if __name__ == "__main__":
    test_edit_tool_shape_audit()
