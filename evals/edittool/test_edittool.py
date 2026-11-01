"""Focused tripwire for the edit-tool shape audit.

Run directly so this remains a lightweight eval check rather than a production
test-suite dependency:
    python3 evals/edittool/test_edittool.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
        hermes_by_task = {record["task_id"]: record for record in hermes}
        wrong_anchor = hermes_by_task["missing_anchor"]
        assert wrong_anchor["passed"] is False
        assert wrong_anchor["status_matches_expected"] is False
        assert wrong_anchor["artifact_correct"] is False
        partial = hermes_by_task["partial_multi_hunk"]
        assert partial["outcome"] == "rejected"
        assert partial["edits_applied"] == 1
        assert partial["partial_write"] is True
        assert partial["artifact_correct"] is False
        assert all("reason" in record for record in hermes)
    finally:
        workspace.cleanup()

    print("edit-tool tripwire: ALL PASS")


def test_report_reads_legacy_status_scorecard() -> None:
    """A pre-v2 scorecard remains readable without inventing an artifact score."""
    payload = {
        "label": "legacy",
        "task_count": 1,
        "arms": {
            "str_replace": [
                {
                    "task_id": "legacy_task",
                    "outcome": "rejected",
                    "expected": "rejected",
                    "passed": True,
                    "reason": "legacy status-only record",
                }
            ]
        },
    }
    with tempfile.TemporaryDirectory(prefix="edittool-legacy-") as directory:
        result_path = Path(directory) / "legacy.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(EVAL_DIR / "report.py"), str(result_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    assert "artifact_score[artifact_correct/v1]=N/A (pre-v2)" in result.stdout


def test_runner_and_report_expose_v2_artifact_results() -> None:
    """The v2 CLI keeps artifact failures, partial writes, and dirty provenance visible."""
    with tempfile.TemporaryDirectory(prefix="edittool-v2-") as directory:
        result_path = Path(directory) / "v2.json"
        run = subprocess.run(
            [
                sys.executable,
                str(EVAL_DIR / "runner.py"),
                "--label",
                "v2-test",
                "--output",
                str(result_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        scorecard = json.loads(result_path.read_text(encoding="utf-8"))
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout != ""
        expected_revision = f"{head}-dirty" if dirty else head
        assert scorecard["provenance"]["measurement_commit"] == expected_revision
        report = subprocess.run(
            [sys.executable, str(EVAL_DIR / "report.py"), str(result_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert report.returncode == 0, report.stderr
    assert "artifact_score[artifact_correct/v1]=" in report.stdout
    assert "WRONG_ARTIFACT" in report.stdout
    assert "partial write" in report.stdout


if __name__ == "__main__":
    test_edit_tool_shape_audit()
    test_report_reads_legacy_status_scorecard()
    test_runner_and_report_expose_v2_artifact_results()
