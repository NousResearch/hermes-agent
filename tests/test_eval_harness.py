import json
from pathlib import Path

from evals.harness import EvalHarness, EvalTask


def test_load_suite_parses_tasks(tmp_path):
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "suite": "smoke",
                "version": "v1",
                "holdout_task_ids": ["t2"],
                "tasks": [
                    {"task_id": "t1", "lane": "coding", "prompt": "Say OK", "expected": "OK"},
                    {"task_id": "t2", "lane": "agentic", "prompt": "Say PLAN", "expected": "PLAN"},
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = EvalHarness.load_suite(suite_file)
    assert suite.suite == "smoke"
    assert suite.version == "v1"
    assert len(suite.tasks) == 2
    assert suite.holdout_task_ids == ["t2"]


def test_run_suite_writes_report(tmp_path):
    harness = EvalHarness(suite_name="hermes-eval", commit="abc123")
    suite = EvalHarness.load_suite("evals/suites/smoke_v1.json")

    def runner(task: EvalTask) -> str:
        if task.task_id == "terminal-1":
            return "READY with shell-safe guidance"
        if task.task_id == "coding-1":
            return "FIXED"
        return "PLAN with two steps"

    report = harness.run_suite(suite, runner=runner, output_dir=tmp_path)

    assert report["summary"]["total"] == 3
    assert report["summary"]["passed"] == 3
    assert report["summary"]["pass_rate"] == 1.0

    report_file = Path(report["report_file"])
    assert report_file.exists()
    loaded = json.loads(report_file.read_text(encoding="utf-8"))
    assert loaded["commit"] == "abc123"
    assert loaded["by_lane"]["coding"]["avg_score"] == 1.0


def test_deterministic_grade_flags_missing_expected():
    task = EvalTask(task_id="t1", lane="coding", prompt="x", expected="PASS")
    passed, score, signature = EvalHarness.deterministic_grade(task, "output without marker")
    assert passed is False
    assert score == 0.0
    assert signature == "missing_expected:PASS"
