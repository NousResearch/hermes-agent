from agent.eval_lab.reporting import render_markdown_report, write_markdown_report
from agent.eval_lab.schemas import EvalScore, TrajectoryAttempt, TrajectoryGroup


def _group():
    return TrajectoryGroup(
        group_id="group-1",
        scenario_id="scenario-1",
        attempts=[
            TrajectoryAttempt(
                attempt_id="a1",
                scenario_id="scenario-1",
                started_at="2026-05-25T10:00:00Z",
                finished_at="2026-05-25T10:00:01Z",
                status="completed",
                final_response="ok",
                steps=[],
                metadata={},
            )
        ],
    )


def test_render_markdown_report_includes_ranked_scores_and_paths():
    report = render_markdown_report(
        run_id="run-1",
        groups=[_group()],
        scores=[EvalScore(attempt_id="a1", total=0.75, criteria={"completed": 1.0}, notes=["ok"])],
        artifact_paths=["trajectory_groups.jsonl", "scores.jsonl"],
    )

    assert "# Hermes Eval Lab Run: run-1" in report
    assert "scenario-1" in report
    assert "a1" in report
    assert "0.7500" in report
    assert "trajectory_groups.jsonl" in report


def test_write_markdown_report_redacts_secret_like_output(tmp_path):
    group = TrajectoryGroup(
        group_id="group-secret",
        scenario_id="scenario-secret",
        attempts=[
            TrajectoryAttempt(
                attempt_id="a-secret",
                scenario_id="scenario-secret",
                started_at="2026-05-25T10:00:00Z",
                finished_at="2026-05-25T10:00:01Z",
                status="completed",
                final_response="Bearer abcdefghijklmnopqrstuvwxyz123456",
                steps=[],
                metadata={},
            )
        ],
    )

    path = write_markdown_report(
        output_path=tmp_path / "report.md",
        run_id="run-secret",
        groups=[group],
        scores=[],
        artifact_paths=[],
    )

    text = path.read_text(encoding="utf-8")
    assert "abcdefghijklmnopqrstuvwxyz123456" not in text
    assert "[REDACTED]" in text
