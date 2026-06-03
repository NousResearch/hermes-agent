import json

from agent.eval_lab.schemas import EvalScore, TrajectoryAttempt, TrajectoryGroup
from agent.eval_lab.storage import EvalRunStorage


def _group():
    attempt = TrajectoryAttempt(
        attempt_id="attempt-1",
        scenario_id="scenario-1",
        started_at="2026-05-25T10:00:00Z",
        finished_at="2026-05-25T10:00:01Z",
        status="completed",
        final_response="ok",
        steps=[],
        metadata={},
    )
    return TrajectoryGroup(group_id="group-1", scenario_id="scenario-1", attempts=[attempt])


def test_storage_writes_replayable_jsonl_artifacts(tmp_path):
    storage = EvalRunStorage(run_id="run-1", base_dir=tmp_path)
    group = _group()
    score = EvalScore(attempt_id="attempt-1", total=1.0, criteria={"ok": 1.0}, notes=[])

    storage.write_group(group)
    storage.write_score(score)

    run_dir = tmp_path / "run-1"
    attempts_lines = (run_dir / "trajectory_groups.jsonl").read_text(encoding="utf-8").splitlines()
    scores_lines = (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()

    assert json.loads(attempts_lines[0])["group_id"] == "group-1"
    assert json.loads(scores_lines[0])["attempt_id"] == "attempt-1"
    assert storage.load_groups() == [group]
    assert storage.load_scores() == [score]


def test_storage_redacts_before_writing(tmp_path):
    attempt = TrajectoryAttempt(
        attempt_id="attempt-secret",
        scenario_id="scenario-secret",
        started_at="2026-05-25T10:00:00Z",
        finished_at="2026-05-25T10:00:01Z",
        status="completed",
        final_response="Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        steps=[],
        metadata={"api_key": "sk-live-secret"},
    )
    storage = EvalRunStorage(run_id="run-secret", base_dir=tmp_path)

    storage.write_group(TrajectoryGroup(group_id="group-secret", scenario_id="scenario-secret", attempts=[attempt]))

    written = (tmp_path / "run-secret" / "trajectory_groups.jsonl").read_text(encoding="utf-8")
    assert "sk-live-secret" not in written
    assert "abcdefghijklmnopqrstuvwxyz123456" not in written
    assert "[REDACTED]" in written
