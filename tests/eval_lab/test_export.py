import json

from agent.eval_lab.export import build_preference_pairs, export_rl_ready_dataset
from agent.eval_lab.schemas import EvalScore, TrajectoryAttempt, TrajectoryGroup


def _group():
    return TrajectoryGroup(
        group_id="group-1",
        scenario_id="scenario-1",
        attempts=[
            TrajectoryAttempt(
                attempt_id="good",
                scenario_id="scenario-1",
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:00:01Z",
                status="completed",
                final_response="Safe answer",
            ),
            TrajectoryAttempt(
                attempt_id="bad",
                scenario_id="scenario-1",
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:00:01Z",
                status="completed",
                final_response="Bad answer Bearer abcdefghijklmnop",
            ),
        ],
    )


def test_build_preference_pairs_uses_best_and_worst_attempts_with_redaction():
    pairs = build_preference_pairs(
        [_group()],
        [EvalScore(attempt_id="good", total=1.0), EvalScore(attempt_id="bad", total=0.2)],
        prompts_by_scenario={"scenario-1": "Prompt Bearer abcdefghijklmnop"},
    )

    assert pairs == [
        {
            "scenario_id": "scenario-1",
            "prompt": "Prompt [REDACTED]",
            "chosen": "Safe answer",
            "rejected": "Bad answer [REDACTED]",
            "chosen_attempt_id": "good",
            "rejected_attempt_id": "bad",
        }
    ]


def test_export_rl_ready_dataset_writes_sft_pairs_and_groups(tmp_path):
    output = export_rl_ready_dataset(
        tmp_path,
        groups=[_group()],
        scores=[EvalScore(attempt_id="good", total=1.0), EvalScore(attempt_id="bad", total=0.2)],
        prompts_by_scenario={"scenario-1": "Prompt"},
    )

    assert {path.name for path in output} == {"sft.jsonl", "preference_pairs.jsonl", "trajectory_groups.jsonl"}
    sft_rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert sft_rows == [{"scenario_id": "scenario-1", "prompt": "Prompt", "response": "Safe answer", "attempt_id": "good"}]
    assert "Bearer" not in (tmp_path / "trajectory_groups.jsonl").read_text()
