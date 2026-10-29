"""Captured turn completion is an observation, not a task acceptance verdict."""

import json

import pytest

from agent.exploration_policy import ExplorationTree, RecordedExplorationPolicy, replay_policy
from agent.trajectory import save_trajectory


@pytest.mark.parametrize(
    ("completed", "response"),
    [
        (True, "Blocked: the required credentials are unavailable."),
        (True, "The task is incomplete; validation still needs to run."),
        (True, "All requested work is complete."),
        (False, "Interrupted before checking the result."),
    ],
)
def test_saved_turn_status_does_not_invent_branch_scores(tmp_path, completed, response):
    primary = tmp_path / "trajectory.jsonl"
    sidecar = tmp_path / "tree.jsonl"
    trajectory = [
        {"from": "human", "value": "Complete and validate the task."},
        {"from": "gpt", "value": '<tool_call>{"name":"terminal"}</tool_call>'},
        {"from": "tool", "value": "The command returned."},
        {"from": "gpt", "value": response},
    ]

    save_trajectory(
        trajectory,
        model="test-model",
        completed=completed,
        filename=str(primary),
        exploration_filename=str(sidecar),
    )

    tree = ExplorationTree.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    assert json.loads(primary.read_text(encoding="utf-8"))["completed"] is completed
    assert tree.completed is completed
    assert all(branch.score is None for event in tree.events for branch in event.branches)

    report = replay_policy(tree, RecordedExplorationPolicy(tree))
    assert report.supported_decisions == len(tree.events)
    assert report.scored_decisions == 0
    assert report.mean_score is None
