import json

import pytest

from agent.exploration_policy import (
    ExplorationBranch,
    ExplorationCost,
    ExplorationDecision,
    ExplorationEvent,
    ExplorationTree,
    PolicyArtifact,
    RecordedExplorationPolicy,
    build_exploration_tree,
    replay_policy,
)
from agent.trajectory import save_trajectory


class FixedPolicy:
    def __init__(self, action):
        self.version = f"fixed-{action}"
        self.action = action

    def decide(self, state):
        return ExplorationDecision(self.action)


def _tree_with_observed_alternatives():
    event = ExplorationEvent(
        decision_id="root",
        parent_id=None,
        policy_version="baseline-v1",
        state_digest="digest-root",
        available_actions=("continue", "stop", "retry"),
        chosen_action="continue",
        branches=(
            ExplorationBranch(action="continue", child_ids=("child",), score=0.5),
            ExplorationBranch(action="stop", terminal=True, score=1.0),
        ),
    )
    return ExplorationTree(
        tree_id="tree-1",
        policy_version="baseline-v1",
        events=(event,),
        completed=True,
    )


def test_replay_reports_score_and_coverage_without_model_calls():
    tree = _tree_with_observed_alternatives()

    baseline = replay_policy(tree, RecordedExplorationPolicy(tree))
    assert baseline.supported_decisions == 1
    assert baseline.out_of_support_decisions == 0
    assert baseline.coverage == 1.0
    assert baseline.mean_score == 0.5

    observed_alternative = replay_policy(tree, FixedPolicy("stop"))
    assert observed_alternative.supported_decisions == 1
    assert observed_alternative.coverage == 1.0
    assert observed_alternative.mean_score == 1.0

    unsupported = replay_policy(tree, FixedPolicy("retry"))
    assert unsupported.supported_decisions == 0
    assert unsupported.out_of_support_decisions == 1
    assert unsupported.mean_score is None


def test_available_but_unobserved_action_is_out_of_support():
    report = replay_policy(_tree_with_observed_alternatives(), FixedPolicy("retry"))
    assert report.to_dict()["coverage"] == 0.0
    assert report.to_dict()["out_of_support_decisions"] == 1


def test_tree_round_trip_is_versioned_and_secret_free():
    secret = "sk-live-do-not-copy"
    trajectory = [
        {"from": "human", "value": f"Use {secret}"},
        {"from": "gpt", "value": '<tool_call>\n{"name":"search","arguments":{}}\n</tool_call>'},
        {"from": "tool", "value": f"result contains {secret}"},
        {"from": "gpt", "value": "finished"},
    ]

    tree = build_exploration_tree(trajectory, completed=True)
    encoded = json.dumps(tree.to_dict(), ensure_ascii=False)
    restored = ExplorationTree.from_dict(tree.to_dict())

    assert tree.schema_version == 1
    assert tree.to_dict() == restored.to_dict()
    assert secret not in encoded
    assert tree.events[0].chosen_action == "continue"
    assert tree.events[1].chosen_action == "stop"
    assert tree.events[0].branches[0].cost.tool_calls == 1


def test_save_trajectory_emits_a_separate_exploration_tree(tmp_path):
    trajectory_path = tmp_path / "trajectory.jsonl"
    tree_path = tmp_path / "trees.jsonl"
    save_trajectory(
        [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "done"}],
        model="test-model",
        completed=True,
        filename=str(trajectory_path),
        exploration_filename=str(tree_path),
    )

    assert trajectory_path.exists()
    assert tree_path.exists()
    saved_tree = json.loads(tree_path.read_text(encoding="utf-8"))
    assert saved_tree["schema_version"] == 1
    assert saved_tree["events"][0]["chosen_action"] == "stop"


def test_policy_artifact_is_inspectable_and_staged():
    artifact = PolicyArtifact(
        policy_version="candidate-2",
        parent_version="baseline-v1",
        policy_name="bounded-stop-policy",
        status="shadow",
        provenance={"source": "offline-replay"},
        replay_metrics={"coverage": 1.0, "mean_score": 0.8},
    )
    encoded = artifact.to_dict()
    assert encoded["status"] == "shadow"
    assert "delegate" in encoded["allowed_actions"]
    assert encoded["replay_metrics"]["mean_score"] == 0.8

    with pytest.raises(ValueError, match="unsupported policy status"):
        PolicyArtifact(policy_version="x", parent_version=None, policy_name="x", status="live")
