from pathlib import Path

import pytest

from agent.managed_agents.policy import (
    PolicyDecision,
    PolicyEngine,
    PolicyEngineError,
    load_policy_engine,
)
from agent.managed_agents.registry import RiskLevel


def test_safety_beats_everything_else(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
  - soul_global_policy
  - managed_agents_policy
  - router_policy
  - skill_policy
  - agent_preference
rules:
  - id: safety_block
    when:
      action_type: delete_file
    decision: deny
    reason: safety
  - id: user_override
    when:
      user_override: claude
    decision: allow
    reason: user
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = load_policy_engine(path)
    decision = engine.evaluate(
        {
            "task_id": "T1",
            "risk_level": "R4",
            "action_type": "delete_file",
            "user_override": "claude",
        }
    )

    assert decision.outcome == "deny"
    assert decision.winner == "safety"
    assert decision.requires_human_approval is True


def test_soul_beats_skill(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
  - soul_global_policy
  - managed_agents_policy
  - router_policy
  - skill_policy
  - agent_preference
rules:
  - id: soul_rule
    when:
      source: soul
    decision: allow
    reason: soul
  - id: skill_rule
    when:
      source: skill
    decision: deny
    reason: skill
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = load_policy_engine(path)
    decision = engine.evaluate({"task_id": "T1", "source": "soul", "risk_level": "R1"})

    assert decision.outcome == "allow"
    assert decision.winner == "soul_global_policy"
    assert decision.record["matched_rule"] == "soul_rule"


def test_r4_requires_human_approval_and_plan(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
  - soul_global_policy
  - managed_agents_policy
  - router_policy
  - skill_policy
  - agent_preference
rules: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = load_policy_engine(path)
    decision = engine.evaluate({"task_id": "T1", "risk_level": "R4"})

    assert decision.requires_human_approval is True
    assert decision.requires_plan is True
    assert decision.requires_review is True
    assert decision.outcome == "allow"


def test_decision_serializes_to_record(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
  - soul_global_policy
  - managed_agents_policy
  - router_policy
  - skill_policy
  - agent_preference
rules: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    engine = load_policy_engine(path)
    decision = engine.evaluate({"task_id": "T1", "risk_level": "R2"})

    assert decision.record["task_id"] == "T1"
    assert decision.record["risk_level"] == "R2"
    assert decision.record["outcome"] == "allow"


def test_invalid_policy_priority_fails(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
version: "2026-05-21"
priority_order:
  - safety
  - user_explicit_instruction
rules: []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyEngineError, match="priority_order"):
        load_policy_engine(path)

