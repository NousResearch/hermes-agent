from __future__ import annotations

from pathlib import Path

import pytest

from agent.managed_agents.event_log import ManagedAgentEventLog
from agent.managed_agents.review_gate import (
    ReviewGate,
    ReviewSeverity,
    load_review_gate,
)
from agent.task_card import CompiledIntent, ExecutionPlan, TaskCard


def _write_review_rules(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "managed_agents"
    path.mkdir(parents=True, exist_ok=True)
    rules_path = path / "review_rules.yaml"
    rules_path.write_text(
        """
version: "2026-05-21"
review_required_when:
  - risk_level: [R2, R3, R4]
  - files_changed_gte: 3
  - touches: [configs/**, agent/managed_agents/**, .env*]
  - actions: [install_dependency, delete_file, git_operation, deploy, modify_permission]
reviewers:
  R1:
    optional: [codex]
  R2:
    required: [codex]
  R3:
    required: [codex, ambrosini]
  R4:
    required: [codex, ambrosini]
    requires_human_approval: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return rules_path


def _make_task(
    *,
    task_id: str = "task-1",
    risk_level: str = "R0",
    files_changed: int = 0,
    changed_files: list[str] | None = None,
    actions: list[str] | None = None,
    owner_agent: str = "claude",
) -> TaskCard:
    task = TaskCard(
        task_id=task_id,
        session_id="sess-1",
        raw_user_request="implement managed review gate",
        compiled_intent=CompiledIntent(real_task="implement managed review gate", task_category="feature"),
        execution_plan=ExecutionPlan(mode="single_agent", agents=[owner_agent], delegation_reason="route"),
        status="reviewing",
    )
    task.risk_level = risk_level
    task.files_changed = files_changed
    task.changed_files = changed_files or []
    task.actions = actions or []
    task.owner_agent = owner_agent
    return task


def _make_gate(tmp_path: Path) -> ReviewGate:
    return load_review_gate(_write_review_rules(tmp_path))


def test_r2_requires_codex_review(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(_make_task(risk_level="R2"))

    assert requirement.requires_review is True
    assert requirement.required_reviewers == ["codex"]
    assert "risk_level:R2,R3,R4" in requirement.matched_triggers


def test_r3_requires_codex_and_ambrosini(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(_make_task(risk_level="R3"))

    assert requirement.requires_review is True
    assert requirement.required_reviewers == ["codex", "ambrosini"]


def test_r4_requires_human_approval(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(_make_task(risk_level="R4"))

    assert requirement.requires_review is True
    assert requirement.requires_human_approval is True


def test_files_changed_gte_triggers_review(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(_make_task(files_changed=3))

    assert requirement.requires_review is True
    assert "files_changed_gte:3" in requirement.matched_triggers


def test_touches_triggers_review(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(
        _make_task(changed_files=["agent/managed_agents/review_gate.py"])
    )

    assert requirement.requires_review is True
    assert any(trigger.startswith("touches:") for trigger in requirement.matched_triggers)


def test_actions_trigger_review(tmp_path):
    gate = _make_gate(tmp_path)
    requirement = gate.build_requirement(_make_task(actions=["deploy"]))

    assert requirement.requires_review is True
    assert any(trigger.startswith("actions:") for trigger in requirement.matched_triggers)


def test_review_completion_writes_events_and_blocks_with_required_fixes(tmp_path):
    event_log = ManagedAgentEventLog.from_db_path(tmp_path / "events.db")
    gate = ReviewGate(rules=load_review_gate(_write_review_rules(tmp_path)).rules, event_log=event_log)
    task = _make_task(task_id="task-2", risk_level="R3")

    gate.request_review(task=task, reviewer="codex", session_id="sess-1", subject_agent="claude")
    result = gate.complete_review(
        task=task,
        reviewer="codex",
        session_id="sess-1",
        decision="pass_with_notes",
        severity=ReviewSeverity(p0=0, p1=0, p2=2, p3=0),
        summary="looks good",
        required_fixes=["fix event replay"],
        optional_fixes=["add more tests"],
        approvers=["codex", "ambrosini"],
    )

    events = event_log.replay_task("task-2")
    assert [event["type"] for event in events] == ["review_requested", "review_completed"]
    assert result.to_dict()["severity"] == {"p0": 0, "p1": 0, "p2": 2, "p3": 0}
    assert gate.is_blocked(result) is True
    assert gate.can_close(result) is False


def test_executer_cannot_be_unique_approver(tmp_path):
    gate = _make_gate(tmp_path)
    task = _make_task(task_id="task-3", owner_agent="claude")
    result = gate.complete_review(
        task=task,
        reviewer="codex",
        session_id="sess-1",
        decision="pass",
        severity=ReviewSeverity(),
        approvers=["claude"],
        task_executor="claude",
    )

    assert gate.can_close(result) is False


def test_review_result_is_serializable(tmp_path):
    gate = _make_gate(tmp_path)
    task = _make_task(task_id="task-4")
    result = gate.complete_review(
        task=task,
        reviewer="codex",
        session_id="sess-1",
        decision="pass",
        severity=ReviewSeverity(p2=1),
        summary="pass with notes",
        required_fixes=[],
        optional_fixes=["tune copy"],
        approvers=["codex", "ambrosini"],
        task_executor="claude",
    )

    payload = result.to_dict()
    assert payload["severity"]["p2"] == 1
    assert payload["optional_fixes"] == ["tune copy"]

