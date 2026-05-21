from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agent.managed_agents.kanban_bridge import (
    KanbanBridgeCard,
    build_kanban_bridge,
    block_card,
    complete_review,
    create_card,
    create_card_from_task,
    deliver_card,
    delegate_card,
    fail_card,
    plan_card,
    request_review,
    resume_work,
    serialize_card,
    should_auto_create_card,
    start_work,
)
from agent.managed_agents.review_gate import ReviewRequirement, ReviewResult, ReviewSeverity


@dataclass
class FakeTask:
    task_id: str = "task-1"
    title: str = "task"
    raw_user_request: str = "task"
    owner_agent: str = "claude"
    steps: int = 2
    agents: list[str] = None
    risk: str = "R1"
    needs_review: bool = False
    requires_cross_session_recovery: bool = False
    changes_include_code: bool = False
    changes_include_test: bool = False
    changes_include_acceptance: bool = False

    def __post_init__(self):
        if self.agents is None:
            self.agents = ["claude"]


def test_should_auto_create_card_triggers_on_steps_agents_risk_review_recovery_and_complexity():
    task = FakeTask()
    assert should_auto_create_card(task) is False

    task.steps = 4
    assert should_auto_create_card(task) is True

    task.steps = 1
    task.agents = ["a", "b"]
    assert should_auto_create_card(task) is True

    task.agents = ["a"]
    task.risk = "R2"
    assert should_auto_create_card(task) is True

    task.risk = "R1"
    task.needs_review = True
    assert should_auto_create_card(task) is True

    task.needs_review = False
    task.requires_cross_session_recovery = True
    assert should_auto_create_card(task) is True

    task.requires_cross_session_recovery = False
    task.changes_include_code = True
    task.changes_include_test = True
    task.changes_include_acceptance = True
    assert should_auto_create_card(task) is True


def test_should_auto_create_card_reads_task_card_execution_plan_agents():
    task = FakeTask(agents=[])
    task.execution_plan = type("Plan", (), {"agents": ["claude", "deepseek-tui"]})()

    assert should_auto_create_card(task) is True


def test_state_machine_transitions():
    card = create_card("task-1")
    assert card.state == "created"
    plan_card(card)
    delegate_card(card, "agent-team")
    start_work(card)
    request_review(card, ["reviewer"])
    assert card.state == "review_pending"
    complete_review(
        card,
        ReviewResult(
            task_id="task-1",
            reviewer="codex",
            decision="changes_requested",
            severity=ReviewSeverity(),
        ),
    )
    assert card.state == "changes_requested"
    resume_work(card)
    assert card.state == "in_progress"
    request_review(card, ["reviewer"])
    complete_review(
        card,
        ReviewResult(
            task_id="task-1",
            reviewer="codex",
            decision="approved",
            severity=ReviewSeverity(),
        ),
    )
    deliver_card(card)
    assert card.state == "done"


def test_blocked_and_failed_require_reasons():
    card = create_card("task-1")
    block_card(card, reason="External dependency missing")
    assert card.state == "blocked"
    assert card.blocked_reason == "External dependency missing"
    fail_card(card, reason="Build failed")
    assert card.state == "failed"
    assert card.failure_reason == "Build failed"


def test_invalid_transition_raises():
    card = create_card("task-1")
    with pytest.raises(ValueError):
        delegate_card(card, "team")


def test_serialize_card_is_json_safe():
    card = create_card("task-1")
    plan_card(card)
    delegate_card(card, "agent-team")
    payload = serialize_card(card)
    assert payload["state"] == "delegated"
    json.dumps(payload)


def test_create_card_from_task_uses_task_fields():
    task = FakeTask(task_id="task-9", raw_user_request="build")
    card = create_card_from_task(task)
    assert card.card_id == "task-9"
    assert card.metadata["source_task"] == "task-9"


def test_runtime_bridge_walks_states():
    task = FakeTask(task_id="task-10", steps=4)
    runtime = build_kanban_bridge(task)
    assert runtime.state == "created"
    runtime.plan()
    runtime.delegate("agent-team")
    runtime.start()
    runtime.request_review(["reviewer"])
    runtime.apply_review_result(
        ReviewResult(
            task_id="task-10",
            reviewer="codex",
            decision="approved",
            severity=ReviewSeverity(),
        )
    )
    runtime.complete()
    assert runtime.state == "done"


def test_runtime_sync_execution_plan_covers_single_agent_and_pipeline():
    task = FakeTask(task_id="task-11")
    runtime = build_kanban_bridge(task)

    single = type("Task", (), {})()
    single.execution_plan = type("Plan", (), {"mode": "single_agent", "agents": ["claude"]})()
    runtime.sync_execution_plan(single)
    assert runtime.state == "in_progress"
    assert runtime.card.assignee == "claude"

    pipeline_runtime = build_kanban_bridge(FakeTask(task_id="task-12"))
    pipeline = type("Task", (), {})()
    pipeline.execution_plan = type("Plan", (), {"mode": "pipeline", "agents": ["claude", "deepseek-tui"]})()
    pipeline_runtime.sync_execution_plan(pipeline)
    assert pipeline_runtime.state == "in_progress"
    assert pipeline_runtime.card.assignee == "claude"


def test_runtime_changes_requested_can_resume_and_review_again():
    runtime = build_kanban_bridge(FakeTask(task_id="task-13"))
    runtime.delegate("claude")
    runtime.start()
    runtime.request_review(["codex"])
    runtime.apply_review_result(
        ReviewResult(
            task_id="task-13",
            reviewer="codex",
            decision="changes_requested",
            severity=ReviewSeverity(),
        )
    )

    resume_work(runtime.card)
    runtime.request_review(["codex"])
    runtime.apply_review_result(
        ReviewResult(
            task_id="task-13",
            reviewer="codex",
            decision="approved",
            severity=ReviewSeverity(),
        )
    )

    assert runtime.complete().state == "done"


def test_runtime_terminal_transitions_do_not_raise_from_reasonable_states():
    approved = build_kanban_bridge(FakeTask(task_id="task-14"))
    approved.delegate("claude")
    approved.start()
    assert approved.approve().state == "done"

    failed = build_kanban_bridge(FakeTask(task_id="task-15"))
    assert failed.fail("Build failed").state == "failed"
    assert failed.card.failure_reason == "Build failed"

    blocked = build_kanban_bridge(FakeTask(task_id="task-16"))
    assert blocked.block("Waiting for dependency").state == "blocked"
    assert blocked.card.blocked_reason == "Waiting for dependency"
