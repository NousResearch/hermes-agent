from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agent.intent_router import IntentClassification, SafetyFlag
from agent.orchestrator_interface import (
    OrchestratorDecision,
    OrchestratorInterface,
    TaskAssignment,
    TaskSpec,
)


def _intent(
    intent_type: str,
    *,
    confidence: float = 0.95,
    routing_strategy: str = "chat_only",
    safety_flags: list[SafetyFlag] | None = None,
) -> IntentClassification:
    return IntentClassification(
        intent_type=intent_type,
        confidence=confidence,
        routing_strategy=routing_strategy,
        safety_flags=safety_flags or [],
    )


def test_orchestrator_decision_to_dict_serializes_plan_and_assignments() -> None:
    decision = OrchestratorDecision(
        plan=[
            TaskSpec(
                task_id="task-1",
                description="Research target",
                assigned_profile="researcher",
                inputs={"query": "target"},
                expected_outputs=["summary"],
                dependencies=["task-0"],
                timeout_s=120,
                requires_user_input=True,
                approval_id="approval-1",
            )
        ],
        task_assignments=[
            TaskAssignment(
                task_id="task-1",
                profile="researcher",
                workspace="/tmp/workspace",
                assigned_at_utc="2026-07-05T00:00:00Z",
                status="assigned",
            )
        ],
        requires_approval=True,
        approval_prompt="Approve research?",
        rationale="test rationale",
        estimated_complexity="simple",
        orchestrator_model="phase2_stub_no_llm",
        decided_at_utc="2026-07-05T00:00:01Z",
    )

    data = decision.to_dict()

    assert data["plan"] == [
        {
            "task_id": "task-1",
            "description": "Research target",
            "assigned_profile": "researcher",
            "inputs": {"query": "target"},
            "expected_outputs": ["summary"],
            "dependencies": ["task-0"],
            "timeout_s": 120,
            "requires_user_input": True,
            "approval_id": "approval-1",
        }
    ]
    assert data["task_assignments"] == [
        {
            "task_id": "task-1",
            "profile": "researcher",
            "workspace": "/tmp/workspace",
            "assigned_at_utc": "2026-07-05T00:00:00Z",
            "status": "assigned",
        }
    ]
    assert data["requires_approval"] is True
    assert data["approval_prompt"] == "Approve research?"
    assert data["rationale"] == "test rationale"
    assert data["estimated_complexity"] == "simple"
    assert data["orchestrator_model"] == "phase2_stub_no_llm"
    assert data["decided_at_utc"] == "2026-07-05T00:00:01Z"


def test_orchestrate_chat_returns_trivial_chat_only_decision() -> None:
    decision = OrchestratorInterface().orchestrate(
        _intent("chat", confidence=0.95, routing_strategy="chat_only")
    )

    assert decision.plan == []
    assert decision.task_assignments == []
    assert decision.requires_approval is False
    assert decision.approval_prompt is None
    assert decision.estimated_complexity == "trivial"
    assert decision.orchestrator_model == "phase2_stub_no_llm"
    assert decision.decided_at_utc.endswith("Z")


def test_orchestrate_orchestration_intents_return_simple_empty_phase2_plan() -> None:
    for intent_type in ("delegate", "research", "code", "kanban"):
        decision = OrchestratorInterface().orchestrate(
            _intent(intent_type, confidence=0.8, routing_strategy="orchestrate")
        )

        assert decision.plan == []
        assert decision.task_assignments == []
        assert decision.requires_approval is False
        assert decision.estimated_complexity == "simple"


def test_orchestrate_composite_returns_moderate_empty_phase2_plan() -> None:
    decision = OrchestratorInterface().orchestrate(
        _intent("composite", confidence=0.8, routing_strategy="orchestrate")
    )

    assert decision.plan == []
    assert decision.task_assignments == []
    assert decision.requires_approval is False
    assert decision.estimated_complexity == "moderate"


def test_orchestrate_requires_approval_by_mode() -> None:
    decision = OrchestratorInterface().orchestrate(
        _intent("chat", routing_strategy="chat_only"),
        mode="approval_required",
    )

    assert decision.requires_approval is True
    assert decision.approval_prompt is not None


def test_orchestrate_requires_approval_by_routing_strategy() -> None:
    decision = OrchestratorInterface().orchestrate(
        _intent("chat", routing_strategy="approval_required")
    )

    assert decision.requires_approval is True
    assert decision.approval_prompt is not None


def test_orchestrate_requires_approval_by_intent_type() -> None:
    decision = OrchestratorInterface().orchestrate(
        _intent("approval", routing_strategy="chat_only")
    )

    assert decision.requires_approval is True
    assert decision.approval_prompt is not None


def test_orchestrate_requires_approval_by_critical_safety_flag() -> None:
    flag = SafetyFlag(
        flag_type="r7_protected",
        severity="critical",
        description="critical test flag",
        blocked=True,
    )

    decision = OrchestratorInterface().orchestrate(
        _intent("chat", routing_strategy="chat_only", safety_flags=[flag])
    )

    assert decision.requires_approval is True
    assert decision.approval_prompt is not None
    assert "r7_protected" in decision.approval_prompt


class _FailingPersistence:
    calls = 0

    def save_event(self, *args, **kwargs):  # pragma: no cover - should never run
        self.calls += 1
        raise AssertionError("OrchestratorInterface must not persist events")


def test_orchestrate_does_not_persist_orchestrator_decided_internally() -> None:
    persistence = _FailingPersistence()

    decision = OrchestratorInterface(conversation_persistence=persistence).orchestrate(
        _intent("research", routing_strategy="orchestrate"),
        conversation_context={"conversation_id": "conv-1"},
    )

    assert decision.estimated_complexity == "simple"
    assert persistence.calls == 0


class _RouterDouble:
    def route(self, message, conversation_context=None):
        return _intent("research", confidence=0.8, routing_strategy="orchestrate")


def test_conversation_api_with_real_orchestrator_persists_single_orchestrator_decided_event() -> None:
    from agent.conversation_api import ConversationAPI
    from agent.conversation_manager import ConversationManager
    from agent.conversation_persistence import ConversationPersistence

    tmpdir = Path(tempfile.mkdtemp(prefix="orchestrator-interface-test-"))
    try:
        persistence = ConversationPersistence(root=tmpdir)
        manager = ConversationManager(persistence=persistence)
        api = ConversationAPI(
            manager=manager,
            intent_router=_RouterDouble(),
            orchestrator_interface=OrchestratorInterface(
                conversation_persistence=persistence
            ),
        )

        result = api.receive(user_id="u", message="research x")
        events = persistence.list_events(result["conversation_id"])
        event_types = [event["event_type"] for event in events]

        assert event_types.count("orchestrator_decided") == 1
        assert result["dispatch_state"] == "ready_for_future_dispatch"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
