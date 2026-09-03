"""Discord specialist work enters the durable orchestration lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature, RegistryResolution
from gateway.specialist_handoff import HandoffSource, create_specialist_handoff
from gateway.specialist_routing import (
    RouteKind,
    SpecialistRouteDecision,
    parse_specialist_response,
)
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for profile in (
        "task-orchestrator",
        "burndown-patch-steward",
        "market-data-authority-auditor",
    ):
        (home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_specialist_handoff_creates_goal_mode_triage_root(kanban_home):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="burndown-patch-steward",
        confidence=1.0,
        reason="explicit request",
        title="Narrow Exception Burndown and Patching",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-1",
    )

    result = create_specialist_handoff(
        decision=decision,
        source=source,
        request="Audit exception burndown and patch confirmed failures.",
        board="exampleproject-burndown",
    )

    assert result.ok, result.reason
    assert result.task_id
    with kb.connect(board="exampleproject-burndown") as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    assert task.status == "triage"
    assert task.goal_mode is True
    assert task.goal_max_turns == 12


def test_specialist_handoff_explicit_board_ignores_database_environment_override(
    kanban_home, monkeypatch, tmp_path
):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="burndown-patch-steward",
        confidence=1.0,
        reason="explicit request",
        title="Patch Exception Burndown",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-env-isolation",
    )
    board = "exampleproject-burndown"
    with kb.connect(board=board):
        pass
    override_path = tmp_path / "override" / "kanban.db"
    with kb.connect(db_path=override_path):
        pass
    monkeypatch.setenv("HERMES_KANBAN_DB", str(override_path))

    result = create_specialist_handoff(
        decision=decision,
        source=source,
        request="Patch the confirmed exception failures.",
        board=board,
    )

    assert result.ok, result.reason
    monkeypatch.delenv("HERMES_KANBAN_DB")
    with kb.connect(board=board) as configured_conn:
        configured_task = kb.get_task(configured_conn, result.task_id)
    with kb.connect(db_path=override_path) as override_conn:
        override_task = kb.get_task(override_conn, result.task_id)
    assert configured_task is not None
    assert override_task is None


def test_router_accepts_task_orchestrator_for_broad_actionable_work():
    decision = parse_specialist_response(
        '{"kind":"specialist","profile":"task-orchestrator",'
        '"confidence":0.91,"reason":"requires planning across roles",'
        '"title":"Implement the requested workflow"}'
    )

    assert decision.dispatches is True
    assert decision.profile == "task-orchestrator"


def test_handoff_rejects_unresolved_candidate_profile_without_a_missing_scope_receipt(kanban_home):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="generated-market-data-candidate",
        confidence=0.95,
        reason="model suggestion",
        title="Generated candidate",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-candidate-without-resolution",
    )

    result = create_specialist_handoff(
        decision=decision,
        source=source,
        request="Audit the supplied evidence.",
    )

    assert result.ok is False
    assert result.reason == "non_dispatch_decision"


def test_handoff_uses_active_registry_profile_before_fixed_classifier_profile(kanban_home, tmp_path):
    signature = CapabilitySignature(
        domain="market-data",
        actions=("audit", "read"),
        evidence_class="diagnostic-only",
        requested_permissions=("market-data:read",),
    )
    registry = CapabilityRegistry(db_path=tmp_path / "capability-registry.db")
    registry.register_fixed_baseline(profile_id="market-data-authority-auditor", signature=signature)
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="market-data-authority-auditor",
        confidence=0.95,
        reason="classifier fixed profile",
        title="Audit market data",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-active-registry-match",
    )

    result = create_specialist_handoff(
        decision=decision,
        source=source,
        request="Audit the supplied market-data evidence.",
        signature=signature,
        registry=registry,
        board=kb.DEFAULT_BOARD,
    )

    assert result.ok, result.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    assert task.assignee == "market-data-authority-auditor"


def test_forged_active_registry_resolution_cannot_create_a_profile_assigned_handoff(kanban_home):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="market-data-authority-auditor",
        confidence=0.95,
        reason="classifier fixed profile",
        title="Audit market data",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-forged-active-resolution",
    )
    forged = RegistryResolution(
        status="active_match",
        profile="forged-profile",
        reason="untrusted caller data",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'resolution'"):
        create_specialist_handoff(
            decision=decision,
            source=source,
            request="Audit the supplied market-data evidence.",
            board=kb.DEFAULT_BOARD,
            resolution=forged,
        )

    with kb.connect() as conn:
        rows = conn.execute("SELECT assignee FROM tasks").fetchall()
    assert rows == []


def test_duck_typed_registry_cannot_authorize_a_profile_assigned_handoff(kanban_home):
    signature = CapabilitySignature(
        domain="market-data",
        actions=("audit", "read"),
        evidence_class="diagnostic-only",
        requested_permissions=("market-data:read",),
    )
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="market-data-authority-auditor",
        confidence=0.95,
        reason="classifier fixed profile",
        title="Audit market data",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-duck-typed-registry",
    )

    class ForgedRegistry:
        def resolve(self, requested_signature):
            assert requested_signature == signature
            return RegistryResolution(
                status="active_match",
                profile="forged-profile",
                reason="untrusted caller data",
            )

    result = create_specialist_handoff(
        decision=decision,
        source=source,
        request="Audit the supplied market-data evidence.",
        board=kb.DEFAULT_BOARD,
        signature=signature,
        registry=ForgedRegistry(),
    )

    assert result.ok is False
    assert result.reason == "non_dispatch_decision"
    with kb.connect() as conn:
        rows = conn.execute("SELECT assignee FROM tasks").fetchall()
    assert rows == []
