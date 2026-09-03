"""Discord specialist work enters the durable orchestration lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda profile: profile in {"patch-steward", "task-orchestrator"},
    )
    kb.init_db()
    return home


def test_specialist_handoff_creates_goal_mode_triage_root(kanban_home):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="patch-steward",
        confidence=1.0,
        reason="explicit request",
        title="Narrow corrective patch",
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
        request="Audit and patch the confirmed failures.",
        board="project-maintenance",
    )

    assert result.ok, result.reason
    assert result.task_id
    with kb.connect(board="project-maintenance") as conn:
        task = kb.get_task(conn, result.task_id)
    assert task is not None
    assert task.status == "triage"
    assert task.goal_mode is True
    assert task.goal_max_turns == 12
    assert task.skills is None


def test_specialist_handoff_explicit_board_ignores_database_environment_override(
    kanban_home, monkeypatch, tmp_path
):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="patch-steward",
        confidence=1.0,
        reason="explicit request",
        title="Patch confirmed failures",
    )
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-env-isolation",
    )
    board = "project-maintenance"
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


def test_router_accepts_only_profiles_from_the_explicit_route_map():
    raw = (
        '{"kind":"specialist","profile":"release-reviewer",'
        '"confidence":0.91,"reason":"release evidence",'
        '"title":"Review release evidence"}'
    )

    rejected = parse_specialist_response(raw)
    accepted = parse_specialist_response(
        raw,
        profiles={"release-reviewer": "release evidence and gate review"},
    )

    assert rejected.dispatches is False
    assert accepted.dispatches is True
    assert accepted.profile == "release-reviewer"


def test_specialist_handoff_rejects_profile_that_is_not_installed(kanban_home):
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="missing-profile",
        confidence=1.0,
        reason="explicit request",
        title="Unavailable route",
    )
    result = create_specialist_handoff(
        decision=decision,
        source=HandoffSource(
            platform="discord",
            chat_id="channel-1",
            chat_type="group",
            user_id="user-1",
            message_id="message-missing-profile",
        ),
        request="Run the unavailable route.",
        board="project-maintenance",
    )

    assert result == type(result)(False, reason="profile_unavailable")
