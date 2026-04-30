"""Tests for the gateway /goal session authority command."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionGoalContract, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner(session_entry: SessionEntry | None = None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False

    if session_entry is None:
        session_entry = SessionEntry(
            session_key=build_session_key(_make_source()),
            session_id="sess-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.update_session = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True

    from gateway.run import GatewayRunner as _GR

    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner, session_entry


def test_goal_command_is_registered_with_subcommands():
    from hermes_cli.commands import resolve_command

    command = resolve_command("goal")

    assert command is not None
    assert command.name == "goal"
    assert command.subcommands == ("status", "new", "lock", "unlock", "clear")


@pytest.mark.asyncio
async def test_goal_new_creates_persisted_contract():
    runner, entry = _make_runner()

    out = await runner._handle_goal_command(_make_event("/goal new Finish the PR"))

    assert "Goal set" in out
    assert entry.goal_contract is not None
    assert entry.goal_contract.current_objective == "Finish the PR"
    assert entry.goal_contract.status == "active"
    assert entry.goal_contract.operator_confirmed is True
    runner.session_store.update_session.assert_called_once_with(entry.session_key)
    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_goal_status_reports_current_contract():
    contract = SessionGoalContract(
        current_objective="Finish the PR",
        locked=True,
        scope_policy="locked",
        allowed_subtasks=["implement", "verify-pr"],
        non_goals=["change unrelated code"],
    )
    runner, _entry = _make_runner(
        SessionEntry(
            session_key=build_session_key(_make_source()),
            session_id="sess-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
            goal_contract=contract,
        )
    )

    out = await runner._handle_goal_command(_make_event("/goal status"))

    assert "Session Goal" in out
    assert "Finish the PR" in out
    assert "Locked:** yes" in out
    assert "implement" in out
    runner.session_store.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_goal_lock_unlock_and_clear_mutate_existing_contract():
    runner, entry = _make_runner()
    entry.goal_contract = SessionGoalContract(current_objective="Finish the PR")

    locked = await runner._handle_goal_command(_make_event("/goal lock"))
    assert "locked" in locked.lower()
    assert entry.goal_contract.locked is True
    assert entry.goal_contract.scope_policy == "locked"
    assert entry.goal_contract.locked_at is not None

    unlocked = await runner._handle_goal_command(_make_event("/goal unlock"))
    assert "unlocked" in unlocked.lower()
    assert entry.goal_contract.locked is False
    assert entry.goal_contract.scope_policy == "soft"

    cleared = await runner._handle_goal_command(_make_event("/goal clear"))
    assert "cleared" in cleared.lower()
    assert entry.goal_contract is None
    assert runner.session_store.update_session.call_count == 3


@pytest.mark.asyncio
async def test_dispatcher_routes_goal_command(monkeypatch):
    import gateway.run as gateway_run

    runner, _entry = _make_runner()
    runner._handle_goal_command = AsyncMock(return_value="goal handler reached")  # type: ignore[attr-defined]
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    result = await runner._handle_message(_make_event("/goal status"))

    assert result == "goal handler reached"
