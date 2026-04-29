"""Tests for the ``/reload-skills`` gateway slash command handler.

Verifies the gateway path that mirrors ``/reload-mcp``:
  * dispatcher routes ``/reload-skills`` to ``_handle_reload_skills_command``
  * the underscored alias ``/reload_skills`` is not flagged as unknown
  * the handler invokes ``agent.skill_commands.reload_skills`` and renders a
    human-readable diff
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


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


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

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
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    return runner


@pytest.mark.asyncio
async def test_reload_skills_handler_renders_added_and_removed(monkeypatch):
    """The handler should call ``reload_skills`` and surface the diff."""
    import gateway.run as gateway_run

    fake_result = {
        "added": ["alpha", "beta"],
        "removed": ["gamma"],
        "unchanged": ["delta"],
        "total": 3,
        "commands": 3,
    }

    def _fake_reload_skills():
        return fake_result

    # Patch the symbol where ``_handle_reload_skills_command`` imports it from.
    import agent.skill_commands as skill_commands_mod
    monkeypatch.setattr(skill_commands_mod, "reload_skills", _fake_reload_skills)

    runner = _make_runner()
    out = await runner._handle_reload_skills_command(_make_event("/reload-skills"))

    assert out is not None
    assert "Skills Reloaded" in out
    assert "alpha" in out and "beta" in out
    assert "gamma" in out
    assert "3 skill(s) available" in out

    # A history note should be appended so the model sees the diff next turn.
    runner.session_store.append_to_transcript.assert_called_once()
    appended = runner.session_store.append_to_transcript.call_args[0][1]
    assert appended["role"] == "user"
    assert "Skills have been reloaded" in appended["content"]


@pytest.mark.asyncio
async def test_reload_skills_handler_reports_no_changes(monkeypatch):
    """When nothing changed, the handler should say so without injecting a note."""
    import agent.skill_commands as skill_commands_mod

    monkeypatch.setattr(
        skill_commands_mod,
        "reload_skills",
        lambda: {
            "added": [],
            "removed": [],
            "unchanged": ["alpha"],
            "total": 1,
            "commands": 1,
        },
    )

    runner = _make_runner()
    out = await runner._handle_reload_skills_command(_make_event("/reload-skills"))

    assert "No changes detected" in out
    assert "1 skill(s) available" in out
    # No history note when nothing changed — preserves prompt cache.
    runner.session_store.append_to_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_routes_reload_skills(monkeypatch):
    """``/reload-skills`` must reach ``_handle_reload_skills_command``."""
    import gateway.run as gateway_run

    runner = _make_runner()
    sentinel = "reload-skills handler reached"
    runner._handle_reload_skills_command = AsyncMock(return_value=sentinel)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload-skills"))
    assert result == sentinel


@pytest.mark.asyncio
async def test_underscored_alias_not_flagged_unknown(monkeypatch):
    """Telegram autocomplete sends ``/reload_skills`` for ``/reload-skills``."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_reload_skills_command = AsyncMock(return_value="ok")  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload_skills"))
    if result is not None:
        assert "Unknown command" not in result
