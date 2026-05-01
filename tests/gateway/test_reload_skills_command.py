"""Tests for the ``/reload-skills`` gateway slash command handler.

Verifies:
  * dispatcher routes ``/reload-skills`` to ``_handle_reload_skills_command``
  * the underscored alias ``/reload_skills`` is not flagged as unknown
  * the handler invokes ``agent.skill_commands.reload_skills`` and renders a
    human-readable diff
  * when any skills changed, a one-shot note is queued on
    ``runner._pending_skills_reload_notes[session_key]`` (the agent loop
    consumes and clears it on the next user turn — see ``gateway/run.py``
    near the ``_has_fresh_tool_tail`` block)
  * the handler does NOT append to the session transcript out-of-band —
    message alternation must not be broken by a phantom user turn
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
    # Use the real _session_key_for_source binding so the key matches what
    # the agent-loop consumer will look up later.
    from gateway.run import GatewayRunner as _GR
    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner


@pytest.mark.asyncio
async def test_reload_skills_handler_queues_note_on_diff(monkeypatch):
    """Diff non-empty → handler queues a one-shot note and does NOT touch transcript."""
    fake_result = {
        "added": [
            {"name": "alpha", "description": "Run alpha to do xyz"},
            {"name": "beta", "description": "Run beta to do abc"},
        ],
        "removed": [
            {"name": "gamma", "description": "Old removed skill"},
        ],
        "unchanged": ["delta"],
        "total": 3,
        "commands": 3,
    }

    import agent.skill_commands as skill_commands_mod
    monkeypatch.setattr(skill_commands_mod, "reload_skills", lambda: fake_result)

    runner = _make_runner()
    event = _make_event("/reload-skills")
    out = await runner._handle_reload_skills_command(event)

    assert out is not None
    assert "Skills Reloaded" in out
    assert "Added Skills:" in out
    assert "- alpha: Run alpha to do xyz" in out
    assert "- beta: Run beta to do abc" in out
    assert "Removed Skills:" in out
    assert "- gamma: Old removed skill" in out
    assert "3 skill(s) available" in out

    # MUST NOT write to the session transcript — that would break alternation.
    runner.session_store.append_to_transcript.assert_not_called()

    # MUST have queued a one-shot note keyed on the session.
    pending = getattr(runner, "_pending_skills_reload_notes", None)
    assert pending is not None
    session_key = runner._session_key_for_source(event.source)
    assert session_key in pending
    note = pending[session_key]
    assert note.startswith("[USER INITIATED SKILLS RELOAD:")
    assert note.endswith("Use skills_list to see the updated catalog.]")
    assert "Added Skills:" in note
    assert "    - alpha: Run alpha to do xyz" in note
    assert "    - beta: Run beta to do abc" in note
    assert "Removed Skills:" in note
    assert "    - gamma: Old removed skill" in note


@pytest.mark.asyncio
async def test_reload_skills_handler_reports_no_changes(monkeypatch):
    """No diff → no queued note, no transcript write."""
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

    assert "No new skills detected" in out
    assert "1 skill(s) available" in out
    runner.session_store.append_to_transcript.assert_not_called()
    # No queued note when nothing changed.
    pending = getattr(runner, "_pending_skills_reload_notes", None)
    assert not pending  # None or empty dict


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
