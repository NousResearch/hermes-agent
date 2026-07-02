"""Regression tests for gateway skill template session-id propagation."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(*, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
    )


def _make_event(
    text: str,
    *,
    thread_id: str | None = None,
    auto_skill=None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(thread_id=thread_id),
        message_id="m1",
        auto_skill=auto_skill,
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-123",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        origin=source,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._busy_ack_ts = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: True
    runner._session_key_for_source = lambda src: build_session_key(src)
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._invalidate_session_run_generation = MagicMock()
    runner._begin_session_run_generation = MagicMock(return_value=1)
    runner._is_session_run_current = MagicMock(return_value=True)
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._format_session_info = MagicMock(return_value="")
    runner._bind_adapter_run_generation = MagicMock()
    runner._telegram_topic_mode_enabled = lambda _source: False
    runner._is_telegram_topic_lane = lambda _source: False
    return runner, session_entry


@pytest.mark.asyncio
async def test_slash_skill_uses_real_session_id(monkeypatch):
    import gateway.run as gateway_run

    runner, session_entry = _make_runner()
    captured = {}

    def _fake_build_skill_invocation_message(
        cmd_key,
        user_instruction="",
        task_id=None,
        runtime_note="",
    ):
        del cmd_key, user_instruction, runtime_note
        captured["task_id"] = task_id
        return f"Session: {task_id}"

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/sess-templated": {"name": "sess-templated"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda command: "/sess-templated" if command == "sess-templated" else None,
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        _fake_build_skill_invocation_message,
    )

    runner._handle_message_with_agent = AsyncMock(return_value="ok")

    await runner._handle_message(_make_event("/sess-templated do it"))

    assert captured["task_id"] == session_entry.session_id
    runner._handle_message_with_agent.assert_awaited_once()
    forwarded_event = runner._handle_message_with_agent.await_args.args[0]
    assert forwarded_event.text == f"Session: {session_entry.session_id}"


@pytest.mark.asyncio
async def test_auto_skill_uses_real_session_id(monkeypatch):
    import gateway.run as gateway_run

    runner, session_entry = _make_runner()
    captured = {}

    def _fake_load_skill_payload(skill_name, task_id=None):
        captured["load_task_id"] = task_id
        return ({"content": "Session: ${HERMES_SESSION_ID}"}, None, skill_name)

    def _fake_build_skill_message(
        loaded_skill,
        skill_dir,
        activation_note,
        user_instruction="",
        runtime_note="",
        session_id=None,
    ):
        del loaded_skill, skill_dir, activation_note, user_instruction, runtime_note
        captured["session_id"] = session_id
        return f"Session: {session_id}"

    monkeypatch.setattr(gateway_run, "build_session_context", lambda *args, **kwargs: object())
    monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda *args, **kwargs: "ctx")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr("agent.skill_commands._load_skill_payload", _fake_load_skill_payload)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", _fake_build_skill_message)

    runner.session_store.load_transcript.side_effect = RuntimeError("stop after auto-skill")

    event = _make_event("hello", auto_skill="sess-templated")
    with pytest.raises(RuntimeError, match="stop after auto-skill"):
        await runner._handle_message_with_agent(
            event,
            event.source,
            build_session_key(event.source),
            1,
        )

    assert captured["load_task_id"] == build_session_key(event.source)
    assert captured["session_id"] == session_entry.session_id
    assert f"Session: {session_entry.session_id}" in event.text

