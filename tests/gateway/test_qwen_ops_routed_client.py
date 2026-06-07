"""Tests for qwen-ops as a routed Hermes gateway client."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.qwen_ops import (
    QWEN_OPS_GRANTED_AUTHORITIES,
    QWEN_OPS_MODEL_COMMAND_TEXT,
    build_qwen_ops_escalation_packet,
)
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.commands import (
    GATEWAY_KNOWN_COMMANDS,
    resolve_command,
    should_bypass_active_session,
    telegram_bot_commands,
)
from neoengine_local.pr_ci_traffic_controller.controller_contract import (
    FRONTIER_ESCALATION_PACKET_FIELDS,
)


_AUTH_ENV_KEYS = (
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_CHATS",
    "TELEGRAM_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


def _clear_auth_env(monkeypatch):
    for key in _AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _make_source(
    *,
    platform: Platform = Platform.TELEGRAM,
    user_id: str | None = "u1",
    chat_id: str = "c1",
    chat_type: str = "dm",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester" if user_id else None,
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=source or _make_source(),
        message_id="m1",
    )


def _make_session_entry(source: SessionSource | None = None) -> SessionEntry:
    source = source or _make_source()
    now = datetime.now()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        platform=source.platform,
        chat_type=source.chat_type,
        total_tokens=0,
    )


def _make_runner(
    session_entry: SessionEntry,
    *,
    platform: Platform = Platform.TELEGRAM,
    platform_extra: dict | None = None,
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra=platform_extra or {},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.clear_resume_pending = MagicMock()
    runner.session_store.reset_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._session_db.get_telegram_topic_binding.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._update_prompt_pending = {}
    runner._session_model_overrides = {}
    runner._agent_cache = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: []
    runner._clear_session_env = lambda _tokens: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._post_turn_goal_continuation = AsyncMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._is_telegram_topic_lane = lambda _source: False
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._telegram_topic_root_lobby_message = lambda: "topic lobby"
    runner._clear_restart_failure_count = MagicMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def test_qwen_ops_is_registered_as_gateway_command_and_aliases():
    for name in ("qwen-ops", "qwen", "qwenops", "qwen_ops"):
        command = resolve_command(name)
        assert command is not None
        assert command.name == "qwen-ops"
        assert name in GATEWAY_KNOWN_COMMANDS
        assert should_bypass_active_session(name) is True

    telegram_commands = dict(telegram_bot_commands())
    assert "qwen_ops" in telegram_commands
    assert "qwen-ops" not in telegram_commands


@pytest.mark.asyncio
async def test_qwen_ops_switches_model_through_shared_model_handler(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [],
        raising=False,
    )
    source = _make_source()
    session_entry = _make_session_entry(source)
    runner = _make_runner(session_entry)
    runner._handle_model_command = AsyncMock(return_value="model switched")

    result = await runner._handle_message(_make_event("/qwen on", source))

    assert "qwen-ops is a routed Hermes client" in result
    assert "model switched" in result
    runner._handle_model_command.assert_awaited_once()
    switch_event = runner._handle_model_command.await_args.args[0]
    assert switch_event.text == QWEN_OPS_MODEL_COMMAND_TEXT


@pytest.mark.asyncio
async def test_qwen_ops_prompt_uses_shared_agent_and_token_accounting(monkeypatch):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "c1")
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [],
        raising=False,
    )
    source = _make_source()
    session_entry = _make_session_entry(source)
    runner = _make_runner(session_entry)
    runner._handle_model_command = AsyncMock(return_value="model switched")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "qwen3-coder",
        }
    )

    result = await runner._handle_message(
        _make_event("/qwen inspect local logs", source)
    )

    assert result == "ok"
    runner._handle_model_command.assert_awaited_once()
    runner._run_agent.assert_awaited_once()
    agent_message = runner._run_agent.await_args.kwargs["message"]
    assert "[qwen-ops routed client scope]" in agent_message
    assert "inspect local logs" in agent_message
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=80,
    )


@pytest.mark.asyncio
async def test_qwen_ops_respects_telegram_group_chat_allowlist(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-100")
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [],
        raising=False,
    )
    from gateway.run import GatewayRunner

    allowed_source = _make_source(chat_id="-100", chat_type="group")
    allowed_entry = _make_session_entry(allowed_source)
    runner = _make_runner(allowed_entry)
    runner._is_user_authorized = GatewayRunner._is_user_authorized.__get__(
        runner,
        GatewayRunner,
    )
    runner._handle_model_command = AsyncMock(return_value="model switched")

    allowed_result = await runner._handle_message(
        _make_event("/qwen on", allowed_source)
    )

    assert "qwen-ops is a routed Hermes client" in allowed_result
    runner._handle_model_command.assert_awaited_once()

    denied_source = _make_source(chat_id="-200", chat_type="group")
    denied_entry = _make_session_entry(denied_source)
    denied_runner = _make_runner(denied_entry)
    denied_runner._is_user_authorized = GatewayRunner._is_user_authorized.__get__(
        denied_runner,
        GatewayRunner,
    )
    denied_runner._handle_model_command = AsyncMock(return_value="model switched")

    denied_result = await denied_runner._handle_message(
        _make_event("/qwen on", denied_source)
    )

    assert denied_result is None
    denied_runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_qwen_ops_respects_shared_slash_access_policy(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [],
        raising=False,
    )
    source = _make_source(user_id="user")
    session_entry = _make_session_entry(source)
    runner = _make_runner(
        session_entry,
        platform_extra={
            "allow_admin_from": ["admin"],
            "user_allowed_commands": ["help"],
        },
    )
    runner._handle_model_command = AsyncMock(return_value="model switched")

    result = await runner._handle_message(_make_event("/qwen on", source))

    assert "/qwen-ops is admin-only here" in result
    runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_qwen_ops_escalation_emits_shared_packet_only(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [],
        raising=False,
    )
    source = _make_source()
    session_entry = _make_session_entry(source)
    runner = _make_runner(session_entry)
    runner._handle_model_command = AsyncMock(return_value="model switched")
    runner._run_agent = AsyncMock()

    result = await runner._handle_message(
        _make_event("/qwen escalate gpt-5.5 protected boundary decision", source)
    )

    assert "Shared escalation packet" in result
    assert "packet only; no qwen-ops execution" in result
    assert "target_model_lane: GPT-5.5" in result
    assert "status: blocked_no_consumer" in result
    for field in FRONTIER_ESCALATION_PACKET_FIELDS:
        assert f"{field}:" in result
    runner._handle_model_command.assert_not_awaited()
    runner._run_agent.assert_not_awaited()


def test_qwen_ops_has_no_extra_write_authority():
    forbidden = {"merge", "approve", "comment", "label", "write"}
    assert QWEN_OPS_GRANTED_AUTHORITIES.isdisjoint(forbidden)

    packet = build_qwen_ops_escalation_packet("escalate codex fix failing test")
    assert "no qwen-ops execution" in packet
    assert "qwen-ops does not execute" in packet
