"""Tests for sender-context passing to plugin slash command handlers.

Plugin slash command handlers historically receive only ``raw_args``. A
handler that declares a second positional parameter (``context``) receives a
dict with the invoking sender's identity — this is what lets identity/RBAC
plugins implement commands like ``/rbac whoami``. One-parameter handlers must
keep working exactly as before (opt-in, backward compatible).
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
        user_id="u42",
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
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _patch_plugin_command(monkeypatch, name: str, handler):
    import gateway.run as gateway_run
    from hermes_cli import plugins as plugins_mod

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_command_handler",
        lambda cmd: handler if cmd == name else None,
    )


@pytest.mark.asyncio
async def test_plugin_command_receives_sender_context(monkeypatch):
    """A two-parameter handler gets (raw_args, context) with sender identity."""
    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )

    captured = {}

    def _handler(raw_args, context):
        captured["raw_args"] = raw_args
        captured["context"] = context
        return "ok"

    _patch_plugin_command(monkeypatch, "myident", _handler)

    result = await runner._handle_message(_make_event("/myident verbose"))

    assert result == "ok"
    assert captured["raw_args"] == "verbose"
    ctx = captured["context"]
    assert ctx["user_id"] == "u42"
    assert ctx["user_name"] == "tester"
    assert ctx["chat_id"] == "c1"
    assert ctx["chat_type"] == "dm"
    assert ctx["platform"] == "telegram"


@pytest.mark.asyncio
async def test_plugin_command_one_param_handler_unchanged(monkeypatch):
    """Legacy one-parameter handlers must keep working (backward compat)."""
    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )

    captured = {}

    def _handler(raw_args):
        captured["raw_args"] = raw_args
        return "legacy ok"

    _patch_plugin_command(monkeypatch, "ping", _handler)

    result = await runner._handle_message(_make_event("/ping hello"))

    assert result == "legacy ok"
    assert captured["raw_args"] == "hello"


@pytest.mark.asyncio
async def test_plugin_command_async_handler_with_context(monkeypatch):
    """Async two-parameter handlers are awaited and receive context."""
    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )

    captured = {}

    async def _handler(raw_args, context):
        captured["context"] = context
        return f"user={context['user_id']}"

    _patch_plugin_command(monkeypatch, "myident", _handler)

    result = await runner._handle_message(_make_event("/myident"))

    assert result == "user=u42"
    assert captured["context"]["platform"] == "telegram"


@pytest.mark.asyncio
async def test_plugin_command_lambda_one_arg_still_legacy(monkeypatch):
    """A lambda registered as handler keeps the legacy one-arg call."""
    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )

    _patch_plugin_command(monkeypatch, "metricas", lambda args: f"metrics {args}")

    result = await runner._handle_message(_make_event("/metricas dias:7"))

    assert result == "metrics dias:7"
