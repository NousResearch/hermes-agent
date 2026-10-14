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
from gateway.platforms.event import MessageEvent
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
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    PluginContext(PluginManifest(name="identity", source="user"), manager).register_command(
        name, handler,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sync", "async", "bound", "partial", "decorated",
                                 "legacy", "optional", "variadic", "all_args",
                                 "async_variadic", "opaque"])
async def test_registered_command_preserves_signature_and_sender(monkeypatch, kind):
    """Real registry -> gateway dispatch delivers context only to an explicit opt-in."""
    from functools import partial, wraps

    captured = {}

    def contextual(raw_args, context):
        captured.update(raw_args=raw_args, context=context)
        return "ok"

    async def async_contextual(raw_args, context):
        return contextual(raw_args, context)

    class Handler:
        def handle(self, raw_args, context):
            return contextual(raw_args, context)

    def prefixed(prefix, raw_args, context):
        assert prefix == "prefix"
        return contextual(raw_args, context)

    @wraps(contextual)
    def decorated(*args, **kwargs):
        return contextual(*args, **kwargs)

    def legacy(raw_args):
        captured["raw_args"] = raw_args
        return "ok"

    def optional(raw_args, option="default"):
        assert option == "default"
        return legacy(raw_args)

    def variadic(raw_args, *args):
        assert len(args) == 1
        return contextual(raw_args, args[0])

    def all_args(*args):
        assert len(args) == 2
        return contextual(*args)

    async def async_variadic(raw_args, *args):
        return variadic(raw_args, *args)

    class OpaqueHandler:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, raw_args):
            return raw_args.upper()

    handlers = {
        "sync": contextual, "async": async_contextual, "bound": Handler().handle,
        "partial": partial(prefixed, "prefix"), "decorated": decorated,
        "legacy": legacy, "optional": optional, "variadic": variadic,
        "all_args": all_args, "async_variadic": async_variadic,
        "opaque": OpaqueHandler(),
    }
    _patch_plugin_command(monkeypatch, "my-ident", handlers[kind])
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("command leaked to agent"))
    event = _make_event("/my_ident user_id=spoofed")
    result = await runner._handle_message(event)

    if kind == "opaque":
        assert result == "USER_ID=SPOOFED"
    else:
        assert result == "ok"
        assert captured["raw_args"] == "user_id=spoofed"
        if kind in {"sync", "async", "bound", "partial", "decorated", "variadic",
                    "all_args", "async_variadic"}:
            assert captured["context"] == {
                "user_id": event.source.user_id, "user_name": event.source.user_name,
                "chat_id": event.source.chat_id, "chat_type": event.source.chat_type,
                "platform": event.source.platform.value,
            }
        else:
            assert "context" not in captured
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_typeerror_does_not_retry_or_leak_to_agent(monkeypatch):
    """A TypeError inside the handler is not a signature-detection mechanism."""
    calls = []

    def handler(raw_args, context):
        calls.append((raw_args, context))
        raise TypeError("plugin failed after a side effect")

    _patch_plugin_command(monkeypatch, "my-ident", handler)
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("command leaked to agent"))
    result = await runner._handle_message(_make_event("/my-ident verbose"))
    assert len(calls) == 1
    assert calls[0][0] == "verbose"
    assert "Unknown command" in result
    runner._run_agent.assert_not_awaited()
