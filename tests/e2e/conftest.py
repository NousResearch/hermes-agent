"""Shared fixtures for Telegram gateway e2e tests.

These tests exercise the full async message flow:
    adapter.handle_message(event)
        → background task
        → GatewayRunner._handle_message (command dispatch)
        → adapter.send() (captured by mock)

No LLM, no real platform connections.
"""

import asyncio
import sys
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Ensure telegram module is available (mock it if not installed)
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return  # Real library installed

    telegram_mod = MagicMock()
    telegram_mod.Update = MagicMock()
    telegram_mod.Update.ALL_TYPES = []
    telegram_mod.Bot = MagicMock
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.ext.Application = MagicMock()
    telegram_mod.ext.Application.builder = MagicMock
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.ext.MessageHandler = MagicMock
    telegram_mod.ext.CommandHandler = MagicMock
    telegram_mod.ext.filters = MagicMock()
    telegram_mod.request.HTTPXRequest = MagicMock

    for name in (
        "telegram",
        "telegram.constants",
        "telegram.ext",
        "telegram.ext.filters",
        "telegram.request",
    ):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# GatewayRunner factory (based on tests/gateway/test_status_command.py)
# ---------------------------------------------------------------------------

def make_runner(session_entry: SessionEntry) -> "GatewayRunner":
    """Create a GatewayRunner with mocked internals for e2e testing.

    Skips __init__ to avoid filesystem/network side effects.
    All command-dispatch dependencies are wired manually.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.reset_session = MagicMock()

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
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **kw: None
    runner._emit_gateway_run_progress = AsyncMock()

    return runner


# ---------------------------------------------------------------------------
# TelegramAdapter factory
# ---------------------------------------------------------------------------

def make_adapter(runner) -> TelegramAdapter:
    """Create a TelegramAdapter wired to *runner*, with send methods mocked.

    connect() is NOT called — no polling, no token lock, no real HTTP.
    """
    config = PlatformConfig(enabled=True, token="e2e-test-token")
    adapter = TelegramAdapter(config)

    # Mock outbound methods so tests can capture what was sent
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-resp-1"))
    adapter.send_typing = AsyncMock()

    # Wire adapter ↔ runner
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.TELEGRAM] = adapter

    return adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source(chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id=user_id,
        user_name="e2e_tester",
        chat_type="dm",
    )


def make_event(text: str, chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=make_source(chat_id, user_id),
        message_id=f"msg-{uuid.uuid4().hex[:8]}",
    )


def make_session_entry(source: SessionSource = None) -> SessionEntry:
    source = source or make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )


async def send_and_capture(adapter: TelegramAdapter, text: str, **event_kwargs) -> AsyncMock:
    """Send a message through the full e2e flow and return the send mock.

    Drives: adapter.handle_message → background task → runner dispatch → adapter.send.
    """
    event = make_event(text, **event_kwargs)
    adapter.send.reset_mock()
    await adapter.handle_message(event)
    # Let the background task complete
    await asyncio.sleep(0.3)
    return adapter.send
