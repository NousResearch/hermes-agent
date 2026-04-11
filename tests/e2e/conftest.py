"""Shared fixtures for gateway e2e tests (Telegram, Discord).

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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key


# Platform library mocks

# Ensure telegram module is available (mock it if not installed)
def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return # Real library installed

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


# Ensure discord module is available (mock it if not installed)
def _ensure_discord_mock():
    """Install mock discord modules so DiscordAdapter can be imported."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return # Real library installed

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus.is_loaded.return_value = True

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)
    sys.modules.setdefault("discord.opus", discord_mod.opus)


def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return  # Real library installed

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()
_ensure_discord_mock()
_ensure_slack_mock()

from gateway.platforms.discord import DiscordAdapter   # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402

import gateway.platforms.slack as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True
from gateway.platforms.slack import SlackAdapter  # noqa: E402


# Platform-generic factories

def make_source(platform: Platform, chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1") -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_name="e2e_tester",
        chat_type="dm",
    )


def make_session_entry(platform: Platform, source: SessionSource = None) -> SessionEntry:
    source = source or make_source(platform)
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )


def make_event(platform: Platform, text: str = "/help", chat_id: str = "e2e-chat-1", user_id: str = "e2e-user-1") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=make_source(platform, chat_id, user_id),
        message_id=f"msg-{uuid.uuid4().hex[:8]}",
    )


def make_runner(platform: Platform, session_entry: SessionEntry = None) -> "GatewayRunner":
    """Create a GatewayRunner with mocked internals for e2e testing.

    Skips __init__ to avoid filesystem/network side effects.
    """
    from gateway.run import GatewayRunner

    if session_entry is None:
        session_entry = make_session_entry(platform)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="e2e-test-token")}
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

    runner.pairing_store = MagicMock()
    runner.pairing_store._is_rate_limited = MagicMock(return_value=False)
    runner.pairing_store.generate_code = MagicMock(return_value="ABC123")

    return runner


def make_adapter(platform: Platform, runner=None):
    """Create a platform adapter wired to *runner*, with send methods mocked."""
    if runner is None:
        runner = make_runner(platform)

    config = PlatformConfig(enabled=True, token="e2e-test-token")

    if platform == Platform.DISCORD:
        with patch.object(DiscordAdapter, "_load_participated_threads", return_value=set()):
            adapter = DiscordAdapter(config)
        platform_key = Platform.DISCORD
    elif platform == Platform.SLACK:
        adapter = SlackAdapter(config)
        platform_key = Platform.SLACK
    else:
        adapter = TelegramAdapter(config)
        platform_key = Platform.TELEGRAM

    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-resp-1"))
    adapter.send_typing = AsyncMock()

    adapter.set_message_handler(runner._handle_message)
    runner.adapters[platform_key] = adapter

    return adapter


async def send_and_capture(adapter, text: str, platform: Platform, **event_kwargs) -> AsyncMock:
    """Send a message through the full e2e flow and return the send mock."""
    event = make_event(platform, text, **event_kwargs)
    adapter.send.reset_mock()
    await adapter.handle_message(event)
    await asyncio.sleep(0.3)
    return adapter.send


# Parametrized fixtures for platform-generic tests
@pytest.fixture(params=[Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK], ids=["telegram", "discord", "slack"])
def platform(request):
    return request.param


@pytest.fixture()
def source(platform):
    return make_source(platform)


@pytest.fixture()
def session_entry(platform, source):
    return make_session_entry(platform, source)


@pytest.fixture()
def runner(platform, session_entry):
    return make_runner(platform, session_entry)


@pytest.fixture()
def adapter(platform, runner):
    return make_adapter(platform, runner)
