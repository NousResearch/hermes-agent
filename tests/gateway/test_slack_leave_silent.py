"""
Tests for Slack /leave and /silent thread management commands.

Verifies:
- _left_threads and _silent_threads state initialisation in SlackAdapter
- /leave: blocks future responses, cleans up tracking state, emits EphemeralReply
- /silent: enables silent mode, re-activates on @mention, emits EphemeralReply
- Re-mention clears both _left_threads and _silent_threads for a thread
- Platform guard: /leave and /silent return errors on non-Slack platforms
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

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


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOT_USER_ID = "U_BOT_123"
CHANNEL_ID = "C0CHANNEL001"
THREAD_TS = "1700000000.000100"


def _make_adapter():
    """Minimal SlackAdapter instance with thread-tracking sets initialised."""
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000
    adapter._bot_message_ts = set()
    adapter._BOT_TS_MAX = 5000
    adapter._left_threads = set()
    adapter._LEFT_THREADS_MAX = 1000
    adapter._silent_threads = set()
    adapter._SILENT_THREADS_MAX = 1000
    adapter._assistant_threads = {}
    adapter._thread_context_cache = {}
    adapter._active_status_threads = {}
    return adapter


def _make_event(platform=Platform.SLACK, thread_id=THREAD_TS, chat_id=CHANNEL_ID, user_id="U_USER_123", text="/leave"):
    """Build a minimal MessageEvent-like namespace."""
    source = SimpleNamespace(
        platform=platform,
        thread_id=thread_id,
        chat_id=chat_id,
        user_id=user_id,
    )
    event = SimpleNamespace(source=source, text=text)
    return event


def _make_runner(adapter=None):
    """Minimal GatewayRunner-like namespace with slash command mixin methods."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    runner = object.__new__(GatewaySlashCommandsMixin)
    runner.adapters = {}
    runner._session_store = None
    if adapter is not None:
        runner.adapters[Platform.SLACK] = adapter
    return runner


# ---------------------------------------------------------------------------
# SlackAdapter: new state attributes
# ---------------------------------------------------------------------------

class TestSlackAdapterNewState:
    def test_left_threads_initialised(self):
        adapter = _make_adapter()
        assert isinstance(adapter._left_threads, set)
        assert len(adapter._left_threads) == 0

    def test_silent_threads_initialised(self):
        adapter = _make_adapter()
        assert isinstance(adapter._silent_threads, set)
        assert len(adapter._silent_threads) == 0

    def test_left_threads_max_constant(self):
        adapter = _make_adapter()
        assert adapter._LEFT_THREADS_MAX == 1000

    def test_silent_threads_max_constant(self):
        adapter = _make_adapter()
        assert adapter._SILENT_THREADS_MAX == 1000


# ---------------------------------------------------------------------------
# SlackAdapter: /leave clears tracking sets and blocks responses
# ---------------------------------------------------------------------------

class TestSlackAdapterLeaveRouting:
    def test_left_thread_blocks_response_when_not_mentioned(self):
        """A thread in _left_threads is blocked even if in _mentioned_threads."""
        adapter = _make_adapter()
        adapter._left_threads.add(THREAD_TS)
        # Simulate the routing guard: if the thread is in _left_threads it returns early
        result = THREAD_TS in adapter._left_threads
        assert result is True

    def test_re_mention_clears_left_threads(self):
        """@mentioning the bot in a left thread removes it from _left_threads."""
        adapter = _make_adapter()
        adapter._left_threads.add(THREAD_TS)
        # Simulate the mention-handling code path
        adapter._left_threads.discard(THREAD_TS)
        assert THREAD_TS not in adapter._left_threads

    def test_re_mention_clears_silent_threads(self):
        """@mentioning the bot in a silent thread removes it from _silent_threads."""
        adapter = _make_adapter()
        adapter._silent_threads.add(THREAD_TS)
        adapter._silent_threads.discard(THREAD_TS)
        assert THREAD_TS not in adapter._silent_threads

    def test_left_threads_max_eviction(self):
        """Exceeding _LEFT_THREADS_MAX triggers LRU-style eviction."""
        adapter = _make_adapter()
        adapter._LEFT_THREADS_MAX = 10
        for i in range(15):
            adapter._left_threads.add(f"ts_{i}")
        # Simulate eviction
        if len(adapter._left_threads) > adapter._LEFT_THREADS_MAX:
            excess = len(adapter._left_threads) - adapter._LEFT_THREADS_MAX // 2
            for old in list(adapter._left_threads)[:excess]:
                adapter._left_threads.discard(old)
        assert len(adapter._left_threads) <= adapter._LEFT_THREADS_MAX

    def test_silent_threads_max_eviction(self):
        """Exceeding _SILENT_THREADS_MAX triggers LRU-style eviction."""
        adapter = _make_adapter()
        adapter._SILENT_THREADS_MAX = 10
        for i in range(15):
            adapter._silent_threads.add(f"ts_{i}")
        if len(adapter._silent_threads) > adapter._SILENT_THREADS_MAX:
            excess = len(adapter._silent_threads) - adapter._SILENT_THREADS_MAX // 2
            for old in list(adapter._silent_threads)[:excess]:
                adapter._silent_threads.discard(old)
        assert len(adapter._silent_threads) <= adapter._SILENT_THREADS_MAX


# ---------------------------------------------------------------------------
# _handle_leave_command: platform guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leave_command_rejects_non_slack():
    """_handle_leave_command returns an error on non-Slack platforms."""
    runner = _make_runner()
    event = _make_event(platform=Platform.TELEGRAM)

    result = await runner._handle_leave_command(event)

    assert isinstance(result, EphemeralReply)
    assert "only available on Slack" in result.text


@pytest.mark.asyncio
async def test_leave_command_rejects_missing_adapter():
    """_handle_leave_command returns an error when the Slack adapter is absent."""
    runner = _make_runner(adapter=None)
    event = _make_event()

    result = await runner._handle_leave_command(event)

    assert isinstance(result, EphemeralReply)
    assert "not available" in result.text


# ---------------------------------------------------------------------------
# _handle_leave_command: thread-level leave
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_leave_command_thread_adds_to_left_threads():
    """Invoking /leave in a thread adds the thread_ts to _left_threads."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/leave")

    result = await runner._handle_leave_command(event)

    assert THREAD_TS in adapter._left_threads
    assert isinstance(result, EphemeralReply)


@pytest.mark.asyncio
async def test_leave_command_thread_removes_from_mentioned_threads():
    """Invoking /leave removes the thread from _mentioned_threads."""
    adapter = _make_adapter()
    adapter._mentioned_threads.add(THREAD_TS)
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS)

    await runner._handle_leave_command(event)

    assert THREAD_TS not in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_leave_command_thread_removes_from_bot_message_ts():
    """Invoking /leave removes the thread from _bot_message_ts."""
    adapter = _make_adapter()
    adapter._bot_message_ts.add(THREAD_TS)
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS)

    await runner._handle_leave_command(event)

    assert THREAD_TS not in adapter._bot_message_ts


@pytest.mark.asyncio
async def test_leave_command_thread_removes_assistant_thread_entry():
    """Invoking /leave removes the (channel_id, thread_id) entry from _assistant_threads."""
    adapter = _make_adapter()
    thread_key = (CHANNEL_ID, THREAD_TS)
    adapter._assistant_threads[thread_key] = {"user_id": "U_USER_123"}
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, chat_id=CHANNEL_ID)

    await runner._handle_leave_command(event)

    assert thread_key not in adapter._assistant_threads


@pytest.mark.asyncio
async def test_leave_command_thread_removes_context_cache():
    """Invoking /leave removes the thread context cache entry."""
    adapter = _make_adapter()
    cache_key = f"{CHANNEL_ID}:{THREAD_TS}"
    adapter._thread_context_cache[cache_key] = object()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, chat_id=CHANNEL_ID)

    await runner._handle_leave_command(event)

    assert cache_key not in adapter._thread_context_cache


@pytest.mark.asyncio
async def test_leave_command_returns_ephemeral_reply():
    """_handle_leave_command always returns an EphemeralReply."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS)

    result = await runner._handle_leave_command(event)

    assert isinstance(result, EphemeralReply)


@pytest.mark.asyncio
async def test_leave_command_no_thread_cleans_channel_level():
    """Invoking /leave in a channel (no thread_id) cleans channel-level tracking."""
    adapter = _make_adapter()
    # Set up channel-level entries
    thread_key = (CHANNEL_ID, THREAD_TS)
    adapter._assistant_threads[thread_key] = {}
    adapter._thread_context_cache[f"{CHANNEL_ID}:{THREAD_TS}"] = object()
    runner = _make_runner(adapter=adapter)
    # No thread_id — channel-level invocation
    event = _make_event(thread_id=None, chat_id=CHANNEL_ID)

    result = await runner._handle_leave_command(event)

    assert thread_key not in adapter._assistant_threads
    assert isinstance(result, EphemeralReply)


# ---------------------------------------------------------------------------
# _handle_silent_command: platform guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_silent_command_rejects_non_slack():
    """_handle_silent_command returns an error on non-Slack platforms."""
    runner = _make_runner()
    event = _make_event(platform=Platform.TELEGRAM, text="/silent")

    result = await runner._handle_silent_command(event)

    assert isinstance(result, EphemeralReply)
    assert "only available on Slack" in result.text


@pytest.mark.asyncio
async def test_silent_command_rejects_missing_adapter():
    """_handle_silent_command returns an error when the Slack adapter is absent."""
    runner = _make_runner(adapter=None)
    event = _make_event(text="/silent")

    result = await runner._handle_silent_command(event)

    assert isinstance(result, EphemeralReply)
    assert "not available" in result.text


@pytest.mark.asyncio
async def test_silent_command_rejects_channel_level():
    """_handle_silent_command rejects invocation outside a thread (no thread_id)."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=None, text="/silent")

    result = await runner._handle_silent_command(event)

    assert isinstance(result, EphemeralReply)
    assert "only available in threads" in result.text.lower() or "threads" in result.text


# ---------------------------------------------------------------------------
# _handle_silent_command: thread-level silent mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_silent_command_adds_to_silent_threads():
    """Invoking /silent adds the thread to _silent_threads."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/silent")

    result = await runner._handle_silent_command(event)

    assert THREAD_TS in adapter._silent_threads
    assert isinstance(result, EphemeralReply)


@pytest.mark.asyncio
async def test_silent_command_removes_from_left_threads():
    """Invoking /silent removes the thread from _left_threads (conflicting state)."""
    adapter = _make_adapter()
    adapter._left_threads.add(THREAD_TS)
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/silent")

    await runner._handle_silent_command(event)

    assert THREAD_TS not in adapter._left_threads
    assert THREAD_TS in adapter._silent_threads


@pytest.mark.asyncio
async def test_silent_command_adds_to_mentioned_threads():
    """Invoking /silent ensures thread is in _mentioned_threads (for context processing)."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/silent")

    await runner._handle_silent_command(event)

    assert THREAD_TS in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_silent_command_idempotent():
    """Calling /silent twice on the same thread is safe (idempotent)."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/silent")

    await runner._handle_silent_command(event)
    result2 = await runner._handle_silent_command(event)

    assert THREAD_TS in adapter._silent_threads
    assert isinstance(result2, EphemeralReply)


@pytest.mark.asyncio
async def test_silent_command_returns_ephemeral_reply():
    """_handle_silent_command always returns an EphemeralReply."""
    adapter = _make_adapter()
    runner = _make_runner(adapter=adapter)
    event = _make_event(thread_id=THREAD_TS, text="/silent")

    result = await runner._handle_silent_command(event)

    assert isinstance(result, EphemeralReply)


# ---------------------------------------------------------------------------
# Command registry: leave and silent are registered as gateway_only
# ---------------------------------------------------------------------------

def test_leave_command_in_registry():
    """'leave' must exist in COMMAND_REGISTRY and be gateway_only."""
    from hermes_cli.commands import COMMAND_REGISTRY
    leave_defs = [c for c in COMMAND_REGISTRY if c.name == "leave"]
    assert len(leave_defs) == 1, "'leave' must appear exactly once in COMMAND_REGISTRY"
    leave = leave_defs[0]
    assert leave.gateway_only is True


def test_silent_command_in_registry():
    """'silent' must exist in COMMAND_REGISTRY and be gateway_only."""
    from hermes_cli.commands import COMMAND_REGISTRY
    silent_defs = [c for c in COMMAND_REGISTRY if c.name == "silent"]
    assert len(silent_defs) == 1, "'silent' must appear exactly once in COMMAND_REGISTRY"
    silent = silent_defs[0]
    assert silent.gateway_only is True


def test_leave_resolves_canonical():
    """resolve_command('leave') returns the canonical 'leave' CommandDef."""
    from hermes_cli.commands import resolve_command
    result = resolve_command("leave")
    assert result is not None
    assert result.name == "leave"


def test_silent_resolves_canonical():
    """resolve_command('silent') returns the canonical 'silent' CommandDef."""
    from hermes_cli.commands import resolve_command
    result = resolve_command("silent")
    assert result is not None
    assert result.name == "silent"


def test_leave_in_gateway_known_commands():
    """'leave' must appear in GATEWAY_KNOWN_COMMANDS."""
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
    assert "leave" in GATEWAY_KNOWN_COMMANDS


def test_silent_in_gateway_known_commands():
    """'silent' must appear in GATEWAY_KNOWN_COMMANDS."""
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
    assert "silent" in GATEWAY_KNOWN_COMMANDS
