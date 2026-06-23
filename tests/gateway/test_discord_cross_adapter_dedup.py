"""Cross-adapter message dedup in GatewayRunner.

Covers the reconnect scenario where a new DiscordAdapter instance is created
after a fatal error and the same Discord message_id is delivered again via
RESUME or at-least-once delivery.  The per-adapter _dedup is fresh on the new
instance and would normally allow the message through; the gateway-level
_platform_dedup must block it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_source(platform=Platform.DISCORD):
    return SessionSource(
        platform=platform,
        chat_id="chan-1",
        chat_type="group",
        user_id="user-99",
    )


def _make_event(message_id="1234567890", platform=Platform.DISCORD):
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=_make_source(platform),
        message_id=message_id,
    )


def _make_runner():
    """Minimal GatewayRunner stub — only the pieces _handle_message needs."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._platform_dedup = {}
    runner._startup_restore_in_progress = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._draining = False
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._update_prompt_pending = {}
    return runner


# ---------------------------------------------------------------------------
# Core dedup behaviour
# ---------------------------------------------------------------------------


def test_same_message_id_blocked_on_second_call():
    """Second delivery of the same message_id for the same platform is dropped."""
    from gateway.platforms.helpers import MessageDeduplicator

    dedup = MessageDeduplicator()
    msg_id = "9876543210"

    assert dedup.is_duplicate(msg_id) is False  # first delivery passes
    assert dedup.is_duplicate(msg_id) is True   # RESUME replay is blocked


@pytest.mark.asyncio
async def test_different_message_ids_both_pass():
    """Two messages with distinct IDs from the same platform are both allowed."""
    from gateway.platforms.helpers import MessageDeduplicator
    dedup = MessageDeduplicator()

    assert dedup.is_duplicate("AAA") is False
    assert dedup.is_duplicate("BBB") is False
    # First IDs still blocked
    assert dedup.is_duplicate("AAA") is True
    assert dedup.is_duplicate("BBB") is True


@pytest.mark.asyncio
async def test_different_platforms_have_independent_dedups():
    """Discord and Telegram dedups are independent — same ID value is fine across platforms."""
    from gateway.platforms.helpers import MessageDeduplicator
    discord_dedup = MessageDeduplicator()
    telegram_dedup = MessageDeduplicator()

    shared_id = "11111111"
    assert discord_dedup.is_duplicate(shared_id) is False  # registered in Discord
    assert telegram_dedup.is_duplicate(shared_id) is False  # independent; passes


@pytest.mark.asyncio
async def test_internal_events_bypass_dedup():
    """Events with internal=True have no message_id and must not be blocked."""
    from gateway.platforms.helpers import MessageDeduplicator
    dedup = MessageDeduplicator()

    # Internal events carry no message_id (None/falsy)
    assert dedup.is_duplicate("") is False   # falsy → pass
    assert dedup.is_duplicate("") is False   # falsy → pass again (never registered)


@pytest.mark.asyncio
async def test_none_message_id_never_registered():
    """is_duplicate(None) and is_duplicate('') return False and never register."""
    from gateway.platforms.helpers import MessageDeduplicator
    dedup = MessageDeduplicator()

    # Calling with None-equivalent IDs multiple times — never blocks
    for _ in range(5):
        assert dedup.is_duplicate("") is False

    # A real ID after None-calls is still fresh
    assert dedup.is_duplicate("real-id-42") is False
    assert dedup.is_duplicate("real-id-42") is True


# ---------------------------------------------------------------------------
# GatewayRunner._platform_dedup initialised in __init__
# ---------------------------------------------------------------------------


def test_platform_dedup_initialised_empty():
    """GatewayRunner starts with an empty _platform_dedup dict."""
    import sys
    import types

    # Stub heavy transitive imports
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **kw: None
    sys.modules.setdefault("dotenv", fake_dotenv)

    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    runner = GatewayRunner(GatewayConfig())
    assert hasattr(runner, "_platform_dedup")
    assert isinstance(runner._platform_dedup, dict)
    assert len(runner._platform_dedup) == 0


# ---------------------------------------------------------------------------
# Reconnect simulation: new adapter sees message already processed by old one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_adapter_cannot_replay_message_seen_by_old_adapter():
    """Simulate: adapter-A processes msg-X, crashes, adapter-B created fresh.

    The per-adapter _dedup on B is empty, but the gateway _platform_dedup
    still has msg-X registered from A's processing — B's delivery is blocked.
    """
    from gateway.platforms.helpers import MessageDeduplicator

    # Shared gateway-level dedup (survives adapter recreation)
    gw_dedup = MessageDeduplicator()

    msg_id = "discord-snowflake-555"

    # Adapter A processes the message
    adapter_a_dedup = MessageDeduplicator()
    assert adapter_a_dedup.is_duplicate(msg_id) is False  # A's per-adapter: pass
    assert gw_dedup.is_duplicate(msg_id) is False          # gateway: pass (first call)

    # Adapter A crashes → new adapter B created with fresh per-adapter dedup
    adapter_b_dedup = MessageDeduplicator()

    # Discord RESUME delivers msg-X to adapter B
    assert adapter_b_dedup.is_duplicate(msg_id) is False  # B's per-adapter: WOULD PASS
    # BUT gateway dedup catches it
    assert gw_dedup.is_duplicate(msg_id) is True           # gateway: BLOCKED ✓


# ---------------------------------------------------------------------------
# RCA-A: content-based dedup (channel vs thread, different snowflake IDs)
# ---------------------------------------------------------------------------


def test_content_dedup_blocks_same_text_within_ttl():
    """Same user text on a different context (different ID) is dropped within TTL."""
    from gateway.platforms.helpers import MessageDeduplicator

    dedup = MessageDeduplicator(max_size=500, ttl_seconds=5)
    user_id = "user-42"
    text = "hello world"
    key = f"u:{user_id}:{text[:500]}"

    assert dedup.is_duplicate(key) is False  # channel delivery → passes
    assert dedup.is_duplicate(key) is True   # thread delivery (different ID) → blocked


def test_content_dedup_allows_different_text_from_same_user():
    """Two different messages from the same user are both allowed."""
    from gateway.platforms.helpers import MessageDeduplicator

    dedup = MessageDeduplicator(max_size=500, ttl_seconds=5)
    user_id = "user-42"
    key1 = f"u:{user_id}:hello world"
    key2 = f"u:{user_id}:different message"

    assert dedup.is_duplicate(key1) is False
    assert dedup.is_duplicate(key2) is False


def test_content_dedup_allows_same_text_different_users():
    """Two users sending identical text are both processed independently."""
    from gateway.platforms.helpers import MessageDeduplicator

    dedup = MessageDeduplicator(max_size=500, ttl_seconds=5)
    key_a = "u:user-A:hello"
    key_b = "u:user-B:hello"

    assert dedup.is_duplicate(key_a) is False
    assert dedup.is_duplicate(key_b) is False  # different user → not a duplicate


@pytest.mark.asyncio
async def test_handle_message_defensive_platform_dedup():
    """GatewayRunner._handle_message handles stubs where _platform_dedup is missing."""
    runner = _make_runner()
    # Deliberately remove the attribute to simulate a bare test stub
    if hasattr(runner, "_platform_dedup"):
        delattr(runner, "_platform_dedup")

    event = _make_event(message_id="msg-999")

    # Under the hood, this will run through our defensive fallback check
    # and shouldn't raise AttributeError.
    from gateway.run import GatewayRunner
    await GatewayRunner._handle_message(runner, event)
    # The call should succeed and initialize the attribute
    assert hasattr(runner, "_platform_dedup")
    assert isinstance(runner._platform_dedup, dict)

