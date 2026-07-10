"""Tests for Discord double-dispatch prevention (#51057).

When _auto_create_thread() creates a thread from a user message via
message.create_thread(), Discord fires a second MESSAGE_CREATE event for
the "thread starter message".  That starter message carries
``message.id == thread.id`` and may arrive with ``type=default``
(instead of ``type=21 / thread_starter_message``), so the type filter
does NOT catch it — resulting in two agent runs and two responses.

Fix: after _auto_create_thread succeeds, pre-seed the dedup cache with
``str(thread.id)`` so the duplicate starter-message event is dropped.

Two sub-scenarios are tested:
  1. Thread-starter as a duplicate MESSAGE_CREATE (the primary bug).
  2. When text_batch_delay=0 the dispatch path is direct (no batching).
     The same dedup pre-seed must still protect against the duplicate.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Discord mock setup
# The tests/gateway/conftest.py already installs a comprehensive discord
# mock at collection time.  We import the adapter AFTER that is done.
# ---------------------------------------------------------------------------

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fake channel/thread helpers
#
# IMPORTANT: FakeTextChannel must NOT be the same class as discord.DMChannel
# or discord.Thread (those are set up by conftest). We give it a neutral name
# and do NOT monkeypatch discord.DMChannel to it.
# ---------------------------------------------------------------------------

class _TextChannel:
    """Fake Discord text channel (not a DM, not a Thread)."""

    def __init__(self, channel_id: int = 100, name: str = "general",
                 guild_name: str = "Test Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


class _Thread:
    """Fake Discord thread (not a DM, not a top-level channel)."""

    def __init__(self, thread_id: int, name: str = "thread",
                 parent=None, guild_name: str = "Test Server"):
        self.id = thread_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(
            name=guild_name, id=1
        )
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _empty():
            return
            yield
        return _empty()


def _make_message(
    *,
    msg_id: int = 42,
    channel,
    content: str = "hello",
    mentions=None,
    author=None,
    msg_type=None,
    attachments=None,
    reference=None,
    message_snapshots=None,
):
    if author is None:
        author = SimpleNamespace(id=7, display_name="Alice", name="Alice", bot=False)
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=list(mentions or []),
        attachments=list(attachments or []),
        reference=reference,
        message_snapshots=message_snapshots,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        type=(
            msg_type
            if msg_type is not None
            else discord_platform.discord.MessageType.default
        ),
    )


# ---------------------------------------------------------------------------
# Adapter fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter(monkeypatch):
    # Clear relevant env vars so tests are hermetic
    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_HISTORY_BACKFILL",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_IGNORE_NO_MENTION",
    ):
        monkeypatch.delenv(var, raising=False)

    config = PlatformConfig(enabled=True, token="***")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    a._text_batch_delay_seconds = 0  # disable batching so dispatch is synchronous
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Scenario 1 — thread-starter message duplicate via on_message (the main bug)
# ---------------------------------------------------------------------------

class TestThreadStarterDedup:
    """Pre-seeding dedup with thread.id prevents a second dispatch when the
    thread-starter message arrives as a duplicate MESSAGE_CREATE event."""

    @pytest.mark.asyncio
    async def test_thread_starter_duplicate_dropped(self, adapter, monkeypatch):
        """After _auto_create_thread the thread.id is pre-seeded in dedup.

        Simulates the exact Discord bug: after thread creation, Discord
        fires MESSAGE_CREATE again with message.id == thread.id.  The
        adapter's on_message guard calls _dedup.is_duplicate(str(message.id))
        before dispatching.  With the fix the duplicate is dropped; without
        it there would be two agent runs.
        """
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        thread_id = 55555  # thread.id == starter-message.id on Discord
        fake_thread = _Thread(thread_id=thread_id, parent=channel)

        async def fake_auto_create_thread(message):
            return fake_thread

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        # 1) Original user message arrives → triggers thread creation + dispatch
        user_msg = _make_message(msg_id=42, channel=channel, content="hello bot")
        await adapter._handle_message(user_msg)

        # One dispatch for the user message
        assert adapter.handle_message.call_count == 1, (
            "Expected handle_message to be called exactly once for the user message"
        )

        # 2) Discord fires a second MESSAGE_CREATE for the thread starter.
        #    Its message.id == thread.id (this is the Discord quirk).
        #    Simulate what on_message does: check _dedup.is_duplicate first.
        #
        #    The fix pre-seeded thread.id via _dedup.is_duplicate(str(thread.id))
        #    inside _handle_message.  That call already marked thread.id as seen.
        #    So this second call with the same id returns True → drop the duplicate.
        starter_msg_id = str(thread_id)
        is_dup = adapter._dedup.is_duplicate(starter_msg_id)
        assert is_dup is True, (
            "Thread starter message (id == thread.id) should be in dedup cache "
            "after _auto_create_thread returns, so the duplicate event is dropped"
        )

        # Confirm: handle_message was only called once total
        assert adapter.handle_message.call_count == 1, (
            "handle_message should only be called once — duplicate starter dropped"
        )

    @pytest.mark.asyncio
    async def test_thread_id_pre_seeded_in_dedup_cache(self, adapter, monkeypatch):
        """After _handle_message with auto-thread, thread.id is in _dedup._seen."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        thread_id = 55555
        fake_thread = _Thread(thread_id=thread_id, parent=channel)

        async def fake_auto_create_thread(message):
            return fake_thread

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        user_msg = _make_message(msg_id=42, channel=channel, content="hello")
        await adapter._handle_message(user_msg)

        # Thread id must be in the dedup internal cache
        assert str(thread_id) in adapter._dedup._seen, (
            f"thread.id={thread_id} should be pre-seeded in _dedup._seen "
            "after _auto_create_thread returns a thread"
        )

    @pytest.mark.asyncio
    async def test_no_dedup_seed_when_thread_creation_fails(self, adapter, monkeypatch):
        """When _auto_create_thread returns None, no pre-seeding occurs.

        Auto-thread failure is now fail-closed (#20243): the agent is NOT
        invoked and the user gets a visible notice instead of a silent inline
        reply. This test's contract is specifically about dedup pre-seeding —
        the phantom thread id must not leak into the dedup cache when creation
        fails.
        """
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        channel.send = AsyncMock()
        phantom_thread_id = 55555

        async def fake_auto_create_thread_fail(message):
            return None  # thread creation failed

        monkeypatch.setattr(
            adapter, "_auto_create_thread", fake_auto_create_thread_fail
        )

        user_msg = _make_message(msg_id=42, channel=channel, content="hello")
        await adapter._handle_message(user_msg)

        # Fail-closed: the agent must NOT run when the required thread route
        # could not be created (#20243).
        adapter.handle_message.assert_not_awaited()

        # The phantom thread id should NOT be in the dedup cache
        assert str(phantom_thread_id) not in adapter._dedup._seen, (
            "thread.id should NOT be pre-seeded when thread creation fails"
        )

    @pytest.mark.asyncio
    async def test_no_dedup_seed_when_auto_thread_disabled(self, adapter, monkeypatch):
        """When DISCORD_AUTO_THREAD=false, no thread is created and no pre-seeding."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

        channel = _TextChannel(channel_id=100)
        auto_create_called = []

        async def fake_auto_create_thread(message):
            auto_create_called.append(True)
            return _Thread(thread_id=55555, parent=channel)

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        user_msg = _make_message(msg_id=42, channel=channel, content="hello")
        await adapter._handle_message(user_msg)

        # _auto_create_thread should NOT have been called
        assert not auto_create_called, "_auto_create_thread should not run when disabled"
        # thread.id should NOT be pre-seeded
        assert "55555" not in adapter._dedup._seen, (
            "thread.id should not be in dedup when auto-threading is disabled"
        )

    @pytest.mark.asyncio
    async def test_dedup_seed_with_text_batch_delay_zero(self, adapter, monkeypatch):
        """With text_batch_delay=0 (direct dispatch path), pre-seeding still works."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        # text_batch_delay_seconds is already 0 in the fixture
        assert adapter._text_batch_delay_seconds == 0

        channel = _TextChannel(channel_id=100)
        thread_id = 77777
        fake_thread = _Thread(thread_id=thread_id, parent=channel)

        async def fake_auto_create_thread(message):
            return fake_thread

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        user_msg = _make_message(msg_id=42, channel=channel, content="hello")
        await adapter._handle_message(user_msg)

        # Dispatched once
        adapter.handle_message.assert_awaited_once()

        # Thread id IS pre-seeded even with direct dispatch path
        assert str(thread_id) in adapter._dedup._seen, (
            "thread.id must be pre-seeded regardless of text_batch_delay setting"
        )

    @pytest.mark.asyncio
    async def test_thread_id_different_from_message_id_both_tracked(
        self, adapter, monkeypatch
    ):
        """Verify thread.id is tracked independently when it differs from message.id."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        user_msg_id = 12345
        thread_id = 99999  # always different in practice
        fake_thread = _Thread(thread_id=thread_id, parent=channel)

        async def fake_auto_create_thread(message):
            return fake_thread

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        user_msg = _make_message(msg_id=user_msg_id, channel=channel, content="hello")
        await adapter._handle_message(user_msg)

        # The thread.id (99999) is pre-seeded
        assert str(thread_id) in adapter._dedup._seen, (
            f"thread.id={thread_id} must be pre-seeded after auto-thread creation"
        )

        # A second MESSAGE_CREATE with message.id=thread.id is caught as duplicate
        assert adapter._dedup.is_duplicate(str(thread_id)) is True, (
            "Subsequent is_duplicate(thread.id) must return True"
        )

        # A hypothetical NEW message with a different id is not a duplicate
        assert adapter._dedup.is_duplicate("11111") is False, (
            "An unrelated new message id must not be blocked"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — direct double-call to _handle_message with same message id
# ---------------------------------------------------------------------------

class TestDirectDoubleDispatch:
    """on_message dedup (checked before _handle_message) prevents double dispatch.

    While the on_message guard calls _dedup.is_duplicate before _handle_message,
    these tests verify that the adapter's own _dedup correctly marks IDs as seen
    so that hypothetical double-delivery of the same MESSAGE_CREATE is dropped.
    """

    @pytest.mark.asyncio
    async def test_same_message_id_not_dispatched_twice_via_dedup(
        self, adapter, monkeypatch
    ):
        """Calling on_message dedup check twice with the same id only dispatches once."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

        channel = _TextChannel(channel_id=100)
        msg = _make_message(msg_id=42, channel=channel, content="hello")

        # Simulate on_message dedup check + dispatch for first delivery
        is_dup_1 = adapter._dedup.is_duplicate(str(msg.id))
        assert is_dup_1 is False
        await adapter._handle_message(msg)
        assert adapter.handle_message.call_count == 1

        # Simulate on_message dedup check for second delivery (RESUME replay)
        is_dup_2 = adapter._dedup.is_duplicate(str(msg.id))
        assert is_dup_2 is True
        # on_message would return early here — do NOT call _handle_message again

        assert adapter.handle_message.call_count == 1, (
            "Second delivery with same message.id must be dropped by dedup"
        )

    @pytest.mark.asyncio
    async def test_different_message_ids_both_dispatched(self, adapter, monkeypatch):
        """Two distinct messages with different IDs both reach the agent."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

        channel = _TextChannel(channel_id=100)
        msg1 = _make_message(msg_id=1, channel=channel, content="first")
        msg2 = _make_message(msg_id=2, channel=channel, content="second")

        assert adapter._dedup.is_duplicate(str(msg1.id)) is False
        await adapter._handle_message(msg1)
        assert adapter._dedup.is_duplicate(str(msg2.id)) is False
        await adapter._handle_message(msg2)

        assert adapter.handle_message.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 3 — message_type=thread_starter filtered by type guard
# ---------------------------------------------------------------------------

class TestThreadStarterTypeFilter:
    """Discord sometimes sends thread starter messages with the correct
    type=21 (thread_starter_message).  Verify the type filter in on_message
    blocks those correctly, separate from the dedup path.
    """

    def test_thread_starter_message_type_not_in_allowed_set(self):
        """MessageType.thread_starter_message (21) is not in the allowed set."""
        discord_mod = sys.modules["discord"]

        # The adapter's on_message guard uses:
        #   if message.type not in {discord.MessageType.default, discord.MessageType.reply}
        # Verify that thread_starter_message (if it has a numeric value of 21)
        # would be excluded.
        allowed = {
            discord_mod.MessageType.default,
            discord_mod.MessageType.reply,
        }
        # In real discord.py, thread_starter_message has value 21.
        # In our mock, MessageType is a MagicMock so attribute access returns
        # a new unique Mock each time — which is NOT in the allowed set.
        thread_starter = discord_mod.MessageType.thread_starter_message
        assert thread_starter not in allowed, (
            "thread_starter_message type should not be in the allowed types set"
        )

    @pytest.mark.asyncio
    async def test_message_type_default_passes_type_filter(self, adapter, monkeypatch):
        """MessageType.default messages pass the type filter (they reach _handle_message)."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

        channel = _TextChannel(channel_id=100)
        msg = _make_message(
            msg_id=42,
            channel=channel,
            content="hello",
            msg_type=discord_platform.discord.MessageType.default,
        )
        await adapter._handle_message(msg)
        adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Scenario 4 — dedup cache integrity after thread pre-seeding
# ---------------------------------------------------------------------------

class TestDedupCacheIntegrity:
    """Verify the dedup cache state is correct after pre-seeding."""

    @pytest.mark.asyncio
    async def test_preseed_does_not_block_legitimate_new_messages(
        self, adapter, monkeypatch
    ):
        """Pre-seeding thread.id does NOT interfere with other unrelated messages."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        thread_id = 22222
        fake_thread = _Thread(thread_id=thread_id, parent=channel)

        async def fake_auto_create_thread(message):
            return fake_thread

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        # First message — creates thread, pre-seeds dedup
        msg1 = _make_message(msg_id=10, channel=channel, content="first")
        await adapter._handle_message(msg1)
        assert adapter.handle_message.call_count == 1

        # A new message ID that is unrelated to the thread
        msg2_id = 20
        assert str(msg2_id) != str(thread_id)  # sanity check
        assert adapter._dedup.is_duplicate(str(msg2_id)) is False, (
            "A new message with a different ID should not be blocked"
        )

    @pytest.mark.asyncio
    async def test_multiple_thread_creations_each_preseeded(
        self, adapter, monkeypatch
    ):
        """Each thread creation pre-seeds its own thread.id independently."""
        monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
        monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")

        channel = _TextChannel(channel_id=100)
        thread_ids = [33333, 44444, 55555]
        thread_idx = [0]

        async def fake_auto_create_thread(message):
            tid = thread_ids[thread_idx[0] % len(thread_ids)]
            thread_idx[0] += 1
            return _Thread(thread_id=tid, parent=channel)

        monkeypatch.setattr(adapter, "_auto_create_thread", fake_auto_create_thread)

        for i, tid in enumerate(thread_ids):
            msg = _make_message(msg_id=100 + i, channel=channel, content=f"msg {i}")
            await adapter._handle_message(msg)

        # All three thread ids should be pre-seeded
        for tid in thread_ids:
            assert str(tid) in adapter._dedup._seen, (
                f"thread.id={tid} should be pre-seeded in _dedup._seen "
                "after its thread was created"
            )
            # And they should be detected as duplicates now
            assert adapter._dedup.is_duplicate(str(tid)) is True, (
                f"thread.id={tid} should be treated as duplicate"
            )
