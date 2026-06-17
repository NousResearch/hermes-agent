"""Test: concurrent on_message events don't race on text batch dicts.

When Discord splits a long user message at 2000 chars, chunks arrive as
concurrent on_message events. _enqueue_text_event mutates
_pending_text_batches and _pending_text_batch_tasks dicts — without a
lock, two concurrent calls can corrupt the dicts (lost update on
existing.text, double task creation).

Uses asyncio.gather to simulate two near-simultaneous text events
with the same batch key.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType


@pytest.mark.asyncio
async def test_concurrent_enqueue_text_events_do_not_race():
    """Two text events with same batch key: the merge wins, no corruption."""

    from plugins.platforms.discord.adapter import DiscordAdapter

    # Build a minimal adapter — enough to exercise _enqueue_text_event
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = "discord"
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._voice_output_callback = None
    adapter._loop = asyncio.get_event_loop()
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_lock = asyncio.Lock()
    adapter._threads = MagicMock()

    # Mock _text_batch_key to return same key for both events
    adapter._text_batch_key = MagicMock(return_value="session:test:123")

    # Mock handle_message to not actually run
    adapter.handle_message = AsyncMock()

    # Build two events with same key (simulating Discord split)
    source = MagicMock()
    source.platform = "discord"
    source.channel_id = "123"

    event1 = MessageEvent(
        source=source,
        text="First chunk of a long message that",
        message_type=MessageType.TEXT,
    )
    event2 = MessageEvent(
        source=source,
        text=" continues here",
        message_type=MessageType.TEXT,
    )

    # Fire both concurrently — like two near-simultaneous on_message
    await asyncio.gather(
        adapter._enqueue_text_event(event1),
        adapter._enqueue_text_event(event2),
    )

    # After gather, exactly one batch should be pending (merged)
    assert len(adapter._pending_text_batches) == 1
    merged = adapter._pending_text_batches.get("session:test:123")
    assert merged is not None
    assert merged.text == "First chunk of a long message that\n continues here"

    # Exactly one flush task should exist
    assert len(adapter._pending_text_batch_tasks) == 1
    task = adapter._pending_text_batch_tasks["session:test:123"]
    assert not task.done()


@pytest.mark.asyncio
async def test_concurrent_events_different_keys_no_interference():
    """Events from different sessions don't interfere under lock."""

    from plugins.platforms.discord.adapter import DiscordAdapter

    # Same minimal adapter setup
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = "discord"
    adapter.config = PlatformConfig(enabled=True, token="t")
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._voice_output_callback = None
    adapter._loop = asyncio.get_event_loop()
    adapter._text_batch_delay_seconds = 0.6
    adapter._text_batch_split_delay_seconds = 2.0
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_lock = asyncio.Lock()
    adapter._threads = MagicMock()

    # Each event gets its own session key
    adapter._text_batch_key = MagicMock(side_effect=lambda e: f"session:{e.source.channel_id}")

    adapter.handle_message = AsyncMock()

    src1 = MagicMock(platform="discord", channel_id="1")
    src2 = MagicMock(platform="discord", channel_id="2")

    ev1 = MessageEvent(source=src1, text="hello from ch1", message_type=MessageType.TEXT)
    ev2 = MessageEvent(source=src2, text="hello from ch2", message_type=MessageType.TEXT)

    await asyncio.gather(
        adapter._enqueue_text_event(ev1),
        adapter._enqueue_text_event(ev2),
    )

    assert len(adapter._pending_text_batches) == 2
    assert adapter._pending_text_batches["session:1"].text == "hello from ch1"
    assert adapter._pending_text_batches["session:2"].text == "hello from ch2"
    assert len(adapter._pending_text_batch_tasks) == 2
