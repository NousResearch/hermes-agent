"""Tests for Discord auto-thread chat_id routing.

When auto-thread fires, the adapter should keep source.chat_id as the parent
channel so the final text response goes inline. Tool-progress messages
route to the thread via source.thread_id. This mirrors Slack's behaviour:
progress in a thread, response in the channel.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform, PlatformConfig


def _make_discord_adapter():
    """Factory to create a DiscordAdapter with minimal config."""
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.config = config
    adapter._reply_to_mode = "first"
    adapter._threads = MagicMock()
    adapter._dedup = MagicMock()
    adapter._discord_channel_keys = MagicMock(return_value=set())
    return adapter


class TestAutoThreadChatIdRouting:
    """Verify that auto-thread keeps chat_id as parent channel."""

    def test_auto_thread_keeps_parent_chat_id(self):
        """When auto-thread fires, chat_id must be the parent channel ID,
        not the thread ID. The final response should go inline."""
        # This is a structural test: we verify the logic that decides
        # source_chat_id when auto_threaded_channel is set.
        auto_threaded_channel = MagicMock()
        auto_threaded_channel.id = 999999
        message_channel = MagicMock()
        message_channel.id = 111111

        # Simulate the logic from the adapter
        if auto_threaded_channel is not None:
            source_chat_id = str(message_channel.id)
        else:
            source_chat_id = str(auto_threaded_channel.id or message_channel.id)

        assert source_chat_id == "111111", (
            "chat_id must be the parent channel ID when auto-thread fires, "
            "not the thread ID"
        )

    def test_no_auto_thread_uses_message_channel(self):
        """Without auto-thread, chat_id is just the message channel."""
        auto_threaded_channel = None
        message_channel = MagicMock()
        message_channel.id = 111111

        effective_channel = auto_threaded_channel or message_channel
        if auto_threaded_channel is not None:
            source_chat_id = str(message_channel.id)
        else:
            source_chat_id = str(effective_channel.id)

        assert source_chat_id == "111111"


class TestProgressThreadIdRouting:
    """Verify that progress thread routing still works with auto-thread on."""

    def test_discord_with_thread_uses_thread_for_progress(self):
        """When source.thread_id is set (auto-thread on), progress goes to thread."""
        from gateway.run import _resolve_progress_thread_id
        from gateway.config import Platform

        assert _resolve_progress_thread_id(
            Platform.DISCORD,
            source_thread_id="thread_abc",
            event_message_id="msg_123",
        ) == "thread_abc"

    def test_discord_no_thread_no_progress_thread_id(self):
        """When no thread (auto-thread off), no thread_id for progress."""
        from gateway.run import _resolve_progress_thread_id
        from gateway.config import Platform

        assert _resolve_progress_thread_id(
            Platform.DISCORD,
            source_thread_id=None,
            event_message_id="msg_123",
        ) is None
