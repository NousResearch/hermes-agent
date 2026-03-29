"""Tests for Telegram channel post handling.

Telegram channels post updates come through update.channel_post (and 
update.edited_channel_post) rather than update.message. The adapter should
handle these appropriately and route them like other messages.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_adapter():
    """Create a minimal TelegramAdapter for testing channel post handling."""
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    # Channel post handlers also need these
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_batch_delay_seconds = 0.1
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    return adapter


def _make_channel_message(text: str = "Hello from channel", channel_id: int = -1001234567890):
    """Create a mock Telegram Message representing a channel post."""
    msg = MagicMock()
    msg.text = text
    msg.caption = None
    msg.message_id = 42
    msg.date = None
    msg.reply_to_message = None
    msg.message_thread_id = None
    msg.forum_topic_created = None
    msg.photo = None
    msg.video = None
    msg.audio = None
    msg.voice = None
    msg.document = None
    msg.sticker = None
    msg.media_group_id = None
    
    # Channel posts have no from_user, but have sender_chat
    msg.from_user = None
    
    # sender_chat is the channel itself
    sender_chat = MagicMock()
    sender_chat.id = channel_id
    sender_chat.title = "Test Channel"
    msg.sender_chat = sender_chat
    
    # Chat is the channel
    chat = MagicMock()
    chat.id = channel_id
    chat.title = "Test Channel"
    chat.type = "channel"
    chat.full_name = None
    msg.chat = chat
    
    return msg


def _make_channel_update(msg):
    """Create a mock Update with channel_post set."""
    update = MagicMock()
    update.message = None
    update.channel_post = msg
    update.edited_channel_post = None
    return update


def _make_edited_channel_update(msg):
    """Create a mock Update with edited_channel_post set."""
    update = MagicMock()
    update.message = None
    update.channel_post = None
    update.edited_channel_post = msg
    return update


class TestChannelPostHandling:
    @pytest.mark.asyncio
    async def test_channel_text_message_processed(self):
        """Channel text posts should be processed and dispatched."""
        adapter = _make_adapter()
        msg = _make_channel_message(text="Breaking news from the channel!")
        update = _make_channel_update(msg)
        context = MagicMock()

        await adapter._handle_channel_text_message(update, context)

        # Wait for flush (text batching)
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "Breaking news from the channel!"
        assert dispatched.source.chat_type == "channel"

    @pytest.mark.asyncio
    async def test_edited_channel_post_processed(self):
        """Edited channel posts should also be processed."""
        adapter = _make_adapter()
        msg = _make_channel_message(text="Edited message")
        update = _make_edited_channel_update(msg)
        context = MagicMock()

        await adapter._handle_channel_text_message(update, context)
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "Edited message"

    @pytest.mark.asyncio
    async def test_channel_command_processed(self):
        """Channel command posts should be processed immediately (no batching)."""
        adapter = _make_adapter()
        msg = _make_channel_message(text="/status check")
        update = _make_channel_update(msg)
        context = MagicMock()

        await adapter._handle_channel_command(update, context)

        # Commands don't use batching, should be immediate
        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "/status check"
        assert dispatched.message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_channel_user_info_from_sender_chat(self):
        """Channel posts should extract user info from sender_chat."""
        adapter = _make_adapter()
        msg = _make_channel_message(channel_id=-1009999888777)
        msg.sender_chat.title = "My News Channel"
        update = _make_channel_update(msg)
        context = MagicMock()

        await adapter._handle_channel_text_message(update, context)
        await asyncio.sleep(0.2)

        dispatched = adapter.handle_message.call_args[0][0]
        # user_id should be the channel ID
        assert dispatched.source.user_id == "-1009999888777"
        # user_name should be the channel title
        assert dispatched.source.user_name == "My News Channel"

    @pytest.mark.asyncio
    async def test_null_channel_post_ignored(self):
        """Updates with null channel_post should be ignored gracefully."""
        adapter = _make_adapter()
        update = MagicMock()
        update.channel_post = None
        update.edited_channel_post = None
        context = MagicMock()

        await adapter._handle_channel_text_message(update, context)

        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_post_without_text_ignored(self):
        """Channel posts without text should be ignored by text handler."""
        adapter = _make_adapter()
        msg = _make_channel_message(text=None)
        msg.text = None
        update = _make_channel_update(msg)
        context = MagicMock()

        await adapter._handle_channel_text_message(update, context)
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()


class TestBuildMessageEventForChannels:
    def test_build_event_with_sender_chat(self):
        """_build_message_event should handle channel posts with sender_chat."""
        from gateway.platforms.telegram import TelegramAdapter
        from telegram.constants import ChatType
        
        # Skip if telegram not available
        try:
            from telegram import Message
        except ImportError:
            pytest.skip("python-telegram-bot not installed")
        
        adapter = _make_adapter()
        adapter.name = "telegram"
        
        msg = _make_channel_message(text="Channel content")
        # Mock ChatType for the test
        msg.chat.type = ChatType.CHANNEL
        
        event = adapter._build_message_event(msg, MessageType.TEXT)
        
        assert event.text == "Channel content"
        assert event.source.chat_type == "channel"
        assert event.source.user_id == str(msg.sender_chat.id)
        assert event.source.user_name == msg.sender_chat.title
