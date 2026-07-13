"""Test Telegram Bot API 10.1 Rich Message extraction and handling.

Telegram Bot API 10.1 introduced Rich Messages (heading, formatted text, list, table)
which may contain their content only in message.rich_message or message.api_kwargs["rich_message"].
These messages have empty message.text and would otherwise be silently ignored.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.skipif(
    not __import__("sys").path,
    reason="tests run in isolated environment"
)
class TestTelegramRichMessageExtraction:
    """Test _extract_rich_message_text method."""

    def test_extract_heading_message(self):
        """Test extraction of a heading block."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        # Mock the config and setup
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {"type": "heading", "size": 2, "text": "Important Heading"}
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "## Important Heading"

    def test_extract_text_message(self):
        """Test extraction of a text block."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {"type": "text", "text": "This is a simple text message."}
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "This is a simple text message."

    def test_extract_list_message(self):
        """Test extraction of a list block."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {
                    "type": "list",
                    "items": [
                        {"text": "First item"},
                        {"text": "Second item"},
                        {"text": "Third item"}
                    ]
                }
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "• First item\n• Second item\n• Third item"

    def test_extract_table_message(self):
        """Test extraction of a table block."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {
                    "type": "table",
                    "rows": [
                        {
                            "cells": [
                                {"text": "Name"},
                                {"text": "Age"}
                            ]
                        },
                        {
                            "cells": [
                                {"text": "Alice"},
                                {"text": "30"}
                            ]
                        }
                    ]
                }
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "Name | Age\nAlice | 30"

    def test_extract_mixed_blocks(self):
        """Test extraction of multiple block types together."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {"type": "heading", "size": 1, "text": "Shopping List"},
                {"type": "list", "items": [{"text": "Milk"}, {"text": "Eggs"}]},
                {"type": "text", "text": "Don't forget the coffee!"}
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "# Shopping List\n• Milk\n• Eggs\nDon't forget the coffee!"

    def test_extract_empty_blocks(self):
        """Test that empty blocks are handled gracefully."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        # Empty blocks dict
        assert adapter._extract_rich_message_text({}) == ""
        # Empty blocks list
        assert adapter._extract_rich_message_text({"blocks": []}) == ""
        # None input
        assert adapter._extract_rich_message_text(None) == ""

    def test_extract_nested_blocks(self):
        """Test extraction of nested blocks."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {
                    "type": "custom",
                    "blocks": [
                        {"type": "text", "text": "Nested text"}
                    ]
                }
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "Nested text"

    def test_extract_heading_size_cap(self):
        """Test that heading size is capped at 6 (Markdown limit)."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {"type": "heading", "size": 10, "text": "Oversized Heading"}
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "###### Oversized Heading"  # Capped at 6 #

    def test_extract_list_string_items(self):
        """Test list with string items (not dict items)."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {
                    "type": "list",
                    "items": ["Item 1", "Item 2"]
                }
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "• Item 1\n• Item 2"

    def test_extract_table_string_cells(self):
        """Test table with string cells (not dict cells)."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = MagicMock(spec=TelegramAdapter)
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)

        rich_message = {
            "blocks": [
                {
                    "type": "table",
                    "rows": [
                        {"cells": ["A", "B"]},
                        {"cells": ["C", "D"]}
                    ]
                }
            ]
        }

        result = adapter._extract_rich_message_text(rich_message)
        assert result == "A | B\nC | D"


@pytest.mark.skipif(
    not __import__("sys").path,
    reason="tests run in isolated environment"
)
class TestTelegramRichMessageHandler:
    """Test _handle_rich_message method."""

    @pytest.mark.asyncio
    async def test_handle_rich_message_from_attribute(self):
        """Test handling rich_message from message.rich_message attribute."""
        from plugins.platforms.telegram.adapter import TelegramAdapter
        from telegram import Update, Message

        # Create a mock adapter
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._effective_update_message = MagicMock(return_value=MagicMock())
        adapter._is_user_authorized_from_message = MagicMock(return_value=True)
        adapter._should_process_message = MagicMock(return_value=True)
        adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
        adapter._ensure_forum_commands = AsyncMock()
        adapter._build_message_event = MagicMock()
        adapter._clean_bot_trigger_text = MagicMock(side_effect=lambda x: x)
        adapter._cache_replied_media = AsyncMock()
        adapter._apply_telegram_group_observe_attribution = MagicMock(side_effect=lambda x: x)
        adapter._enqueue_text_event = MagicMock()

        # Bind the real methods
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)
        adapter._handle_rich_message = TelegramAdapter._handle_rich_message.__get__(adapter, TelegramAdapter)

        # Create a mock message with rich_message attribute
        msg = MagicMock()
        msg.rich_message = {
            "blocks": [{"type": "heading", "size": 1, "text": "Test Heading"}]
        }
        adapter._effective_update_message.return_value = msg

        # Mock the event object
        event = MagicMock()
        adapter._build_message_event.return_value = event

        # Call the handler
        update = MagicMock()
        context = MagicMock()
        await adapter._handle_rich_message(update, context)

        # Verify the event was enqueued with extracted text
        adapter._enqueue_text_event.assert_called_once()
        assert event.text == "# Test Heading"

    @pytest.mark.asyncio
    async def test_handle_rich_message_from_api_kwargs(self):
        """Test handling rich_message from message.api_kwargs fallback."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        # Create a mock adapter
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._effective_update_message = MagicMock(return_value=MagicMock())
        adapter._is_user_authorized_from_message = MagicMock(return_value=True)
        adapter._should_process_message = MagicMock(return_value=True)
        adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
        adapter._ensure_forum_commands = AsyncMock()
        adapter._build_message_event = MagicMock()
        adapter._clean_bot_trigger_text = MagicMock(side_effect=lambda x: x)
        adapter._cache_replied_media = AsyncMock()
        adapter._apply_telegram_group_observe_attribution = MagicMock(side_effect=lambda x: x)
        adapter._enqueue_text_event = MagicMock()

        # Bind the real methods
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)
        adapter._handle_rich_message = TelegramAdapter._handle_rich_message.__get__(adapter, TelegramAdapter)

        # Create a mock message with api_kwargs fallback (no rich_message attribute)
        msg = MagicMock(spec=object)
        # Simulate getattr returning None for rich_message
        type(msg).rich_message = property(lambda self: None)
        msg.api_kwargs = {
            "rich_message": {
                "blocks": [{"type": "text", "text": "Fallback text"}]
            }
        }
        adapter._effective_update_message.return_value = msg

        # Mock the event object
        event = MagicMock()
        adapter._build_message_event.return_value = event

        # Call the handler
        update = MagicMock()
        context = MagicMock()
        await adapter._handle_rich_message(update, context)

        # Verify the event was enqueued with extracted text
        adapter._enqueue_text_event.assert_called_once()
        assert event.text == "Fallback text"

    @pytest.mark.asyncio
    async def test_handle_rich_message_unauthorized_user(self):
        """Test that unauthorized users are rejected."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        # Create a mock adapter
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._effective_update_message = MagicMock(return_value=MagicMock())
        adapter._is_user_authorized_from_message = MagicMock(return_value=False)

        # Bind the real methods
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)
        adapter._handle_rich_message = TelegramAdapter._handle_rich_message.__get__(adapter, TelegramAdapter)

        # Create a mock message
        msg = MagicMock()
        msg.rich_message = {"blocks": [{"type": "text", "text": "Unauthorized"}]}
        msg.from_user = MagicMock(id=123)
        msg.chat = MagicMock(id=456)
        adapter._effective_update_message.return_value = msg

        # Call the handler
        update = MagicMock()
        context = MagicMock()
        await adapter._handle_rich_message(update, context)

        # Verify _should_process_message was not called (early return)
        adapter._should_process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_rich_message_no_extractable_text(self):
        """Test that messages with no extractable text are ignored."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        # Create a mock adapter
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._effective_update_message = MagicMock(return_value=MagicMock())
        adapter._is_user_authorized_from_message = MagicMock(return_value=True)

        # Bind the real methods
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)
        adapter._handle_rich_message = TelegramAdapter._handle_rich_message.__get__(adapter, TelegramAdapter)

        # Create a mock message with empty blocks
        msg = MagicMock()
        msg.rich_message = {"blocks": []}
        adapter._effective_update_message.return_value = msg

        # Call the handler
        update = MagicMock()
        context = MagicMock()
        await adapter._handle_rich_message(update, context)

        # Verify _should_process_message was not called (early return)
        adapter._should_process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_rich_message_no_rich_message(self):
        """Test that messages without rich_message are ignored."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        # Create a mock adapter
        adapter = MagicMock(spec=TelegramAdapter)
        adapter._effective_update_message = MagicMock(return_value=MagicMock())

        # Bind the real methods
        adapter._extract_rich_message_text = TelegramAdapter._extract_rich_message_text.__get__(adapter, TelegramAdapter)
        adapter._handle_rich_message = TelegramAdapter._handle_rich_message.__get__(adapter, TelegramAdapter)

        # Create a mock message without rich_message
        msg = MagicMock(spec=object)
        type(msg).rich_message = property(lambda self: None)
        msg.api_kwargs = {}
        adapter._effective_update_message.return_value = msg

        # Call the handler
        update = MagicMock()
        context = MagicMock()
        await adapter._handle_rich_message(update, context)

        # Verify early return (no rich_message found)
        adapter._is_user_authorized_from_message.assert_not_called()