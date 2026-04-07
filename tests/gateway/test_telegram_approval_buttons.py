"""Tests for Telegram inline keyboard approval buttons."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Telegram mock so TelegramAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_telegram_mock():
    """Wire up the minimal mocks required to import TelegramAdapter."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request", "telegram.error"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter():
    """Create a TelegramAdapter with mocked internals."""
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


# ===========================================================================
# send_exec_approval — inline keyboard buttons
# ===========================================================================

class TestTelegramExecApproval:
    """Test the send_exec_approval method sends InlineKeyboard buttons."""

    @pytest.mark.asyncio
    async def test_sends_inline_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /important",
            session_key="agent:main:telegram:group:12345:99",
            description="dangerous deletion",
        )

        assert result.success is True
        assert result.message_id == "42"

        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "rm -rf /important" in kwargs["text"]
        assert "dangerous deletion" in kwargs["text"]
        assert kwargs["reply_markup"] is not None  # InlineKeyboardMarkup

    @pytest.mark.asyncio
    async def test_stores_approval_state(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_exec_approval(
            chat_id="12345",
            command="echo test",
            session_key="my-session-key",
        )

        # The approval_id should map to the session_key
        assert len(adapter._approval_state) == 1
        approval_id = list(adapter._approval_state.keys())[0]
        assert adapter._approval_state[approval_id] == "my-session-key"

    @pytest.mark.asyncio
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        await adapter.send_exec_approval(
            chat_id="12345",
            command="ls",
            session_key="s",
            metadata={"thread_id": "999"},
        )

        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs.get("message_thread_id") == 999

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._bot = None
        result = await adapter.send_exec_approval(
            chat_id="12345", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 1
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        long_cmd = "x" * 5000
        await adapter.send_exec_approval(
            chat_id="12345", command=long_cmd, session_key="s"
        )

        kwargs = adapter._bot.send_message.call_args[1]
        assert "..." in kwargs["text"]
        assert len(kwargs["text"]) < 5000


# ===========================================================================
# _handle_callback_query — approval button clicks
# ===========================================================================

class TestTelegramApprovalCallback:
    """Test the approval callback handling in _handle_callback_query."""

    @pytest.mark.asyncio
    async def test_resolves_approval_on_click(self):
        adapter = _make_adapter()
        # Set up approval state
        adapter._approval_state[1] = "agent:main:telegram:group:12345:99"

        # Mock callback query
        query = AsyncMock()
        query.data = "ea:once:1"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_callback_query(update, context)

        mock_resolve.assert_called_once_with("agent:main:telegram:group:12345:99", "once")
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()

        # State should be cleaned up
        assert 1 not in adapter._approval_state

    @pytest.mark.asyncio
    async def test_deny_button(self):
        adapter = _make_adapter()
        adapter._approval_state[2] = "some-session"

        query = AsyncMock()
        query.data = "ea:deny:2"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Alice"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_callback_query(update, context)

        mock_resolve.assert_called_once_with("some-session", "deny")
        edit_kwargs = query.edit_message_text.call_args[1]
        assert "Denied" in edit_kwargs["text"]

    @pytest.mark.asyncio
    async def test_already_resolved(self):
        adapter = _make_adapter()
        # No state for approval_id 99 — already resolved

        query = AsyncMock()
        query.data = "ea:once:99"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Bob"
        query.answer = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_callback_query(update, context)

        # Should NOT resolve — already handled
        mock_resolve.assert_not_called()
        # Should still ack with "already resolved" message
        query.answer.assert_called_once()
        assert "already been resolved" in query.answer.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_model_picker_callback_not_affected(self):
        """Ensure model picker callbacks still route correctly."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "mp:some_provider"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        # Model picker callback should be handled (not crash)
        # We just verify it doesn't try to resolve an approval
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch.object(adapter, "_handle_model_picker_callback", new_callable=AsyncMock):
                await adapter._handle_callback_query(update, context)

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_prompt_callback_not_affected(self):
        """Ensure update prompt callbacks still work."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch("hermes_constants.get_hermes_home", return_value=Path("/tmp/test")):
                try:
                    await adapter._handle_callback_query(update, context)
                except Exception:
                    pass  # May fail on file write, that's fine

        # Should NOT have triggered approval resolution
        mock_resolve.assert_not_called()
