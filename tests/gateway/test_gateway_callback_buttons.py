"""Tests for the generic inline-button surface and the gateway_callback
fall-through dispatch added for the workflow Stop button (Lane A).

The ``telegram`` mock is installed by tests/gateway/conftest.py at collection
time (shared, overwrite semantics), so this file does NOT define its own — a
per-file mock using setdefault can leave a polluted MagicMock in sys.modules
that breaks sibling telegram tests when files share a process.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


# ===========================================================================
# Generic buttons= kwarg on send / edit_message
# ===========================================================================

class TestButtonsKwarg:
    @pytest.mark.asyncio
    async def test_send_builds_reply_markup_from_buttons(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)
        result = await adapter.send(
            "12345", "hi",
            buttons=[{"text": "⏹ Stop", "callback_data": "wf:stop:wg1"}],
        )
        assert result.success is True
        assert sent.get("reply_markup") is not None

    @pytest.mark.asyncio
    async def test_send_without_buttons_has_no_markup(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)
        await adapter.send("12345", "hi")
        assert "reply_markup" not in sent

    @pytest.mark.asyncio
    async def test_edit_sets_markup_when_buttons_given(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_edit(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.edit_message_text = AsyncMock(side_effect=mock_edit)
        await adapter.edit_message(
            "12345", "7", "updated",
            buttons=[{"text": "⏹ Stop", "callback_data": "wf:stop:wg1"}],
        )
        assert sent.get("reply_markup") is not None

    @pytest.mark.asyncio
    async def test_edit_clears_markup_with_empty_list(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_edit(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.edit_message_text = AsyncMock(side_effect=mock_edit)
        # [] -> explicitly clear keyboard
        await adapter.edit_message("12345", "7", "done", buttons=[])
        assert "reply_markup" in sent
        assert sent["reply_markup"] is None

    @pytest.mark.asyncio
    async def test_edit_none_buttons_leaves_markup_untouched(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_edit(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=7)

        adapter._bot.edit_message_text = AsyncMock(side_effect=mock_edit)
        await adapter.edit_message("12345", "7", "mid-run")  # buttons defaults None
        assert "reply_markup" not in sent

    def test_inline_keyboard_helper_shapes(self):
        # Flat list -> one row; malformed entries dropped; empty -> None.
        assert TelegramAdapter._inline_keyboard_from_buttons(None) is None
        assert TelegramAdapter._inline_keyboard_from_buttons([]) is None
        assert TelegramAdapter._inline_keyboard_from_buttons(
            [{"text": "x"}]  # missing callback_data
        ) is None
        markup = TelegramAdapter._inline_keyboard_from_buttons(
            [{"text": "a", "callback_data": "p:1"}]
        )
        assert markup is not None


# ===========================================================================
# gateway_callback fall-through dispatch
# ===========================================================================

def _wf_query(data="wf:stop:wg1"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.message_thread_id = None
    query.message.chat.type = "supergroup"
    query.message.reply_markup = MagicMock()
    query.from_user = MagicMock()
    query.from_user.id = "12345"
    query.from_user.first_name = "Alfredo"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


class TestGatewayCallbackDispatch:
    @pytest.mark.asyncio
    async def test_handled_strip_only_uses_markup_only_edit(self):
        adapter = _make_adapter()
        update, query = _wf_query()
        directive = {"handled": True, "answer": "⏹ Stopping…", "edit_text": None, "strip_buttons": True}
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_cli.plugins.invoke_hook", return_value=[directive]):
                await adapter._handle_callback_query(update, MagicMock())
        query.answer.assert_awaited()
        # None + strip -> markup-only edit, NOT edit_message_text
        query.edit_message_reply_markup.assert_awaited_once()
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_handled_edit_text_with_strip(self):
        adapter = _make_adapter()
        update, query = _wf_query()
        directive = {"handled": True, "answer": "done", "edit_text": "Stopped.", "strip_buttons": True}
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_cli.plugins.invoke_hook", return_value=[directive]):
                await adapter._handle_callback_query(update, MagicMock())
        query.edit_message_text.assert_awaited_once()
        edit_kwargs = query.edit_message_text.call_args[1]
        assert edit_kwargs["reply_markup"] is None

    @pytest.mark.asyncio
    async def test_unhandled_directive_silent_ack(self):
        adapter = _make_adapter()
        update, query = _wf_query(data="zz:unknown")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
                await adapter._handle_callback_query(update, MagicMock())
        # Silent ack clears the spinner; no edit.
        query.answer.assert_awaited_once()
        query.edit_message_text.assert_not_called()
        query.edit_message_reply_markup.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_flag_passed_to_hook(self):
        adapter = _make_adapter()
        update, query = _wf_query()
        seen = {}

        def fake_invoke(name, **kwargs):
            seen.update(kwargs)
            return [{"handled": True, "answer": "ok", "edit_text": None, "strip_buttons": True}]

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke):
                await adapter._handle_callback_query(update, MagicMock())
        assert seen.get("authorized") is True
        assert seen.get("data") == "wf:stop:wg1"
        assert seen.get("platform") == "telegram"

    @pytest.mark.asyncio
    async def test_unauthorized_flag_when_user_blocked(self):
        adapter = _make_adapter()
        update, query = _wf_query()
        seen = {}

        def fake_invoke(name, **kwargs):
            seen.update(kwargs)
            return [{"handled": True, "answer": "⛔", "edit_text": None, "strip_buttons": False}]

        # Empty allowlist + no GATEWAY_ALLOW_ALL_USERS -> fail closed (unauthorized).
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "", "GATEWAY_ALLOW_ALL_USERS": ""}, clear=False):
            with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke):
                await adapter._handle_callback_query(update, MagicMock())
        assert seen.get("authorized") is False

    @pytest.mark.asyncio
    async def test_existing_ea_prefix_still_routes(self):
        """Regression: the fall-through must not steal built-in prefixes."""
        adapter = _make_adapter()
        adapter._approval_state[1] = "agent:main:telegram:group:12345:99"
        update, query = _wf_query(data="ea:once:1")
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                with patch("hermes_cli.plugins.invoke_hook") as mock_hook:
                    await adapter._handle_callback_query(update, MagicMock())
        mock_resolve.assert_called_once()
        # gateway_callback must NOT fire for a built-in prefix.
        mock_hook.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
