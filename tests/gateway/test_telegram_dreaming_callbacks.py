"""Tests for the Telegram dreaming-engine inline-button callback (dr:verb:rec_id).

Revives the handler that was bundled into the 6 Aug autostash and never
shipped. Mirrors the gt: (gmail-triage) callback test shape — mocks the
subprocess call so tests never touch the real accept.sh/reject.sh scripts or
~/.hermes/dreaming/pending.json.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import unauthorized_action_notice

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _make_query(data: str, user_id: str = "777"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.chat.type = "private"
    query.message.text = "Skill candidate: repeated board checks"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


def _run_dispatch(query):
    update = MagicMock()
    update.callback_query = query
    context = MagicMock()
    adapter = _make_adapter()

    async def _go():
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            await adapter._handle_callback_query(update, context)

    return adapter, _go


class TestDreamingCallback:
    @pytest.mark.asyncio
    async def test_accept_success_edits_message_and_strips_keyboard(self):
        query = _make_query("dr:accept:b7032ccd")
        adapter, go = _run_dispatch(query)

        fake_proc = AsyncMock()
        fake_proc.returncode = 0
        fake_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("pathlib.Path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)):
            await go()

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == "✅ Accepted"
        query.edit_message_text.assert_called_once()
        assert query.edit_message_text.call_args[1]["reply_markup"] is None
        assert "Accepted" in query.edit_message_text.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_script_failure_reports_error_and_leaves_keyboard(self):
        query = _make_query("dr:reject:b7032ccd")
        adapter, go = _run_dispatch(query)

        fake_proc = AsyncMock()
        fake_proc.returncode = 1
        fake_proc.communicate = AsyncMock(return_value=(b"", b"already resolved\n"))

        with patch("pathlib.Path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fake_proc)):
            await go()

        query.answer.assert_called_once()
        assert "already resolved" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_script_answers_and_does_not_edit(self):
        query = _make_query("dr:accept:b7032ccd")
        adapter, go = _run_dispatch(query)

        with patch("pathlib.Path.exists", return_value=False):
            await go()

        query.answer.assert_called_once()
        assert "missing" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_verb_rejected(self):
        query = _make_query("dr:snooze:b7032ccd")
        adapter, go = _run_dispatch(query)
        await go()

        query.answer.assert_called_once()
        assert "Unknown verb" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected_without_invoking_script(self):
        query = _make_query("dr:accept:b7032ccd", user_id="999")
        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        adapter = _make_adapter()

        class _DenyRunner:
            async def _handle_message(self, event):
                return None
            def _is_user_authorized(self, source):
                return False

        adapter._message_handler = _DenyRunner()._handle_message

        with patch("asyncio.create_subprocess_exec", AsyncMock()) as mock_exec:
            await adapter._handle_callback_query(update, context)
            mock_exec.assert_not_called()

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")
        query.edit_message_text.assert_not_called()
