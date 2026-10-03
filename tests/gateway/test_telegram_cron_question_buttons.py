"""Telegram rendering + routing for cron report questions (#107138).

``send_cron_questions`` mirrors ``send_clarify``'s shape but must NOT block: the
report is already delivered, so a tap is recorded in the durable store and then
handed to the conversation as an ordinary user turn. These tests pin the render,
the ``cq:`` callback contract, and the replay/unauthorized paths.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from cron.questions import (  # noqa: E402
    PendingQuestion,
    Question,
    QuestionOption,
    get_question,
    record_questions,
)
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture
def store(monkeypatch, tmp_path):
    import cron.questions as questions_mod

    monkeypatch.setattr(questions_mod, "QUESTIONS_FILE", tmp_path / "cron" / "questions.db")
    return questions_mod


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _question():
    return Question(
        text="Merge PR #1827?",
        options=[QuestionOption("\u2705 Yes", recommended=True), QuestionOption("\u23f8\ufe0f Not yet")],
    )


def _query(data, *, user_id="777", chat_id=12345):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = chat_id
    query.message.message_id = 500
    query.message.chat.id = chat_id
    query.message.chat.type = "private"
    query.message.text = "Merge PR #1827?"
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = "Tester"
    query.from_user.full_name = "Tester Person"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return query, update


async def _tap(adapter, data, *, user_id="777", chat_id=12345, allowed="*"):
    """Drive ``_handle_callback_query`` with a button tap on ``data``."""
    query, update = _query(data, user_id=user_id, chat_id=chat_id)
    # The env gate is read while the handler runs, not when the coroutine is built.
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": allowed}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())
    return query


def _buttons_sent():
    """``(label, callback_data)`` for every button the adapter built (PTB is a MagicMock here)."""
    from telegram import InlineKeyboardButton

    return [
        (call.args[0] if call.args else None, call.kwargs.get("callback_data"))
        for call in InlineKeyboardButton.call_args_list
    ]


class TestSendCronQuestions:
    @pytest.mark.asyncio
    async def test_one_button_per_option_carrying_its_token(self):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        InlineKeyboardButton.reset_mock()
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=100))

        result = await adapter.send_cron_questions(
            "12345", [PendingQuestion(token="tok1", question="Merge PR #1827?", options=["\u2705 Yes", "\u23f8\ufe0f Not yet"])])

        assert result.success is True
        kwargs = adapter._bot.send_message.call_args[1]
        assert "Merge PR #1827?" in kwargs["text"]
        # One option per row, carrying the token the tap resolves through.
        rows = InlineKeyboardMarkup.call_args[0][0]
        assert all(len(row) == 1 for row in rows)
        assert _buttons_sent() == [("\u2705 Yes", "cq:tok1:0"), ("\u23f8\ufe0f Not yet", "cq:tok1:1")]

    @pytest.mark.asyncio
    async def test_multiple_questions_number_their_rows(self):
        from telegram import InlineKeyboardButton

        InlineKeyboardButton.reset_mock()
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=101))

        await adapter.send_cron_questions("12345", [
            PendingQuestion(token="tokA", question="Merge?", options=["yes", "no"]),
            PendingQuestion(token="tokB", question="Deploy?", options=["now", "later"]),
        ])

        assert [label for label, _ in _buttons_sent()] == [
            "1. yes", "1. no", "2. now", "2. later"]
        assert [data for _, data in _buttons_sent()] == [
            "cq:tokA:0", "cq:tokA:1", "cq:tokB:0", "cq:tokB:1"]

    @pytest.mark.asyncio
    async def test_question_text_is_html_escaped(self):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=102))

        await adapter.send_cron_questions(
            "12345", [PendingQuestion(token="tokC", question="<script>x</script>", options=["ok"])])

        text = adapter._bot.send_message.call_args[1]["text"]
        assert "<script>" not in text
        assert "&lt;script&gt;" in text


class TestCronQuestionCallback:
    @pytest.mark.asyncio
    async def test_tap_records_the_answer_and_reinjects_it_as_a_user_turn(self, store):
        adapter = _make_adapter()
        (token,) = record_questions("job9", "telegram", "12345", [_question()])
        injected = []

        async def _handle(event):
            injected.append(event)

        adapter.handle_message = _handle
        adapter._message_handler = AsyncMock()
        query = await _tap(adapter, f"cq:{token}:0")

        assert get_question(token)["answer_text"] == "\u2705 Yes"
        query.answer.assert_awaited()
        assert len(injected) == 1
        event = injected[0]
        assert "Merge PR #1827?" in event.text
        assert "\u2705 Yes" in event.text
        # The answer re-enters as the tapping user, in the chat the report was delivered to.
        assert event.source.chat_id == "12345"
        assert event.source.user_id == "777"
        assert event.reply_to_is_own_message is True
        # A button label is not a command surface.
        assert event.allow_gateway_control is False

    @pytest.mark.asyncio
    async def test_second_tap_does_not_answer_twice(self, store):
        adapter = _make_adapter()
        (token,) = record_questions("job9", "telegram", "12345", [_question()])
        injected = []

        async def _handle(event):
            injected.append(event)

        adapter.handle_message = _handle
        adapter._message_handler = AsyncMock()
        await _tap(adapter, f"cq:{token}:0")
        second_query = await _tap(adapter, f"cq:{token}:1")

        assert len(injected) == 1
        assert get_question(token)["answer_text"] == "\u2705 Yes"
        assert "Already answered" in second_query.answer.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_unauthorized_tap_changes_nothing(self, store):
        adapter = _make_adapter()
        (token,) = record_questions("job9", "telegram", "12345", [_question()])
        adapter.handle_message = AsyncMock()
        adapter._message_handler = AsyncMock()

        query = await _tap(adapter, f"cq:{token}:0", user_id="999", allowed="777")

        assert get_question(token)["answered_at"] is None
        adapter.handle_message.assert_not_awaited()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()

    @pytest.mark.asyncio
    async def test_expired_token_is_reported_not_swallowed(self, store):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        query = await _tap(adapter, "cq:deadbeefdead:0")

        adapter.handle_message.assert_not_awaited()
        assert "no longer available" in query.answer.call_args[1]["text"]

    @pytest.mark.asyncio
    async def test_other_prefixes_are_not_hijacked(self, store):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        await _tap(adapter, "cl:someid:0")

        adapter.handle_message.assert_not_awaited()


class TestBaseAdapterFallback:
    @pytest.mark.asyncio
    async def test_buttonless_adapter_renders_numbered_text(self):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Stub(BasePlatformAdapter):
            name = "stub"

            def __init__(self):
                self.sent = []

            async def connect(self, *, is_reconnect: bool = False): pass
            async def disconnect(self): pass

            async def send(self, chat_id, content, **kw):
                self.sent.append(content)
                return SendResult(success=True, message_id="1")

            async def edit(self, *a, **k): return SendResult(success=False)
            async def get_history(self, *a, **k): return []
            async def get_chat_info(self, *a, **k): return {}

        adapter = _Stub()
        result = await adapter.send_cron_questions(
            "c", [PendingQuestion(token="t", question="Merge PR #1827?", options=["Yes", "Not yet"])])

        assert result.success is True
        text = adapter.sent[0]
        assert "Merge PR #1827?" in text
        assert "1. Yes" in text
        assert "2. Not yet" in text
