"""Regression tests for the Telegram ``/model`` picker callback acknowledgment.

Bug (#125171): ``_picker_edit()`` awaited ``edit_message_text()`` before
``query.answer()`` with no guard, so a failing edit (stale/deleted message,
``"message is not modified"``, network error, MarkdownV2 render failure) left
the callback query unanswered — Telegram kept showing the tap as pending
forever with no visible response and no error.

The fix answers the callback FIRST, then attempts the edit inside try/except
(logging a warning instead of propagating).
"""

import pytest

from plugins.platforms.telegram import adapter as telegram_adapter

pytestmark = pytest.mark.asyncio


class _FakeCallbackQuery:
    """Minimal stand-in for a PTB CallbackQuery."""

    def __init__(self, edit_error: Exception | None = None):
        self.answered = 0
        self.answer_texts: list = []
        self.edited: list = []
        self._edit_error = edit_error

    async def answer(self, text=None, show_alert=None, **kwargs):
        self.answered += 1
        self.answer_texts.append(text)

    async def edit_message_text(self, **kwargs):
        if self._edit_error is not None:
            raise self._edit_error
        self.edited.append(kwargs)


@pytest.fixture()
def picker() -> object:
    """A TelegramAdapter-ish object exposing only what _picker_edit touches."""
    return telegram_adapter.TelegramAdapter.__new__(telegram_adapter.TelegramAdapter)


async def test_picker_edit_answers_callback_before_editing(picker):
    query = _FakeCallbackQuery()
    await picker._picker_edit(query, "⚙ *Model Configuration*", keyboard=None)
    assert query.answered == 1
    assert query.edited and query.edited[0]["text"]


async def test_picker_edit_still_acks_when_edit_fails(picker):
    query = _FakeCallbackQuery(edit_error=RuntimeError("message to edit not found"))
    await picker._picker_edit(query, "⚙ *Model Configuration*", keyboard=None)
    assert query.answered == 1, "callback must be acknowledged even when the edit fails"


async def test_picker_edit_ack_precedes_edit_call(picker):
    calls: list = []

    class _OrderProbe(_FakeCallbackQuery):
        async def answer(self, text=None, show_alert=None, **kwargs):
            calls.append("answer")
            await super().answer(text)

        async def edit_message_text(self, **kwargs):
            calls.append("edit")
            await super().edit_message_text(**kwargs)

    await picker._picker_edit(_OrderProbe(), "text", keyboard=None)
    assert calls == ["answer", "edit"], "answer() must run before edit_message_text()"
