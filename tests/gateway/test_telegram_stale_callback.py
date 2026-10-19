"""A tap on an expired Telegram card must not die on the answer it can no longer deliver.

Telegram refuses ``answerCallbackQuery`` once the query is older than its window
("Query is too old and response timeout expired or query id is invalid"). Every button handler
awaited that call unguarded, so the raise escaped into PTB, whatever the handler had left to do
never ran, and the tap produced no feedback at all (#141587).

These cases drive the real adapter with a query that refuses its answer, the way Telegram does
after the card has been sitting in the chat. ``_stale_error`` builds a real PTB ``BadRequest``
when python-telegram-bot is installed and a text-only stand-in otherwise, so the cases hold in
both environments.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.callback_answer import answer_callback, is_stale_query_error

STALE = "Bad Request: Query is too old and response timeout expired or query id is invalid"
EDIT_GONE = "Bad Request: message to edit not found"


def _error(text: str) -> Exception:
    """A PTB ``BadRequest`` with *text* when the library is importable, else a stand-in."""
    try:
        from telegram.error import BadRequest
    except ImportError:
        return RuntimeError(text)
    return BadRequest(text)


class _Bot:
    def __init__(self, sent):
        self.sent = sent

    async def send_message(self, chat_id=None, text=None, **kwargs):
        self.sent.append((chat_id, text))


class _RefusingQuery:
    """``CallbackQuery`` stand-in whose ``answer()`` Telegram refuses."""

    def __init__(self, error: str = STALE):
        self.data = "mc:0"
        self.message = SimpleNamespace(chat_id=12345, text="picker")
        self.from_user = SimpleNamespace(id=1, first_name="tester")
        self.notes: list = []
        self.edits: list = []
        self._error = error

    async def answer(self, text=None, **kwargs):
        raise _error(self._error)

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)

    def get_bot(self):
        return _Bot(self.notes)


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    return adapter


def _picker_state() -> dict:
    return {
        "providers": [{"slug": "acme", "name": "Acme", "models": [{"id": "acme/big"}]}],
        "model_list": [{"id": "acme/big"}],
        "current_model": "acme/small",
        "current_provider": "acme",
        "selected_provider": "acme",
        "session_key": "s",
        "on_model_selected": AsyncMock(return_value="Switched to acme/big"),
        "msg_id": 42,
    }


class TestExpiredCallbackAnswer:
    @pytest.mark.asyncio
    async def test_stale_tap_on_an_expired_picker_is_not_raised_and_tells_the_user(self):
        adapter = _adapter()
        query = _RefusingQuery()

        await adapter._handle_model_picker_callback(query, "mc:0", "12345")

        # The tap can no longer be acked as a toast; the toast text is still a usable next step.
        assert len(query.notes) == 1
        assert query.notes[0][0] == 12345
        assert "/model" in query.notes[0][1]

    @pytest.mark.asyncio
    async def test_stale_tap_on_an_expired_choice_picker_also_answers_in_the_chat(self):
        adapter = _adapter()
        query = _RefusingQuery()

        await adapter._handle_choice_picker_callback(query, "cp:0", "12345")

        assert len(query.notes) == 1

    @pytest.mark.asyncio
    async def test_stale_answer_on_a_model_switch_still_switches_and_clears_the_picker(self):
        adapter = _adapter()
        state = _picker_state()
        adapter._model_picker_state["12345"] = state
        query = _RefusingQuery()

        await adapter._handle_model_picker_callback(query, "mc:0", "12345")

        state["on_model_selected"].assert_awaited_once()
        assert "12345" not in adapter._model_picker_state
        assert query.edits, "the switch result must still replace the card"

    @pytest.mark.asyncio
    async def test_other_answer_errors_still_raise(self):
        adapter = _adapter()
        adapter._model_picker_state["12345"] = _picker_state()
        query = _RefusingQuery(EDIT_GONE)

        with pytest.raises(Exception, match="message to edit not found"):
            await adapter._handle_model_picker_callback(query, "mc:0", "12345")

    @pytest.mark.asyncio
    async def test_answer_callback_reports_whether_the_answer_landed(self):
        class Query:
            def __init__(self, error=None):
                self.error = error
                self.answered = []

            async def answer(self, text=None, **kwargs):
                if self.error:
                    raise self.error
                self.answered.append(text)

        assert await answer_callback(Query(), "hi") is True
        assert await answer_callback(Query(_error(STALE)), "hi") is False


class TestStaleQueryDetection:
    def test_matches_the_wording_telegram_uses_for_an_expired_query(self):
        assert is_stale_query_error(_error(STALE))
        assert is_stale_query_error(_error("Bad Request: query id is invalid"))
        assert is_stale_query_error(_error("Bad Request: response timeout expired"))

    def test_does_not_match_unrelated_telegram_errors(self):
        assert not is_stale_query_error(_error(EDIT_GONE))
        assert not is_stale_query_error(RuntimeError("boom"))
