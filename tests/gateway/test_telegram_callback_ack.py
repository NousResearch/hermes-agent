"""Every Telegram button tap is answered, so the client never keeps a spinning button.

Driven through the adapter's real callback entry point (``_handle_callback_query``).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Query:
    def __init__(self, data, *, edit_error=None):
        self.data = data
        self.message = SimpleNamespace(chat_id=111, chat=SimpleNamespace(type="private"), message_thread_id=None,
                                       message_id=5, text="picker")
        self.from_user = SimpleNamespace(id=111, first_name="tester")
        self.events: list = []
        self._edit_error = edit_error

    async def answer(self, text=None, **kwargs):
        self.events.append(("answer", text))

    async def edit_message_text(self, **kwargs):
        if self._edit_error is not None:
            self.events.append(("edit_failed", None))
            raise self._edit_error
        self.events.append(("edit", kwargs.get("text")))


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    adapter._is_callback_user_authorized = lambda *a, **k: True
    return adapter


async def _tap(adapter, query):
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), SimpleNamespace())


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [None, "", "zz:stale-build-button"])
async def test_unhandled_and_data_less_taps_are_answered(data):
    query = _Query(data)
    await _tap(_adapter(), query)
    assert query.events == [("answer", None)]


@pytest.mark.asyncio
async def test_provider_tap_is_answered_even_when_the_picker_edit_fails():
    adapter = _adapter()
    adapter._model_picker_state["111"] = {
        "providers": [{"slug": "acme", "name": "Acme", "models": ["acme/big"]}],
        "current_model": "acme/small", "current_provider": "acme", "session_key": "s",
        "on_model_selected": AsyncMock(), "msg_id": 5}
    from telegram.error import BadRequest
    query = _Query("mp:acme", edit_error=BadRequest("Message can't be edited"))

    await _tap(adapter, query)

    assert query.events[0] == ("answer", None), query.events
    assert ("edit_failed", None) in query.events
    assert adapter._model_picker_state["111"]["selected_provider"] == "acme"
