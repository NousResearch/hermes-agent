"""Core picker delivery preserves scope, callbacks, and text fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.choice_picker import ChoicePage
from gateway.platforms.base import SendResult
from gateway.slash_commands_model import GatewayModelCommandsMixin


def choices(count=12):
    return [{"label": f"Choice {i}", "value": f"identity-{i}"} for i in range(count)]


class Picker:
    supports_choice_pages = True

    def __init__(self):
        self.send_choice_picker = AsyncMock(
            return_value=SendResult(success=True, message_id="menu")
        )

    async def send_choice_picker(self, **kwargs):
        pass


class OldPicker(Picker):
    supports_choice_pages = False


def runner(adapter):
    instance = GatewayModelCommandsMixin()
    instance._adapter_for_source = lambda source: adapter
    instance._thread_metadata_for_source = lambda source, anchor: {
        "thread_id": "topic",
        "reply_to_message_id": anchor,
    }
    instance._reply_anchor_for_event = lambda event: "command"
    return instance


@pytest.mark.asyncio
async def test_shared_entry_opt_in_preserves_scope_and_original_callback():
    adapter = Picker()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="chat", user_id="owner"))
    callback = AsyncMock(return_value=ChoicePage("Next", choices()))
    assert await runner(adapter)._try_send_choice_picker(
        event, "session", "Files", choices(), callback, reusable=True
    )
    call = adapter.send_choice_picker.await_args.kwargs
    assert call["on_choice_selected"] is callback
    assert call["session_key"] == "session"
    assert call["metadata"] == {
        "thread_id": "topic",
        "reply_to_message_id": "command",
        "requester_user_id": "owner",
        "choice_pages": True,
    }
    assert len(call["choices"]) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [None, object(), OldPicker()])
async def test_unsupported_adapters_explicitly_fall_back(adapter):
    event = SimpleNamespace(source=SimpleNamespace(chat_id="chat", user_id="owner"))
    assert not await runner(adapter)._try_send_choice_picker(
        event, "s", "Files", choices(), AsyncMock(), reusable=True
    )


@pytest.mark.asyncio
async def test_one_shot_does_not_require_new_capability_or_change_callback():
    adapter = OldPicker()
    callback = AsyncMock(return_value="Done")
    event = SimpleNamespace(source=SimpleNamespace(chat_id="chat", user_id="owner"))
    assert await runner(adapter)._try_send_choice_picker(
        event, "s", "Setting", choices(1), callback
    )
    call = adapter.send_choice_picker.await_args.kwargs
    assert "choice_pages" not in call["metadata"]
    assert await call["on_choice_selected"]("chat", "identity-0") == "Done"


@pytest.mark.asyncio
async def test_reusable_requires_requester_before_native_send():
    adapter = Picker()
    event = SimpleNamespace(source=SimpleNamespace(chat_id="chat", user_id=None))
    assert not await runner(adapter)._try_send_choice_picker(
        event, "s", "Files", choices(), AsyncMock(), reusable=True
    )
    adapter.send_choice_picker.assert_not_awaited()
