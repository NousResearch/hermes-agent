"""A picker card left visible after a restart must not act on the next picker in that chat.

Related to #152586 and the approval-card restart fix in #153418. Telegram model and
choice pickers keep their sent message id, but previously dispatched every callback
through the latest chat-scoped state without checking which message was tapped.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(message_id: int) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=message_id))
    adapter._app = MagicMock()
    return adapter


async def _send_picker(adapter: TelegramAdapter, picker: str, selected: AsyncMock) -> None:
    if picker == "choice":
        result = await adapter.send_choice_picker(
            chat_id="12345", title="Reasoning", choices=[{"value": "high", "label": "High"}],
            session_key="agent:main:telegram:dm:12345", on_choice_selected=selected,
        )
    else:
        result = await adapter.send_model_picker(
            chat_id="12345", providers=[{"slug": "demo", "name": "Demo", "models": ["demo/one"],
                                         "total_models": 1}],
            current_model="demo/one", current_provider="demo",
            session_key="agent:main:telegram:dm:12345", on_model_selected=selected,
        )
    assert result.success


def _tap(message_id: int, data: str):
    query = AsyncMock(data=data)
    query.message = SimpleNamespace(
        chat_id=12345, message_id=message_id, message_thread_id=None,
        chat=SimpleNamespace(type="private"),
    )
    query.from_user = SimpleNamespace(id=12345, first_name="Ana")
    return SimpleNamespace(callback_query=query)


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", ["choice", "model"])
async def test_old_picker_after_restart_cannot_answer_the_new_card(picker, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    old = _adapter(41)
    new = _adapter(42)
    selected = AsyncMock(return_value="Applied")
    await _send_picker(old, picker, selected)
    await _send_picker(new, picker, selected)
    state_name = "_choice_picker_state" if picker == "choice" else "_model_picker_state"
    assert getattr(old, state_name)["12345"]["msg_id"] == 41
    assert getattr(new, state_name)["12345"]["msg_id"] == 42

    data = "cp:0" if picker == "choice" else "mx"
    await new._handle_callback_query(_tap(41, data), MagicMock())

    selected.assert_not_awaited()
    assert "12345" in getattr(new, state_name), "old card consumed the new picker's state"

    await new._handle_callback_query(_tap(42, data), MagicMock())
    assert "12345" not in getattr(new, state_name)
    if picker == "choice":
        selected.assert_awaited_once_with("12345", "high")
