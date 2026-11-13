"""An approval card still visible after a gateway restart must not answer the first new card.

Card ids ride in Telegram's ``callback_data``; a fresh adapter (a restart) used to number its cards from
1 again, so the old card's button carried the new card's id and resolved the new request.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

_SESSION = "agent:main:telegram:dm:12345"


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    adapter._app = MagicMock()
    return adapter


def _once_data(adapter: TelegramAdapter) -> str:
    (card_id,) = adapter._approval_state  # the id the card's buttons carry in callback_data
    return f"ea:once:{card_id}"


def _tap(data: str) -> MagicMock:
    query = AsyncMock(data=data)
    query.message = MagicMock(chat_id=12345, text_html="<b>Dangerous command</b>")
    query.from_user = MagicMock(first_name="Ana", id="12345")
    return MagicMock(callback_query=query)


@pytest.mark.asyncio
async def test_old_card_tapped_after_restart_does_not_resolve_the_new_card():
    before = _adapter()
    await before.send_exec_approval(chat_id="12345", command="ls", session_key=_SESSION)
    old_card = _once_data(before)

    after = _adapter()  # the gateway restarted: a new adapter, the old card still in the chat
    await after.send_exec_approval(chat_id="12345", command="rm -rf /srv", session_key=_SESSION)
    new_card = _once_data(after)
    assert old_card != new_card, "a restarted adapter reissued the old card's id"

    resolve = MagicMock(return_value=1)
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
            patch("hermes_agent.tools.approval.resolve_gateway_approval_card", resolve):
        await after._handle_callback_query(_tap(old_card), MagicMock())
        resolve.assert_not_called()
        await after._handle_callback_query(_tap(new_card), MagicMock())  # control: the new card answers
    assert resolve.call_count == 1 and resolve.call_args.args[:2] == (_SESSION, "once")


def _update_card_data(adapter: TelegramAdapter, answer: str) -> str:
    # The id the card's buttons carry (python-telegram-bot's keyboard is stubbed in this suite).
    return f"update_prompt:{answer}:{getattr(adapter, '_update_prompt_id', '')}"


@pytest.mark.asyncio
async def test_old_update_card_tapped_after_restart_does_not_answer_the_new_prompt(tmp_path):
    before = _adapter()
    await before.send_update_prompt(chat_id="12345", prompt="Restore stash?", session_key=_SESSION)
    old_yes = _update_card_data(before, "y")

    after = _adapter()
    await after.send_update_prompt(chat_id="12345", prompt="Overwrite config?", session_key=_SESSION)
    response = tmp_path / ".update_response"
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
            patch("hermes_agent.hermes_constants.get_hermes_home", return_value=tmp_path):
        await after._handle_callback_query(_tap(old_yes), MagicMock())
        assert not response.exists(), "the old /update card answered the new prompt"
        new_yes, new_no = _update_card_data(after, "y"), _update_card_data(after, "n")
        await after._handle_callback_query(_tap(new_no), MagicMock())  # control: the new card answers
        assert response.read_text() == "n"
        response.unlink()
        await after._handle_callback_query(_tap(new_yes), MagicMock())
        assert not response.exists(), "an answered /update card answered again"


@pytest.mark.asyncio
async def test_update_card_stays_answerable_when_the_answer_could_not_be_written(tmp_path):
    adapter = _adapter()
    await adapter.send_update_prompt(chat_id="12345", prompt="Restore stash?", session_key=_SESSION)
    card = _update_card_data(adapter, "y")
    missing = tmp_path / "gone"  # the response write fails: its directory does not exist
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        with patch("hermes_agent.hermes_constants.get_hermes_home", return_value=missing):
            await adapter._handle_callback_query(_tap(card), MagicMock())
        with patch("hermes_agent.hermes_constants.get_hermes_home", return_value=tmp_path):
            await adapter._handle_callback_query(_tap(card), MagicMock())  # the retry still answers
    assert (tmp_path / ".update_response").read_text() == "y"


@pytest.mark.asyncio
async def test_old_update_card_cannot_answer_while_the_new_card_awaits_its_send_receipt(tmp_path):
    before = _adapter()
    await before.send_update_prompt(chat_id="12345", prompt="Restore stash?", session_key=_SESSION)
    old_yes = _update_card_data(before, "y")

    after = _adapter()
    posted, receipt = asyncio.Event(), asyncio.Event()

    async def slow_receipt(**_kwargs):
        posted.set()
        await receipt.wait()
        return MagicMock(message_id=43)

    after._bot.send_message = AsyncMock(side_effect=slow_receipt)
    sending = asyncio.create_task(
        after.send_update_prompt(chat_id="12345", prompt="Overwrite config?", session_key=_SESSION))
    response = tmp_path / ".update_response"
    try:
        await asyncio.wait_for(posted.wait(), 2)
        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
                patch("hermes_agent.hermes_constants.get_hermes_home", return_value=tmp_path):
            await after._handle_callback_query(_tap(old_yes), MagicMock())
        assert not response.exists(), "the old /update card answered while the new card was sending"
    finally:
        receipt.set()
        await sending
