from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.wisdom_command import WisdomView
from gateway.wisdom_publication_cards import refresh


@pytest.mark.asyncio
async def test_idle_gateway_edits_without_agent_or_new_send(monkeypatch):
    job = {
        "receipt": {"message_id": "123"},
        "view": WisdomView(title="Published"),
        "state": "published",
    }
    updater = Mock()
    updater.claim.return_value = [job]
    monkeypatch.setattr(
        "hermes_wisdom.publication_cards.PublicationCards", lambda _: updater
    )
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Mock())

    async def scoped(fn):
        return fn()

    adapter = SimpleNamespace(
        platform="telegram",
        _run_wisdom_profile_operation=scoped,
        edit_wisdom_publication=AsyncMock(),
    )
    await refresh({"telegram": adapter})
    adapter.edit_wisdom_publication.assert_awaited_once_with(
        job["receipt"], job["view"]
    )
    updater.finish.assert_called_once_with(job, success=True)


@pytest.mark.asyncio
async def test_failed_edit_releases_claim_for_durable_retry(monkeypatch):
    job = {
        "receipt": {"message_id": "123"},
        "view": WisdomView(title="Published"),
        "state": "published",
    }
    updater = Mock()
    updater.claim.return_value = [job]
    monkeypatch.setattr(
        "hermes_wisdom.publication_cards.PublicationCards", lambda _: updater
    )
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Mock())

    async def scoped(fn):
        return fn()

    adapter = SimpleNamespace(
        platform="slack",
        _run_wisdom_profile_operation=scoped,
        edit_wisdom_publication=AsyncMock(side_effect=TimeoutError),
    )
    await refresh({"slack": adapter})
    updater.finish.assert_called_once_with(job, success=False)


@pytest.mark.asyncio
async def test_telegram_edits_original_message_and_treats_replay_as_success():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from telegram.error import BadRequest

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._bot = SimpleNamespace(
        do_api_request=AsyncMock(), edit_message_text=AsyncMock()
    )
    receipt = {"destination": "123456", "message_id": "789"}
    view = WisdomView(title="Collective Wisdom", summary="Pending moderation")
    await adapter.edit_wisdom_publication(receipt, view)
    args = adapter._bot.do_api_request.call_args
    assert args.args == ("editMessageText",)
    assert args.kwargs["api_kwargs"]["message_id"] == 789
    adapter._bot.do_api_request.side_effect = BadRequest("Message is not modified")
    await adapter.edit_wisdom_publication(receipt, view)
    adapter._bot.edit_message_text.assert_not_called()


@pytest.mark.asyncio
async def test_slack_edits_original_workspace_message():
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter.__new__(SlackAdapter)
    client = SimpleNamespace(chat_update=AsyncMock())
    adapter._get_client = Mock(return_value=client)
    await adapter.edit_wisdom_publication(
        {"destination": "D123", "message_id": "123.456", "scope_id": "T123"},
        WisdomView(title="Collective Wisdom", summary="Published"),
    )
    adapter._get_client.assert_called_once_with("D123", team_id="T123")
    assert client.chat_update.call_args.kwargs["ts"] == "123.456"
