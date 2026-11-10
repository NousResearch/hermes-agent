"""WhatsApp 👀/✅/❌ processing reactions and opt-in online presence."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter

KEY = {"remoteJid": "120363@g.us", "id": "ABC", "participant": "3162@s.whatsapp.net", "fromMe": False}


def _event():
    source = SessionSource(platform="whatsapp", chat_id="120363@g.us", chat_type="group", user_id="3162@s.whatsapp.net")
    return MessageEvent(text="hi", source=source, raw_message={"readReceiptKey": KEY}, message_id="ABC")


def _reacting_adapter(enabled=True):
    adapter = _make_adapter()
    adapter._reactions = enabled
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    return adapter


def _emojis(adapter):
    return [c.kwargs["json"]["emoji"] for c in adapter._http_session.post.call_args_list]


@pytest.mark.asyncio
async def test_reaction_follows_processing_outcome_with_the_inbound_key():
    adapter = _reacting_adapter()
    event = _event()

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    assert _emojis(adapter) == ["👀", "✅", "❌", ""]
    first = adapter._http_session.post.call_args_list[0]
    assert first.args[0].endswith("/react")
    assert first.kwargs["json"]["key"] == KEY
    assert first.kwargs["json"]["chatId"] == "120363@g.us"


@pytest.mark.asyncio
async def test_reactions_are_off_by_default():
    adapter = _reacting_adapter(enabled=False)

    await adapter.on_processing_start(_event())
    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)

    adapter._http_session.post.assert_not_called()


@pytest.mark.parametrize("raw, expected", [(None, False), (True, True), ("yes", True), ("false", False), (False, False)])
def test_config_flag_parsing(raw, expected):
    assert WhatsAppAdapter._flag(raw) is expected


@pytest.mark.parametrize("mark_online", [True, False])
def test_bridge_env_carries_mark_online(mark_online):
    adapter = _make_adapter()
    adapter._send_read_receipts = False
    adapter._mark_online = mark_online
    adapter._dm_policy = adapter._group_policy = "open"
    adapter._allow_from = adapter._group_allow_from = set()

    assert adapter._bridge_env()["WHATSAPP_MARK_ONLINE"] == ("true" if mark_online else "false")
