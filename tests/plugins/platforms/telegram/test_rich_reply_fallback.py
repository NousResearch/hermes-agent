"""Rich replies retain the sent-index fallback without confusing marker-like text."""

import pytest

pytest.importorskip("telegram")
from telegram import Update

from gateway import rich_sent_store
from gateway.platforms.event import MessageType
from tests.plugins.test_telegram_update_admission import connected, update


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {"blocks": []}, {"blocks": [{"type": "future"}]}])
async def test_non_content_rich_reply_uses_original_sent_text(monkeypatch, payload):
    async with connected(monkeypatch) as (adapter, app, _):
        reply = update(app.bot).message.to_dict()
        reply.pop("text")
        reply["rich_message"] = payload
        raw = update(app.bot, 11).to_dict()
        raw["message"]["reply_to_message"] = reply
        incoming = Update.de_json(raw, app.bot)
        rich_sent_store.record("42", "472", "Original sent content")
        event = adapter._build_message_event(incoming.message, MessageType.TEXT)
        assert event.reply_to_text == "Original sent content"


def test_url_destination_without_label_is_still_reply_content():
    from plugins.platforms.telegram.telegram_rich_inbound import rich_reply_text
    reply = {"rich_message": {"blocks": [{"type": "paragraph", "text": {
        "type": "url", "text": "", "url": "https://example.com"}}]}}
    assert rich_reply_text(reply) == " (https://example.com)"


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ["[Empty Telegram rich message]", "[Invalid Telegram rich content]",
                                     "[Unsupported Telegram rich content]"])
async def test_literal_marker_text_is_real_content_not_projection_status(monkeypatch, literal):
    async with connected(monkeypatch) as (adapter, app, _):
        reply = update(app.bot).message.to_dict()
        reply.pop("text")
        reply["rich_message"] = {"blocks": [{"type": "paragraph", "text": literal}]}
        raw = update(app.bot, 11).to_dict()
        raw["message"]["reply_to_message"] = reply
        incoming = Update.de_json(raw, app.bot)
        rich_sent_store.record("42", "472", "Stale cached content")
        event = adapter._build_message_event(incoming.message, MessageType.TEXT)
        assert event.reply_to_text == literal
