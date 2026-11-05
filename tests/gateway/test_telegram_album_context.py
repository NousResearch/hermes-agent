"""Album items under require_mention: an addressed album (or a reply to one item) carries every item.

Telegram sends an album as one message per item with the caption on a single item; the mention gate
used to judge each item alone, so the agent only ever saw the addressed/replied-to file.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.telegram_albums import RecentAlbumItems
from tests.gateway.test_telegram_group_gating import _group_message, _make_adapter, _mention_entity


def _album_doc(message_id, *, caption=None, mention=False, media_group_id="album-1", reply_to=None):
    msg = _group_message(
        text=None, caption=caption, caption_entities=[_mention_entity(caption)] if mention else None)
    msg.message_id = message_id
    msg.media_group_id = media_group_id
    msg.document = SimpleNamespace(file_name=f"file-{message_id}.pdf", mime_type="application/pdf", file_size=10)
    for attr in ("photo", "video", "audio", "voice", "sticker"):
        setattr(msg, attr, None)
    msg.reply_to_message = reply_to
    return msg


def _media_adapter():
    adapter = _make_adapter(require_mention=True)
    queued = []

    async def _fake_cache_document(msg, event):
        event.media_urls.append(f"/cache/{msg.document.file_name}")
        event.media_types.append("application/pdf")
        return False

    async def _fake_queue(media_group_id, event):
        queued.append((media_group_id, event))

    adapter._cache_inbound_document = _fake_cache_document
    adapter._queue_media_group_event = _fake_queue
    adapter.handle_message = AsyncMock()
    return adapter, queued


def _update(msg, update_id=1):
    return SimpleNamespace(message=msg, update_id=update_id, effective_message=msg)


def test_album_caption_mention_pulls_in_earlier_unaddressed_item():
    async def _run():
        adapter, queued = _media_adapter()
        await adapter._handle_media_message(_update(_album_doc(500), 1), None)
        assert queued == []
        await adapter._handle_media_message(
            _update(_album_doc(501, caption="@hermes_bot preencha", mention=True), 2), None)
        assert [event.media_urls for _gid, event in queued] == [["/cache/file-500.pdf"], ["/cache/file-501.pdf"]]
        assert {gid for gid, _event in queued} == {"album-1"}

    asyncio.run(_run())


def test_album_items_after_the_addressed_one_bypass_the_gate():
    async def _run():
        adapter, queued = _media_adapter()
        await adapter._handle_media_message(
            _update(_album_doc(600, caption="@hermes_bot olha", mention=True, media_group_id="album-2"), 1), None)
        await adapter._handle_media_message(_update(_album_doc(601, media_group_id="album-2"), 2), None)
        assert [event.media_urls[0] for _gid, event in queued] == ["/cache/file-600.pdf", "/cache/file-601.pdf"]

    asyncio.run(_run())


def test_unaddressed_album_stays_dropped():
    async def _run():
        adapter, queued = _media_adapter()
        await adapter._handle_media_message(_update(_album_doc(700, media_group_id="album-3"), 1), None)
        await adapter._handle_media_message(_update(_album_doc(701, caption="sem menção", media_group_id="album-3"), 2), None)
        assert queued == []
        adapter.handle_message.assert_not_awaited()

    asyncio.run(_run())


def test_reply_to_one_album_item_attaches_the_others_and_their_caption():
    async def _run():
        adapter, _queued = _media_adapter()
        pdf = _album_doc(332)
        docx = _album_doc(333, caption="Preencha a etiqueta de envio")
        docx.document = SimpleNamespace(file_name="ETIQUETA.docx", mime_type="application/msword", file_size=10)
        await adapter._handle_media_message(_update(pdf, 1), None)
        await adapter._handle_media_message(_update(docx, 2), None)

        async def _fake_download(msg, what):
            name = msg.document.file_name
            return "ok", SimpleNamespace(
                path=f"/cache/{name}", media_type="application/octet-stream", kind="document", display_name=name)

        adapter._download_observed_media = _fake_download
        # Bot API reply_to_message is a fresh copy; media_group_id may be absent on it.
        quoted = _album_doc(332, media_group_id=None)
        trigger = _group_message("@hermes_bot", entities=[_mention_entity("@hermes_bot")])
        trigger.reply_to_message = quoted
        event = MessageEvent(
            text="", message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group"))
        await adapter._cache_replied_media(trigger, event)

        assert event.media_urls == ["/cache/file-332.pdf", "/cache/ETIQUETA.docx"]
        assert "Same album as the replied-to message" in event.text
        assert "Preencha a etiqueta de envio" in event.text

    asyncio.run(_run())


def test_recent_album_items_expire_and_cap():
    now = [0.0]
    items = RecentAlbumItems(ttl=10, max_entries=2, triggered_ttl=5, clock=lambda: now[0])
    for mid in (1, 2, 3):
        items.remember("-100", SimpleNamespace(message_id=mid, media_group_id="g"))
    assert [m.message_id for m, _ in items.siblings("-100", "g")] == [2, 3]
    assert items.mark_triggered("-100", "g") is True
    assert items.mark_triggered("-100", "g") is False
    now[0] = 11
    assert items.siblings("-100", "g") == []
    assert items.is_triggered("-100", "g") is False

