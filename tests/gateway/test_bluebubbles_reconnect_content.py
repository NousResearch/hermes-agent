"""Admission/content ownership through the runner's real reconnect handoff seams."""
import asyncio

import pytest

from gateway.platforms.helpers import carry_inbound_dedup, hand_over_held_inbound
from tests.gateway.bluebubbles_test_support import Request, adapter, settle


def replace(old):
    new = adapter()
    carry_inbound_dedup(old, new)
    hand_over_held_inbound(old, new)
    return new


@pytest.mark.asyncio
async def test_admitted_text_is_not_replayed_after_adapter_rebuild():
    old = adapter()
    records = []

    async def record(event):
        records.append(event.text)

    old.set_message_handler(record)
    await old._handle_webhook(Request())
    await settle(old)
    new = replace(old)
    new.set_message_handler(record)
    await new._handle_webhook(Request())
    await settle(new)
    assert records == ["caption"]


@pytest.mark.asyncio
async def test_unaccepted_text_remains_retryable_after_rebuild():
    old = adapter()
    await old._handle_webhook(Request())
    await settle(old)
    new = replace(old)
    records = []

    async def record(event):
        records.append(event.text)

    new.set_message_handler(record)
    await new._handle_webhook(Request())
    await settle(new)
    assert records == ["caption"]


@pytest.mark.asyncio
async def test_late_photo_after_rebuild_keeps_caption_suppressed(monkeypatch):
    old = adapter()
    records = []

    async def admit(event):
        event._gateway_accepted = True
        records.append((event.text, list(event.media_urls)))

    monkeypatch.setattr(old, "handle_message", admit)
    await old._handle_webhook(Request())
    await settle(old)
    new = replace(old)
    monkeypatch.setattr(new, "handle_message", admit)

    async def download(guid, metadata):
        return "/synthetic/cache/photo.png"

    monkeypatch.setattr(new, "_download_attachment", download)
    await new._handle_webhook(Request(event="updated-message", attachments=[{
        "guid": "new-photo", "mimeType": "image/png"}]))
    await settle(new)
    assert records == [("caption", []), ("(attachment)", ["/synthetic/cache/photo.png"])]


@pytest.mark.asyncio
async def test_accepted_photo_replay_after_rebuild_does_not_download_again(monkeypatch):
    old = adapter()
    downloads, records = [], []

    async def download(guid, metadata):
        downloads.append(guid)
        return "/synthetic/cache/photo.png"

    async def admit(event):
        event._gateway_accepted = True
        records.append(event.text)

    monkeypatch.setattr(old, "handle_message", admit)
    monkeypatch.setattr(old, "_download_attachment", download)
    request = lambda: Request(attachments=[{"guid": "photo", "mimeType": "image/png"}])
    await old._handle_webhook(request())
    await settle(old)
    new = replace(old)
    monkeypatch.setattr(new, "handle_message", admit)
    monkeypatch.setattr(new, "_download_attachment", download)
    await new._handle_webhook(request())
    await settle(new)
    assert records == ["caption"]
    assert downloads == ["photo"]


@pytest.mark.asyncio
async def test_cancellation_after_admission_survives_rebuild(monkeypatch):
    old = adapter()

    async def accepted_then_cancelled(event):
        event._gateway_accepted = True
        raise asyncio.CancelledError

    monkeypatch.setattr(old, "handle_message", accepted_then_cancelled)
    with pytest.raises(asyncio.CancelledError):
        await old._handle_webhook(Request())
    new = replace(old)
    admitted = []

    async def admit(event):
        event._gateway_accepted = True
        admitted.append(event.text)

    monkeypatch.setattr(new, "handle_message", admit)
    await new._handle_webhook(Request())
    await settle(new)
    assert admitted == []


@pytest.mark.asyncio
async def test_partial_download_cache_survives_rebuild(monkeypatch):
    old = adapter()
    calls = []
    records = []

    async def download(guid, metadata):
        calls.append(guid)
        return "/synthetic/cache/first.png" if guid == "first" else None

    async def admit(event):
        event._gateway_accepted = True
        records.append(list(event.media_urls))

    monkeypatch.setattr(old, "handle_message", admit)
    monkeypatch.setattr(old, "_download_attachment", download)
    request = lambda: Request(attachments=[
        {"guid": "first", "mimeType": "image/png"}, {"guid": "second", "mimeType": "image/png"}])
    assert (await old._handle_webhook(request())).status == 503
    new = replace(old)

    async def recovered_download(guid, metadata):
        calls.append(guid)
        return "/synthetic/cache/second.png" if guid == "second" else None

    monkeypatch.setattr(new, "handle_message", admit)
    monkeypatch.setattr(new, "_download_attachment", recovered_download)
    assert (await new._handle_webhook(request())).status == 200
    assert records == [["/synthetic/cache/first.png", "/synthetic/cache/second.png"]]
    assert calls.count("first") == 1
