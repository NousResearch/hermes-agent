"""Per-response inode dedup at the real outbound delivery boundaries."""
import os

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import build_session_key
from tests.gateway.test_media_resend_dedup import (
    _DummyAdapter, _hold_typing, _make_event,
)


@pytest.mark.parametrize("suffix,as_document", [(".pdf", False), (".png", False), (".png", True)])
@pytest.mark.asyncio
async def test_aliases_send_once_per_turn_but_explicit_resends_survive(
    tmp_path, monkeypatch, suffix, as_document,
):
    real = tmp_path / f"attachment{suffix}"
    alias = tmp_path / f"alias{suffix}"
    real.write_bytes(b"synthetic attachment")
    os.link(real, alias)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    monkeypatch.setattr("gateway.platforms.base.LOCAL_DELIVERY_SAFE_ROOTS", (tmp_path,), raising=False)
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing

    async def handler(_event):
        directive = "[[as_document]]\n" if as_document else ""
        return f"{directive}MEDIA:{real}\nThe same attachment is at {alias}."

    adapter.set_message_handler(handler)
    event = _make_event()
    key = build_session_key(event.source)
    await adapter._process_message_background(event, key)
    sent = adapter.documents if suffix == ".pdf" or as_document else adapter.images_sent
    assert len(sent) == 1
    await adapter._process_message_background(event, key)
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_streamed_aliases_send_once_and_resend_on_next_turn(tmp_path, monkeypatch):
    real = tmp_path / "attachment.pdf"
    alias = tmp_path / "alias.pdf"
    real.write_bytes(b"synthetic attachment")
    os.link(real, alias)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    adapter = _DummyAdapter(Platform.TELEGRAM)
    # Build a runner without booting adapters, services, or credentials.
    runner = GatewayRunner.__new__(GatewayRunner)
    event = _make_event(Platform.TELEGRAM)
    response = f"MEDIA:{real}\nMEDIA:{alias}"
    await runner._deliver_media_from_response(response, event, adapter)
    assert adapter.documents == [str(real)]
    await runner._deliver_media_from_response(response, event, adapter)
    assert adapter.documents == [str(real), str(real)]


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.asyncio
async def test_rejected_first_alias_does_not_hide_allowed_file(tmp_path, monkeypatch, streamed):
    denied = tmp_path / "denied.pdf"
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    allowed = allowed_dir / "report.pdf"
    denied.write_bytes(b"synthetic attachment")
    os.link(denied, allowed)
    os.utime(denied, (1, 1))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (allowed_dir,))
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing
    event = _make_event(Platform.TELEGRAM)
    response = f"MEDIA:{denied}\nMEDIA:{allowed}"
    assert adapter.validate_media_delivery_path(str(denied)) is None
    assert adapter.validate_media_delivery_path(str(allowed)) == str(allowed)
    if streamed:
        runner = GatewayRunner.__new__(GatewayRunner)
        await runner._deliver_media_from_response(response, event, adapter)
    else:
        async def handler(_event):
            return response
        adapter.set_message_handler(handler)
        await adapter._process_message_background(event, build_session_key(event.source))
    assert adapter.documents == [str(allowed)]


@pytest.mark.asyncio
async def test_aliases_in_different_dispatch_buckets_send_once(tmp_path, monkeypatch):
    image = tmp_path / "chart.png"
    alias = tmp_path / "chart.pdf"
    image.write_bytes(b"synthetic image")
    os.link(image, alias)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    monkeypatch.setattr("gateway.platforms.base.LOCAL_DELIVERY_SAFE_ROOTS", (tmp_path,), raising=False)
    adapter = _DummyAdapter()
    adapter._keep_typing = _hold_typing
    async def handler(_event):
        return f"MEDIA:{image}\nThe same attachment is at {alias}."
    adapter.set_message_handler(handler)
    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))
    assert len(adapter.images_sent) == 1
    assert adapter.documents == []
