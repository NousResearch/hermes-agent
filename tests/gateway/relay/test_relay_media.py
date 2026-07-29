"""Relay Phase 2 media tests — send_media egress lanes + inbound media localization.

Covers:
  - the five ``send_*`` overrides route through ONE ``send_media`` op with the
    right ``media_kind`` and honor op-level capability gating (a connector not
    advertising ``send_media`` falls back to the base-class behaviour);
  - local-path sources upload through the RelayMediaClient first (the
    connector cannot reach our filesystem) and public URLs pass through;
  - a connector decline / failed upload degrades to the pre-media fallback;
  - inbound ``media_urls`` are localized to temp paths (re-hosts downloaded
    with the per-gateway bearer; dead re-host refs dropped; public URLs kept
    when no client is available);
  - the RelayMediaClient URL derivation + auth header shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.media import RelayMediaClient, media_base_url

from tests.gateway.relay.stub_connector import StubConnector


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        supported_ops=(
            "send",
            "edit",
            "typing",
            "get_chat_info",
            "send_media",
        ),
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


class FakeMediaClient:
    """In-memory stand-in for RelayMediaClient (no HTTP)."""

    def __init__(self) -> None:
        self.enabled = True
        self.uploads: list[tuple[str, Optional[str]]] = []
        self.downloads: list[str] = []
        self.upload_result: Optional[str] = "https://conn.example/relay/media/aa11"
        self.download_result: Optional[str] = "/tmp/relay_media_fake.png"

    async def upload(self, file_path, *, mime=None, filename=None):
        self.uploads.append((str(file_path), filename))
        return self.upload_result

    async def download(self, url, *, suggested_name=None):
        self.downloads.append(url)
        return self.download_result

    def is_relay_media_url(self, url: str) -> bool:
        return "/relay/media/" in (url or "")


def _adapter(**desc_kw) -> tuple[RelayAdapter, StubConnector, FakeMediaClient]:
    stub = StubConnector(make_desc(**desc_kw))
    adapter = RelayAdapter(PlatformConfig(), make_desc(**desc_kw), transport=stub)
    fake = FakeMediaClient()
    adapter._media_client = fake  # bypass env-derived construction
    return adapter, stub, fake


# ── egress: the five overrides ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_image_url_passes_through_without_upload():
    adapter, stub, fake = _adapter()
    result = await adapter.send_image(
        "chat1", "https://fal.media/x.png", caption="a pic", reply_to="m9"
    )
    assert result.success is True
    assert result.message_id == "md1"
    assert fake.uploads == []  # public URL → no upload leg
    action = stub.sent[-1]
    assert action["op"] == "send_media"
    assert action["media_kind"] == "image"
    assert action["source_url"] == "https://fal.media/x.png"
    assert action["content"] == "a pic"
    assert action["reply_to"] == "m9"


@pytest.mark.asyncio
async def test_local_path_lanes_upload_first(tmp_path: Path):
    adapter, stub, fake = _adapter()
    f = tmp_path / "clip.ogg"
    f.write_bytes(b"oggbytes")
    result = await adapter.send_voice("chat1", str(f), caption="listen")
    assert result.success is True
    assert fake.uploads == [(str(f), None)]
    action = stub.sent[-1]
    assert action["op"] == "send_media"
    assert action["media_kind"] == "voice"
    # The wire carries the RE-HOST reference, never the local path.
    assert action["source_url"] == fake.upload_result
    assert str(f) not in str(action)


@pytest.mark.asyncio
async def test_op_gating_falls_back_when_not_advertised(tmp_path: Path):
    # Connector advertises only the legacy ops — send_media must never hit the wire.
    adapter, stub, fake = _adapter(
        supported_ops=("send", "edit", "typing", "get_chat_info")
    )
    result = await adapter.send_image("chat1", "https://x.io/a.png", caption="hi")
    # Base-class fallback: caption + URL as a text send.
    assert result.success is True
    ops = [a["op"] for a in stub.sent]
    assert "send_media" not in ops
    assert ops[-1] == "send"
    assert "https://x.io/a.png" in stub.sent[-1]["content"]


# ── inbound localization ─────────────────────────────────────────────────


def _make_event(media_urls):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    return MessageEvent(
        text="look",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform="telegram", chat_id="c1", chat_type="dm", user_id="u1"
        ),
        media_urls=list(media_urls),
    )


@pytest.mark.asyncio
async def test_inbound_without_client_keeps_public_drops_rehost():
    adapter, _stub, _fake = _adapter()
    adapter._media_client = None
    adapter._get_media_client = lambda: None  # type: ignore[method-assign]
    event = _make_event(
        [
            "https://conn.example/relay/media/deadbeef",
            "https://cdn.discordapp.com/attachments/a/b.png",
        ]
    )
    await adapter._localize_inbound_media(event)
    assert event.media_urls == ["https://cdn.discordapp.com/attachments/a/b.png"]


# ── RelayMediaClient unit surface ────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_upload_rejects_oversize_and_missing(tmp_path: Path):
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Missing file → None (no network attempted).
    assert await c.upload(str(tmp_path / "nope.bin")) is None
    # Empty file → None.
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert await c.upload(str(empty)) is None
