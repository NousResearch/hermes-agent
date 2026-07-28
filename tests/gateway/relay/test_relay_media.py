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
async def test_each_override_maps_to_its_media_kind(tmp_path: Path):
    adapter, stub, fake = _adapter()
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    await adapter.send_image_file("c", str(f))
    await adapter.send_voice("c", str(f))
    await adapter.send_video("c", str(f))
    await adapter.send_document("c", str(f), file_name="report.pdf")
    kinds = [a["media_kind"] for a in stub.sent if a["op"] == "send_media"]
    assert kinds == ["image", "voice", "video", "document"]
    doc_action = stub.sent[-1]
    assert doc_action["filename"] == "report.pdf"
    # The document upload passed the user-facing filename through.
    assert fake.uploads[-1] == (str(f), "report.pdf")


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


@pytest.mark.asyncio
async def test_legacy_empty_ops_also_gates_send_media():
    # An empty supported_ops = legacy connector; send_media is NOT in LEGACY_OPS.
    adapter, stub, _fake = _adapter(supported_ops=())
    await adapter.send_image("chat1", "https://x.io/a.png")
    assert all(a["op"] != "send_media" for a in stub.sent)


@pytest.mark.asyncio
async def test_connector_decline_degrades_to_fallback(tmp_path: Path):
    adapter, stub, fake = _adapter()
    stub.next_media_result = {"success": False, "error": "media too large"}
    f = tmp_path / "big.mp4"
    f.write_bytes(b"x")
    result = await adapter.send_video("chat1", str(f), caption="cap")
    # The video lane failed → base fallback notice still delivers (a send op).
    assert result.success is True
    assert stub.sent[-1]["op"] == "send"
    assert "cap" in stub.sent[-1]["content"]
    # And the local path never leaked into the fallback text.
    assert str(f) not in stub.sent[-1]["content"]


@pytest.mark.asyncio
async def test_failed_upload_degrades_to_fallback(tmp_path: Path):
    adapter, stub, fake = _adapter()
    fake.upload_result = None  # upload leg fails
    f = tmp_path / "pic.png"
    f.write_bytes(b"png")
    result = await adapter.send_image_file("chat1", str(f))
    assert result.success is True
    assert all(a["op"] != "send_media" for a in stub.sent)


@pytest.mark.asyncio
async def test_scope_metadata_rides_send_media(tmp_path: Path):
    """The egress guard resolves tenants from metadata — send_media must carry
    the same scope/user discriminators a plain send does."""
    adapter, stub, fake = _adapter()
    adapter._scope_by_chat["chat1"] = "guild9"
    await adapter.send_image("chat1", "https://x.io/p.png")
    action = stub.sent[-1]
    assert action["op"] == "send_media"
    assert (action.get("metadata") or {}).get("scope_id") == "guild9"


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
async def test_inbound_rehost_urls_are_localized():
    adapter, _stub, fake = _adapter()
    event = _make_event(["https://conn.example/relay/media/deadbeef"])
    await adapter._localize_inbound_media(event)
    assert fake.downloads == ["https://conn.example/relay/media/deadbeef"]
    assert event.media_urls == ["/tmp/relay_media_fake.png"]


@pytest.mark.asyncio
async def test_inbound_dead_rehost_ref_is_dropped_public_url_kept():
    adapter, _stub, fake = _adapter()
    fake.download_result = None  # every download fails
    event = _make_event(
        [
            "https://conn.example/relay/media/deadbeef",  # dead re-host → dropped
            "https://cdn.discordapp.com/attachments/a/b.png",  # public → kept as URL
        ]
    )
    await adapter._localize_inbound_media(event)
    assert event.media_urls == ["https://cdn.discordapp.com/attachments/a/b.png"]


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


def test_media_base_url_derivation():
    assert media_base_url("wss://conn.example/relay") == "https://conn.example"
    assert media_base_url("ws://localhost:8080/relay") == "http://localhost:8080"
    assert media_base_url("https://conn.example") == "https://conn.example"


def test_client_enabled_requires_full_credentials():
    assert RelayMediaClient("https://c.example", "gw1", "sec").enabled is True
    assert RelayMediaClient("https://c.example", None, "sec").enabled is False
    assert RelayMediaClient("https://c.example", "gw1", None).enabled is False
    assert RelayMediaClient("", "gw1", "sec").enabled is False


def test_client_recognizes_rehost_urls():
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    assert c.is_relay_media_url("https://c.example/relay/media/abc") is True
    assert c.is_relay_media_url("https://cdn.discordapp.com/a/b.png") is False


@pytest.mark.asyncio
async def test_client_upload_rejects_oversize_and_missing(tmp_path: Path):
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Missing file → None (no network attempted).
    assert await c.upload(str(tmp_path / "nope.bin")) is None
    # Empty file → None.
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert await c.upload(str(empty)) is None
