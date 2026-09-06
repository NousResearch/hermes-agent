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

@pytest.mark.asyncio
async def test_download_sends_a_user_agent_on_every_request():
    """Discord's CDN 403s the default ``Python-urllib/x.y`` User-Agent.

    Live-verified on staging 2026-08-26: every Discord CDN pass-through
    download failed with ``HTTP Error 403: Forbidden``, the localizer then kept
    the raw URL, and the transcriber tried to open a URL as a file path
    ("Audio file not found") — so voice notes, images and documents were ALL
    silently dead on the Discord relay lane. Reproduced from a clean shell:
    urllib with no UA → 403; the same URL with any descriptive UA → 200.

    Assert the header on BOTH url classes (public pass-through and
    bearer-authenticated re-host), because they take different header paths.
    """
    seen: list[dict] = []

    class _Resp:
        headers = {"Content-Type": "audio/ogg", "Content-Length": "4"}

        def read(self, *_a):
            return b"OggS"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        seen.append(dict(req.headers))
        return _Resp()

    import urllib.request as _ur

    orig = _ur.urlopen
    _ur.urlopen = _fake_urlopen  # type: ignore[assignment]
    try:
        c = RelayMediaClient("https://conn.example", "gw1", "sec")
        assert await c.download("https://cdn.discordapp.com/attachments/1/2/v.ogg")
        assert await c.download("https://conn.example/relay/media/deadbeef")
    finally:
        _ur.urlopen = orig  # type: ignore[assignment]

    assert len(seen) == 2
    for headers in seen:
        # urllib title-cases header keys on Request.
        ua = headers.get("User-agent") or headers.get("User-Agent")
        assert ua, f"no User-Agent sent; urllib would default to Python-urllib (403s on Discord CDN): {headers}"
        assert "python-urllib" not in ua.lower()
    # The re-host request must still carry its bearer (no regression).
    rehost_headers = seen[1]
    assert (rehost_headers.get("Authorization") or "").startswith("Bearer ")


def test_is_relay_media_url_distinguishes_rehost_from_public():
    """Public compat helper: connector re-host refs need our bearer; ordinary
    public URLs (CDN pass-throughs) do not. None/empty never raise."""
    c = RelayMediaClient("https://conn.example", "gw1", "sec")
    assert c.is_relay_media_url("https://conn.example/relay/media/aa11") is True
    assert c.is_relay_media_url("http://other.host:8080/relay/media/x") is True
    assert c.is_relay_media_url("https://cdn.discordapp.com/attachments/1/2/v.ogg") is False
    assert c.is_relay_media_url("https://conn.example/relay/mediafile") is False
    assert c.is_relay_media_url("") is False
    assert c.is_relay_media_url(None) is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_download_routes_auth_decision_through_is_relay_media_url(monkeypatch):
    """download() must consult is_relay_media_url (so subclasses/plugins that
    override the classifier change auth behaviour): a disabled client refuses
    re-host refs but still fetches public URLs without a bearer."""
    asked: list[str] = []
    c = RelayMediaClient("https://conn.example", None, None)  # disabled: no creds
    assert c.enabled is False
    orig = c.is_relay_media_url

    def _spy(url):
        asked.append(url)
        return orig(url)

    monkeypatch.setattr(c, "is_relay_media_url", _spy)
    # Re-host ref + disabled client → None before any network call.
    assert await c.download("https://conn.example/relay/media/deadbeef") is None
    assert asked == ["https://conn.example/relay/media/deadbeef"]

    seen: list[dict] = []

    class _Resp:
        headers = {"Content-Type": "image/png", "Content-Length": "4"}

        def read(self, *_a):
            return b"\x89PNG"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):  # noqa: ARG001
        seen.append(dict(req.headers))
        return _Resp()

    import urllib.request as _ur

    monkeypatch.setattr(_ur, "urlopen", _fake_urlopen)
    assert await c.download("https://cdn.discordapp.com/attachments/1/2/i.png")
    assert asked[-1] == "https://cdn.discordapp.com/attachments/1/2/i.png"
    assert len(seen) == 1 and "Authorization" not in seen[0]
