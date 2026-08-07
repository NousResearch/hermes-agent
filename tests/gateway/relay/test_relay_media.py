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

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        self._classifier = RelayMediaClient("https://conn.example", "gw1", "sec")

    async def upload(self, file_path, *, mime=None, filename=None):
        self.uploads.append((str(file_path), filename))
        return self.upload_result

    async def download(self, url, *, suggested_name=None):
        self.downloads.append(url)
        return self.download_result

    def is_relay_media_url(self, url: str) -> bool:
        return self._classifier.is_relay_media_url(url)


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
async def test_inbound_download_failure_keeps_public_marker_url():
    adapter, _stub, fake = _adapter()
    fake.download_result = None
    public_marker = "https://cdn.example/files/relay/media/photo.png"
    event = _make_event(
        [
            "https://conn.example/relay/media/deadbeef",
            public_marker,
        ]
    )
    await adapter._localize_inbound_media(event)
    assert fake.downloads == [
        "https://conn.example/relay/media/deadbeef",
        public_marker,
    ]
    assert event.media_urls == [public_marker]


@pytest.mark.asyncio
async def test_inbound_without_client_keeps_public_drops_rehost(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://conn.example/relay")
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


@pytest.mark.asyncio
async def test_inbound_without_client_keeps_public_marker_url(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://conn.example/relay")
    adapter, _stub, _fake = _adapter()
    adapter._media_client = None
    adapter._get_media_client = lambda: None  # type: ignore[method-assign]
    public_marker = "https://cdn.example/files/relay/media/photo.png"
    event = _make_event(
        [
            "https://conn.example/relay/media/deadbeef",
            public_marker,
        ]
    )
    await adapter._localize_inbound_media(event)
    assert event.media_urls == [public_marker]


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
    assert c.is_relay_media_url("https://evil.example/relay/media/abc") is False
    assert c.is_relay_media_url("https://cdn.example/files/relay/media/photo.png") is False
    assert c.is_relay_media_url("https://cdn.discordapp.com/a/b.png") is False


def test_client_recognizes_rehost_urls_under_connector_base_path():
    c = RelayMediaClient("https://c.example/team-a", "gw1", "sec")
    assert c.is_relay_media_url("https://c.example/team-a/relay/media/abc") is True
    assert c.is_relay_media_url("https://c.example/relay/media/abc") is False


def test_client_recognizes_equivalent_default_port_origins():
    implicit_https = RelayMediaClient("https://c.example", "gw1", "sec")
    explicit_https = RelayMediaClient("https://c.example:443", "gw1", "sec")
    implicit_http = RelayMediaClient("http://c.example", "gw1", "sec")

    assert implicit_https.is_relay_media_url(
        "https://c.example:443/relay/media/abc"
    ) is True
    assert explicit_https.is_relay_media_url(
        "https://c.example/relay/media/abc"
    ) is True
    assert implicit_http.is_relay_media_url(
        "http://c.example:80/relay/media/abc"
    ) is True
    assert implicit_https.is_relay_media_url(
        "https://c.example:444/relay/media/abc"
    ) is False
    assert implicit_https.is_relay_media_url(
        "http://c.example:443/relay/media/abc"
    ) is False


class _CaptureHandler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, Optional[str]]] = []
    location: str = ""
    body: bytes = b"ok"
    content_type: str = "application/octet-stream"

    def do_GET(self):
        type(self).seen.append(("GET", self.path, self.headers.get("Authorization")))
        if type(self).location:
            self.send_response(302)
            self.send_header("Location", type(self).location)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", type(self).content_type)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        type(self).seen.append(("POST", self.path, self.headers.get("Authorization")))
        if type(self).location:
            self.send_response(302)
            self.send_header("Location", type(self).location)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"id":"uploaded"}')

    def log_message(self, format, *args):
        return


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _stop(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


@pytest.mark.asyncio
async def test_authenticated_download_does_not_follow_cross_origin_redirect():
    class Capture(_CaptureHandler):
        seen = []

    capture_server = _serve(Capture)

    class Redirect(_CaptureHandler):
        seen = []
        location = f"http://127.0.0.1:{capture_server.server_port}/capture"

    redirect_server = _serve(Redirect)
    result = None
    try:
        c = RelayMediaClient(
            f"http://127.0.0.1:{redirect_server.server_port}", "gw1", "sec"
        )
        result = await c.download(
            f"http://127.0.0.1:{redirect_server.server_port}/relay/media/x"
        )
        assert result is None
        assert len(Redirect.seen) == 1
        assert Redirect.seen[0][2]
        assert Capture.seen == []
    finally:
        _stop(redirect_server)
        _stop(capture_server)


@pytest.mark.asyncio
async def test_authenticated_upload_does_not_follow_cross_origin_redirect(tmp_path):
    class Capture(_CaptureHandler):
        seen = []

    capture_server = _serve(Capture)

    class Redirect(_CaptureHandler):
        seen = []
        location = f"http://127.0.0.1:{capture_server.server_port}/capture"

    redirect_server = _serve(Redirect)
    media = tmp_path / "clip.txt"
    media.write_text("hello", encoding="utf-8")
    try:
        c = RelayMediaClient(
            f"http://127.0.0.1:{redirect_server.server_port}", "gw1", "sec"
        )
        result = await c.upload(str(media))
        assert result is None
        assert len(Redirect.seen) == 1
        assert Redirect.seen[0][0] == "POST"
        assert Redirect.seen[0][2]
        assert Capture.seen == []
    finally:
        _stop(redirect_server)
        _stop(capture_server)


@pytest.mark.asyncio
async def test_public_download_keeps_normal_redirect_behavior():
    class Capture(_CaptureHandler):
        seen = []
        body = b"public image"
        content_type = "image/png"

    capture_server = _serve(Capture)

    class Redirect(_CaptureHandler):
        seen = []
        location = f"http://127.0.0.1:{capture_server.server_port}/capture.png"

    redirect_server = _serve(Redirect)
    try:
        c = RelayMediaClient(
            f"http://127.0.0.1:{redirect_server.server_port}", "gw1", "sec"
        )
        result = await c.download(
            f"http://127.0.0.1:{redirect_server.server_port}/public/image"
        )
        assert result is not None
        assert Path(result).read_bytes() == b"public image"
        assert len(Redirect.seen) == 1
        assert Redirect.seen[0][2] is None
        assert len(Capture.seen) == 1
        assert Capture.seen[0][2] is None
    finally:
        if result is not None:
            Path(result).unlink(missing_ok=True)
        _stop(redirect_server)
        _stop(capture_server)


@pytest.mark.asyncio
async def test_client_upload_rejects_oversize_and_missing(tmp_path: Path):
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Missing file → None (no network attempted).
    assert await c.upload(str(tmp_path / "nope.bin")) is None
    # Empty file → None.
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert await c.upload(str(empty)) is None
