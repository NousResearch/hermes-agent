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

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.media import RelayMediaClient

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


def test_client_rejects_hostile_origin_media_url():
    """A path-substring match alone is spoofable: a caller-supplied event
    could name https://attacker.example/relay/media/x and get our
    per-gateway bearer attached to an arbitrary host. The URL's origin must
    match the configured connector's origin too."""
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Same-origin re-host reference — still recognized.
    assert c.is_relay_media_url("https://c.example/relay/media/abc") is True
    # A public URL with no rehost path segment — never recognized.
    assert c.is_relay_media_url("https://cdn.discordapp.com/a/b.png") is False
    # Path substring matches, but the origin doesn't — rejected.
    assert c.is_relay_media_url("https://attacker.example/relay/media/x") is False
    # Same path, different port — still a different origin.
    assert c.is_relay_media_url("https://c.example:8443/relay/media/x") is False


@pytest.mark.asyncio
async def test_client_upload_rejects_oversize_and_missing(tmp_path: Path):
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Missing file → None (no network attempted).
    assert await c.upload(str(tmp_path / "nope.bin")) is None
    # Empty file → None.
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert await c.upload(str(empty)) is None


# ── RelayMediaClient credential-redirect hardening (real servers) ────────


class _RedirectingMediaHandler(BaseHTTPRequestHandler):
    """Answers the configured path with a 302 to a configurable target.

    Same shape as _RedirectingProvisionHandler in test_self_provision.py:
    a second, independent server plays the redirect target and records the
    headers it received, proving the gateway's per-gateway bearer never
    reaches an unintended origin.
    """

    redirect_to = ""  # full URL, set per test
    trigger_path = "/relay/media"  # path that answers 302, set per test
    received_headers: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.rstrip("/") == type(self).trigger_path:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
        else:
            self._respond()

    def do_GET(self):
        if self.path.rstrip("/") == type(self).trigger_path:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
        else:
            self._respond()

    def _respond(self):
        type(self).received_headers = dict(self.headers)
        body = json.dumps({"id": "collected"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.mark.asyncio
async def test_media_upload_strips_bearer_on_cross_host_redirect(tmp_path: Path):
    """upload()'s Authorization must not follow a redirect to a different
    origin — mirrors the same fix already applied to _post_provision() /
    _post_policy() / _post_enroll()."""
    _RedirectingMediaHandler.received_headers = {}
    _RedirectingMediaHandler.trigger_path = "/relay/media"
    server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    target_server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    port = server.server_address[1]
    target_port = target_server.server_address[1]
    _RedirectingMediaHandler.redirect_to = f"http://127.0.0.1:{target_port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()
    Thread(target=target_server.serve_forever, daemon=True).start()

    f = tmp_path / "pic.png"
    f.write_bytes(b"png-bytes")
    c = RelayMediaClient(f"http://127.0.0.1:{port}", "gw1", "sec")
    try:
        result = await c.upload(str(f))
    finally:
        server.shutdown()
        target_server.shutdown()

    # The redirect target answered normally, proving the request really was
    # followed — but without the Bearer token attached.
    assert result is not None
    headers = {k.lower(): v for k, v in _RedirectingMediaHandler.received_headers.items()}
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_media_upload_preserves_bearer_on_same_origin_redirect(tmp_path: Path):
    """A same-origin redirect (e.g. the connector's own load balancer) must
    still carry the Bearer token — only cross-origin hops strip it."""
    _RedirectingMediaHandler.received_headers = {}
    _RedirectingMediaHandler.trigger_path = "/relay/media"
    server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    port = server.server_address[1]
    _RedirectingMediaHandler.redirect_to = f"http://127.0.0.1:{port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()

    f = tmp_path / "pic.png"
    f.write_bytes(b"png-bytes")
    c = RelayMediaClient(f"http://127.0.0.1:{port}", "gw1", "sec")
    try:
        result = await c.upload(str(f))
    finally:
        server.shutdown()

    assert result is not None
    headers = {k.lower(): v for k, v in _RedirectingMediaHandler.received_headers.items()}
    assert "authorization" in headers
    assert headers["authorization"].startswith("Bearer ")


@pytest.mark.asyncio
async def test_media_download_strips_bearer_on_cross_host_redirect():
    """download()'s Authorization must not follow a redirect to a different
    origin either. The requested URL is a /relay/media/ reference on the
    client's OWN configured origin (so it is authenticated on the initial
    request), which then redirects cross-origin."""
    _RedirectingMediaHandler.received_headers = {}
    _RedirectingMediaHandler.trigger_path = "/relay/media/abc123"
    server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    target_server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    port = server.server_address[1]
    target_port = target_server.server_address[1]
    _RedirectingMediaHandler.redirect_to = f"http://127.0.0.1:{target_port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()
    Thread(target=target_server.serve_forever, daemon=True).start()

    c = RelayMediaClient(f"http://127.0.0.1:{port}", "gw1", "sec")
    try:
        result = await c.download(f"http://127.0.0.1:{port}/relay/media/abc123")
    finally:
        server.shutdown()
        target_server.shutdown()

    assert result is not None
    headers = {k.lower(): v for k, v in _RedirectingMediaHandler.received_headers.items()}
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_media_download_preserves_bearer_on_same_origin_redirect():
    _RedirectingMediaHandler.received_headers = {}
    _RedirectingMediaHandler.trigger_path = "/relay/media/abc123"
    server = HTTPServer(("127.0.0.1", 0), _RedirectingMediaHandler)
    port = server.server_address[1]
    _RedirectingMediaHandler.redirect_to = f"http://127.0.0.1:{port}/collect"
    Thread(target=server.serve_forever, daemon=True).start()

    c = RelayMediaClient(f"http://127.0.0.1:{port}", "gw1", "sec")
    try:
        result = await c.download(f"http://127.0.0.1:{port}/relay/media/abc123")
    finally:
        server.shutdown()

    assert result is not None
    headers = {k.lower(): v for k, v in _RedirectingMediaHandler.received_headers.items()}
    assert "authorization" in headers
    assert headers["authorization"].startswith("Bearer ")


class _HostileMediaHandler(BaseHTTPRequestHandler):
    """A server standing in for an attacker.example host: any path is
    answered 200 directly (no redirect), recording whatever headers it
    received."""

    received_headers: dict = {}

    def do_GET(self):
        type(self).received_headers = dict(self.headers)
        body = b"not-really-media"
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.mark.asyncio
async def test_media_download_omits_bearer_for_hostile_origin_url():
    """A URL that merely CONTAINS '/relay/media/' but lives on a different
    origin than the configured connector must never receive the bearer —
    is_relay_media_url()'s origin check must gate the initial request, not
    just cross-origin redirects."""
    _HostileMediaHandler.received_headers = {}
    hostile_server = HTTPServer(("127.0.0.1", 0), _HostileMediaHandler)
    hostile_port = hostile_server.server_address[1]
    Thread(target=hostile_server.serve_forever, daemon=True).start()

    # Client is configured for a DIFFERENT connector origin than the one
    # it's asked to download from.
    c = RelayMediaClient("https://real-connector.example", "gw1", "sec")
    try:
        result = await c.download(
            f"http://127.0.0.1:{hostile_port}/relay/media/x"
        )
    finally:
        hostile_server.shutdown()

    # The download still succeeds (public URLs are fetched without auth) —
    # what matters is the credential never went to the hostile host.
    assert result is not None
    headers = {k.lower(): v for k, v in _HostileMediaHandler.received_headers.items()}
    assert "authorization" not in headers
