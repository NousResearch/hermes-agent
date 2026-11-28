"""Upload filenames survive the real urllib/http.client header serialization."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import unquote

import pytest

from gateway.relay.media import RelayMediaClient
from tests.gateway.relay.test_relay_media import _adapter


@pytest.fixture
def media_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.headers.get("X-Media-Filename", "")
            # Decode only the explicit ext-value marker, and only once: a raw
            # legacy name may contain percent escapes that are literal text.
            name = unquote(raw[7:], encoding="utf-8", errors="strict") if raw.startswith("UTF-8''") else raw
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((raw, name, body, self.headers["Content-Type"], self.path))
            self.send_response(201)
            self.end_headers()
            self.wfile.write(b'{"id":"uploaded","size":4}')

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        yield RelayMediaClient(base, "gw1", "test-secret"), received, base
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "header"),
    [
        ("отчёт.pdf", "UTF-8''%D0%BE%D1%82%D1%87%D1%91%D1%82.pdf"),
        ("報告.pdf", "UTF-8''%E5%A0%B1%E5%91%8A.pdf"),
        ("é.pdf", "UTF-8''%C3%A9.pdf"),
        ("report.pdf", "report.pdf"),
        ("100%.txt", "UTF-8''100%25.txt"),
        ("a%2F.txt", "UTF-8''a%252F.txt"),
        ("line\r\nbreak\t\x7f.txt", "UTF-8''line%0D%0Abreak%09%7F.txt"),
        ("я" * 300, "UTF-8''" + "%D1%8F" * 255),
        ("a" * 254 + "😀tail", "UTF-8''" + "a" * 254 + "%F0%9F%98%80"),
        ("😀" * 300, "UTF-8''" + "%F0%9F%98%80" * 255),
    ],
    ids=["cyrillic", "cjk", "latin1", "ascii", "percent", "escaped-slash",
         "controls", "truncated-cyrillic", "astral-boundary", "maximum-header"],
)
async def test_upload_filename_roundtrip(media_server, tmp_path, name, header):
    client, received, base = media_server
    path = tmp_path / "payload.pdf"
    path.write_bytes(b"%PDF")

    reference = await client.upload(str(path), filename=name, mime="application/pdf")

    assert reference == f"{base}/relay/media/uploaded"
    assert received == [(header, name[:255], b"%PDF", "application/pdf", "/relay/media")]
    assert len(header.encode("ascii")) <= 3067


@pytest.mark.asyncio
async def test_cyrillic_document_sends_media_without_fallback(media_server, tmp_path):
    client, received, base = media_server
    adapter, stub, _fake = _adapter()
    adapter._media_client = client
    path = tmp_path / "отчёт.pdf"
    path.write_bytes(b"%PDF")

    result = await adapter.send_document("chat1", str(path))

    assert result.success is True
    assert received[0][1] == "отчёт.pdf"
    assert [action["op"] for action in stub.sent] == ["send_media"]
    assert stub.sent[0]["source_url"] == f"{base}/relay/media/uploaded"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["report.pdf", "é.pdf", "a%2F.txt"])
async def test_receiver_accepts_legacy_raw_names(media_server, name):
    import asyncio
    import urllib.request

    _client, received, base = media_server

    def post():
        request = urllib.request.Request(
            f"{base}/relay/media", data=b"%PDF",
            headers={"X-Media-Filename": name, "Content-Type": "application/pdf"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 201

    await asyncio.get_running_loop().run_in_executor(None, post)
    assert received[0][:2] == (name, name)
