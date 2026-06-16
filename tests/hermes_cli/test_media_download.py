from __future__ import annotations

import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli.media_download import ACCESS_RESTRICTED, DownloadError, download_public_media
from hermes_cli.commands import resolve_command


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + (b"\0" * 32)
MP3_BYTES = b"ID3\x04\x00\x00\x00\x00\x00\x10" + b"public mp3 bytes"


class PublicMediaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def do_HEAD(self):
        self._serve(send_body=False)

    def do_GET(self):
        self._serve(send_body=True)

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None, send_body: bool = True):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _serve(self, send_body: bool):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/media/movie.mp4")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/media/movie.mp4":
            self._send(
                200,
                MP4_BYTES,
                "video/mp4",
                {"Content-Disposition": 'attachment; filename="movie.mp4"', "Accept-Ranges": "bytes"},
                send_body,
            )
            return
        if self.path == "/page":
            body = b'<html><body><audio controls src="/media/song.mp3"></audio></body></html>'
            self._send(200, body, "text/html; charset=utf-8", send_body=send_body)
            return
        if self.path == "/media/song.mp3":
            self._send(200, MP3_BYTES, "audio/mpeg", send_body=send_body)
            return
        if self.path == "/login":
            body = b"<html><body><h1>Log in</h1><form>Sign in to continue. Forgot password?</form></body></html>"
            self._send(200, body, "text/html", send_body=send_body)
            return
        if self.path == "/private.mp4":
            self._send(403, b"forbidden", "text/plain", send_body=send_body)
            return
        self._send(404, b"not found", "text/plain", send_body=send_body)


@pytest.fixture
def media_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), PublicMediaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_command_registered():
    cmd = resolve_command("download")
    assert cmd is not None
    assert cmd.name == "download"
    assert cmd.args_hint == "<url>"


def test_download_direct_media_after_redirect(media_server, tmp_path):
    result = download_public_media(f"{media_server}/redirect", output_dir=tmp_path)

    saved = Path(result.saved_path)
    assert saved.exists()
    assert saved.stat().st_size == len(MP4_BYTES)
    assert saved.suffix == ".mp4"
    assert result.media_type == "video/mp4"
    assert result.content_type == "video/mp4"
    assert result.final_url == f"{media_server}/media/movie.mp4"
    assert result.metadata["content_length"] == len(MP4_BYTES)
    assert result.metadata["accept_ranges"] == "bytes"


def test_download_public_html_direct_media_link(media_server, tmp_path):
    result = download_public_media(f"{media_server}/page", output_dir=tmp_path)

    saved = Path(result.saved_path)
    assert saved.exists()
    assert saved.read_bytes() == MP3_BYTES
    assert saved.suffix == ".mp3"
    assert result.media_type == "audio/mpeg"
    assert result.final_url == f"{media_server}/media/song.mp3"


@pytest.mark.parametrize("path", ["/login", "/private.mp4"])
def test_download_access_restricted_stops(media_server, tmp_path, path):
    with pytest.raises(DownloadError) as excinfo:
        download_public_media(f"{media_server}{path}", output_dir=tmp_path)

    assert str(excinfo.value) == ACCESS_RESTRICTED
    assert not [p for p in tmp_path.iterdir() if p.is_file()]
