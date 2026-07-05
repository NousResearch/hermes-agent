"""Direct tests for ``agent.image_gen_provider.save_url_image`` (#26942).

These exercise the helper against a real in-process HTTP server — no
``requests.get`` mocking — so we catch the kinds of issues a mocked
unit test won't: content-type parsing, partial-write cleanup, the
oversize cap, the empty-body refusal, and the cache directory it
actually writes to.

Pre-fix the helper didn't exist; xAI URL responses were returned bare
and the gateway 404'd at ``send_photo`` time.
"""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest


PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de00000010494441547801635c0e000000feff03000006000557bfabd400"
    "00000049454e44ae426082"
)


class _TinyImageHandler(http.server.BaseHTTPRequestHandler):
    """Tiny HTTP server that mimics the shapes save_url_image must handle."""

    def do_GET(self):
        if self.path == "/image.png":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG_1PX)))
            self.end_headers()
            self.wfile.write(PNG_1PX)
        elif self.path == "/image.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(PNG_1PX)  # bytes don't have to be a real jpeg
        elif self.path == "/oversize":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            chunk = b"\x00" * 65536
            for _ in range(64):  # 4 MiB
                self.wfile.write(chunk)
        elif self.path == "/empty":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/404":
            self.send_response(404)
            self.end_headers()
        elif self.path == "/no-type-with-url-ext.jpg":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(PNG_1PX)
        elif self.path == "/no-type-no-ext":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(PNG_1PX)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kw):
        return


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    """Spin up a localhost HTTP server and isolate HERMES_HOME under tmp_path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    # Force the constants/image cache helpers to re-read HERMES_HOME.
    import sys
    for mod in list(sys.modules):
        if mod.startswith("hermes_constants") or mod.startswith("agent.image_gen_provider"):
            sys.modules.pop(mod, None)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _TinyImageHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", httpd
    httpd.shutdown()


class TestSaveUrlImage:
    def test_writes_real_bytes_to_hermes_home_cache(self, http_server):
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        path = save_url_image(f"{base}/image.png", prefix="xai_test")

        assert path.exists()
        assert path.read_bytes() == PNG_1PX
        # The cache directory must be under HERMES_HOME — gateway cleanup
        # relies on this being the canonical location.
        assert "cache/images" in str(path)
        assert path.suffix == ".png"

    def test_extension_inferred_from_content_type(self, http_server):
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        path = save_url_image(f"{base}/image.jpg", prefix="xai_test")
        assert path.suffix == ".jpg", "image/jpeg → .jpg"

    def test_extension_falls_back_to_url_suffix(self, http_server):
        """Some CDNs send ``application/octet-stream`` — the URL suffix wins then."""
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        path = save_url_image(f"{base}/no-type-with-url-ext.jpg", prefix="xai_test")
        assert path.suffix == ".jpg"

    def test_extension_defaults_to_png_when_unknowable(self, http_server):
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        path = save_url_image(f"{base}/no-type-no-ext", prefix="xai_test")
        assert path.suffix == ".png"

    def test_404_raises(self, http_server):
        """HTTP errors must propagate — caller decides whether to fall back."""
        base, _ = http_server
        from agent.image_gen_provider import save_url_image
        import requests as req_lib

        with pytest.raises(req_lib.HTTPError):
            save_url_image(f"{base}/404")

    def test_empty_body_raises_without_writing_file(self, http_server):
        """0-byte responses are not images — refuse to cache."""
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        with pytest.raises(ValueError, match="0 bytes"):
            save_url_image(f"{base}/empty")

    def test_oversize_raises_and_cleans_up(self, http_server, tmp_path):
        """Oversize downloads must NOT leak a partial file into the cache."""
        base, _ = http_server
        from agent.image_gen_provider import save_url_image, _images_cache_dir

        cache_dir = _images_cache_dir()
        before = set(cache_dir.glob("*"))
        with pytest.raises(ValueError, match="exceeds"):
            save_url_image(f"{base}/oversize", max_bytes=1024 * 1024)
        after = set(cache_dir.glob("*"))
        assert after == before, "partial file leaked into cache after oversize cap"

    def test_unique_filenames_avoid_collision(self, http_server):
        """Two back-to-back saves of the same URL must produce different paths."""
        base, _ = http_server
        from agent.image_gen_provider import save_url_image

        path1 = save_url_image(f"{base}/image.png", prefix="xai_collision")
        path2 = save_url_image(f"{base}/image.png", prefix="xai_collision")
        assert path1 != path2, "filename collision — uuid suffix isn't doing its job"
