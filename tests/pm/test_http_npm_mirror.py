"""A plain-http npm mirror (#150032) is bounded: redirects, size, and the archive fallback."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import threading

import pytest

from pm import artifact_mirror, downloader
from pm.artifact_mirror import pinned_source
from pm.downloader import Download, Source
from tests.pm._range_server import RangeHandler, url
from tests.pm._range_server import dl_server as dl_server  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import

_PNPM_PATH = "/@pnpm/exe.linux-x64/-/exe.linux-x64-12.15.0.tgz"
_BODY = b"pinned pnpm tarball"
_DIGEST = hashlib.sha256(_BODY).hexdigest()
_CAP = 1024


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """urllib honours proxy variables in any case; a developer's proxy must not carry 127.0.0.2."""
    for key in [k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")]:
        monkeypatch.delenv(key)


@pytest.fixture
def archive(dl_server, monkeypatch):
    """Hermes's pinned archive, the fallback behind every mirrored npm artifact."""
    monkeypatch.setattr(artifact_mirror, "PUBLIC_PREFIX", url(dl_server, "/archive/"))
    RangeHandler.payloads["/archive/" + _DIGEST] = _BODY
    return dl_server


@contextlib.contextmanager
def _serve(handler, host="127.0.0.2"):
    """127.0.0.2 is plain http the downloader does not treat as loopback, like a LAN registry."""
    server = HTTPServer((host, 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://{host}:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _lan_mirror(handler, monkeypatch):
    with _serve(handler) as origin:
        monkeypatch.setenv("NPM_CONFIG_REGISTRY", origin + "/")
        yield origin



class _RedirectingMirror(RangeHandler):
    location = ""

    def do_GET(self):
        if self.path == _PNPM_PATH:
            self.send_response(302)
            self.send_header("Location", self.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


@pytest.mark.platforms("linux")  # all of 127/8 answers on Linux; macOS binds only 127.0.0.1
@pytest.mark.parametrize("target", ["same origin", "blob store on another port"])
def test_http_mirror_redirects_are_followed(tmp_path, monkeypatch, archive, target):
    """Nexus/Artifactory 30x tarball requests; the pin, not the redirect target, decides the bytes."""
    RangeHandler.payloads["/blobs/pnpm"] = _BODY
    del RangeHandler.payloads["/archive/" + _DIGEST]  # the mirror alone must deliver
    with _serve(RangeHandler) as blobs, _lan_mirror(_RedirectingMirror, monkeypatch):
        monkeypatch.setattr(_RedirectingMirror, "location", "/blobs/pnpm" if target == "same origin" else blobs + "/blobs/pnpm")
        source = pinned_source("https://registry.npmjs.org" + _PNPM_PATH, tmp_path / "pnpm.tgz", _DIGEST)
        assert Download([source], partials_dir=tmp_path / "partials").run() == [source.dest]
    assert source.dest.read_bytes() == _BODY


@pytest.mark.parametrize("origin", [
    "loopback",
    pytest.param("http mirror", marks=pytest.mark.platforms("linux")),  # macOS binds only 127.0.0.1
])
def test_refused_redirect_falls_back_to_the_archive(tmp_path, monkeypatch, archive, origin):
    """A redirect the policy refuses (http to another host) is a mirror miss, not the end of the download."""
    with _serve(RangeHandler, "127.0.0.3" if origin == "http mirror" else "127.0.0.1") as elsewhere:
        if origin == "loopback":
            # Loopback test servers may not bounce to a non-loopback http host.
            monkeypatch.setattr(_RedirectingMirror, "location", "http://192.0.2.1/tool")
            with _serve(_RedirectingMirror, "127.0.0.1") as primary:
                source = Source(primary + _PNPM_PATH, tmp_path / "tool", _DIGEST,
                                fallbacks=(artifact_mirror.mirror_url(_DIGEST),))
                Download([source], partials_dir=tmp_path / "partials").run()
        else:
            RangeHandler.payloads["/elsewhere"] = b"not what the pin names"
            monkeypatch.setattr(_RedirectingMirror, "location", elsewhere + "/elsewhere")
            with _lan_mirror(_RedirectingMirror, monkeypatch):
                source = pinned_source("https://registry.npmjs.org" + _PNPM_PATH, tmp_path / "tool", _DIGEST)
                Download([source], partials_dir=tmp_path / "partials").run()
    assert source.dest.read_bytes() == _BODY


class _OversizedMirror(BaseHTTPRequestHandler):
    """An on-path attacker's answer: a huge declared range total, or an endless stream."""

    ranged = True

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.ranged:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes 0-0/{1 << 30}")
            self.send_header("Content-Length", "1")
            self.end_headers()
            self.wfile.write(b"x")
            return
        self.send_response(200)
        self.end_headers()
        with contextlib.suppress(OSError):
            for _ in range(4):
                self.wfile.write(b"x" * _CAP)


@pytest.mark.platforms("linux")  # all of 127/8 answers on Linux; macOS binds only 127.0.0.1
@pytest.mark.parametrize("ranged", [True, False], ids=["declared total", "unbounded stream"])
def test_http_mirror_cannot_fill_the_disk(tmp_path, monkeypatch, archive, ranged):
    """Bytes from plain http are bounded before the hash check; the archive then serves the pin."""
    monkeypatch.setattr(downloader, "_HTTP_MIRROR_MAX_BYTES", _CAP)
    monkeypatch.setattr(_OversizedMirror, "ranged", ranged)
    partials = tmp_path / "partials"
    with _lan_mirror(_OversizedMirror, monkeypatch):
        source = pinned_source("https://registry.npmjs.org" + _PNPM_PATH, tmp_path / "pnpm.tgz", _DIGEST)
        try:
            Download([source], partials_dir=partials).run()
        finally:
            assert all(path.stat().st_size <= _CAP for path in partials.rglob("*") if path.is_file())
    assert source.dest.read_bytes() == _BODY
