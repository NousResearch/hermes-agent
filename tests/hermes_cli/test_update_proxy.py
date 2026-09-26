"""Updater proxy support: channel reads and git honor ``updates.proxy`` (#124022).

The updater ran with no proxy configuration at all: it never set
``http_proxy``/``https_proxy`` and read no proxy setting from config, so on
networks that cut direct TLS every update died mid-handshake
(``SSL: UNEXPECTED_EOF_WHILE_READING`` / ``server closed abruptly``) while the
shipped ``hermes-update-with-vpn.bat`` wrapper (which just exports proxy env
for the child) succeeded on the same machine.
"""
import importlib.util
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

# The HTTP channel fixture is shared without making tests/ an importable package.
_spec = importlib.util.spec_from_file_location(
    "channel_http_fixture_124022", Path(__file__).parents[1] / "scripts/test_release_channels.py")
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
object_server = _fixture.object_server

PROXY_ENV_KEYS = ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY",
                  "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY")


@pytest.fixture(autouse=True)
def _contained_proxy_env():
    # Contain ensure_update_proxy_env()'s real os.environ writes: snapshot the
    # proxy vars, start each test with none set, and restore exactly on teardown
    # (monkeypatch.delenv alone does not undo a later real setenv).
    saved = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _updates_config(monkeypatch, section):
    import hermes_cli.update_proxy as update_proxy
    monkeypatch.setattr(update_proxy, "_updates_config", lambda: dict(section))


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        # A proxy receives the absolute URI in the request line.
        self.server.hits.append(self.path)
        body = b'{"proxied": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def recording_proxy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyHandler)
    server.hits = []
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_channel_read_routes_through_configured_proxy(monkeypatch, recording_proxy):
    """Channel reads must go through ``updates.proxy`` instead of direct TLS."""
    _updates_config(monkeypatch, {"proxy": f"http://127.0.0.1:{recording_proxy.server_port}"})
    from hermes_cli.release_channels import ChannelReader
    with object_server() as (url, objects, headers, requests, faults):
        objects["releases/direct-only.json"] = b'{"direct": true}'
        reader = ChannelReader(url + "/bucket", repository="example/hermes-agent")
        body = reader.read_bytes("releases/direct-only.json")
    assert body == b'{"proxied": true}'
    assert recording_proxy.hits, "channel read bypassed the configured proxy"
    assert any(url in hit for hit in recording_proxy.hits), recording_proxy.hits


def test_update_git_fetch_inherits_configured_proxy(tmp_path, monkeypatch):
    """The updater's network git calls must carry the configured proxy env."""
    _updates_config(monkeypatch, {"proxy": "http://127.0.0.1:10808"})
    from hermes_cli import update_proxy
    from hermes_cli.update_cmd import _git_run
    assert update_proxy.ensure_update_proxy_env() == "http://127.0.0.1:10808"
    witness = tmp_path / "git-env.json"
    fake_git = tmp_path / "fake-git.py"
    fake_git.write_text(
        "import json, os\n"
        f"open({str(witness)!r}, 'w').write(json.dumps({{k: os.environ.get(k) for k in {PROXY_ENV_KEYS!r}}}))\n",
        encoding="utf-8")
    result = _git_run([sys.executable, str(fake_git)], ["fetch", "origin", "main"],
                      cwd=tmp_path, network=True)
    assert result.returncode == 0
    seen = json.loads(witness.read_text(encoding="utf-8"))
    assert seen["http_proxy"] == "http://127.0.0.1:10808", seen
    assert seen["https_proxy"] == "http://127.0.0.1:10808", seen


def test_invalid_proxy_value_falls_back_to_direct(monkeypatch):
    """A malformed ``updates.proxy`` must never brick updates: warn and go direct."""
    _updates_config(monkeypatch, {"proxy": "not a url"})
    from hermes_cli import update_proxy
    assert update_proxy.resolve_update_proxy() is None
    assert update_proxy.ensure_update_proxy_env() is None
    assert os.environ.get("http_proxy") is None
    assert os.environ.get("https_proxy") is None
