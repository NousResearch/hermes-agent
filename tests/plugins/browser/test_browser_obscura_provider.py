"""Tests for the Obscura local browser provider plugin.

Obscura is the only *local* browser backend: ``create_session`` spawns a real
``obscura serve`` subprocess and returns its CDP endpoint. These tests follow
the repo rule of using real objects, not mocks: a tiny fake ``obscura`` binary
(a real Python script that serves ``/json/version`` over HTTP, exactly like the
real engine) stands in for the Rust binary, so the full spawn / poll / teardown
path is exercised for real without depending on the binary being installed.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import time

import pytest
import requests

from plugins.browser.obscura.provider import ObscuraBrowserProvider

# A real Python program that mimics ``obscura serve --port N``: it serves the
# ``/json/version`` document the provider polls for, then runs until killed.
# When FAKE_ARGV_FILE is set it records its argv so a test can assert the CLI
# shape (e.g. that ``--stealth`` was forwarded).
_FAKE_OBSCURA = """#!/usr/bin/env python3
import sys, os, json, http.server, socketserver

args = sys.argv[1:]
argv_file = os.environ.get("FAKE_ARGV_FILE")
if argv_file:
    with open(argv_file, "w") as f:
        f.write(" ".join(args))

if not args or args[0] != "serve":
    sys.exit(2)
port = None
for i, a in enumerate(args):
    if a == "--port" and i + 1 < len(args):
        port = int(args[i + 1])
if port is None:
    sys.exit(2)

ws = "ws://127.0.0.1:%d/devtools/browser" % port


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/json/version":
            body = json.dumps(
                {"Browser": "Obscura/fake", "webSocketDebuggerUrl": ws}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


with socketserver.TCPServer(("127.0.0.1", port), Handler) as srv:
    srv.serve_forever()
"""

# A fake that never serves anything: it just sleeps, so the provider's readiness
# poll times out and create_session raises.
_FAKE_OBSCURA_NEVER_READY = """#!/usr/bin/env python3
import time
time.sleep(60)
"""


def _write_fake(path, body: str) -> str:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "OBSCURA_BIN",
        "OBSCURA_STEALTH",
        "OBSCURA_PORT",
        "OBSCURA_STARTUP_TIMEOUT",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def fake_obscura(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = _write_fake(tmp_path / "obscura", _FAKE_OBSCURA)
    monkeypatch.setenv("OBSCURA_BIN", path)
    return path


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


def test_is_available_false_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OBSCURA_BIN", "/nonexistent/path/to/obscura")
    assert ObscuraBrowserProvider().is_available() is False


def test_is_available_true_with_resolvable_binary(fake_obscura: str) -> None:
    assert ObscuraBrowserProvider().is_available() is True


def test_identity() -> None:
    p = ObscuraBrowserProvider()
    assert p.name == "obscura"
    assert p.display_name == "Obscura"
    schema = p.get_setup_schema()
    assert schema["post_setup"] == "agent_browser"
    assert any(v["key"] == "OBSCURA_BIN" for v in schema["env_vars"])


# ---------------------------------------------------------------------------
# create_session / close_session lifecycle (real subprocess)
# ---------------------------------------------------------------------------


def test_create_session_spawns_and_returns_live_cdp_url(fake_obscura: str) -> None:
    provider = ObscuraBrowserProvider()
    session = provider.create_session("task-abc")
    try:
        assert session["session_name"].startswith("hermes_task-abc_")
        assert session["bb_session_id"]
        assert session["features"] == {"stealth": False, "local": True}

        cdp_url = session["cdp_url"]
        port = int(re.search(r":(\d+)/", cdp_url).group(1))
        assert cdp_url == f"ws://127.0.0.1:{port}/devtools/browser"

        # The endpoint the provider returned is actually reachable.
        resp = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=3)
        assert resp.ok
        assert resp.json()["webSocketDebuggerUrl"] == cdp_url
    finally:
        assert provider.close_session(session["bb_session_id"]) is True

    # After close the process is gone, so the port stops serving.
    time.sleep(0.5)
    with pytest.raises(requests.RequestException):
        requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1)


def test_close_unknown_session_returns_false(fake_obscura: str) -> None:
    assert ObscuraBrowserProvider().close_session("nope") is False


def test_create_session_raises_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OBSCURA_BIN", "/nonexistent/path/to/obscura")
    with pytest.raises(ValueError, match="Obscura binary not found"):
        ObscuraBrowserProvider().create_session("task-1")


def test_create_session_raises_when_cdp_never_ready(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_fake(tmp_path / "obscura", _FAKE_OBSCURA_NEVER_READY)
    monkeypatch.setenv("OBSCURA_BIN", path)
    monkeypatch.setenv("OBSCURA_STARTUP_TIMEOUT", "0.5")
    with pytest.raises(RuntimeError, match="did not become ready"):
        ObscuraBrowserProvider().create_session("task-2")


def test_stealth_flag_forwarded(
    fake_obscura: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_ARGV_FILE", str(argv_file))
    monkeypatch.setenv("OBSCURA_STEALTH", "true")
    provider = ObscuraBrowserProvider()
    session = provider.create_session("task-3")
    try:
        assert session["features"]["stealth"] is True
        assert "--stealth" in argv_file.read_text().split()
    finally:
        provider.close_session(session["bb_session_id"])


def test_port_override_is_honored(
    fake_obscura: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _free_port()
    monkeypatch.setenv("OBSCURA_PORT", str(port))
    provider = ObscuraBrowserProvider()
    session = provider.create_session("task-4")
    try:
        assert session["cdp_url"] == f"ws://127.0.0.1:{port}/devtools/browser"
    finally:
        provider.close_session(session["bb_session_id"])
