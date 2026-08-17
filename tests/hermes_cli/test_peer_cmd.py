"""Tests for ``hermes peer`` — cross-machine bot-to-bot DMs."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli.subcommands import peer as peer_cmd


# ── target parsing ───────────────────────────────────────────────────────────


def test_parse_target_bare_peer():
    assert peer_cmd._parse_target("spark") == ("spark", None)


def test_parse_target_peer_and_profile():
    assert peer_cmd._parse_target("spark/researcher") == ("spark", "researcher")


def test_parse_target_rejects_empty():
    with pytest.raises(ValueError):
        peer_cmd._parse_target("")


def test_parse_target_rejects_bad_profile():
    with pytest.raises(ValueError):
        peer_cmd._parse_target("spark/../etc")


# ── url scoping ──────────────────────────────────────────────────────────────


def test_base_url_bare_and_profile():
    peer = {"url": "http://spark.lan:8377/"}
    assert peer_cmd._base_url(peer, None) == "http://spark.lan:8377"
    assert peer_cmd._base_url(peer, "researcher") == "http://spark.lan:8377/p/researcher"


# ── registry round-trip (isolated config) ────────────────────────────────────


def test_add_list_remove_roundtrip(monkeypatch, capsys):
    store = {}

    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: dict(store))

    def fake_save(peers):
        store.clear()
        store.update(peers)

    monkeypatch.setattr(peer_cmd, "_save_peers", fake_save)
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "k" * 20)

    rc = peer_cmd.cmd_peer(
        SimpleNamespace(peer_action="add", name="spark", url="http://spark.lan:8377", key="", note="")
    )
    assert rc == 0
    assert store["spark"]["url"] == "http://spark.lan:8377"

    rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="list"))
    assert rc == 0
    assert "spark" in capsys.readouterr().out

    rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="remove", name="spark"))
    assert rc == 0
    assert "spark" not in store


def test_add_rejects_bad_name_and_url(monkeypatch):
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {})
    monkeypatch.setattr(peer_cmd, "_save_peers", lambda peers: None)

    assert peer_cmd.cmd_peer(SimpleNamespace(peer_action="add", name="Bad Name!", url="http://x", key="", note="")) == 2
    assert peer_cmd.cmd_peer(SimpleNamespace(peer_action="add", name="ok", url="ftp://x", key="", note="")) == 2


def test_dm_unknown_peer_and_missing_key(monkeypatch):
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": "http://x"}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "")

    assert peer_cmd.cmd_peer(SimpleNamespace(peer_action="dm", target="nope", message="hi", json=False)) == 1
    assert peer_cmd.cmd_peer(SimpleNamespace(peer_action="dm", target="spark", message="hi", json=False)) == 1


# ── live HTTP dm flow (real loopback server, fake peer gateway) ──────────────


class _FakePeer(BaseHTTPRequestHandler):
    sessions: list = []
    chats: list = []
    auth_seen: list = []

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).auth_seen.append(self.headers.get("Authorization", ""))
        if self.path.startswith("/api/sessions"):
            data = [{"id": s, "title": "Bot Chat"} for s in type(self).sessions]
            return self._json({"object": "list", "data": data})
        return self._json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        type(self).auth_seen.append(self.headers.get("Authorization", ""))
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path == "/api/sessions":
            type(self).sessions.append("bc_1")
            return self._json({"id": "bc_1", "title": body.get("title")}, 201)

        if self.path.startswith("/api/sessions/") and self.path.endswith("/chat"):
            type(self).chats.append(body.get("message"))
            return self._json(
                {
                    "object": "hermes.session.chat.completion",
                    "session_id": "bc_1",
                    "message": {"role": "assistant", "content": "reply from the other machine"},
                }
            )

        return self._json({"error": {"message": "not found"}}, 404)

    def log_message(self, *args):  # noqa: D102 — silence test server logging
        pass


@pytest.fixture()
def fake_peer_server():
    _FakePeer.sessions = []
    _FakePeer.chats = []
    _FakePeer.auth_seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakePeer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_dm_creates_bot_chat_then_chats(monkeypatch, capsys, fake_peer_server):
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": fake_peer_server}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "secret-key-123456")

    rc = peer_cmd.cmd_peer(
        SimpleNamespace(
            peer_action="dm",
            target="spark",
            message="Message from 🤖 dixie (@dixie): disk status?",
            json=False,
        )
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "reply from the other machine" in out
    # One Bot Chat was created (none existed), then the chat turn ran on it.
    assert _FakePeer.sessions == ["bc_1"]
    assert _FakePeer.chats == ["Message from 🤖 dixie (@dixie): disk status?"]
    # Every request carried the peer key.
    assert all(a == "Bearer secret-key-123456" for a in _FakePeer.auth_seen)


def test_dm_reuses_existing_bot_chat(monkeypatch, capsys, fake_peer_server):
    _FakePeer.sessions = ["bc_existing"]
    monkeypatch.setattr(peer_cmd, "_load_peers", lambda: {"spark": {"url": fake_peer_server}})
    monkeypatch.setattr(peer_cmd, "_peer_secret", lambda name: "secret-key-123456")

    rc = peer_cmd.cmd_peer(SimpleNamespace(peer_action="dm", target="spark", message="ping", json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reply"] == "reply from the other machine"
    # No new session was created — the existing canonical chat was reused.
    assert _FakePeer.sessions == ["bc_existing"]
