"""Tests for the google_meet node primitive.

Covers protocol helpers, the file-backed registry, the server's
token-and-dispatch machinery, a mocked client, and the CLI plumbing.
We never open a real socket — websockets.serve / websockets.sync.client
are fully mocked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


# ---------------------------------------------------------------------------
# protocol.py
# ---------------------------------------------------------------------------

def test_protocol_encode_decode_roundtrip():
    from plugins.google_meet.node import protocol

    msg = protocol.make_request("ping", "tok", {"x": 1}, req_id="abc")
    raw = protocol.encode(msg)
    out = protocol.decode(raw)
    assert out == msg
    assert out["type"] == "ping"
    assert out["id"] == "abc"
    assert out["token"] == "tok"
    assert out["payload"] == {"x": 1}


def test_protocol_make_request_autogenerates_id():
    from plugins.google_meet.node import protocol

    a = protocol.make_request("ping", "tok", {})
    b = protocol.make_request("ping", "tok", {})
    assert a["id"] != b["id"]
    assert len(a["id"]) >= 16  # uuid4 hex


def test_protocol_make_request_rejects_bad_input():
    from plugins.google_meet.node import protocol

    with pytest.raises(ValueError):
        protocol.make_request("", "tok", {})
    with pytest.raises(ValueError):
        protocol.make_request("unknown_type", "tok", {})
    with pytest.raises(ValueError):
        protocol.make_request("ping", "tok", "not a dict")  # type: ignore[arg-type]


def test_protocol_decode_raises_on_malformed():
    from plugins.google_meet.node import protocol

    with pytest.raises(ValueError):
        protocol.decode("not json at all")
    with pytest.raises(ValueError):
        protocol.decode("[]")  # list, not object
    with pytest.raises(ValueError):
        protocol.decode(json.dumps({"id": "x"}))  # missing type
    with pytest.raises(ValueError):
        protocol.decode(json.dumps({"type": "ping"}))  # missing id


def test_protocol_validate_request_happy_path():
    from plugins.google_meet.node import protocol

    msg = protocol.make_request("status", "secret", {})
    ok, reason = protocol.validate_request(msg, "secret")
    assert ok is True
    assert reason == ""


def test_protocol_validate_request_rejects_bad_token():
    from plugins.google_meet.node import protocol

    msg = protocol.make_request("status", "wrong", {})
    ok, reason = protocol.validate_request(msg, "right")
    assert ok is False
    assert "token" in reason.lower()


def test_protocol_validate_request_rejects_unknown_type():
    from plugins.google_meet.node import protocol

    raw = {"type": "nope", "id": "1", "token": "t", "payload": {}}
    ok, reason = protocol.validate_request(raw, "t")
    assert ok is False
    assert "unknown" in reason.lower()


def test_protocol_validate_request_rejects_missing_id():
    from plugins.google_meet.node import protocol

    raw = {"type": "ping", "token": "t", "payload": {}}
    ok, reason = protocol.validate_request(raw, "t")
    assert ok is False
    assert "id" in reason.lower()


def test_protocol_validate_request_rejects_non_dict_payload():
    from plugins.google_meet.node import protocol

    raw = {"type": "ping", "id": "1", "token": "t", "payload": "oops"}
    ok, reason = protocol.validate_request(raw, "t")
    assert ok is False


def test_protocol_error_envelope_shape():
    from plugins.google_meet.node import protocol

    err = protocol.make_error("abc", "nope")
    assert err == {"type": "error", "id": "abc", "error": "nope"}


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------

def test_registry_add_get_roundtrip_persists(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    p = tmp_path / "nodes.json"
    r = NodeRegistry(path=p)
    r.add("mac", "ws://mac.local:18789", "deadbeef")

    # Second instance sees it.
    r2 = NodeRegistry(path=p)
    entry = r2.get("mac")
    assert entry is not None
    assert entry["name"] == "mac"
    assert entry["url"] == "ws://mac.local:18789"
    assert entry["token"] == "deadbeef"
    assert "added_at" in entry


def test_registry_get_returns_none_when_missing(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    assert r.get("ghost") is None


def test_registry_remove(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    r.add("a", "ws://a", "t")
    assert r.remove("a") is True
    assert r.get("a") is None
    assert r.remove("a") is False  # idempotent


def test_registry_list_all_sorted(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    r.add("zeta", "ws://z", "t1")
    r.add("alpha", "ws://a", "t2")
    names = [n["name"] for n in r.list_all()]
    assert names == ["alpha", "zeta"]


def test_registry_resolve_auto_picks_single(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    r.add("mac", "ws://mac", "t")
    picked = r.resolve(None)
    assert picked is not None
    assert picked["name"] == "mac"


def test_registry_resolve_ambiguous_returns_none(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    r.add("a", "ws://a", "t")
    r.add("b", "ws://b", "t")
    assert r.resolve(None) is None


def test_registry_resolve_empty_returns_none(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    assert r.resolve(None) is None


def test_registry_resolve_by_name(tmp_path):
    from plugins.google_meet.node.registry import NodeRegistry

    r = NodeRegistry(path=tmp_path / "n.json")
    r.add("a", "ws://a", "t")
    r.add("b", "ws://b", "t")
    picked = r.resolve("b")
    assert picked is not None
    assert picked["name"] == "b"
    assert r.resolve("ghost") is None


def test_registry_defaults_to_hermes_home(tmp_path, monkeypatch):
    from plugins.google_meet.node.registry import NodeRegistry

    # _isolate_home already set HERMES_HOME to tmp_path/.hermes; the
    # registry default path must live inside that tree.
    r = NodeRegistry()
    r.add("x", "ws://x", "t")
    expected = Path(tmp_path) / ".hermes" / "workspace" / "meetings" / "nodes.json"
    assert expected.is_file()


# ---------------------------------------------------------------------------
# server.py — token + dispatch
# ---------------------------------------------------------------------------

def test_server_ensure_token_generates_and_persists(tmp_path):
    from plugins.google_meet.node.server import NodeServer

    p = tmp_path / "tok.json"
    s1 = NodeServer(token_path=p)
    t1 = s1.ensure_token()
    assert isinstance(t1, str) and len(t1) == 32

    # Reuse on a fresh instance.
    s2 = NodeServer(token_path=p)
    t2 = s2.ensure_token()
    assert t1 == t2

    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["token"] == t1
    assert "generated_at" in data


def test_server_get_token_is_idempotent(tmp_path):
    from plugins.google_meet.node.server import NodeServer

    s = NodeServer(token_path=tmp_path / "t.json")
    assert s.get_token() == s.get_token()


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_server_handle_request_rejects_bad_token(tmp_path):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol

    s = NodeServer(token_path=tmp_path / "t.json")
    s.ensure_token()
    bad = protocol.make_request("ping", "not-the-token", {})
    resp = asyncio.run(s._handle_request(bad))
    assert resp["type"] == "error"
    assert "token" in resp["error"].lower()


def test_server_handle_request_ping(tmp_path):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol

    s = NodeServer(token_path=tmp_path / "t.json", display_name="node-x")
    tok = s.ensure_token()
    req = protocol.make_request("ping", tok, {})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "pong"
    assert resp["id"] == req["id"]
    assert resp["payload"]["display_name"] == "node-x"


def test_server_handle_request_status_dispatches_to_pm(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    monkeypatch.setattr(pm, "status",
                        lambda: {"ok": True, "alive": True, "meetingId": "abc"})

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("status", tok, {})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert resp["id"] == req["id"]
    assert resp["payload"] == {"ok": True, "alive": True, "meetingId": "abc"}


def test_server_handle_request_start_bot_dispatches(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "pid": 42, "meeting_id": "abc-defg-hij"}

    monkeypatch.setattr(pm, "start", fake_start)

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("start_bot", tok, {
        "url": "https://meet.google.com/abc-defg-hij",
        "guest_name": "Bot",
        "duration": "30m",
    })
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert resp["payload"]["ok"] is True
    assert captured["url"] == "https://meet.google.com/abc-defg-hij"
    assert captured["guest_name"] == "Bot"
    assert captured["duration"] == "30m"


def test_server_handle_request_start_bot_missing_url(tmp_path):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("start_bot", tok, {"guest_name": "x"})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "error"
    assert "url" in resp["error"]


def test_server_handle_request_stop_dispatches(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    got = {}

    def fake_stop(*, reason="requested"):
        got["reason"] = reason
        return {"ok": True, "reason": reason}

    monkeypatch.setattr(pm, "stop", fake_stop)

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("stop", tok, {"reason": "user-cancel"})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert got["reason"] == "user-cancel"


def test_server_handle_request_transcript(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    got = {}

    def fake_transcript(last=None):
        got["last"] = last
        return {"ok": True, "lines": ["a", "b"], "total": 2}

    monkeypatch.setattr(pm, "transcript", fake_transcript)

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("transcript", tok, {"last": 5})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert resp["payload"]["lines"] == ["a", "b"]
    assert got["last"] == 5


def test_server_handle_request_say_enqueues_when_active(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    out = tmp_path / "meet-out"
    out.mkdir()
    monkeypatch.setattr(pm, "_read_active",
                        lambda: {"pid": 1, "meeting_id": "m", "out_dir": str(out)})

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("say", tok, {"text": "hello"})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert resp["payload"]["ok"] is True
    assert resp["payload"]["enqueued"] is True
    q = (out / "say_queue.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(q) == 1
    assert json.loads(q[0])["text"] == "hello"


def test_server_handle_request_say_without_active_still_ok(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    monkeypatch.setattr(pm, "_read_active", lambda: None)

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("say", tok, {"text": "hi"})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "response"
    assert resp["payload"]["ok"] is True
    assert resp["payload"]["enqueued"] is False


def test_server_handle_request_wraps_pm_exceptions(tmp_path, monkeypatch):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol
    from plugins.google_meet import process_manager as pm

    def boom():
        raise ValueError("kaboom")

    monkeypatch.setattr(pm, "status", boom)

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("status", tok, {})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "error"
    assert "kaboom" in resp["error"]


# ---------------------------------------------------------------------------
# client.py
# ---------------------------------------------------------------------------

class _FakeWS:
    """Minimal context-manager stand-in for websockets.sync.client.connect."""

    def __init__(self, reply_builder):
        self._reply_builder = reply_builder
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def send(self, raw):
        self.sent.append(raw)

    def recv(self, timeout=None):
        return self._reply_builder(self.sent[-1])


def _install_fake_ws(monkeypatch, reply_builder):
    fake_ws_holder = {}

    def _connect(url, **kwargs):
        ws = _FakeWS(reply_builder)
        fake_ws_holder["ws"] = ws
        fake_ws_holder["url"] = url
        fake_ws_holder["kwargs"] = kwargs
        return ws

    # Patch the concrete import site inside client._rpc
    import websockets.sync.client as wsc  # type: ignore
    monkeypatch.setattr(wsc, "connect", _connect)
    return fake_ws_holder


def test_client_rpc_sends_correct_envelope_and_parses_response(monkeypatch):
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node import protocol

    def reply(raw_out):
        req = protocol.decode(raw_out)
        return protocol.encode(protocol.make_response(req["id"], {"ok": True, "echo": req["type"]}))

    holder = _install_fake_ws(monkeypatch, reply)

    c = NodeClient("ws://remote:1", "tok123")
    out = c._rpc("ping", {"hello": 1})
    assert out == {"ok": True, "echo": "ping"}

    sent = json.loads(holder["ws"].sent[0])
    assert sent["type"] == "ping"
    assert sent["token"] == "tok123"
    assert sent["payload"] == {"hello": 1}
    assert sent["id"]  # non-empty
    assert holder["url"] == "ws://remote:1"


def test_client_rpc_raises_on_error_envelope(monkeypatch):
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node import protocol

    def reply(raw_out):
        req = protocol.decode(raw_out)
        return protocol.encode(protocol.make_error(req["id"], "nope"))

    _install_fake_ws(monkeypatch, reply)

    c = NodeClient("ws://x", "t")
    with pytest.raises(RuntimeError, match="nope"):
        c._rpc("ping", {})


def test_client_rpc_raises_on_id_mismatch(monkeypatch):
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node import protocol

    def reply(raw_out):
        return protocol.encode(protocol.make_response("different-id", {"ok": True}))

    _install_fake_ws(monkeypatch, reply)

    c = NodeClient("ws://x", "t")
    with pytest.raises(RuntimeError, match="mismatch"):
        c._rpc("ping", {})


def test_client_convenience_methods_hit_correct_types(monkeypatch):
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node import protocol

    seen = []

    def reply(raw_out):
        req = protocol.decode(raw_out)
        seen.append((req["type"], req["payload"]))
        return protocol.encode(protocol.make_response(req["id"], {"ok": True}))

    _install_fake_ws(monkeypatch, reply)

    c = NodeClient("ws://x", "t")
    c.start_bot("https://meet.google.com/a-b-c", guest_name="G", duration="10m")
    c.stop()
    c.status()
    c.transcript(last=3)
    c.say("hi")
    c.ping()

    types = [t for t, _ in seen]
    assert types == ["start_bot", "stop", "status", "transcript", "say", "ping"]
    # Check specific payload routing
    assert seen[0][1]["url"] == "https://meet.google.com/a-b-c"
    assert seen[0][1]["guest_name"] == "G"
    assert seen[0][1]["duration"] == "10m"
    assert seen[3][1]["last"] == 3
    assert seen[4][1]["text"] == "hi"


def test_client_init_rejects_bad_args():
    from plugins.google_meet.node.client import NodeClient

    with pytest.raises(ValueError):
        NodeClient("", "t")
    with pytest.raises(ValueError):
        NodeClient("ws://x", "")


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------

def _build_parser():
    from plugins.google_meet.node.cli import register_cli

    parser = argparse.ArgumentParser(prog="meet-node-test")
    register_cli(parser)
    return parser


def test_cli_approve_list_remove(capsys):
    from plugins.google_meet.node.registry import NodeRegistry

    p = _build_parser()

    args = p.parse_args(["approve", "mac", "ws://mac:1", "tok"])
    rc = args.func(args)
    assert rc == 0
    assert NodeRegistry().get("mac") is not None

    args = p.parse_args(["list"])
    rc = args.func(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "mac" in out
    assert "ws://mac:1" in out

    args = p.parse_args(["remove", "mac"])
    rc = args.func(args)
    assert rc == 0
    assert NodeRegistry().get("mac") is None


def test_cli_list_empty(capsys):
    p = _build_parser()
    args = p.parse_args(["list"])
    rc = args.func(args)
    assert rc == 0
    assert "no nodes" in capsys.readouterr().out


def test_cli_remove_missing_returns_nonzero():
    p = _build_parser()
    args = p.parse_args(["remove", "ghost"])
    rc = args.func(args)
    assert rc == 1


def test_cli_status_pings_via_node_client(capsys, monkeypatch):
    from plugins.google_meet.node.registry import NodeRegistry
    from plugins.google_meet.node import cli as node_cli

    NodeRegistry().add("mac", "ws://mac:1", "tok")

    class _FakeClient:
        def __init__(self, url, token):
            assert url == "ws://mac:1"
            assert token == "tok"

        def ping(self):
            return {"type": "pong", "display_name": "hermes-meet-node"}

    monkeypatch.setattr(node_cli, "NodeClient", _FakeClient)

    p = _build_parser()
    args = p.parse_args(["status", "mac"])
    rc = args.func(args)
    assert rc == 0
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["ok"] is True
    assert data["node"] == "mac"


def test_cli_status_unknown_node_fails(capsys):
    p = _build_parser()
    args = p.parse_args(["status", "ghost"])
    rc = args.func(args)
    assert rc == 1


def test_cli_status_reports_client_error(capsys, monkeypatch):
    from plugins.google_meet.node.registry import NodeRegistry
    from plugins.google_meet.node import cli as node_cli

    NodeRegistry().add("mac", "ws://mac:1", "tok")

    class _FakeClient:
        def __init__(self, url, token):
            pass

        def ping(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(node_cli, "NodeClient", _FakeClient)

    p = _build_parser()
    args = p.parse_args(["status", "mac"])
    rc = args.func(args)
    assert rc == 1
    data = json.loads(capsys.readouterr().out.strip())
    assert data["ok"] is False
    assert "connection refused" in data["error"]
