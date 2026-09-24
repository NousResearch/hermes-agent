"""Tests for the google_meet node primitive.

Covers protocol helpers, the file-backed registry, the server's
token-and-dispatch machinery, a mocked client, a real localhost WebSocket
round trip, and the CLI plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

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


def test_server_handle_request_start_bot_missing_url(tmp_path):
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node import protocol

    s = NodeServer(token_path=tmp_path / "t.json")
    tok = s.ensure_token()
    req = protocol.make_request("start_bot", tok, {"guest_name": "x"})
    resp = asyncio.run(s._handle_request(req))
    assert resp["type"] == "error"
    assert "url" in resp["error"]


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


def test_client_server_transcript_privacy_and_speech_readiness(tmp_path):
    from plugins.google_meet import process_manager as pm
    from plugins.google_meet.node import protocol
    from plugins.google_meet.node.client import NodeClient
    from plugins.google_meet.node.server import NodeServer
    from plugins.google_meet.node.registry import NodeRegistry
    from plugins.google_meet.tools import handle_meet_say, handle_meet_transcript
    from websockets.sync.server import serve

    node = NodeServer(token_path=tmp_path / "token.json", display_name="local-node")
    token = node.ensure_token()
    out_dir = (
        Path(os.environ["HERMES_HOME"]) / "workspace" / "meetings" / "abc-defg-hij"
    )
    out_dir.mkdir(parents=True)
    lines = ["[10:00:00] Alice: one", "[10:00:01] Bob: two", "[10:00:02] Alice: three"]
    (out_dir / "transcript.txt").write_text("\n".join(lines))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pm._write_active(
        {
            "pid": process.pid,
            "meeting_id": "abc-defg-hij",
            "out_dir": str(out_dir),
            "session_id": "owner",
            "mode": "realtime",
        }
    )

    def handler(connection):
        request = protocol.decode(connection.recv())
        response = asyncio.run(node._handle_request(request))
        connection.send(protocol.encode(response))

    pump = None
    try:
        pump = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with serve(handler, "127.0.0.1", 0) as server:
            port = server.socket.getsockname()[1]
            NodeRegistry().add("local-node", f"ws://127.0.0.1:{port}", token)
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.start()
            try:
                client = NodeClient(f"ws://127.0.0.1:{port}", token, timeout=5.0)
                assert client.ping()["display_name"] == "local-node"
                assert client.status()["alive"] is True
                assert (
                    json.loads(
                        handle_meet_transcript(
                            {"node": "local-node", "last": 2}, session_id="owner"
                        )
                    )["lines"]
                    == lines[-2:]
                )
                assert (
                    json.loads(
                        handle_meet_say({"text": "not ready", "node": "local-node"})
                    )["success"]
                    is False
                )
                queue = out_dir / "say_queue.jsonl"
                assert not queue.exists()
                status = {
                    "inCall": True,
                    "realtime": True,
                    "realtimeReady": True,
                    "realtimeAudioPumpStatus": "ready",
                    "realtimeAudioPumpPid": pump.pid,
                    "localMicrophoneOn": True,
                }
                (out_dir / "status.json").write_text(json.dumps(status))
                queued = json.loads(
                    handle_meet_say({"text": "ready to speak", "node": "local-node"})
                )
                assert queued["success"] is True
                expected = [{"id": queued["enqueued_id"], "text": "ready to speak"}]
                assert [
                    json.loads(line) for line in queue.read_text().splitlines()
                ] == expected
                status["localMicrophoneOn"] = False
                (out_dir / "status.json").write_text(json.dumps(status))
                assert (
                    json.loads(
                        handle_meet_say({"text": "muted", "node": "local-node"})
                    )["success"]
                    is False
                )
                assert [
                    json.loads(line) for line in queue.read_text().splitlines()
                ] == expected
                pump.terminate()
                pump.wait(timeout=5)
                status["localMicrophoneOn"] = True
                (out_dir / "status.json").write_text(json.dumps(status))
                rejected = json.loads(
                    handle_meet_say({"text": "dead route", "node": "local-node"})
                )
                assert rejected["success"] is False
                assert "audio pump" in rejected["reason"]
                assert [
                    json.loads(line) for line in queue.read_text().splitlines()
                ] == expected
                process.terminate()
                process.wait(timeout=5)
                assert client.stop(reason="requested")["ok"] is True
                assert client.transcript()["ok"] is False
                assert (
                    client.transcript(include_finished=True, session_id="intruder")[
                        "ok"
                    ]
                    is False
                )
                finished = client.transcript(
                    last=1, include_finished=True, session_id="owner"
                )
                assert finished["lines"] == lines[-1:]
                assert finished["fromLast"] is True
                assert finished["active"] is False
                assert finished["leaveReason"] == "requested"
            finally:
                server.shutdown()
                server_thread.join(timeout=5)
    finally:
        if pump is not None and pump.poll() is None:
            pump.terminate()
            pump.wait(timeout=5)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


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
