"""A real manual summary must not park other conversations on Desktop's shared socket.

Only the loopback model is a fake: the daemon, authority, SQLite, compression preparation,
receipt publication and WebSocket dispatch are real. The same-session control stays FIFO.
"""

import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

import pytest

from tests.gateway.fixtures.local_recovery_probe import (
    child_env,
    daemon,
    rpc,
    websocket,
)
from tests.waiting import async_wait_until


class ParkedSummaryModel(BaseHTTPRequestHandler):
    """Answer local chat requests, holding only the manual summary until released."""

    def log_message(self, format, *args):
        pass

    def _send(self, status, payload, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        server = self.server
        assert isinstance(server, ParkedSummaryServer)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(body)
        text = next(
            (
                m.get("content", "")
                for m in reversed(body.get("messages", []))
                if m["role"] == "user"
            ),
            "",
        )
        if body.get("model") == "summary":
            server.blocked.set()
            if not server.release.wait(90):
                server.expired.set()
                return self._send(504, b"{}", "application/json")
            reply = "SUMMARY_RETAINED_FACTS"
        else:
            reply = "ACK_" + str(text)[:40] + " historical detail" * 100
        message = {"role": "assistant", "content": reply}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        if body.get("stream"):
            choice["delta"] = choice.pop("message")
            payload = "data: " + json.dumps({"id": "local", "choices": [choice]})
            return self._send(
                200, (payload + "\n\ndata: [DONE]\n\n").encode(), "text/event-stream"
            )
        payload = {
            "id": "local",
            "model": body.get("model"),
            "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        self._send(200, json.dumps(payload).encode(), "application/json")


class ParkedSummaryServer(ThreadingHTTPServer):
    """Loopback model that records each request and parks the summary until released."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), ParkedSummaryModel)
        self.requests: list[dict] = []
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.expired = threading.Event()


@pytest.mark.platforms("linux")
def test_manual_compress_keeps_other_session_responsive_on_shared_socket(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ParkedSummaryServer()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{peer.server_port}/v1"
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "gateway": {"multiplex_profiles": False, "managed_workers": False},
                "model": {
                    "provider": "custom",
                    "default": "original",
                    "base_url": url,
                    "context_length": 1000000,
                },
                "auxiliary": {
                    "title_generation": {"enabled": False},
                    "compression": {
                        "provider": "custom",
                        "model": "summary",
                        "base_url": url,
                    },
                },
                # Warmup must not compress automatically: only the explicit mutation may reach summary.
                "compression": {
                    "protect_first_n": 1,
                    "protect_last_n": 2,
                    "min_tail_user_messages": 1,
                    "threshold_tokens": 900000,
                    "threshold": 0.9,
                    "in_place": False,
                },
                "platform_toolsets": {"cli": []},
            }
        )
    )
    (home / "models_dev_cache.json").write_text(
        json.dumps(
            {
                "custom": {
                    "id": "custom",
                    "models": {
                        name: {
                            "id": name,
                            "name": name,
                            "limit": {"context": 1000000, "output": 4000},
                        }
                        for name in ("original", "summary")
                    },
                }
            }
        )
    )
    # The child starts with cwd=root; no inherited Python path, credentials or real user home.
    env = child_env() | dict(
        HOME=str(user),
        USERPROFILE=str(user),
        HERMES_HOME=str(home),
        TMPDIR=str(tmp_path),
        OPENAI_API_KEY="loopback-only",
        OPENAI_BASE_URL=url,
        PYTHONUNBUFFERED="1",
    )
    metrics: dict[str, Any] = {}

    def rows(sql, args=()):
        with closing(
            sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)
        ) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, args)]

    def terminal(input_id):
        return rows(
            "SELECT admission_id,target_session_id,status,outcome FROM session_admissions "
            "WHERE request_id=? AND status='terminal'",
            (input_id,),
        )

    async def run(desc):
        async with websocket(home, desc) as ws:

            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert "result" in response, response
                return response["result"]

            sessions = {}
            for name in ("A", "B"):
                sessions[name] = (
                    await call(
                        "session.create",
                        source="cli",
                        request_id="create-" + name,
                        model="original",
                        provider="custom",
                        base_url=url,
                        toolsets=[],
                        cwd=str(home),
                    )
                )["session_id"]
            for i in range(6):
                input_id = f"history-{i}"
                await call(
                    "prompt.submit",
                    session_id=sessions["A"],
                    input_id=input_id,
                    text=f"HISTORY_{i} " + "retain this fact " * 40,
                )
                settled = await async_wait_until(
                    lambda: terminal(input_id), timeout=40, what=f"{input_id} terminal"
                )
                assert settled[0]["outcome"] == "completed", settled
            before = await call("session.resume", session_id=sessions["A"])
            assert len(before["messages"]) >= 12
            assert not peer.blocked.is_set(), (
                "warmup unexpectedly triggered automatic compression"
            )
            params = dict(
                session_id=sessions["A"],
                request_id="manual-compress-A",
                operation="compress",
                expected_revision=before["revision"],
                expected_generation=before["execution_generation"],
                payload={},
            )
            sent, replies, received, order = {}, {}, {}, []

            async def send(frame_id, method, **payload):
                sent[frame_id] = time.monotonic()
                await ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": frame_id,
                            "method": method,
                            "params": payload,
                        }
                    )
                )

            async def collect():
                # One receiver correlates all in-flight replies; rpc() would discard other IDs.
                async for raw in ws:
                    reply = json.loads(raw)
                    request_id = reply.get("id")
                    if request_id in sent:
                        received[request_id] = time.monotonic()
                        replies[request_id] = reply
                        order.append(request_id)

            reader = asyncio.create_task(collect())
            try:
                await send("compress-A", "session.mutate", **params)
                await async_wait_until(
                    peer.blocked.is_set, timeout=30, what="summary HTTP request parked"
                )
                metrics["summary_started_seconds"] = (
                    time.monotonic() - sent["compress-A"]
                )
                assert "compress-A" not in replies, replies
                # A is queued FIRST: B replies prove dispatch advanced past the held A control.
                await send("info-A", "session.info", session_id=sessions["A"])
                await send(
                    "submit-B",
                    "prompt.submit",
                    session_id=sessions["B"],
                    input_id="B-while-parked",
                    text="B_WHILE_A_COMPRESSES",
                )
                await send("info-B", "session.info", session_id=sessions["B"])
                try:
                    await async_wait_until(
                        lambda: {"submit-B", "info-B"} <= replies.keys(),
                        timeout=10,
                        what="both B replies while A summary is parked",
                    )
                finally:
                    metrics.update(
                        parked_observation_seconds=time.monotonic() - sent["submit-B"],
                        b_reply_seconds={
                            key: received[key] - sent[key] if key in received else None
                            for key in ("submit-B", "info-B")
                        },
                        replies_before_release=list(order),
                        summary_parked=peer.blocked.is_set()
                        and not peer.release.is_set()
                        and not peer.expired.is_set(),
                    )
                assert metrics["summary_parked"], metrics
                assert not reader.done(), (
                    "WebSocket reader ended before summary release"
                )
                assert "compress-A" not in replies and "info-A" not in replies, replies
                for key in ("submit-B", "info-B"):
                    assert "result" in replies[key], replies[key]
                assert replies["info-B"]["result"]["model"] == "original"
                admitted = replies["submit-B"]["result"]
                assert admitted["admission_id"]

                released = time.monotonic()
                peer.release.set()
                await async_wait_until(
                    lambda: {"compress-A", "info-A"} <= replies.keys(),
                    timeout=40,
                    what="compress receipt then same-session info",
                )
                for key in ("compress-A", "info-A"):
                    assert "result" in replies[key], replies[key]
                assert order.index("compress-A") < order.index("info-A"), order
                result = replies["compress-A"]["result"]
                # Canonical commit receipts carry advanced fences and a published target, not
                # a "committed" status string (preview/noop have status but no such receipt).
                assert result["revision"] == before["revision"] + 1, result
                assert (
                    result["execution_generation"] == before["execution_generation"] + 1
                ), result
                assert result["previous_target_session_id"] == sessions["A"], result
                assert result["target_session_id"] != sessions["A"], result
                assert replies["info-A"]["result"]["model"] == "original"
                settled = await async_wait_until(
                    lambda: terminal("B-while-parked"),
                    timeout=40,
                    what="B turn completes normally",
                )
                assert settled == [
                    {
                        "admission_id": admitted["admission_id"],
                        "target_session_id": sessions["B"],
                        "status": "terminal",
                        "outcome": "completed",
                    }
                ], settled
                # Read actual SQLite publication rather than accepting only a plausible RPC reply.
                assert rows(
                    "SELECT end_reason FROM sessions WHERE id=?", (sessions["A"],)
                ) == [{"end_reason": "compression"}]
                assert rows(
                    "SELECT id FROM sessions WHERE parent_session_id=?",
                    (sessions["A"],),
                ) == [{"id": result["target_session_id"]}]
                history = rows(
                    "SELECT content FROM messages WHERE session_id=? AND active=1",
                    (result["target_session_id"],),
                )
                assert "SUMMARY_RETAINED_FACTS" in json.dumps(history), history
                b_history = rows(
                    "SELECT content FROM messages WHERE session_id=? AND active=1",
                    (sessions["B"],),
                )
                assert "ACK_B_WHILE_A_COMPRESSES" in json.dumps(b_history), b_history
                assert sum(r.get("model") == "summary" for r in peer.requests) == 1
                metrics.update(
                    compress_after_release_seconds=received["compress-A"] - released,
                    a_info_after_release_seconds=received["info-A"] - released,
                    final_reply_order=order,
                    compress_receipt=result,
                    b_outcome=settled[0]["outcome"],
                )
            finally:
                peer.release.set()
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)

    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(run(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
        print("JOURNEY_METRICS " + json.dumps(metrics, sort_keys=True))
