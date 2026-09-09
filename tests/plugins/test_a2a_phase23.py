"""
Streaming / push / anti-loop / task-store tests for the A2A plugin (v1.0).

Tests cover:
- v1.0 SSE StreamResponse format (member-name discrimination, no kind/final)
- message/stream and tasks/subscribe end-to-end against a live server
- Push notification HMAC signing
- Anti-loop ping-pong protection (TurnTracker + live rejection)
- Rate limiting (per-identity sliding window)
- Metrics collection (real latency)
- Task store (idempotent completion, watchers, orphan handling)
- Dynamic Agent Cards from the live tool registry
- Capability-based routing with fan-out (a2a_orchestrate)
- SSRF protection for push callback URLs
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, security, tools


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_live_adapter(monkeypatch, reply_fn=None):
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True))

    async def fake_handle_message(event):
        reply = "ECHO: " + event.text if reply_fn is None else reply_fn(event)
        if reply is not None:
            await adapter.send(event.source.chat_id, reply, metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore
    adapter._message_handler = object()
    return adapter, f"http://127.0.0.1:{port}"


def _post_sse(url, body):
    """POST a JSON-RPC request and return the parsed SSE stream as
    (data_payloads, event_names).  Unwraps the JSON-RPC envelope from
    each data frame so callers see bare StreamResponse objects."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode("utf-8")
    payloads, events = [], []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("event: "):
                events.append(line[len("event:"):].strip())
            elif line.startswith("data: "):
                data = line[len("data: "):].strip()
                if data:
                    obj = json.loads(data)
                    # Unwrap JSON-RPC envelope: {"jsonrpc":"2.0","id":...,"result":{...}}
                    if isinstance(obj, dict) and "jsonrpc" in obj and "result" in obj:
                        payloads.append(obj["result"])
                    else:
                        payloads.append(obj)
            # SSE comment lines (": done") are ignored — not data frames.
    return payloads, events


def _post_json(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _send_body(text, ctx="", method="message/send"):
    return {
        "jsonrpc": "2.0", "id": "1", "method": method,
        "params": {"message": protocol.text_message(protocol.ROLE_USER, text, context_id=ctx)},
    }


# ═════════════════════════════════════════════════════════════════════════════
# v1.0 SSE StreamResponse format
# ═════════════════════════════════════════════════════════════════════════════


class TestStreamResponseFormat:
    def test_status_update_shape(self):
        ev = protocol.status_update("task-1", "ctx-1", protocol.STATE_WORKING)
        assert set(ev.keys()) == {"statusUpdate"}
        su = ev["statusUpdate"]
        assert su["taskId"] == "task-1"
        assert su["contextId"] == "ctx-1"
        assert su["status"]["state"] == "TASK_STATE_WORKING"
        assert "kind" not in su and "final" not in su

    def test_status_update_with_message(self):
        ev = protocol.status_update("t", "c", protocol.STATE_INPUT_REQUIRED, "which one?")
        msg = ev["statusUpdate"]["status"]["message"]
        assert msg["role"] == "ROLE_AGENT"
        assert protocol.extract_text(msg) == "which one?"

    def test_artifact_update_shape(self):
        ev = protocol.artifact_update("task-1", "ctx-1", "the result")
        assert set(ev.keys()) == {"artifactUpdate"}
        au = ev["artifactUpdate"]
        assert au["taskId"] == "task-1"
        part = au["artifact"]["parts"][0]
        assert part == {"text": "the result", "mediaType": "text/plain"}
        assert "kind" not in au and "final" not in au

    def test_sse_data_framing(self):
        chunk = protocol.sse_data({"statusUpdate": {"taskId": "t"}})
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        # No event-name line: v1.0 discriminates by member presence.
        assert "event:" not in chunk

    def test_sse_data_jsonrpc_envelope(self):
        """A2A v1.0 §9.4: SSE frames must be JSON-RPC-wrapped when req_id is
        provided.  Bare StreamResponse (REST binding) breaks a2a-sdk clients."""
        chunk = protocol.sse_data({"statusUpdate": {"taskId": "t"}}, req_id="42")
        assert chunk.startswith("data: ")
        obj = json.loads(chunk[len("data: "):].strip())
        assert obj["jsonrpc"] == "2.0"
        assert obj["id"] == "42"
        assert "result" in obj
        assert obj["result"]["statusUpdate"]["taskId"] == "t"

    def test_sse_data_no_envelope_without_req_id(self):
        """Without req_id, sse_data falls back to bare payload for legacy callers."""
        chunk = protocol.sse_data({"statusUpdate": {"taskId": "t"}})
        obj = json.loads(chunk[len("data: "):].strip())
        assert "jsonrpc" not in obj
        assert obj["statusUpdate"]["taskId"] == "t"

    def test_sse_done_marker(self):
        """v1.0 signals stream completion by closing the stream.  The done
        marker is an SSE comment (``: done``), not a parseable data frame —
        emitting ``data: {}`` breaks JSON-RPC clients that try to parse it."""
        done = protocol.sse_done()
        assert ": done" in done
        assert "data:" not in done  # no data frame for SDK to parse
        assert done.endswith("\n\n")


@pytest.mark.integration
class TestStreamingEndToEnd:
    def test_message_stream_v1_events(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            payloads, events = await asyncio.to_thread(
                _post_sse, base + "/", _send_body("stream me", method="message/stream"))

            # Discrimination is by member name; every payload is a StreamResponse.
            # v1.0 streaming begins with the current Task (or a direct Message),
            # followed by status/artifact updates until terminal closure.
            for p in payloads:
                assert set(p.keys()) <= {"task", "message", "statusUpdate", "artifactUpdate"}
                assert "kind" not in json.dumps(p)
            assert "task" in payloads[0]
            assert payloads[0]["task"]["status"]["state"] == "TASK_STATE_SUBMITTED"

            states = [p["statusUpdate"]["status"]["state"]
                      for p in payloads if "statusUpdate" in p]
            assert states[0] == "TASK_STATE_WORKING"
            assert "TASK_STATE_WORKING" in states
            assert states[-1] == "TASK_STATE_COMPLETED"
            # No v0.3 'final' flag anywhere; closure is the terminal signal.
            assert all("final" not in p.get("statusUpdate", {}) for p in payloads)

            artifacts = [p["artifactUpdate"] for p in payloads if "artifactUpdate" in p]
            assert len(artifacts) == 1
            assert "ECHO:" in protocol.extract_text(artifacts[0]["artifact"])

            assert events == []  # v1.0: stream closure is the terminal signal, no event frame
            await adapter.disconnect()

        asyncio.run(run())

    def test_tasks_subscribe_replays_terminal_state(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_post_json, base + "/", _send_body("hello"))
            task = resp["result"]

            payloads, events = await asyncio.to_thread(_post_sse, base + "/", {
                "jsonrpc": "2.0", "id": "2", "method": "tasks/subscribe",
                "params": {"taskId": task["id"]},
            })
            states = [p["statusUpdate"]["status"]["state"]
                      for p in payloads if "statusUpdate" in p]
            assert "TASK_STATE_COMPLETED" in states
            artifacts = [p for p in payloads if "artifactUpdate" in p]
            assert artifacts and "ECHO:" in protocol.extract_text(
                artifacts[0]["artifactUpdate"]["artifact"])
            assert events == []  # v1.0: stream closure is the terminal signal, no event frame
            await adapter.disconnect()

        asyncio.run(run())

    def test_tasks_subscribe_unknown_task(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "2", "method": "tasks/subscribe",
                "params": {"taskId": "ghost"},
            })
            assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
            await adapter.disconnect()

        asyncio.run(run())

    def test_agent_card_advertises_streaming(self):
        card = protocol.build_agent_card(
            name="test", url="http://localhost:9900/",
            description="test", streaming=True, push_notifications=True,
        )
        assert card["capabilities"]["streaming"] is True
        assert card["capabilities"]["pushNotifications"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Push notification signing
# ═════════════════════════════════════════════════════════════════════════════


class TestPushSigning:
    def test_sign_push_payload_deterministic(self, monkeypatch):
        monkeypatch.setenv("A2A_PUSH_SECRET", "test-secret-123")
        payload = {"statusUpdate": {"taskId": "task-1"}}
        sig = security.A2ASecurityContext.capture().sign_push_payload(payload)
        assert sig
        import hashlib
        import hmac as hmac_mod
        expected = hmac_mod.new(
            b"test-secret-123",
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(),
            hashlib.sha256,
        ).hexdigest()
        assert sig == expected

    def test_no_secret_means_unsigned(self, monkeypatch):
        monkeypatch.delenv("A2A_PUSH_SECRET", raising=False)
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        assert security.A2ASecurityContext.capture().sign_push_payload({"x": 1}) == ""

    def test_falls_back_to_bearer_token(self, monkeypatch):
        monkeypatch.delenv("A2A_PUSH_SECRET", raising=False)
        monkeypatch.setenv("A2A_BEARER_TOKEN", "bearer-as-push-secret")
        assert security.A2ASecurityContext.capture().sign_push_payload({"x": 1})


# ═════════════════════════════════════════════════════════════════════════════
# Anti-loop ping-pong protection
# ═════════════════════════════════════════════════════════════════════════════


class TestAntiLoopProtection:
    def test_track_turn_increments(self):
        turns = protocol.TurnTracker()
        assert turns.track("c1") == 1
        assert turns.track("c1") == 2
        assert turns.track("c1") == 3
        assert turns.track("c2") == 1  # separate context

    def test_reset_turns_clears(self):
        turns = protocol.TurnTracker()
        for _ in range(5):
            turns.track("c1")
        turns.reset("c1")
        assert turns.track("c1") == 1

    def test_max_pingpong_turns_default(self, monkeypatch):
        monkeypatch.delenv("A2A_MAX_PINGPONG_TURNS", raising=False)
        assert protocol.max_pingpong_turns() == 5

    def test_max_pingpong_turns_env_override(self, monkeypatch):
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "10")
        assert protocol.max_pingpong_turns() == 10
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "50")
        assert protocol.max_pingpong_turns() == 20  # hard cap
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "0")
        assert protocol.max_pingpong_turns() == 1  # min 1

    @pytest.mark.integration
    def test_loop_rejected_live(self, monkeypatch):
        """The turn past the limit is REJECTED (v1.0 state), not failed."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_MAX_PINGPONG_TURNS", "2")
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            states = []
            for _ in range(3):
                resp = await asyncio.to_thread(
                    _post_json, base + "/", _send_body("ping", ctx="ctx-pingpong"))
                states.append(resp["result"]["status"]["state"])
            assert states[0] == "TASK_STATE_COMPLETED"
            assert states[1] == "TASK_STATE_COMPLETED"
            assert states[2] == "TASK_STATE_REJECTED"
            await adapter.disconnect()

        asyncio.run(run())


# ═════════════════════════════════════════════════════════════════════════════
# Rate limiting
# ═════════════════════════════════════════════════════════════════════════════


class TestRateLimiting:
    def test_allows_under_limit(self, monkeypatch):
        monkeypatch.setenv("A2A_RATE_LIMIT", "10")
        rl = protocol.RateLimiter()
        for _ in range(10):
            assert rl.allow("peer-1") is True

    def test_blocks_over_limit(self, monkeypatch):
        monkeypatch.setenv("A2A_RATE_LIMIT", "3")
        rl = protocol.RateLimiter()
        assert rl.allow("peer-2") is True
        assert rl.allow("peer-2") is True
        assert rl.allow("peer-2") is True
        assert rl.allow("peer-2") is False  # 4th blocked

    def test_separate_per_identity(self, monkeypatch):
        monkeypatch.setenv("A2A_RATE_LIMIT", "2")
        rl = protocol.RateLimiter()
        assert rl.allow("peer-a") is True
        assert rl.allow("peer-a") is True
        assert rl.allow("peer-a") is False
        assert rl.allow("peer-b") is True  # different bucket
        assert rl.allow("peer-b") is True

    @pytest.mark.integration
    def test_rate_limit_live_returns_429(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_RATE_LIMIT", "2")
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True

            def _burst():
                codes = []
                for _ in range(3):
                    try:
                        _post_json(base + "/", _send_body("hi"))
                        codes.append(200)
                    except urllib.error.HTTPError as e:
                        codes.append(e.code)
                        err = json.loads(e.read().decode())
                        assert err["error"]["code"] == protocol.ERR_RATE_LIMITED
                return codes

            codes = await asyncio.to_thread(_burst)
            assert codes[:2] == [200, 200]
            assert codes[2] == 429
            await adapter.disconnect()

        asyncio.run(run())


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_metrics_snapshot_has_fields(self):
        m = protocol.metrics.snapshot()
        for field in ("uptime_seconds", "inbound_total", "outbound_total",
                      "streams_started", "push_sent", "push_failed",
                      "tasks_completed", "tasks_failed", "anti_loop_triggers",
                      "rate_limit_triggers", "avg_latency_ms"):
            assert field in m

    def test_record_latency_updates_average(self):
        m = protocol.Metrics()
        m.record_latency(0.1)
        m.record_latency(0.3)
        assert 0.19 <= m.avg_latency() <= 0.21

    @pytest.mark.integration
    def test_latency_is_actually_recorded_live(self, monkeypatch):
        """The avg latency metric must be fed by real elapsed time, not a
        hardcoded 0."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

        def slow_reply(event):
            time.sleep(0.05)
            return "done"

        adapter, base = _make_live_adapter(monkeypatch, reply_fn=slow_reply)

        async def run():
            assert await adapter.connect() is True
            before = len(protocol.metrics._latencies)
            await asyncio.to_thread(_post_json, base + "/", _send_body("time me"))
            new = list(protocol.metrics._latencies)[before:]
            assert new and new[-1] >= 0.05
            await adapter.disconnect()

        asyncio.run(run())


# ═════════════════════════════════════════════════════════════════════════════
# Task store
# ═════════════════════════════════════════════════════════════════════════════


class TestTaskStore:
    def test_create_and_get(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "peer-1")
        rec = store.get("t1")
        assert rec["state"] == protocol.STATE_SUBMITTED
        assert rec["context_id"] == "c1"
        assert rec["peer"] == "peer-1"

    def test_complete_keeps_task_queryable(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        store.complete("t1", protocol.STATE_COMPLETED, "the reply")
        rec = store.get("t1")
        assert rec is not None
        assert rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == "the reply"

    def test_complete_is_idempotent(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        assert store.complete("t1", protocol.STATE_COMPLETED, "first") is not None
        # Second terminal transition is refused (prevents double-counting).
        assert store.complete("t1", protocol.STATE_FAILED, "second") is None
        assert store.get("t1")["state"] == protocol.STATE_COMPLETED
        assert store.complete("ghost", protocol.STATE_FAILED) is None

    def test_watch_resolves_on_complete(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        fut = store.watch("t1")
        assert not fut.done()
        store.complete("t1", protocol.STATE_COMPLETED, "answer")
        assert fut.result(timeout=0) == (protocol.STATE_COMPLETED, "answer")

    def test_watch_terminal_resolves_immediately(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        store.complete("t1", protocol.STATE_FAILED, "err")
        fut = store.watch("t1")
        assert fut.result(timeout=0) == (protocol.STATE_FAILED, "err")
        assert store.watch("ghost") is None

    def test_fail_orphans(self):
        store = protocol.TaskStore()
        store.create("t-old", "c1", "p")
        store.create("t-new", "c1", "p")
        store._tasks["t-old"]["created_at"] = time.time() - 600
        failed = store.fail_orphans(timeout_seconds=300)
        assert failed == ["t-old"]
        assert store.get("t-old")["state"] == protocol.STATE_FAILED
        assert store.get("t-new")["state"] == protocol.STATE_SUBMITTED
        # Second sweep does nothing (already terminal).
        assert store.fail_orphans(timeout_seconds=300) == []

    def test_list_newest_first_with_filters(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        store.create("t2", "c2", "p")
        store.create("t3", "c1", "p")
        store.complete("t1", protocol.STATE_COMPLETED)
        recs, _ = store.list(context_id="c1")
        assert [r["task_id"] for r in recs] == ["t3", "t1"]
        recs, _ = store.list(state=protocol.STATE_SUBMITTED)
        assert {r["task_id"] for r in recs} == {"t2", "t3"}

    def test_push_config_lifecycle(self):
        store = protocol.TaskStore()
        store.create("t1", "c1", "p")
        cfg = store.set_push_config("t1", "https://example.com/hook")
        assert cfg["configId"].startswith("cfg-")
        assert cfg["createdAt"]
        assert store.pop_push_url("t1") == "https://example.com/hook"
        assert store.pop_push_url("t1") == ""  # consumed
        assert store.set_push_config("ghost", "https://x/") is None


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic Agent Cards
# ═════════════════════════════════════════════════════════════════════════════


class TestDynamicAgentCards:
    def test_skills_reflect_live_tool_registry(self, monkeypatch):
        """The Agent Card is built from the real tool registry at serve time."""
        from tools.registry import registry
        from gateway.config import PlatformConfig
        from plugins.platforms.a2a.adapter import A2AAdapter

        monkeypatch.setattr(registry, "get_registered_toolset_names",
                            lambda: ["webz", "termz"])
        monkeypatch.setattr(registry, "get_tool_names_for_toolset",
                            lambda ts: {"webz": ["web_search"], "termz": ["terminal"]}[ts])

        adapter = A2AAdapter(PlatformConfig(enabled=True))
        card = adapter._build_card()
        by_name = {s["name"]: s for s in card["skills"]}
        assert set(by_name) == {"webz", "termz"}
        assert "web_search" in by_name["webz"]["tags"]

    def test_advertised_toolsets_restrict_card(self, monkeypatch):
        from tools.registry import registry
        from gateway.config import PlatformConfig
        from plugins.platforms.a2a.adapter import A2AAdapter

        monkeypatch.setattr(registry, "get_registered_toolset_names",
                            lambda: ["webz", "termz", "secretz"])
        monkeypatch.setattr(registry, "get_tool_names_for_toolset", lambda ts: [])
        monkeypatch.setenv("A2A_ADVERTISED_TOOLSETS", "webz")

        adapter = A2AAdapter(PlatformConfig(enabled=True))
        card = adapter._build_card()
        assert [s["name"] for s in card["skills"]] == ["webz"]


# ═════════════════════════════════════════════════════════════════════════════
# Capability-based routing (a2a_orchestrate)
# ═════════════════════════════════════════════════════════════════════════════


_TWO_PEERS = {
    "a2a_agents": {
        "researcher": {"url": "http://localhost:9991", "capabilities": ["research"]},
        "coder": {"url": "http://localhost:9992", "capabilities": ["code"]},
        "generalist": {"url": "http://localhost:9993", "capabilities": ["research", "code"]},
    }
}


class TestA2AOrchestrate:
    def test_requires_capability_and_message(self):
        assert "capability" in tools.a2a_orchestrate({"message": "do something"})
        assert "message" in tools.a2a_orchestrate({"capability": "research"})

    def test_no_matching_peers(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: {})
        result = tools.a2a_orchestrate({"capability": "research", "message": "search X"})
        assert "no configured peers" in result

    def test_match_peers_by_capability(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        matches = tools._match_peers_by_capability("research")
        assert {m[0] for m in matches} == {"researcher", "generalist"}
        assert len(tools._match_peers_by_capability("*")) == 3

    def test_all_mode_returns_every_reply(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, f"reply from {name}"))
        out = tools.a2a_orchestrate({"capability": "research", "message": "go"})
        assert "reply from researcher" in out
        assert "reply from generalist" in out

    def test_best_mode_picks_longest_success(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        replies = {
            "researcher": "short",
            "generalist": "a much longer and more detailed reply",
        }
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, replies[name]))
        out = tools.a2a_orchestrate({"capability": "research", "message": "go", "mode": "best"})
        assert out.startswith("[best: generalist]")

    def test_best_mode_ignores_error_replies(self, monkeypatch):
        """A long error must not beat a short success (old max() heuristic bug)."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        replies = {
            "researcher": "ok",
            "generalist": "Error: " + "x" * 500,
        }
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, replies[name]))
        out = tools.a2a_orchestrate({"capability": "research", "message": "go", "mode": "best"})
        assert out.startswith("[best: researcher]")
        assert "ok" in out

    def test_best_mode_all_errors_reports_failure(self, monkeypatch):
        """All-error edge: report the failures instead of returning one error
        with a misleading [best: ...] header."""
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, "Error: connection refused"))
        out = tools.a2a_orchestrate({"capability": "research", "message": "go", "mode": "best"})
        assert out.startswith("All peers failed:")
        assert "[best:" not in out

    def test_first_mode_all_errors_reports_failure(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, "Error: nope"))
        out = tools.a2a_orchestrate({"capability": "code", "message": "go", "mode": "first"})
        assert out.startswith("All peers failed:")

    def test_first_mode_returns_a_success(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: _TWO_PEERS)
        monkeypatch.setattr(tools, "_call_peer_sync",
                            lambda name, entry, msg, ctx="": (name, f"win {name}"))
        out = tools.a2a_orchestrate({"capability": "code", "message": "go", "mode": "first"})
        assert out.startswith("[first: ")
        assert "win" in out


# ═════════════════════════════════════════════════════════════════════════════
# SSRF protection for push callbacks
# ═════════════════════════════════════════════════════════════════════════════


class TestSSRFProtection:
    def test_safe_public_urls_allowed(self):
        assert security.is_safe_callback_url("https://example.com/webhook") is True
        assert security.is_safe_callback_url("http://example.com/webhook") is True

    def test_localhost_blocked_in_remote_mode(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")  # remote mode
        assert security.is_safe_callback_url("http://127.0.0.1:8080/hook") is False
        assert security.is_safe_callback_url("http://localhost:8080/hook") is False

    def test_localhost_allowed_in_local_mode(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        assert security.is_safe_callback_url("http://127.0.0.1:8080/hook") is True
        assert security.is_safe_callback_url("http://localhost:8080/hook") is True

    def test_aws_metadata_blocked(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")
        assert security.is_safe_callback_url("http://169.254.169.254/latest/meta-data/") is False

    def test_private_ranges_blocked(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "tok")
        assert security.is_safe_callback_url("http://10.0.0.1/hook") is False
        assert security.is_safe_callback_url("http://192.168.1.1/hook") is False
        assert security.is_safe_callback_url("http://172.16.0.1/hook") is False

    def test_non_http_schemes_blocked(self):
        assert security.is_safe_callback_url("file:///etc/passwd") is False
        assert security.is_safe_callback_url("ftp://example.com/file") is False

    def test_empty_url_blocked(self):
        assert security.is_safe_callback_url("") is False
        assert security.is_safe_callback_url(None) is False
