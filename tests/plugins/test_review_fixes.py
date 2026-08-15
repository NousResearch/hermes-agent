"""Tests for the PR #86369 review-fix pass (fixes 1-5).

Semantic changes under test:
  1. Truncated stream (EOF after WORKING, no terminal frame) -> _A2aTransportError
  2. Wall-clock deadline -> _A2aTransportError after timeout + read-timeout grace
  3. URLError / TimeoutError / http.client exceptions -> fallback to message/send
  4. JSON-RPC error frame on healthy stream -> ValueError, NO fallback (no resubmit)
  5. HTTP 404/405/501 on streaming endpoint -> fallback; other HTTP errors propagate

Plus: multi-artifact accumulation, peer taskId keying.
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, tools


# ---------------------------------------------------------------------------
# Frame builders (ground-truth shapes from a live Hermes peer)
# ---------------------------------------------------------------------------

def rpc(result, msg_id=1):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}

def err(code, message, msg_id=1):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

def submitted(task_id="task-1", ctx="ctx-1"):
    return rpc({"task": {"id": task_id, "contextId": ctx,
                          "status": {"state": protocol.STATE_SUBMITTED}}})

def working(task_id="task-1", ctx="ctx-1"):
    return rpc({"statusUpdate": {"taskId": task_id, "contextId": ctx,
                                  "status": {"state": protocol.STATE_WORKING}}})

def artifact(text, task_id="task-1", ctx="ctx-1"):
    return rpc({"artifactUpdate": {"taskId": task_id, "contextId": ctx,
                                    "artifact": {"artifactId": "a1",
                                                  "parts": [{"text": text}]}}})

def terminal(state, text=None, task_id="task-1", ctx="ctx-1"):
    status = {"state": state}
    if text:
        status["message"] = {"parts": [{"text": text}]}
    return rpc({"statusUpdate": {"taskId": task_id, "contextId": ctx, "status": status}})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_io(monkeypatch):
    """Block real network; neutralize persistence/metrics side effects."""
    from plugins.platforms.a2a import security
    def boom(*a, **k):
        raise AssertionError("network access in unit test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(protocol, "persist_message", lambda *a, **k: None)
    monkeypatch.setattr(security, "audit", lambda *a, **k: None)
    monkeypatch.setattr(security, "redact_outbound", lambda m: m)


@pytest.fixture
def stream(monkeypatch):
    """Return a function feed(frames) that patches _http_post_sse to yield frames."""
    def feed(frames, exc=None):
        def fake(*a, **k):
            for fr in frames:
                yield fr
            if exc is not None:
                raise exc
        monkeypatch.setattr(tools, "_http_post_sse", fake)
    return feed


def run_stream(frames, exc=None, **kw):
    """Drive _send_task_stream with a canned frame sequence."""
    orig = tools._http_post_sse

    def fake(*a, **k):
        for fr in frames:
            yield fr
        if exc is not None:
            raise exc

    tools._http_post_sse = fake
    try:
        return tools._send_task_stream(
            kw.get("label", "peer-x"), kw.get("url", "https://x"),
            kw.get("body", {"jsonrpc": "2.0", "id": "req-1", "params": {}}),
            kw.get("headers", {}), kw.get("timeout", 30),
            kw.get("ctx", "ctx-1"), kw.get("task_id", "req-1"))
    finally:
        tools._http_post_sse = orig


# ---------------------------------------------------------------------------
# Fix 1: truncated stream must fail loud
# ---------------------------------------------------------------------------

class TestTruncatedStream:
    def test_eof_after_working_raises(self):
        """WORKING snapshot then EOF (no terminal) -> _A2aTransportError."""
        with pytest.raises(tools._A2aTransportError, match="terminal state"):
            run_stream([submitted(), working()])

    def test_eof_after_zero_frames_raises(self):
        with pytest.raises(tools._A2aTransportError, match="without a result"):
            run_stream([])

    def test_eof_after_task_snapshot_no_terminal_raises(self):
        """Task snapshot carries SUBMITTED state; EOF without terminal -> raise."""
        with pytest.raises(tools._A2aTransportError, match="terminal state"):
            run_stream([submitted()])

    def test_terminal_frame_succeeds(self):
        """Sanity: full happy path still returns normally."""
        reply, ctx, state = run_stream(
            [submitted(), working(), artifact("DONE"),
             terminal(protocol.STATE_COMPLETED)])
        assert reply == "DONE"
        assert state == protocol.STATE_COMPLETED


# ---------------------------------------------------------------------------
# Fix 2: wall-clock deadline
# ---------------------------------------------------------------------------

class TestDeadline:
    def test_deadline_constant_present(self):
        assert hasattr(tools, "_STREAM_READ_TIMEOUT_S")
        assert tools._STREAM_READ_TIMEOUT_S > 0

    def test_sse_reader_has_deadline_logic(self):
        """Verify the deadline check exists in _http_post_sse source."""
        import inspect
        src = inspect.getsource(tools._http_post_sse)
        assert "deadline" in src
        assert "time.monotonic" in src


# ---------------------------------------------------------------------------
# Fix 3: transport exceptions fall back
# ---------------------------------------------------------------------------

class TestTransportFallback:
    """_send_task catches URLError/TimeoutError/HTTPException -> fallback."""

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError("connection refused"),
        TimeoutError("read timed out"),
        http.client.IncompleteRead(b""),
        http.client.RemoteDisconnected(),
        http.client.BadStatusLine("garbage"),
    ], ids=["URLError", "TimeoutError", "IncompleteRead",
            "RemoteDisconnected", "BadStatusLine"])
    def test_transport_exception_triggers_fallback(self, monkeypatch, exc):
        """Streaming path raises transport exc -> _send_task falls back."""
        fallback_called = []

        def fake_stream(*a, **k):
            raise exc

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                                "contextId": "ctx-fb",
                                "artifacts": [{"parts": [{"text": "FALLBACK OK"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        reply, ctx, state = tools._send_task("peer-x", peer, "hi", "")
        assert fallback_called
        assert "FALLBACK OK" in reply

    def test_rpc_error_does_not_fallback(self, monkeypatch):
        """JSON-RPC error frame -> ValueError raised, NO fallback resubmit."""
        fallback_called = []

        def fake_stream(*a, **k):
            raise ValueError("Peer 'x' returned an error: rate limited")

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        with pytest.raises(ValueError, match="rate limited"):
            tools._send_task("peer-x", peer, "hi", "")
        assert not fallback_called, "RPC error must NOT trigger fallback"


# ---------------------------------------------------------------------------
# Fix 4: RPC error semantics inside _send_task_stream
# ---------------------------------------------------------------------------

class TestRpcErrorInStream:
    def test_error_frame_raises_valueerror_not_transport(self):
        """Application-level error -> ValueError (not _A2aTransportError)."""
        with pytest.raises(ValueError) as exc_info:
            run_stream([submitted(), working(), err(-32000, "rate limited")])
        assert not isinstance(exc_info.value, tools._A2aTransportError)
        assert "rate limited" in str(exc_info.value)
        assert "peer-x" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Fix 5: HTTP status-based fallback
# ---------------------------------------------------------------------------

class TestHttpStatusFallback:
    @pytest.mark.parametrize("code", [404, 405, 501])
    def test_streaming_endpoint_missing_triggers_fallback(self, monkeypatch, code):
        fallback_called = []
        http_exc = urllib.error.HTTPError(
            "https://x", code, "Not Found", {}, None)

        def fake_stream(*a, **k):
            raise http_exc

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                                "contextId": "ctx-fb",
                                "artifacts": [{"parts": [{"text": "OK"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        reply, ctx, state = tools._send_task("peer-x", peer, "hi", "")
        assert fallback_called

    @pytest.mark.parametrize("code", [403, 500, 502])
    def test_other_http_errors_propagate(self, monkeypatch, code):
        http_exc = urllib.error.HTTPError(
            "https://x", code, "Error", {}, None)

        def fake_stream(*a, **k):
            raise http_exc

        def fake_post_json(*a, **k):
            raise AssertionError("should not fall back")

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        with pytest.raises(urllib.error.HTTPError):
            tools._send_task("peer-x", peer, "hi", "")


# ---------------------------------------------------------------------------
# Multi-artifact accumulation + peer taskId keying
# ---------------------------------------------------------------------------

class TestArtifactAndKeying:
    def test_multiple_artifacts_accumulated(self):
        """All artifact parts survive, not just the last one."""
        reply, ctx, state = run_stream([
            submitted(), working(),
            artifact("part one"), artifact("part two"),
            terminal(protocol.STATE_COMPLETED)])
        assert "part one" in reply
        assert "part two" in reply

    def test_peer_task_id_used_for_persistence(self, monkeypatch):
        """Peer-assigned taskId keys the history, not the request id."""
        persisted_ids = []
        monkeypatch.setattr(protocol, "persist_message",
                            lambda ctx, role, text, tid: persisted_ids.append(tid))
        # Patch at module level since run_stream patches _http_post_sse
        import types
        def fake(*a, **k):
            yield submitted(task_id="peer-task-99", ctx="ctx-1")
            yield working(task_id="peer-task-99", ctx="ctx-1")
            yield artifact("X", task_id="peer-task-99", ctx="ctx-1")
            yield terminal(protocol.STATE_COMPLETED, task_id="peer-task-99", ctx="ctx-1")
        orig = tools._http_post_sse
        tools._http_post_sse = fake
        try:
            tools._send_task_stream("peer-x", "https://x",
                                    {"jsonrpc": "2.0", "id": "req-1", "params": {}},
                                    {}, 30, "ctx-1", "req-1")
        finally:
            tools._http_post_sse = orig
        assert "peer-task-99" in persisted_ids
        assert "req-1" not in persisted_ids
