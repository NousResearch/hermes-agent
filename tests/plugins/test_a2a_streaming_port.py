"""Tests for the streaming client surgical port onto fork main.

The streaming send path (SSE + card gate + zero-frame fallback) is ported
verbatim from PR #86369 (feat/a2a-streaming-client). One semantic composition
is deliberate and diverges from the PR branch: fork main owns per-peer
idempotency (deterministic task_id + replay-safe fallback). Composed contract:

  - Zero-frame stream failure  -> message/send fallback (clean first dispatch)
  - Frames-received stream death -> INDETERMINATE (no fallback) by default;
    falls back ONLY when the peer config asserts ``idempotency: true``
    (the peer dedupes on resend).

The PR branch treated the opt-in as inert after frames; here it gates the
replay. Tests below cover both branches.
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
# Truncated stream must fail loud
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

    def test_artifact_only_death_is_frames_received(self):
        """Artifact-only stream (no task/status/message frame) still counts
        as frames-received: the peer was demonstrably executing (it emitted
        artifact parts), so a fallback would replay the task."""
        with pytest.raises(tools._A2aTransportError) as exc_info:
            run_stream([artifact("partial", ctx="ctx-real")])
        assert exc_info.value.frames_received is True
        assert exc_info.value.frame_count == 1

    def test_eof_without_terminal_preserves_frames_and_ctx(self):
        """Run the real _send_task_stream to EOF-without-terminal: the error
        must carry the true frame_count and the peer-established contextId."""
        with pytest.raises(tools._A2aTransportError) as exc_info:
            run_stream([working(ctx="ctx-real"), working(ctx="ctx-real")])
        assert exc_info.value.frames_received is True
        assert exc_info.value.frame_count == 2
        assert exc_info.value.seen_ctx == "ctx-real"

    def test_terminal_frame_succeeds(self):
        """Sanity: full happy path still returns normally."""
        reply, ctx, state = run_stream(
            [submitted(), working(), artifact("DONE"),
             terminal(protocol.STATE_COMPLETED)])
        assert reply == "DONE"
        assert state == protocol.STATE_COMPLETED


# ---------------------------------------------------------------------------
# Wall-clock deadline
# ---------------------------------------------------------------------------

class TestDeadline:
    def test_deadline_constant_present(self):
        assert hasattr(tools, "_STREAM_READ_TIMEOUT_S")
        assert tools._STREAM_READ_TIMEOUT_S > 0

    def test_stream_exceeding_wall_clock_deadline_raises_transport_error(self, monkeypatch):
        """A peer that only sends keepalives never trips the per-read
        timeout; the wall-clock deadline must still fire and raise a
        transport error (-> message/send fallback), not hang forever."""
        clock = {"t": 0.0}

        def fake_monotonic():
            clock["t"] += (tools._STREAM_READ_TIMEOUT_S * 2)
            return clock["t"]

        class _Resp:
            headers = {"Content-Type": "text/event-stream"}

            def __iter__(self):
                while True:
                    yield b": keepalive\n"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            return _Resp()

        monkeypatch.setattr(tools.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(tools, "_open_url_no_redirect_leak",
                            lambda req, timeout, allowed_origins=(): fake_urlopen(req, timeout))
        with pytest.raises(tools._A2aTransportError, match="deadline"):
            list(tools._http_post_sse("https://x", {"m": 1}, {}, timeout=30))


# ---------------------------------------------------------------------------
# Transport exceptions fall back
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
# RPC error semantics inside _send_task_stream
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
# HTTP status-based fallback
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


# ---------------------------------------------------------------------------
# Non-SSE 200 with a non-JSON body -> transport error, fallback runs
# ---------------------------------------------------------------------------

class TestNonSseBodyGuard:
    def _resp(self, body: bytes, ctype: str):
        class _Resp:
            headers = {"Content-Type": ctype}

            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return _Resp()

    def test_html_body_raises_transport_error(self, monkeypatch):
        """200 + text/html (proxy error page) -> _A2aTransportError."""
        monkeypatch.setattr(
            tools, "_open_url_no_redirect_leak",
            lambda req, timeout, allowed_origins=(): self._resp(
                b"<html><body>Bad Gateway</body></html>", "text/html"))
        with pytest.raises(tools._A2aTransportError, match="not valid JSON-RPC"):
            list(tools._http_post_sse("https://x", {"m": 1}, {}, timeout=30))

    def test_empty_body_raises_transport_error(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_open_url_no_redirect_leak",
            lambda req, timeout, allowed_origins=(): self._resp(b"", "application/json"))
        with pytest.raises(tools._A2aTransportError, match="not valid JSON-RPC"):
            list(tools._http_post_sse("https://x", {"m": 1}, {}, timeout=30))

    def test_valid_jsonrpc_body_still_yields(self, monkeypatch):
        """Plain JSON-RPC response on a non-SSE content type still works."""
        monkeypatch.setattr(
            tools, "_open_url_no_redirect_leak",
            lambda req, timeout, allowed_origins=(): self._resp(
                b'{"jsonrpc": "2.0", "id": 1, "result": {}}', "application/json"))
        frames = list(tools._http_post_sse("https://x", {"m": 1}, {}, timeout=30))
        assert frames == [{"jsonrpc": "2.0", "id": 1, "result": {}}]

    def test_transport_error_triggers_fallback_end_to_end(self, monkeypatch):
        """The guard's (zero-frame) error is caught by _send_task -> fallback."""
        fallback_called = []

        def fake_stream(*a, **k):
            raise tools._A2aTransportError("non-SSE response was not valid JSON-RPC")

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                               "contextId": "ctx-fb",
                               "artifacts": [{"parts": [{"text": "RECOVERED"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")
        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        reply, ctx, state = tools._send_task("peer-x", peer, "hi", "")
        assert fallback_called
        assert "RECOVERED" in reply


# ---------------------------------------------------------------------------
# ContextId from ANY frame, not just the terminal one
# ---------------------------------------------------------------------------

class TestStreamContextTracking:
    def test_early_context_survives_absent_terminal_context(self):
        """Context established in an early frame, omitted later -> kept."""
        reply, ctx, state = run_stream([
            working(ctx="ctx-real"),
            terminal("TASK_STATE_COMPLETED", text="DONE", ctx=""),
        ])
        assert ctx == "ctx-real"

    def test_caller_context_used_when_no_frame_carries_one(self):
        reply, ctx, state = run_stream(
            [terminal("TASK_STATE_COMPLETED", text="DONE", ctx="")],
            ctx="ctx-caller")
        assert ctx == "ctx-caller"

    def test_latest_frame_context_wins(self):
        reply, ctx, state = run_stream([
            working(ctx="ctx-a"),
            terminal("TASK_STATE_COMPLETED", text="DONE", ctx="ctx-b"),
        ])
        assert ctx == "ctx-b"


# ---------------------------------------------------------------------------
# Fallback visibility: frames-received death must NOT resubmit by default
# ---------------------------------------------------------------------------

class TestFallbackVisibility:
    def test_zero_frame_fallback_stays_debug(self, caplog, monkeypatch):
        import logging

        def fake_stream(*a, **k):
            raise tools._A2aTransportError(
                "stream closed without a result", frames_received=False)

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", lambda *a, **k: {
            "result": {"status": {"state": "TASK_STATE_COMPLETED"},
                       "contextId": "ctx-fb",
                       "artifacts": [{"parts": [{"text": "FALLBACK"}]}]}})
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")
        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.a2a.tools"):
            reply, _, _ = tools._send_task("peer-x", peer, "hi", "")
        assert "FALLBACK" in reply
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "falling back" in r.message]
        assert not warnings, "zero-frame fallback should stay at DEBUG"

    def test_frames_received_death_with_optin_falls_back(self, caplog, monkeypatch):
        """Port composition: frames-received death + ``idempotency: true``
        -> deliberate message/send fallback (dedup on resend assertion as
        retry), logged at WARNING."""
        import logging

        def fake_stream(*a, **k):
            raise tools._A2aTransportError(
                "stream closed without a terminal state", frames_received=True)

        fallback_called = {"n": 0}

        def fake_post(*a, **k):
            fallback_called["n"] += 1
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                               "contextId": "ctx-fb",
                               "artifacts": [{"parts": [{"text": "FALLBACK"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")
        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30,
                "idempotency": True}
        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.a2a.tools"):
            reply, _, _ = tools._send_task("peer-x", peer, "hi", "")
        assert fallback_called["n"] == 1, "idempotent peer must get the fallback replay"
        assert "FALLBACK" in reply
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "asserted idempotency" in r.message]
        assert warnings, "opt-in replay after frames must log at WARNING"


# ---------------------------------------------------------------------------
# Indeterminate-outcome contract (no opt-in): frames-received death
# ---------------------------------------------------------------------------

class TestIndeterminateOutcome:
    def test_frames_received_death_without_optin_no_fallback(self, caplog, monkeypatch):
        """Frames received, no terminal, no opt-in -> indeterminate result,
        the message/send fallback POST must never fire, and the outcome is
        logged at WARNING."""
        import logging

        def fake_stream(*a, **k):
            raise tools._A2aTransportError(
                "stream closed without a terminal state",
                frames_received=True, seen_ctx="ctx-real", frame_count=3)

        fallback_called = []

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.a2a.tools"):
            reply, reply_ctx, state = tools._send_task("peer-x", peer, "hi", "ctx-caller")

        assert not fallback_called, "message/send fallback must NOT run"
        assert "peer-x" in reply
        assert "3 frames" in reply
        assert "may have already run" in reply
        assert reply_ctx == "ctx-real", "stream's contextId must be preserved"
        assert state == "", "outcome is non-terminal"
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "NOT falling back" in r.message]
        assert warnings, "expected a WARNING-level indeterminate record"

    @pytest.mark.parametrize("exc", [
        tools._A2aTransportError("stream exceeded total deadline of 30s"),
        TimeoutError("read timed out"),
        urllib.error.URLError("connection reset by peer"),
    ], ids=["wall-clock-deadline", "read-timeout", "urlerror-reset"])
    def test_midstream_transport_death_no_fallback(self, monkeypatch, exc):
        """WORKING frame then transport death -> indeterminate, NO fallback.

        Covers the remaining frames-received death modes: the wall-clock
        deadline _A2aTransportError (raised with no frames_received flag) and
        raw mid-stream URLError/TimeoutError. Both must be re-qualified as
        frames-received by _send_task_stream so _send_task returns an
        indeterminate outcome instead of resubmitting via message/send.
        """
        fallback_called = []

        def fake_sse(*a, **k):
            yield working(ctx="ctx-real")
            raise exc

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {}}

        monkeypatch.setattr(tools, "_http_post_sse", fake_sse)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        reply, reply_ctx, state = tools._send_task("peer-x", peer, "hi", "ctx-caller")

        assert not fallback_called, "message/send fallback must NOT run"
        assert "peer-x" in reply
        assert "1 frame" in reply
        assert "may have already run" in reply
        assert reply_ctx == "ctx-real", "stream's contextId must be preserved"
        assert state == "", "outcome is non-terminal"

    def test_zero_frame_failure_with_optin_still_falls_back(self, monkeypatch):
        """idempotency opt-in does not change zero-frame semantics: still falls back."""
        fallback_called = []

        def fake_stream(*a, **k):
            raise tools._A2aTransportError(
                "stream closed without a result", frames_received=False)

        def fake_post_json(*a, **k):
            fallback_called.append(True)
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                               "contextId": "ctx-fb",
                               "artifacts": [{"parts": [{"text": "FALLBACK OK"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {"streaming": True}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")

        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30,
                "idempotency": True}
        reply, reply_ctx, state = tools._send_task("peer-x", peer, "hi", "")
        assert fallback_called, "zero-frame failure must still fall back"
        assert "FALLBACK OK" in reply

    def test_resolve_peer_passes_idempotency_through(self, monkeypatch):
        """Fork main's resolver forwards the operator's idempotency assertion
        (it gates the frames-received stream fallback)."""
        monkeypatch.setattr(tools, "_load_config", lambda: {
            "a2a_agents": {
                "researcher": {"url": "http://p/", "idempotency": True},
                "plain": {"url": "http://q/"},
            },
        })
        researcher = tools._resolve_peer("researcher")
        plain = tools._resolve_peer("plain")
        assert researcher is not None and researcher.get("idempotency") is True
        assert plain is not None and not plain.get("idempotency")


# ---------------------------------------------------------------------------
# Non-streaming peer: card without streaming -> straight message/send
# ---------------------------------------------------------------------------

class TestNoStreamingCard:
    def test_card_without_streaming_skips_stream_path(self, monkeypatch):
        """Peer whose card does not advertise streaming goes straight to
        message/send; the streaming code never runs."""
        stream_called = []

        def fake_stream(*a, **k):
            stream_called.append(True)
            raise AssertionError("stream must not be attempted")

        def fake_post_json(*a, **k):
            return {"result": {"status": {"state": "TASK_STATE_COMPLETED"},
                               "contextId": "ctx-fb",
                               "artifacts": [{"parts": [{"text": "PLAIN"}]}]}}

        monkeypatch.setattr(tools, "_send_task_stream", fake_stream)
        monkeypatch.setattr(tools, "_http_post_json", fake_post_json)
        monkeypatch.setattr(tools, "_fetch_card",
                            lambda *a, **k: {"capabilities": {}})
        monkeypatch.setattr(tools, "_rpc_url", lambda *a, **k: "https://x")
        peer = {"url": "https://x", "auth": {}, "headers": {}, "timeout": 30}
        reply, ctx, state = tools._send_task("peer-x", peer, "hi", "")
        assert not stream_called
        assert "PLAIN" in reply
