"""Unit tests for the A2A SSE stream frame parser.

Covers ``_send_task_stream()`` and ``_http_post_sse()`` in
``plugins/platforms/a2a/tools.py`` (client-side SendStreamingMessage support).

Frame shapes are member-discriminated A2A v1.0 StreamResponse objects captured
LIVE from a real Hermes peer:

  task snapshot    {"result": {"task": {...}}}
  status update    {"result": {"statusUpdate": {...}}}
  artifact update  {"result": {"artifactUpdate": {...}}}
  bare message     {"result": {"message": {...}}}
  error            {"error": {"code": -32000, "message": "..."}}

SSE transport: ``data: {json}\\n\\n`` frames, ``: keepalive`` comments,
stream ends with ``: done``. Some peers ignore Accept and return plain JSON.

Stdlib + pytest only. No network: urlopen / _http_post_sse are monkeypatched.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, tools


# ---------------------------------------------------------------------------
# Frame builders (ground-truth shapes from a live Hermes peer)
# ---------------------------------------------------------------------------

def rpc(result, msg_id=1):
    """JSON-RPC wrapper around a StreamResponse."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def error_frame(code=-32000, message="Task not found", msg_id=1):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def task_snapshot(state="TASK_STATE_SUBMITTED", ctx="ctx-abc", task_id="task-1",
                  status_message=None):
    status = {"state": state, "timestamp": "2026-08-14T12:00:00Z"}
    if status_message is not None:
        status["message"] = status_message
    return rpc({"task": {"id": task_id, "contextId": ctx, "status": status}})


def status_update(state, ctx="ctx-abc", task_id="task-1", message=None):
    status = {"state": state, "timestamp": "2026-08-14T12:00:01Z"}
    if message is not None:
        status["message"] = message
    return rpc({"statusUpdate": {"taskId": task_id, "contextId": ctx, "status": status}})


def artifact_update(text, ctx="ctx-abc", task_id="task-1", artifact_id="art-1"):
    return rpc({"artifactUpdate": {
        "taskId": task_id,
        "contextId": ctx,
        "artifact": {"artifactId": artifact_id,
                     "parts": [{"text": text, "mediaType": "text/plain"}]},
    }})


def bare_message(text, ctx="ctx-abc", role="ROLE_AGENT"):
    return rpc({"message": {"role": role, "contextId": ctx, "messageId": "msg-1",
                            "parts": [{"text": text, "mediaType": "text/plain"}]}})


def sse_bytes(frames, keepalives=True, end_done=True):
    """Encode frames as an SSE byte stream, interleaving keepalive comments."""
    out = []
    if keepalives:
        out.append(": keepalive\n\n")
    for f in frames:
        out.append("data: " + json.dumps(f) + "\n\n")
        if keepalives:
            out.append(": keepalive\n\n")
    if end_done:
        out.append(": done\n\n")
    return "".join(out).encode("utf-8")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal urllib response: header dict + iterable of byte lines."""

    def __init__(self, lines=(), content_type="text/event-stream", body=None):
        self._lines = list(lines)
        self.headers = {"Content-Type": content_type}
        self._body = body if body is not None else b"".join(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body


def install_urlopen(monkeypatch, response, capture=None):
    """Replace urllib.request.urlopen with a fake returning ``response``."""

    def fake_urlopen(req, timeout=None):
        if capture is not None:
            capture.append(req)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


class FrameSource:
    """Replacement for _http_post_sse that records how many frames were pulled.

    ``frames`` is the list of yielded values; ``consumed`` tracks iterator
    advancement so tests can assert the parser stopped at the terminal frame.
    """

    def __init__(self, frames):
        self.frames = list(frames)
        self.consumed = 0

    def __call__(self, url, body, headers, timeout):
        source = self

        def gen():
            for f in source.frames:
                source.consumed += 1
                yield f

        return gen()


@pytest.fixture(autouse=True)
def no_disk_side_effects(monkeypatch, tmp_path):
    """Keep persistence / audit off the real disk and reset metric counters."""
    monkeypatch.setattr(tools.protocol, "persist_message",
                        lambda *a, **k: None)
    monkeypatch.setattr(tools.security, "audit", lambda *a, **k: None)
    monkeypatch.setattr(tools.security, "redact_outbound", lambda s: s)
    m = tools.protocol.metrics
    monkeypatch.setattr(m, "inbound_total", 0)
    monkeypatch.setattr(m, "outbound_total", 0)


def run_stream(frames, agent="researcher", ctx="ctx-abc", task_id="task-1"):
    """Drive _send_task_stream against a fake _http_post_sse (no network)."""
    src = FrameSource(frames)
    real = tools._http_post_sse
    tools._http_post_sse = src
    try:
        out = tools._send_task_stream(agent, "http://peer/x", {"params": {}},
                                      {}, 5, ctx, task_id)
    finally:
        tools._http_post_sse = real
    return out, src


# ---------------------------------------------------------------------------
# _http_post_sse: SSE transport layer
# ---------------------------------------------------------------------------

class TestHttpPostSse:

    def test_parses_data_frames_and_skips_keepalive_comments(self, monkeypatch):
        frames = [task_snapshot(), status_update("TASK_STATE_WORKING")]
        resp = FakeResponse(lines=[l.encode() for l in
                                   sse_bytes(frames).decode().splitlines(keepends=True)])
        install_urlopen(monkeypatch, resp)
        got = list(tools._http_post_sse("http://p/", {}, {}, 5))
        assert got == frames

    def test_sends_accept_event_stream_header_and_post(self, monkeypatch):
        captured = []
        install_urlopen(monkeypatch,
                        FakeResponse(lines=[b"data: {}\n\n"]), capture=captured)
        list(tools._http_post_sse("http://p/", {"x": 1}, {"Authorization": "Bearer t"}, 7))
        req = captured[0]
        assert req.get_method() == "POST"
        accept = {k.lower(): v for k, v in req.header_items()}["accept"]
        assert accept == "text/event-stream"
        assert json.loads(req.data) == {"x": 1}

    def test_non_sse_response_yielded_as_single_frame(self, monkeypatch):
        body = json.dumps(task_snapshot("TASK_STATE_COMPLETED")).encode()
        resp = FakeResponse(content_type="application/json", body=body)
        install_urlopen(monkeypatch, resp)
        got = list(tools._http_post_sse("http://p/", {}, {}, 5))
        assert got == [json.loads(body)]

    def test_malformed_data_line_skipped(self, monkeypatch):
        # Malformed/hostile frames are skipped silently (skip, do not abort);
        # the reader never yields None.
        lines = [b": keepalive\n", b"\n", b"data: {not json\n", b"\n",
                 b'data: "ok"\n', b"\n"]
        install_urlopen(monkeypatch, FakeResponse(lines=lines))
        assert list(tools._http_post_sse("http://p/", {}, {}, 5)) == ["ok"]

    def test_empty_stream_yields_nothing(self, monkeypatch):
        lines = [b": keepalive\n\n", b": done\n\n"]
        install_urlopen(monkeypatch, FakeResponse(lines=lines))
        assert list(tools._http_post_sse("http://p/", {}, {}, 5)) == []


# ---------------------------------------------------------------------------
# _send_task_stream: happy path + member-discriminated shapes
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_submitted_working_artifact_completed(self):
        frames = [
            task_snapshot("TASK_STATE_SUBMITTED"),
            status_update("TASK_STATE_WORKING"),
            artifact_update("PROBE DONE"),          # artifact BEFORE terminal
            status_update("TASK_STATE_COMPLETED"),
            status_update("TASK_STATE_COMPLETED"),  # must never be pulled
        ]
        (reply, ctx, state), src = run_stream(frames)
        assert reply == "PROBE DONE"          # reply text from the artifact
        assert ctx == "ctx-abc"
        assert state == "TASK_STATE_COMPLETED"
        assert src.consumed == 4               # terminal break: 5th frame unread

    def test_reply_from_terminal_status_message_when_no_artifact(self):
        final_msg = {"role": "ROLE_AGENT", "messageId": "m9",
                     "parts": [{"text": "all done, no artifact",
                                "mediaType": "text/plain"}]}
        frames = [status_update("TASK_STATE_WORKING"),
                  status_update("TASK_STATE_COMPLETED", message=final_msg)]
        (reply, ctx, state), _ = run_stream(frames)
        assert reply == "all done, no artifact"
        assert state == "TASK_STATE_COMPLETED"

    def test_bare_message_without_terminal_raises(self):
        # Stream closing after a bare message frame never delivered a terminal
        # state — task outcome unknown, so fail loud (review fix 1) instead of
        # returning partial speech as success.
        frames = [status_update("TASK_STATE_WORKING"), bare_message("hi from peer")]
        with pytest.raises(tools._A2aTransportError, match="terminal state"):
            run_stream(frames)


class TestTerminalBreak:

    @pytest.mark.parametrize("state", [
        "TASK_STATE_COMPLETED", "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED", "TASK_STATE_REJECTED",
    ])
    def test_every_terminal_state_breaks_the_loop(self, state):
        frames = [status_update("TASK_STATE_WORKING"),
                  status_update(state),
                  artifact_update("never read")]
        (reply, _ctx, out_state), src = run_stream(frames)
        assert out_state == state
        assert src.consumed == 2
        assert reply == ""    # no artifact was consumed, no status message

    def test_artifact_before_terminal_survives(self):
        """The live ordering quirk: artifactUpdate arrives BEFORE the terminal
        statusUpdate. The parser must still prefer the artifact text."""
        frames = [artifact_update("EARLY ARTIFACT"),
                  status_update("TASK_STATE_COMPLETED")]
        (reply, _ctx, state), _ = run_stream(frames)
        assert reply == "EARLY ARTIFACT"
        assert state == "TASK_STATE_COMPLETED"

    def test_multiple_artifacts_accumulated(self):
        # Multi-artifact / chunked streams keep every part (review fix).
        frames = [artifact_update("first", artifact_id="a-1"),
                  artifact_update("second draft", artifact_id="a-2"),
                  artifact_update("final answer", artifact_id="a-3"),
                  status_update("TASK_STATE_COMPLETED")]
        (reply, _ctx, _state), _ = run_stream(frames)
        assert "first" in reply
        assert "second draft" in reply
        assert "final answer" in reply


class TestErrorAndEdgeCases:

    def test_error_frame_raises_value_error_with_peer_name(self):
        frames = [status_update("TASK_STATE_WORKING"),
                  error_frame(-32000, "Task not found")]
        with pytest.raises(ValueError) as ei:
            run_stream(frames)
        assert "researcher" in str(ei.value)
        assert "Task not found" in str(ei.value)

    def test_stream_closed_without_result_raises(self):
        with pytest.raises(ValueError, match="stream closed without a result"):
            run_stream([None, "not-a-dict", ": keepalive"])

    def test_non_dict_and_unparsable_frames_skipped_until_terminal(self):
        frames = [None, 42, "junk", artifact_update("survived"),
                  status_update("TASK_STATE_COMPLETED")]
        (reply, _ctx, _state), _ = run_stream(frames)
        assert reply == "survived"

    def test_artifact_with_empty_parts_does_not_override(self):
        empty = rpc({"artifactUpdate": {
            "taskId": "task-1", "contextId": "ctx-abc",
            "artifact": {"artifactId": "a-0", "parts": []}}})
        final_msg = {"parts": [{"text": "from status", "mediaType": "text/plain"}]}
        frames = [empty, status_update("TASK_STATE_COMPLETED", message=final_msg)]
        (reply, _ctx, _state), _ = run_stream(frames)
        assert reply == "from status"


class TestSendTaskIntegration:
    """_send_task: streaming path selection + error propagation."""

    PEER = {"url": "http://peer/x", "auth": {}, "timeout": 5}
    STREAMING_CARD = {"url": "http://peer/x", "capabilities": {"streaming": True}}

    def _patch_card(self, monkeypatch, card):
        monkeypatch.setattr(tools, "_fetch_card", lambda *a, **k: card)

    def test_http_error_not_swallowed_by_fallback(self, monkeypatch):
        """An HTTPError from the streaming path must propagate, not trigger
        the ValueError-only fallback to message/send."""
        self._patch_card(monkeypatch, self.STREAMING_CARD)

        def raising_sse(url, body, headers, timeout):
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)

        monkeypatch.setattr(tools, "_http_post_sse", raising_sse)
        monkeypatch.setattr(tools, "_http_post_json",
                            lambda *a, **k: pytest.fail("fallback must not run"))
        with pytest.raises(urllib.error.HTTPError) as ei:
            tools._send_task("researcher", dict(self.PEER), "hi", "")
        assert ei.value.code == 502

    def test_rpc_error_does_not_fallback(self, monkeypatch):
        """Application-level JSON-RPC error on a healthy stream must NOT fall
        back to message/send (would resubmit the task) — review fix 4."""
        self._patch_card(monkeypatch, self.STREAMING_CARD)

        def bad_stream(url, body, headers, timeout):
            def gen():
                yield error_frame(-32000, "stream unsupported")
            return gen()

        monkeypatch.setattr(tools, "_http_post_sse", bad_stream)
        monkeypatch.setattr(tools, "_http_post_json",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError("must not fall back")))
        with pytest.raises(ValueError, match="stream unsupported"):
            tools._send_task("researcher", dict(self.PEER), "hi", "")

    def test_streaming_path_chosen_when_card_advertises(self, monkeypatch):
        self._patch_card(monkeypatch, self.STREAMING_CARD)
        seen = {}

        def fake_sse(url, body, headers, timeout):
            seen["url"], seen["body"] = url, body

            def gen():
                yield status_update("TASK_STATE_COMPLETED")
            return gen()

        monkeypatch.setattr(tools, "_http_post_sse", fake_sse)
        monkeypatch.setattr(tools, "_http_post_json",
                            lambda *a, **k: pytest.fail("must stream, not send"))
        tools._send_task("researcher", dict(self.PEER), "hi", "")
        assert seen["body"]["method"] == "SendStreamingMessage"
        assert tools.protocol.metrics.inbound_total == 1
