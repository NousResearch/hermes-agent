"""``agent.stream_max_lifetime`` covers the codex_responses and bedrock_converse wires.

#80701 wired the cap into ``StreamingWaitMonitor``, which only the chat_completions /
anthropic_messages path runs. The other two wires dispatch elsewhere
(``_stream_codex_passthrough`` and ``_BedrockStream``), and every watchdog they own —
TTFB, event-idle, progress, stale — resets on provider activity, so a provider that keeps
emitting events but never finishes was never cut. These are the red tests for that gap:
the cap case hangs on main, the guard cases must keep working either way.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import run_agent
from agent.chat_completion_helpers import interruptible_streaming_api_call
from agent.stream_max_lifetime import StreamLifetimeExceeded


def _settle(fn, deadline: float):
    """Run ``fn`` on a daemon thread so a never-ending stream fails the test instead of
    hanging the suite (and so a regression reads as a plain assertion, not a CI timeout)."""
    box: dict[str, object] = {}

    def worker():
        try:
            box["value"] = fn()
        except BaseException as exc:  # the test inspects whatever came out
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(deadline)
    return thread.is_alive(), box


def _set_cap(monkeypatch, tmp_path, cap):
    """``agent.stream_max_lifetime`` in a temp HERMES_HOME config, read by the real loader."""
    (tmp_path / "config.yaml").write_text(f"agent:\n  stream_max_lifetime: {cap}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The cap is floored at the stale timeout: keep stale below the cap in tests, and high
    # enough that a fast drip never trips it (so only the cap can end these calls).
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "1.0")
    monkeypatch.setenv("HERMES_BEDROCK_STALE_TIMEOUT", "1.0")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")


# --------------------------------------------------------------------------- Bedrock


def _bedrock_events(limit: int | None):
    """Converse event iterator: a text delta every 50 ms, then messageStop (or never)."""
    yield {"messageStart": {"role": "assistant"}}
    sent = 0
    while limit is None or sent < limit:
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "."}}}
        sent += 1
        # health: allow HX013 -- the 50 ms drip IS the behaviour under test (a provider that keeps yielding forever)
        time.sleep(0.05)
    yield {"messageStop": {"stopReason": "end_turn"}}


def _bedrock_agent(monkeypatch, limit: int | None):
    import agent.chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_bedrock_converse_call", lambda *a, **k: _bedrock_events(limit))
    agent = run_agent.AIAgent(
        api_key="k", model="anthropic.claude-3-5-sonnet", provider="bedrock", platform="cli",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=1,
    )
    agent.api_mode = "bedrock_converse"
    return agent, {"modelId": "anthropic.claude-3-5-sonnet",
                   "messages": [{"role": "user", "content": [{"text": "hi"}]}]}


def test_bedrock_converse_stream_that_never_finishes_is_cut_by_the_cap(monkeypatch, tmp_path):
    _set_cap(monkeypatch, tmp_path, 1.5)
    agent, kwargs = _bedrock_agent(monkeypatch, limit=None)
    started = time.time()
    alive, box = _settle(lambda: interruptible_streaming_api_call(agent, kwargs), 8.0)
    assert not alive, (
        "bedrock_converse stream was never cut: still streaming after 8s with "
        "agent.stream_max_lifetime=1.5 (the stale check resets on every event)"
    )
    error = box.get("error")
    assert isinstance(error, StreamLifetimeExceeded), error
    assert "agent.stream_max_lifetime" in str(error)
    assert time.time() - started < 8.0


def test_bedrock_converse_stream_that_finishes_inside_the_cap_is_untouched(monkeypatch, tmp_path):
    _set_cap(monkeypatch, tmp_path, 30)
    agent, kwargs = _bedrock_agent(monkeypatch, limit=3)
    started = time.time()
    response = interruptible_streaming_api_call(agent, kwargs)
    assert response is not None
    assert time.time() - started < 8.0


# ----------------------------------------------------------------------------- Codex


class _ResponsesDrip:
    """Real Responses-wire SSE server: one output_text delta every 50 ms, then completion
    (or never)."""

    def __init__(self, deltas: int | None):
        limit = deltas

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", 0)) or 0)
                if not self.path.endswith("/responses"):
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()

                def send(event: dict) -> None:
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

                try:
                    send({"type": "response.created",
                          "response": {"id": "r1", "object": "response", "status": "in_progress"}})
                    sent = 0
                    while limit is None or sent < limit:
                        sent += 1
                        send({"type": "response.output_text.delta", "delta": ".",
                              "output_index": 0, "content_index": 0,
                              "item_id": "msg_1", "sequence_number": sent})
                        # health: allow HX013 -- same reason: the drip is the stream this test feeds.
                        time.sleep(0.05)
                    if limit is not None:
                        send({"type": "response.completed", "response": {
                            "id": "r1", "object": "response", "status": "completed",
                            "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                                        "status": "completed",
                                        "content": [{"type": "output_text", "text": "."}]}],
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}})
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def responses_wire():
    wires: list[_ResponsesDrip] = []

    def make(deltas):
        wire = _ResponsesDrip(deltas)
        wires.append(wire)
        return wire

    yield make
    for wire in wires:
        wire.close()


def _codex_agent(wire):
    agent = run_agent.AIAgent(
        api_key="k", base_url=wire.base_url, model="m", provider="custom", platform="cli",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=1,
    )
    agent.api_mode = "codex_responses"
    return agent


def test_codex_responses_stream_that_never_finishes_is_cut_by_the_cap(
    responses_wire, monkeypatch, tmp_path
):
    _set_cap(monkeypatch, tmp_path, 1.5)
    agent = _codex_agent(responses_wire(None))
    started = time.time()
    alive, box = _settle(
        lambda: interruptible_streaming_api_call(agent, {"model": "m", "input": "hi"}), 8.0)
    assert not alive, (
        "codex_responses stream was never cut: still streaming after 8s with "
        "agent.stream_max_lifetime=1.5 (TTFB/idle/progress watchdogs all reset on events)"
    )
    error = box.get("error")
    assert isinstance(error, StreamLifetimeExceeded), error
    assert "agent.stream_max_lifetime" in str(error)
    assert time.time() - started < 8.0


def test_codex_responses_stream_that_completes_inside_the_cap_is_untouched(
    responses_wire, monkeypatch, tmp_path
):
    _set_cap(monkeypatch, tmp_path, 30)
    agent = _codex_agent(responses_wire(3))
    started = time.time()
    response = interruptible_streaming_api_call(agent, {"model": "m", "input": "hi"})
    assert response is not None
    assert time.time() - started < 8.0


# ------------------------------------------------------------------------------ guards


def test_cap_off_arms_nothing_on_either_wire(monkeypatch, tmp_path):
    """Cap 0 (the default) must not introduce a clock of its own: every poll stays a no-op."""
    from agent.stream_max_lifetime import StreamLifetime, resolve_stream_max_lifetime

    _set_cap(monkeypatch, tmp_path, 0)
    assert resolve_stream_max_lifetime(600.0) == 0.0
    assert StreamLifetime(0.0).exceeded("attempt") is None
