"""The --run-budget deadline bounds the model, not only its silence (#127773).

A stream that keeps producing never trips the stale timeout the budget caps, so a run with
``agent.run_budget_seconds`` set kept streaming past its deadline, and new model calls kept
starting after it. Past the deadline the model now gets one grace call; a live stream stops there
with its text kept. These turns run with no stream callback, the way cron and one-shot
``hermes chat --run-budget`` call the model.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

_BUDGET_S = 3.0
_TURN_BOUND_S = 30.0


class _FakeEndpoint:
    """Local endpoint replaying one scripted reply per request (the last one repeats)."""

    def __init__(self, replies) -> None:
        self.replies, self.bodies = replies, []
        self._stop = threading.Event()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # model metadata probes
                body = json.dumps({"object": "list", "data": [
                    {"id": "fixture-local", "object": "model", "context_length": 131072}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                endpoint.bodies.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode())
                reply = endpoint.replies[min(len(endpoint.bodies), len(endpoint.replies)) - 1]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for frame in reply(endpoint._stop, anthropic=self.path.endswith("/messages")):
                        self.wfile.write(frame.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()


def _chat(delta, finish_reason=None):
    return "data: " + json.dumps({
        "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture-local",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


def _event(kind, **payload):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"


def _endless_distinct_text(stop, *, anthropic):
    """A healthy reply that never ends: every sentence differs, so only a deadline stops it."""
    if anthropic:
        yield _event("message_start", message={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture-local", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}})
        yield _event("content_block_start", index=0, content_block={"type": "text", "text": ""})
    step = 0
    while not stop.is_set():
        text = f"Step {step} of the migration plan is drafted. "
        yield (_event("content_block_delta", index=0, delta={"type": "text_delta", "text": text}) if anthropic
               else _chat({"content": text}))
        step += 1
        time.sleep(0.02)


def _read_file_call(stop, *, anthropic):
    yield _chat({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_fixture", "type": "function",
                                                       "function": {"name": "read_file", "arguments": "{\"path\": \"plan.md\"}"}}]})
    yield _chat({}, "tool_calls")
    yield "data: [DONE]\n\n"


def _final_answer(stop, *, anthropic):
    yield _chat({"content": "All steps are drafted."})
    yield _chat({}, "stop")
    yield "data: [DONE]\n\n"


@pytest.fixture()
def endpoint(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    started = []

    def start(*replies) -> _FakeEndpoint:
        started.append(_FakeEndpoint(list(replies)))
        return started[-1]

    yield start
    for server in started:
        server.close()


def _run_turn(server: _FakeEndpoint, *, api_mode: str = "chat_completions", toolsets=()) -> dict:
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture", api_mode=api_mode,
        quiet_mode=True, skip_memory=True, skip_context_files=True, enabled_toolsets=list(toolsets),
        max_iterations=10, run_budget_seconds=_BUDGET_S,
    )
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = agent.run_conversation("Draft the migration plan.")
        except BaseException as exc:  # surfaced below, on the test thread
            outcome["error"] = exc

    turn = threading.Thread(target=run, daemon=True)
    turn.start()
    turn.join(_TURN_BOUND_S)
    if turn.is_alive():
        agent.interrupt("test bound reached")
        turn.join(10)
        pytest.fail(f"turn still running after {_TURN_BOUND_S:.0f}s with a {_BUDGET_S:.0f}s run budget")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_live_stream_stops_at_the_deadline_and_keeps_its_text(endpoint, api_mode):
    server = endpoint(_endless_distinct_text)

    result = _run_turn(server, api_mode=api_mode)

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert result["final_response"].startswith("Step 0 of the migration plan is drafted.")
    assert "Run budget reached" in result["final_response"]
    assert len(server.bodies) == 1
    tail = result["messages"][-1]
    assert tail["role"] == "assistant" and tail["content"].startswith("Step 0 of the migration plan")


def test_one_grace_call_after_the_deadline_then_no_call_starts(endpoint):
    server = endpoint(_read_file_call, _read_file_call, _final_answer)
    dispatched = []

    def slow_first_tool(*args, **kwargs):
        dispatched.append(time.monotonic())
        if len(dispatched) == 1:
            time.sleep(_BUDGET_S + 0.5)  # the deadline passes inside the tool
        return json.dumps({"content": "1| plan"})

    with patch("model_tools.handle_function_call", side_effect=slow_first_tool):
        result = _run_turn(server, toolsets=["file"])

    assert len(dispatched) == 2
    assert len(server.bodies) == 2  # the grace call ran; the call after it never started
    assert "run time budget nearly exhausted" in server.bodies[1].lower()  # and it carried the wrap-up
    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert "Run budget reached" in result["final_response"]
    assert [m["role"] for m in result["messages"][-2:]] == ["tool", "assistant"]
