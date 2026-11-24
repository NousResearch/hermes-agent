"""After a reply is cut for repetition, the turn re-issues the call with feedback, bounded per turn.

The cut reply is dropped, never replayed (#112764), and the re-issued request says how often the
model restated itself without quoting the reasoning. ``repetition_guard`` in config.yaml turns the circling watch
off, bounds the recoveries, and can send the re-issued call with reasoning off. Every turn here is
a real ``AIAgent`` against a scripted local endpoint; the circling text is synthetic.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.fakes.circling_text import ANSWER as _ANSWER
from tests.fakes.circling_text import PLAN as _PLAN
from tests.fakes.circling_text import restated_plan as _restated_plan

_CHAT, _ANTHROPIC = "chat_completions", "anthropic_messages"
_FEEDBACK_MARK = "kept restating the same statements"


@dataclass
class Reply:
    """One streamed reply: an optional tool call (streamed first), then reasoning, then text."""

    reasoning: str = ""
    content: str = ""
    tool_call: tuple[str, dict] | None = None


@dataclass
class _Endpoint:
    """Streams ``replies[n]`` to request n (the last reply repeats) on the chat or Anthropic wire."""

    replies: list[Reply]
    bodies: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                body = json.dumps({"object": "list", "data": [
                    {"id": "fixture-local", "object": "model", "context_length": 131072}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                endpoint.bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
                reply = endpoint.replies[min(len(endpoint.bodies), len(endpoint.replies)) - 1]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                frames = _anthropic_frames(reply) if self.path.endswith("/messages") else _chat_frames(reply)
                try:
                    for frame in frames:
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

    @property
    def requests(self) -> list[str]:
        return [json.dumps(body, ensure_ascii=False) for body in self.bodies]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _pieces(text: str):
    return (text[i:i + 200] for i in range(0, len(text), 200))


def _chat_frames(reply: Reply):
    def chunk(delta, finish_reason=None):
        return "data: " + json.dumps({
            "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture-local",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }) + "\n\n"

    if reply.tool_call:
        name, args = reply.tool_call
        yield chunk({"tool_calls": [{"index": 0, "id": "call_fixture_1", "type": "function",
                                     "function": {"name": name, "arguments": json.dumps(args)}}]})
    yield from (chunk({"reasoning_content": piece}) for piece in _pieces(reply.reasoning))
    if reply.content:
        yield chunk({"content": reply.content})
    yield chunk({}, "tool_calls" if reply.tool_call else "stop")
    yield "data: [DONE]\n\n"


def _anthropic_frames(reply: Reply):
    def event(kind, **payload):
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"

    yield event("message_start", message={
        "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture-local", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}})
    index = 0
    if reply.reasoning:
        yield event("content_block_start", index=index, content_block={"type": "thinking", "thinking": "", "signature": ""})
        for piece in _pieces(reply.reasoning):
            yield event("content_block_delta", index=index, delta={"type": "thinking_delta", "thinking": piece})
        yield event("content_block_delta", index=index, delta={"type": "signature_delta", "signature": "sig"})
        yield event("content_block_stop", index=index)
        index += 1
    if reply.content:
        yield event("content_block_start", index=index, content_block={"type": "text", "text": ""})
        yield event("content_block_delta", index=index, delta={"type": "text_delta", "text": reply.content})
        yield event("content_block_stop", index=index)
    yield event("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 1})
    yield event("message_stop")


@pytest.fixture()
def endpoint(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    started: list[_Endpoint] = []

    def start(*replies: Reply) -> _Endpoint:
        started.append(_Endpoint(list(replies)))
        return started[-1]

    yield start
    for server in started:
        server.close()


@pytest.fixture()
def guard_decisions(monkeypatch):
    """The loop-guard metric calls the repetition path makes, as ``(action, code)``."""
    from hermes_cli.observability import shared_metrics_harness

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(shared_metrics_harness, "record_guardrail_decision",
                        lambda _agent, action, code: seen.append((action, code)))
    return seen


def _write_config(text: str) -> None:
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(text, encoding="utf-8")


def _agent(server: _Endpoint, api_mode: str = _CHAT, toolsets: list[str] | None = None):
    from run_agent import AIAgent

    return AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture", api_mode=api_mode,
        quiet_mode=True, skip_memory=True, skip_context_files=True, enabled_toolsets=toolsets or [], max_iterations=6,
    )


def _run(agent, message: str = "Count the molecules.", history: list | None = None) -> dict:
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = agent.run_conversation(message, conversation_history=history)
        except BaseException as exc:  # surfaced below, on the test thread
            outcome["error"] = exc

    turn = threading.Thread(target=run, daemon=True)
    turn.start()
    turn.join(60)
    if turn.is_alive():
        agent.interrupt("test bound reached")
        turn.join(10)
        pytest.fail("turn still running after 60s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _assert_recovered(server: _Endpoint, result: dict, requests: int = 2) -> None:
    assert result["completed"] is True
    assert result["final_response"] == _ANSWER
    assert len(server.bodies) == requests
    retry = server.requests[1]
    assert _FEEDBACK_MARK in retry
    assert not any(statement.lower() in retry.lower() for statement in _PLAN)  # the cut reasoning is never replayed
    assert not any(statement.lower() in json.dumps(m, ensure_ascii=False).lower()
                   for m in result["messages"] for statement in _PLAN)  # nor stored in the user's row


@pytest.mark.parametrize("api_mode", [_CHAT, _ANTHROPIC])
def test_circling_reasoning_is_cut_and_the_turn_recovers_with_feedback(endpoint, guard_decisions, api_mode):
    server = endpoint(Reply(reasoning=_restated_plan(80)), Reply(content=_ANSWER))

    result = _run(_agent(server, api_mode))

    _assert_recovered(server, result)
    repetition = result["repetition"]
    assert (repetition["cuts"], repetition["recoveries"]) == (1, 1)
    assert 8_000 <= repetition["cut_at_chars"][0] < len(_restated_plan(80))
    assert any(statement.lower() in unit for unit in repetition["repeated"] for statement in _PLAN)
    assert guard_decisions == [("block", "circling_reasoning_cut"), ("retry", "repetition_recovery")]


def test_the_recovery_call_keeps_the_output_cap_of_the_cut_call(endpoint):
    server = endpoint(Reply(reasoning=_restated_plan(80)), Reply(content=_ANSWER))

    _assert_recovered(server, _run(_agent(server)))

    cut, retry = ({k: v for k, v in body.items() if k in ("max_tokens", "max_completion_tokens")} for body in server.bodies)
    assert retry == cut  # no cap on the cut call, none on the recovery


def test_a_model_that_circles_again_ends_the_turn_after_one_recovery(endpoint, guard_decisions):
    server = endpoint(Reply(reasoning=_restated_plan(80)))

    result = _run(_agent(server))

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert "Repetition Detected" in result["final_response"]
    assert "restated the same statements" in result["final_response"]
    assert not any(statement.lower() in result["final_response"].lower() for statement in _PLAN)  # reasoning stays private
    assert len(server.bodies) == 2  # the first call and one recovery, never a third
    assert (result["repetition"]["cuts"], result["repetition"]["recoveries"]) == (2, 1)
    assert ("halt", "repetition_recovery_exhausted") in guard_decisions


def test_each_user_turn_gets_its_own_recovery(endpoint):
    plan = Reply(reasoning=_restated_plan(80))
    server = endpoint(plan, Reply(content=_ANSWER), plan, Reply(content=_ANSWER))
    agent = _agent(server)

    first = _run(agent)
    second = _run(agent, "And the next cell?", history=first["messages"])

    assert first["completed"] and second["completed"]
    assert second["final_response"] == _ANSWER
    assert len(server.bodies) == 4
    assert _FEEDBACK_MARK in server.requests[3]
    assert second["repetition"]["recoveries"] == 1


def test_recovery_call_answering_with_reasoning_only_still_reaches_an_answer(endpoint):
    thinking_only = Reply(reasoning="I should report the blocker instead of trying again.")
    server = endpoint(Reply(reasoning=_restated_plan(80)), thinking_only, Reply(content=_ANSWER))

    result = _run(_agent(server))

    assert result["completed"] is True
    assert result["final_response"] == _ANSWER
    assert len(server.bodies) == 3
    assert _FEEDBACK_MARK in server.requests[1]


def test_a_tool_call_streamed_before_the_cut_is_never_dispatched(endpoint, tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    cut = Reply(reasoning=_restated_plan(80), tool_call=("write_file", {"path": str(sentinel), "content": "x"}))
    server = endpoint(cut, Reply(content=_ANSWER))

    result = _run(_agent(server, toolsets=["file"]))

    _assert_recovered(server, result)
    assert not sentinel.exists()
    assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in result["messages"])


# ---- configuration (repetition_guard in config.yaml) ---------------------------------------------

def test_circling_disabled_lets_a_restated_plan_stream_to_the_end(endpoint, guard_decisions):
    _write_config("repetition_guard:\n  circling:\n    enabled: false\n")
    server = endpoint(Reply(reasoning=_restated_plan(80)), Reply(content=_ANSWER))

    result = _run(_agent(server))

    assert result["completed"] is True
    assert _FEEDBACK_MARK not in server.requests[1]  # no cut: the thinking-only path answered instead
    assert "repetition" not in result
    assert guard_decisions == []


def test_zero_recovery_attempts_end_the_turn_on_the_first_cut(endpoint, guard_decisions):
    _write_config("repetition_guard:\n  recovery:\n    max_attempts: 0\n")
    server = endpoint(Reply(reasoning=_restated_plan(80)), Reply(content=_ANSWER))

    result = _run(_agent(server))

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert len(server.bodies) == 1
    assert (result["repetition"]["cuts"], result["repetition"]["recoveries"]) == (1, 0)
    assert guard_decisions == [("block", "circling_reasoning_cut")]


def test_two_recovery_attempts_allow_a_second_retry(endpoint):
    _write_config("repetition_guard:\n  recovery:\n    max_attempts: 2\n")
    plan = Reply(reasoning=_restated_plan(80))
    server = endpoint(plan, plan, Reply(content=_ANSWER))

    result = _run(_agent(server))

    _assert_recovered(server, result, requests=3)
    assert result["repetition"]["recoveries"] == 2


def test_recovery_with_reasoning_off_disables_reasoning_on_the_retry_only(endpoint):
    _write_config("agent:\n  reasoning_effort: high\nrepetition_guard:\n  recovery:\n    reasoning: off\n")
    server = endpoint(Reply(reasoning=_restated_plan(80)), Reply(content=_ANSWER), Reply(content=_ANSWER))
    agent = _agent(server)

    first = _run(agent)
    _run(agent, "Thanks.", history=first["messages"])

    assert first["completed"] is True and _FEEDBACK_MARK in server.requests[1]
    assert len(server.bodies) == 3
    cut, retry, next_turn = (_reasoning_fields(body) for body in server.bodies)
    assert retry != cut
    assert next_turn == cut  # one-shot: the next request is back to the configured reasoning


def _reasoning_fields(body: dict) -> dict:
    return {key: value for key, value in body.items() if "reason" in key or key == "thinking"} | {
        key: value for key, value in (body.get("extra_body") or {}).items() if "reason" in key}
