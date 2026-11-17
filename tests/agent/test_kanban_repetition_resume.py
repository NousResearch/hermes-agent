"""A kanban worker whose streamed reply the repetition guard cut resumes the SAME session.

The abort copy tells a reader to resend the message ("your conversation history is preserved").
A dispatcher-owned worker is a one-shot run with nobody to resend anything, so it exited rc=1 on a
cut and the board booked the death ``crashed``, throwing away an intact transcript and the run
time behind it. The resume re-issues the call once in the same session, after a note naming the
discarded loop; a second cut ends the run and books the model's own fault, so the dispatcher parks
the card once instead of spending its failure budget on the same loop.

The turns below run through the real streaming client against a local endpoint, so the cut is the
one ``RunawayStreamWatch`` makes rather than a stubbed verdict.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_TURN_BOUND_S = 60.0
# A degenerate reply: one status line, re-emitted forever by the endpoint.
_LOOP = "- **Status** : `NOTES_LOCAL.md` (draft notes, pending review). Not part of the stock layout.\n"
_ANSWER = "Backlog summarised: three cards are waiting on review.\n"


class _Endpoint:
    """Local chat-completions endpoint: the first request loops, every later one answers."""

    def __init__(self, *, loop_every_request: bool = False) -> None:
        self.requests = 0
        self.loop_every_request = loop_every_request
        self.bodies: list[dict] = []
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
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                endpoint.requests += 1
                try:
                    endpoint.bodies.append(json.loads(raw or b"{}"))
                except ValueError:
                    endpoint.bodies.append({})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    for frame in endpoint._frames():
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

    def _frames(self):
        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps({
                "id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 0,
                "model": "fixture-local",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"

        if self.loop_every_request or self.requests == 1:
            while not self._stop.is_set():
                yield chunk({"content": _LOOP})
                self._stop.wait(0.001)  # stream pace; returns at once on close()
            return
        yield chunk({"content": _ANSWER})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    def user_rows(self, request_index: int) -> list[str]:
        """Contents of the user rows the provider actually received on one request."""
        rows = self.bodies[request_index].get("messages") or []
        return [str(r.get("content") or "") for r in rows if r.get("role") == "user"]

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture(autouse=True)
def _isolated_identity(monkeypatch):
    """No inherited worker or child identity, and no proxies: this is the whole point of the test."""
    for var in ("HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_KANBAN_STOP_NUDGE",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.fixture()
def endpoint():
    server = _Endpoint()
    yield server
    server.close()


@pytest.fixture()
def looping_endpoint():
    """Every request loops: the model repeats itself on the transcript it was resumed with."""
    server = _Endpoint(loop_every_request=True)
    yield server
    server.close()


def _drive(server: _Endpoint, *, kanban_task: str = "") -> tuple[dict, str]:
    """Run one turn (as a dispatcher-owned worker when named) and return (result, session id)."""
    from run_agent import AIAgent

    agent = AIAgent(
        model="fixture-local", provider="custom", base_url=server.url, api_key="fixture",
        api_mode="chat_completions", quiet_mode=True, skip_memory=True, skip_context_files=True,
        enabled_toolsets=[], max_iterations=3,
    )
    session_id = agent.session_id
    if kanban_task:
        # Worker identity is the env the dispatcher spawns with; the guard reads it per turn.
        os.environ["HERMES_KANBAN_TASK"] = kanban_task
        os.environ["HERMES_KANBAN_STOP_NUDGE"] = "0"  # the stop nudge is a different guard
    try:
        outcome: dict = {}

        def run():
            try:
                outcome["result"] = agent.run_conversation("Summarize the backlog.")
            except BaseException as exc:  # surfaced below, on the test thread
                outcome["error"] = exc

        turn = threading.Thread(target=run, daemon=True)
        turn.start()
        turn.join(_TURN_BOUND_S)
        if turn.is_alive():
            agent.interrupt("test bound reached")
            turn.join(10)
            pytest.fail(f"turn still running after {_TURN_BOUND_S:.0f}s")
        if "error" in outcome:
            raise outcome["error"]
    finally:
        if kanban_task:
            os.environ.pop("HERMES_KANBAN_TASK", None)
            os.environ.pop("HERMES_KANBAN_STOP_NUDGE", None)
    return outcome["result"], session_id


def test_a_worker_cut_mid_loop_resumes_the_same_session(endpoint):
    """(1) the cut does not end the run, and (2) the retried call is the same session, whole."""
    result, session_id = _drive(endpoint, kanban_task="t_fixture")

    assert result["completed"] is True, result.get("failure_reason")
    assert result["final_response"].strip() == _ANSWER.strip()
    assert endpoint.requests == 2, "the cut is resumed exactly once"

    # The retried request carries everything that came before the cut, plus the one note.
    resumed = endpoint.user_rows(1)
    assert resumed[0].startswith("Summarize the backlog."), resumed
    note = resumed[-1]
    assert "cut by the repetition guard" in note
    assert "repetition loop and was interrupted" in note  # REPETITION_LOOP_INTERRUPTED
    assert len(resumed) == 2

    # The looped bytes never entered the transcript the next call replays.
    unit = _LOOP.strip()[:40]
    assert not any(unit in json.dumps(m, ensure_ascii=False) for m in result["messages"])
    assert session_id  # one session: the run never asked for a new one

    # (1) as the worker's exit code: a completed turn is rc 0, not the rc 1 that books a crash.
    from hermes_cli.turn_exit import turn_exit_code
    assert turn_exit_code(result, kanban_worker=True) == 0


def test_a_second_cut_ends_the_run_as_the_model_fault(looping_endpoint):
    """(3) the model looping again on the same transcript is booked as its own fault, once."""
    from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    from hermes_cli.turn_exit import turn_exit_code

    result, _ = _drive(looping_endpoint, kanban_task="t_fixture")

    assert looping_endpoint.requests == 2, "one resume, then the run ends"
    assert result.get("partial") is True
    assert result.get("failure_reason") == "model_repetition"
    assert "Repetition Detected" in str(result.get("final_response") or "")
    # The board books the model's fault: one spawn, the terminal provider lane, not a plain crash.
    assert turn_exit_code(result, kanban_worker=True) == KANBAN_TERMINAL_PROVIDER_EXIT_CODE


def test_a_person_keeps_the_abort_they_had(endpoint):
    """Control: the same cut outside a worker is unchanged - one call, rc 1, the resend copy."""
    from hermes_cli.turn_exit import turn_exit_code

    result, _ = _drive(endpoint)

    assert endpoint.requests == 1
    assert result["completed"] is False
    assert result["failure_reason"] == "truncated"
    assert turn_exit_code(result, kanban_worker=True) == 1
