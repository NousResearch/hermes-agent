"""Clarify prompts must reach the SSE queue of the session-stream run that asked them.

A ``/api/sessions/{id}/chat/stream`` turn never subscribes to the gateway's live event
stream, so a prompt registered only there parks the turn in ``tools.clarify_gateway``
until the clarify timeout and the question is never seen by the client.
``APIServerAdapter.notify_clarify_prompt`` hands the prompt to that run's own SSE queue
instead — and must pick the NEWEST live run, because every turn on a session appends a row
to ``_run_statuses`` (dict order is oldest-first, so the first match is usually a finished
turn whose queue nobody reads).

The answer path is covered too: such a turn has no admission row, so the authority
projection refuses ``POST /v1/runs/{run_id}/clarify``; the request id is exact, so the
handler resolves the waiting entry in the clarify registry directly.
"""

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

from gateway.platforms import api_server_runs
from gateway.platforms.api_server import APIServerAdapter


class _Queue:
    """Stand-in for the run's SSE queue — records what would be written to the client."""

    def __init__(self):
        self.events = []

    def enqueue(self, name, payload):
        self.events.append((name, payload))


def _adapter(runs):
    """A duck-typed adapter: only the two dicts notify_clarify_prompt touches."""
    fake = SimpleNamespace()
    fake._run_statuses = {}
    fake._run_event_queues = {}
    for rid, status, session_id, created, updated in runs:
        fake._run_statuses[rid] = {
            "status": status,
            "session_id": session_id,
            "created_at": created,
            "updated_at": updated,
        }
        fake._run_event_queues[rid] = _Queue()
    return fake


def _entry():
    return SimpleNamespace(
        clarify_id="clarify-1",
        question="Which colour do you prefer?",
        choices=["Red", "Green"],
        multi_select=False,
    )


def test_prompt_goes_to_the_newest_live_run_of_the_session():
    fake = _adapter([
        ("run-old", "completed", "sess-1", 1.0, 2.0),
        ("run-live", "running", "sess-1", 3.0, 4.0),
    ])

    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-1", entry=_entry(), generation=2, session_key="sess-1")

    live = fake._run_event_queues["run-live"]
    assert [name for name, _ in live.events] == ["clarify.request"]
    payload = live.events[0][1]
    assert payload["kind"] == "clarify"
    assert payload["request_id"] == "clarify-1"
    assert payload["question"] == "Which colour do you prefer?"
    assert payload["choices"] == ["Red", "Green"]
    assert payload["multi_select"] is False
    assert payload["execution_generation"] == 2
    # the finished turn's queue is pruned, never written to (nobody reads it any more)
    assert "run-old" not in fake._run_event_queues


def test_finished_runs_are_pruned_and_never_targeted():
    fake = _adapter([
        ("run-done", "completed", "sess-2", 1.0, 2.0),
    ])

    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-2", entry=_entry(), generation=1, session_key="sess-2")

    assert "run-done" not in fake._run_event_queues
    assert fake._run_event_queues == {}


def test_a_run_from_another_session_is_not_targeted():
    fake = _adapter([
        ("run-other", "running", "sess-other", 5.0, 6.0),
    ])

    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-mine", entry=_entry(), generation=1, session_key="sess-mine")

    assert fake._run_event_queues["run-other"].events == []


def test_no_live_run_is_a_no_op():
    fake = _adapter([])

    # must not raise: a session can lose its run between the prompt and the delivery
    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-3", entry=_entry(), generation=1, session_key="sess-3")

    assert fake._run_event_queues == {}


def test_payload_carries_the_live_stream_prompt_id_alias():
    """The live session-event stream labels the clarify id ``prompt_id``; the SSE path and the
    answer body use ``request_id``. A client rendering either surface must find the id under the
    key it knows, so both keys travel with the same value."""
    fake = _adapter([("run-live", "running", "sess-alias", 3.0, 4.0)])

    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-alias", entry=_entry(), generation=2, session_key="sess-alias")

    payload = fake._run_event_queues["run-live"].events[0][1]
    assert payload["request_id"] == "clarify-1"
    assert payload["prompt_id"] == "clarify-1"


def test_question_and_choices_are_redacted_like_the_live_path():
    """The live publisher redacts with force=True (#48456); a clarify question can quote tool-call
    context, so the SSE copy must not ship a credential-shaped value verbatim."""
    entry = SimpleNamespace(
        clarify_id="clarify-2",
        question="Run this: export AWS_SECRET_ACCESS_KEY=abcd1234efgh5678",
        choices=["Use export AWS_SECRET_ACCESS_KEY=abcd1234efgh5678", "Skip"],
        multi_select=False,
    )
    fake = _adapter([("run-live", "running", "sess-redact", 3.0, 4.0)])

    APIServerAdapter.notify_clarify_prompt(
        fake, session_id="sess-redact", entry=entry, generation=1, session_key="sess-redact")

    payload = fake._run_event_queues["run-live"].events[0][1]
    assert "abcd1234efgh5678" not in payload["question"]
    assert "abcd1234efgh5678" not in " ".join(payload["choices"])
    assert "***" in payload["question"]
    # a choice with nothing to redact is passed through untouched
    assert payload["choices"][1] == "Skip"


class _Request:
    """Opaque request stand-in — the handler is patched out below."""


class _Handler:
    def __init__(self, body):
        self._body = body

    async def _read_json_body(self, request):
        return self._body, None


def _run_clarify(monkeypatch, body, *, authority_status, resolved):
    """Drive the real handler with the authority projection refusing the answer."""
    calls = {}

    monkeypatch.setattr(api_server_runs, "_load_owned_run",
                        lambda *a, **k: ("run-1", None, None, None, None))

    async def _refuse(*a, **k):
        return web.Response(status=authority_status)

    monkeypatch.setattr(api_server_runs, "_respond_authority_run", _refuse)
    monkeypatch.setattr(api_server_runs, "_mark_run_event",
                        lambda *a, **k: calls.setdefault("marked", k))

    from tools import clarify_gateway

    def _resolve(request_id, answer):
        calls["resolved"] = (request_id, answer)
        return resolved

    monkeypatch.setattr(clarify_gateway, "resolve_gateway_clarify", _resolve)

    handler = _Handler(body)
    return asyncio.run(api_server_runs._handle_run_clarify(handler, _Request(),
                                                           _api_server=None)), calls


def test_answer_resolves_the_clarify_registry_when_the_authority_refuses(monkeypatch):
    response, calls = _run_clarify(
        monkeypatch, {"request_id": "clarify-1", "answer": "Green"},
        authority_status=409, resolved=True)

    assert calls["resolved"] == ("clarify-1", "Green")
    assert response.status == 200
    body = response.body if isinstance(response.body, (bytes, bytearray)) else b""
    assert b'"resolved": true' in body


def test_answer_keeps_the_refusal_when_nothing_is_waiting(monkeypatch):
    response, calls = _run_clarify(
        monkeypatch, {"request_id": "clarify-expired", "answer": "Green"},
        authority_status=409, resolved=False)

    assert calls["resolved"] == ("clarify-expired", "Green")
    # an expired/unknown id must not be reported as answered
    assert response.status == 409


@pytest.mark.parametrize("body", [
    {"request_id": "", "answer": "Green"},
    {"request_id": "clarify-1"},
    {"answer": "Green"},
])
def test_incomplete_answer_body_keeps_the_refusal(monkeypatch, body):
    response, calls = _run_clarify(monkeypatch, body, authority_status=409, resolved=True)

    assert "resolved" not in calls
    assert response.status == 409
