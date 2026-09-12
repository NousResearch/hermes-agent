"""#107475: ``agent_close`` must not hard-close the shared OpenAI/httpx client under an in-flight request.

The per-request wire client runs under an in-use guard: ``_close_cached_request_openai_client`` reads the
slot's ``in_use`` flag and refuses to hard-close a client a worker has checked out (it aborts the sockets
and lets the owning worker close). The shared primary client — driven IN PLACE by requests that get no
per-request clone (codex-direct streams, iteration-limit summaries) — had no such check: ``close()``
unconditionally closed it, so a request that was mid-stream lost its transport without an error and hung
until a watchdog.

These tests pin the reuse of the same guard:

1. an in-flight request keeps its transport across a concurrent ``agent.close()`` and still completes;
2. the parked close then runs from the request's own thread (the FD owner), once it releases;
3. with nothing in flight the shared client is closed immediately, as before.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from run_agent import AIAgent


def _make_agent():
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    return agent


def _delta_event(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _completed_event():
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(usage=None, id="resp_1", status="completed"),
    )


class _StallingStream:
    """Codex SSE stand-in: one delta, then a provider pause the test controls."""

    def __init__(self, client):
        self._client = client
        self.streaming = threading.Event()  # first event delivered: the request is on the wire
        self.release = threading.Event()    # let the provider deliver the rest

    def __iter__(self):
        yield _delta_event("partial ")
        self.streaming.set()
        while not self.release.is_set():
            if self._client.closed_event.is_set():
                # Transport torn down under the reader: the provider's remaining events never
                # arrive and nothing raises — the silent hang the report describes. Bounded so a
                # broken fix fails the assertions below instead of hanging the suite.
                self.release.wait(timeout=5.0)
                return
            time.sleep(0.005)
        yield _delta_event("more")
        yield _completed_event()


class _SharedClient:
    """Shared primary client stand-in (not a Mock: ``_is_openai_client_closed`` reads ``is_closed``)."""

    def __init__(self):
        self.is_closed = False
        self.close_calls = 0
        self.close_thread = None
        self.closed_event = threading.Event()
        self.created = threading.Event()  # responses.create() reached: the request is on the wire
        self.last_stream = None
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.last_stream = _StallingStream(self)
        self.created.set()
        return self.last_stream

    def close(self):
        self.close_calls += 1
        self.close_thread = threading.current_thread().ident
        self.closed_event.set()
        self.is_closed = True


class _ForceCloseSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, client):
        self.calls.append(client)
        return 0


def test_agent_close_leaves_the_transport_of_an_in_flight_request_alone():
    from agent.codex_runtime import run_codex_stream

    agent = _make_agent()
    client = _SharedClient()
    agent.client = client
    force_close = _ForceCloseSpy()
    agent._force_close_tcp_sockets = force_close
    result = {}

    def _request():
        try:
            result["response"] = run_codex_stream(agent, {"model": "test/model"})
        except BaseException as exc:  # noqa: BLE001 - asserted below
            result["error"] = exc

    worker = threading.Thread(target=_request, daemon=True)
    worker.start()
    # No fixed sleep: wait until the request is genuinely mid-stream (first delta consumed), so the
    # close below really interleaves with an in-flight request.
    assert client.created.wait(timeout=30.0), f"request never reached the wire: {result.get('error')!r}"
    assert client.last_stream.streaming.wait(timeout=30.0), (
        f"request never consumed its first delta: {result.get('error')!r}"
    )

    agent.close()

    assert client.close_calls == 0, "agent_close closed the shared client of an in-flight request"
    assert force_close.calls == [], "agent_close force-shut the in-flight request's sockets"
    assert agent.client is None, "the shared client must be dropped from the agent"

    # The provider finishes normally because its transport was never touched.
    client.last_stream.release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "the in-flight request hung instead of completing"
    assert "error" not in result, f"in-flight request failed: {result.get('error')!r}"
    assert result["response"].output_text == "partial more"

    # ...and the deferred close ran once, from the request's own thread (FD owner).
    assert client.close_calls == 1
    assert client.close_thread == worker.ident


def test_agent_close_closes_the_shared_client_when_nothing_is_in_flight():
    agent = _make_agent()
    client = _SharedClient()
    agent.client = client

    agent.close()

    assert (client.close_calls, client.close_thread) == (1, threading.current_thread().ident)
    assert agent.client is None


def test_release_clients_parks_the_shared_retirement_too():
    """``release_clients`` (gateway cache evict) shares the guard: same in-flight deferral."""
    agent = _make_agent()
    client = _SharedClient()
    agent.client = client
    retired = []
    agent._retire_shared_openai_client = lambda c, *, reason: retired.append((c, reason))

    with agent._shared_client_checkout(reason="test_request"):
        agent.release_clients()
        assert (retired, agent.client) == ([], None), "retirement must be parked for the in-flight request"

    assert [reason for _c, reason in retired] == ["cache_evict"]
    assert retired[0][0] is client


def test_shared_checkout_releases_on_failure():
    agent = _make_agent()
    agent.client = _SharedClient()

    with pytest.raises(RuntimeError):
        with agent._shared_client_checkout(reason="test_request"):
            raise RuntimeError("request blew up")

    assert agent._shared_in_flight() == 0
