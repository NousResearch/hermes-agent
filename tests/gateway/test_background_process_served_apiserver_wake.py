"""A served profile's background-process completion wakes its ``api_server`` session in-process.

A served (multiplexed) profile's ``api_server`` turn binds the RAW session id as its session key, so
the completion / watch event it leaves behind names no profile at all. The wake path therefore
self-posted it to the unprefixed shared listener with the PRIMARY key — resuming the session in the
DEFAULT profile's store — and an event whose source did name a route-only served profile resolved
no adapter and was deferred forever.

Invariant (mirrors the Kanban one): the served profile whose own session store holds that exact
session is woken in-process, in that profile's scope, through the one shared adapter, without a
secondary credential; the default profile keeps its HTTP self-post; a served profile that does not
own the session never gets a wake in anyone's store.
"""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import GatewayRunner
from tests.gateway.test_kanban_notifier_served_apiserver_wake import (
    RecordingApiServerAdapter, _FakeHttpSession, _own_session, served,  # noqa: F401 (fixture)
)

SESSION = "20260918_090000_aa11bb"  # a served profile's api_server session (raw id)


def _make_runner(*, adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._profile_adapters = {"builder": {}, "atlas": {}}
    runner._profile_failed_platforms = {}
    runner._primary_profile_name = "default"
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=())
    return runner


def _completion_event(session_id, session_key=None):
    """Exactly what ``_bind_api_server_session`` leaves on a process event: the raw id, no profile."""
    return {"type": "completion", "session_id": "proc_1", "command": "make build", "exit_code": 0,
            "session_key": session_key if session_key is not None else session_id,
            "platform": "api_server", "chat_id": session_id, "user_id": "", "user_name": "", "thread_id": ""}


def _wake(runner, evt):
    return asyncio.run(runner._inject_watch_notification("[SYSTEM: make build exited 0]", evt))


def test_served_profile_completion_wakes_in_process_only_for_the_session_it_owns(served, monkeypatch):
    import aiohttp

    _own_session(served.builder, SESSION, "builder")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    adapter = RecordingApiServerAdapter()
    adapter._api_key, adapter._host, adapter._port, adapter._model_name = "k" * 20, "127.0.0.1", 8642, "hermes"

    # Raw event (no profile anywhere): the owning store is the proof.
    assert _wake(_make_runner(adapter=adapter), _completion_event(SESSION)) is True
    assert [t["session_id"] for t in adapter.turns] == [SESSION]
    assert adapter.homes == [str(served.builder)] and adapter.profiles == ["builder"]
    assert _FakeHttpSession.calls == []  # no HTTP self-post, so no secondary API_SERVER_KEY

    # Event whose source names the route-only served profile: same in-process wake, not deferred.
    adapter.turns.clear()
    assert _wake(_make_runner(adapter=adapter), _completion_event(SESSION, f"agent:builder:api_server:dm:{SESSION}")) is True
    assert [t["session_id"] for t in adapter.turns] == [SESSION] and adapter.profiles[-1] == "builder"

    # A served profile that does not own the session: no wake in anyone's store (retryable False).
    adapter.turns.clear()
    _own_session(served.atlas, "atlas-owned", "atlas")
    assert _wake(_make_runner(adapter=adapter), _completion_event("atlas-owned", "agent:builder:api_server:dm:atlas-owned")) is False
    assert adapter.turns == [] and _FakeHttpSession.calls == []

    # Control: the default profile's own session keeps the HTTP self-post.
    _own_session(served.root, "default-owned", "default")
    assert _wake(_make_runner(adapter=adapter), _completion_event("default-owned")) is True
    assert adapter.turns == []
    assert [c["headers"]["X-Hermes-Session-Id"] for c in _FakeHttpSession.calls] == ["default-owned"]
    assert _FakeHttpSession.calls[0]["url"].endswith("/v1/chat/completions")


def test_single_profile_gateway_keeps_the_http_self_post(served, monkeypatch):
    """Standalone (non-multiplex) gateway: no store scan, the historical HTTP self-post."""
    import aiohttp

    _own_session(served.builder, SESSION, "builder")
    _FakeHttpSession.calls = []
    monkeypatch.setattr(aiohttp, "ClientSession", _FakeHttpSession)
    adapter = RecordingApiServerAdapter()
    adapter._api_key, adapter._host, adapter._port, adapter._model_name = "k" * 20, "127.0.0.1", 8642, "hermes"
    runner = _make_runner(adapter=adapter)
    runner.config = SimpleNamespace(multiplex_profiles=False, profile_routes=())
    assert _wake(runner, _completion_event(SESSION)) is True
    assert adapter.turns == []
    assert [c["headers"]["X-Hermes-Session-Id"] for c in _FakeHttpSession.calls] == [SESSION]
