"""POST /v1/runs/{id}/pause and /resume.

Covers both run kinds (a ``POST /v1/runs`` run and a ``/api/sessions/{id}/chat/stream`` turn), with
and without the session authority the live gateway always runs: pause -> paused -> resume, steer
while paused, stop while paused, the 404/409 contract, ``run.paused`` / ``run.resumed`` plus SSE
keepalives while paused, and ``/v1/capabilities``.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.interrupt_control import InterruptControlMixin
from agent.run_pause import RunPauseMixin, wait_while_paused
from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RunStream
from tests.waiting import async_wait_until


class PausableAgent(InterruptControlMixin, RunPauseMixin):
    """The agent surface the endpoints reach. ``turn()`` stands in for the conversation loop: its
    in-flight tool call runs until ``go`` is set, then the loop hits its iteration boundary."""

    def __init__(self):
        self._pending_steer = None
        self._interrupt_requested = False
        self._current_turn_id = "turn-1"
        self.go = threading.Event()

    def interrupt(self, *args, **kwargs):
        self._interrupt_requested = True
        return True

    def hard_interrupt(self, *args, **kwargs):
        self._interrupt_requested = True

    def turn(self) -> dict:
        assert self.go.wait(10), "test never released the in-flight tool call"
        wait_while_paused(self)
        if self._interrupt_requested:
            return {"final_response": "", "interrupted": True, "messages": []}
        return {"final_response": f"steer={self._drain_pending_steer()}", "messages": []}


def _claim(adapter, run_id):
    request = MagicMock()
    request.headers = {}
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)


def _app(adapter):
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/") or path.startswith("/api/sessions"):
            app.router.add_route(method, path, handler)
    return app


async def _status(cli, run_id, headers=None):
    return (await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json())["status"]


async def _read_until(stream, marker: str) -> str:
    seen = ""
    while marker not in seen:
        line = await asyncio.wait_for(stream.content.readline(), 10)
        assert line, f"stream ended before {marker!r}: {seen}"
        seen += line.decode()
    return seen


def _event_names(adapter, run_id):
    return [event["event"] for _, event in adapter._run_streams[run_id].backlog if event]


# ---------------------------------------------------------------------------
# Without a session authority: the agent is the adapter's _active_run_agents entry
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    yield adapter
    adapter._response_store.close()
    adapter._run_idempotency_store.close()


def _live_run(adapter, run_id="run_p"):
    agent = PausableAgent()
    _claim(adapter, run_id)
    adapter._set_run_status(run_id, "running")
    adapter._run_streams[run_id] = _RunStream()
    adapter._active_run_agents[run_id] = agent
    return run_id, agent


@pytest.mark.asyncio
async def test_v1_run_pause_paused_steer_resume(adapter):
    run_id, agent = _live_run(adapter)
    async with TestClient(TestServer(_app(adapter))) as cli:
        turn = asyncio.create_task(asyncio.to_thread(agent.turn))
        paused = await cli.post(f"/v1/runs/{run_id}/pause")
        assert paused.status == 200, await paused.text()
        assert await paused.json() == {"object": "hermes.run.pause", "run_id": run_id, "status": "pausing",
                                       "max_pause_seconds": 1800.0}
        assert await _status(cli, run_id) == "pausing"
        assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 409

        agent.go.set()  # the in-flight tool call finishes; the loop reaches its boundary
        await async_wait_until(lambda: agent.pause_state() == "paused", what="the run to block")
        await async_wait_until(lambda: "run.paused" in _event_names(adapter, run_id), what="run.paused")
        assert await _status(cli, run_id) == "paused"

        steered = await cli.post(f"/v1/runs/{run_id}/steer", json={"input": "use staging"})
        assert steered.status == 200, await steered.text()
        assert await _status(cli, run_id) == "paused"
        assert not turn.done()

        resumed = await cli.post(f"/v1/runs/{run_id}/resume")
        assert resumed.status == 200, await resumed.text()
        assert (await resumed.json())["status"] == "running"
        result = await asyncio.wait_for(turn, 10)
        assert result["final_response"] == "steer=use staging"
        await async_wait_until(lambda: "run.resumed" in _event_names(adapter, run_id), what="run.resumed")
        assert await _status(cli, run_id) == "running"
        assert (await cli.post(f"/v1/runs/{run_id}/resume")).status == 409
    names = _event_names(adapter, run_id)
    assert names.index("run.paused") < names.index("run.steered") < names.index("run.resumed")


@pytest.mark.asyncio
async def test_v1_run_stop_while_paused_wakes_and_stops(adapter):
    run_id, agent = _live_run(adapter)
    async with TestClient(TestServer(_app(adapter))) as cli:
        turn = asyncio.create_task(asyncio.to_thread(agent.turn))
        assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 200
        agent.go.set()
        await async_wait_until(lambda: agent.pause_state() == "paused", what="the run to block")

        stopped = await cli.post(f"/v1/runs/{run_id}/stop")
        assert stopped.status == 200, await stopped.text()
        result = await asyncio.wait_for(turn, 10)
        assert result["interrupted"] is True
        assert (await cli.post(f"/v1/runs/{run_id}/resume")).status == 409
    assert "run.resumed" not in _event_names(adapter, run_id)


@pytest.mark.asyncio
async def test_resume_before_the_boundary_cancels_the_pause(adapter):
    run_id, agent = _live_run(adapter)
    async with TestClient(TestServer(_app(adapter))) as cli:
        turn = asyncio.create_task(asyncio.to_thread(agent.turn))
        assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 200
        assert (await cli.post(f"/v1/runs/{run_id}/resume")).status == 200
        agent.go.set()
        assert (await asyncio.wait_for(turn, 10))["final_response"] == "steer=None"
        assert await _status(cli, run_id) == "running"


@pytest.mark.asyncio
async def test_pause_contract_404_and_409(adapter):
    _claim(adapter, "run_done")
    adapter._set_run_status("run_done", "completed")
    _claim(adapter, "run_queued")
    adapter._set_run_status("run_queued", "running")  # status says running, but no agent here
    async with TestClient(TestServer(_app(adapter))) as cli:
        for path in ("pause", "resume"):
            gone = await cli.post(f"/v1/runs/run_missing/{path}")
            assert gone.status == 404
            assert (await gone.json())["error"]["code"] == "run_not_found"
        done = await cli.post("/v1/runs/run_done/pause")
        assert done.status == 409
        assert (await done.json())["error"]["code"] == "run_not_pausable"
        assert (await cli.post("/v1/runs/run_queued/pause")).status == 409
        not_paused = await cli.post("/v1/runs/run_done/resume")
        assert (await not_paused.json())["error"]["code"] == "run_not_paused"


@pytest.mark.asyncio
async def test_an_old_run_id_never_resumes_a_later_turns_pause(adapter):
    """The gateway caches agents across turns: a pause asked through run A whose turn ended before
    its boundary must not let run A's id resume the pause a later run B put on the same agent."""
    run_a, agent = _live_run(adapter, "run_a")
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await cli.post(f"/v1/runs/{run_a}/pause")).status == 200
        adapter._active_run_agents.pop(run_a)
        agent._current_turn_id = "turn-2"  # run A's turn ended; the cached agent runs run B
        run_b = "run_b"
        _claim(adapter, run_b)
        adapter._set_run_status(run_b, "running")
        adapter._run_streams[run_b] = _RunStream()
        adapter._active_run_agents[run_b] = agent
        assert (await cli.post(f"/v1/runs/{run_b}/pause")).status == 200
        refused = await cli.post(f"/v1/runs/{run_a}/resume")
        assert refused.status == 409
        assert (await refused.json())["error"]["code"] == "run_not_paused"
        assert agent.pause_state() == "pausing"
        assert (await cli.post(f"/v1/runs/{run_b}/resume")).status == 200


@pytest.mark.asyncio
async def test_run_events_keep_alive_while_paused(adapter, monkeypatch):
    monkeypatch.setattr(api_server_mod, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.05)
    run_id, agent = _live_run(adapter)
    async with TestClient(TestServer(_app(adapter))) as cli:
        turn = asyncio.create_task(asyncio.to_thread(agent.turn))
        events = await cli.get(f"/v1/runs/{run_id}/events")
        assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 200
        agent.go.set()
        await _read_until(events, "run.paused")
        await _read_until(events, ": keepalive")
        assert agent.pause_state() == "paused"
        await cli.post(f"/v1/runs/{run_id}/resume")
        await _read_until(events, "run.resumed")
        await asyncio.wait_for(turn, 10)
        events.close()


@pytest.mark.asyncio
async def test_session_chat_stream_turn_pauses_and_resumes(adapter, monkeypatch):
    monkeypatch.setattr(api_server_mod, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.05)
    agent = PausableAgent()

    async def run_agent(*, active_run_id, **kwargs):
        adapter._active_run_agents[active_run_id] = agent
        return await asyncio.to_thread(agent.turn), {}

    async with TestClient(TestServer(_app(adapter))) as cli:
        with (
            patch.object(adapter, "_get_existing_session_or_404", new=AsyncMock(return_value=({"id": "s1"}, None))),
            patch.object(adapter, "_conversation_history_for_session", new=AsyncMock(return_value=[])),
            patch.object(adapter, "_run_agent", new=run_agent),
        ):
            stream = await cli.post("/api/sessions/s1/chat/stream", json={"message": "hi"})
            await _read_until(stream, "event: message.started")
            (run_id,) = list(adapter._run_statuses)
            assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 200
            agent.go.set()
            seen = await _read_until(stream, "event: run.paused")
            seen += await _read_until(stream, ": keepalive")
            assert await _status(cli, run_id) == "paused"
            assert (await cli.post(f"/v1/runs/{run_id}/steer", json={"input": "be brief"})).status == 200
            assert (await cli.post(f"/v1/runs/{run_id}/resume")).status == 200
            seen += await stream.text()
    assert seen.index("event: run.paused") < seen.index("event: run.resumed") < seen.index("event: run.completed")
    assert "steer=be brief" in seen
    assert f'"run_id": "{run_id}"' in seen.split("event: run.paused", 1)[1].split("\n\n", 1)[0]


@pytest.mark.asyncio
async def test_session_chat_stream_stop_while_paused(adapter):
    agent = PausableAgent()

    async def run_agent(*, active_run_id, **kwargs):
        adapter._active_run_agents[active_run_id] = agent
        return await asyncio.to_thread(agent.turn), {}

    async with TestClient(TestServer(_app(adapter))) as cli:
        with (
            patch.object(adapter, "_get_existing_session_or_404", new=AsyncMock(return_value=({"id": "s1"}, None))),
            patch.object(adapter, "_conversation_history_for_session", new=AsyncMock(return_value=[])),
            patch.object(adapter, "_run_agent", new=run_agent),
        ):
            stream = await cli.post("/api/sessions/s1/chat/stream", json={"message": "hi"})
            await _read_until(stream, "event: message.started")
            (run_id,) = list(adapter._run_statuses)
            assert (await cli.post(f"/v1/runs/{run_id}/pause")).status == 200
            agent.go.set()
            await _read_until(stream, "event: run.paused")
            assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
            rest = await stream.text()
    assert "event: run.cancelled" in rest
    assert "event: run.resumed" not in rest


# ---------------------------------------------------------------------------
# Capabilities and config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_advertise_run_pause(adapter):
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    async with TestClient(TestServer(app)) as cli:
        body = await (await cli.get("/v1/capabilities")).json()
    assert body["features"]["run_pause"] is True
    assert body["endpoints"]["run_pause"] == {"method": "POST", "path": "/v1/runs/{run_id}/pause"}
    assert body["endpoints"]["run_resume"] == {"method": "POST", "path": "/v1/runs/{run_id}/resume"}


def test_max_pause_seconds_config(monkeypatch):
    from gateway.platforms import api_server_run_pause as run_pause

    config = {}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.delenv("API_SERVER_MAX_PAUSE_SECONDS", raising=False)
    assert run_pause.max_pause_seconds() == 1800.0
    config["gateway"] = {"api_server": {"max_pause_seconds": 0}}
    assert run_pause.max_pause_seconds() == 0.0
    monkeypatch.setenv("API_SERVER_MAX_PAUSE_SECONDS", "60")
    assert run_pause.max_pause_seconds() == 60.0


# ---------------------------------------------------------------------------
# Under the session authority (every live gateway): the agent is the claim's adopted agent
# ---------------------------------------------------------------------------


def _adopting_turn(owner, agent, adopted: asyncio.Event):
    async def handle(event):
        from gateway.session_results import execution_result

        sid = event.source.chat_id
        generation = owner._handle(SimpleNamespace(profile_id=owner.profile_id, session_id=sid)).execution_generation
        await asyncio.to_thread(owner.adopt_agent, sid, generation, agent)
        adopted.set()
        result = await asyncio.to_thread(agent.turn)
        execution_result.get()["result"] = result
        return result["final_response"]
    return handle


@pytest.mark.asyncio
async def test_authority_v1_run_pause_steer_resume(api, owner):
    agent, adopted = PausableAgent(), asyncio.Event()
    owner.runner._handle_message = _adopting_turn(owner, agent, adopted)
    api._api_key = "request-scope-key-long-enough"
    headers = {"Authorization": "Bearer " + api._api_key}
    async with TestClient(TestServer(_app(api))) as cli:
        started = await cli.post("/v1/runs", json={"input": "build it", "session_id": "api-session"}, headers=headers)
        assert started.status == 202, await started.text()
        run_id = (await started.json())["run_id"]
        await asyncio.wait_for(adopted.wait(), 10)

        paused = await cli.post(f"/v1/runs/{run_id}/pause", headers=headers)
        assert paused.status == 200, await paused.text()
        assert await _status(cli, run_id, headers) == "pausing"
        agent.go.set()
        await async_wait_until(lambda: agent.pause_state() == "paused", what="the run to block")
        assert await _status(cli, run_id, headers) == "paused"
        assert (await cli.post(f"/v1/runs/{run_id}/steer", json={"input": "staging"}, headers=headers)).status == 200
        assert (await cli.post(f"/v1/runs/{run_id}/resume", headers=headers)).status == 200
        await asyncio.wait_for(owner.sessions["api-session"].task, 10)
        await async_wait_until(lambda: "run.completed" in _event_names(api, run_id), what="run.completed")
    names = _event_names(api, run_id)
    assert names.index("run.paused") < names.index("run.resumed") < names.index("run.completed")
    assert agent._pending_steer is None  # the steer drained after resume


@pytest.mark.asyncio
async def test_authority_v1_run_stop_while_paused(api, owner):
    agent, adopted = PausableAgent(), asyncio.Event()
    owner.runner._handle_message = _adopting_turn(owner, agent, adopted)
    api._api_key = "request-scope-key-long-enough"
    headers = {"Authorization": "Bearer " + api._api_key}
    async with TestClient(TestServer(_app(api))) as cli:
        started = await cli.post("/v1/runs", json={"input": "build it", "session_id": "api-session"}, headers=headers)
        run_id = (await started.json())["run_id"]
        await asyncio.wait_for(adopted.wait(), 10)
        assert (await cli.post(f"/v1/runs/{run_id}/pause", headers=headers)).status == 200
        agent.go.set()
        await async_wait_until(lambda: agent.pause_state() == "paused", what="the run to block")
        stopped = await cli.post(f"/v1/runs/{run_id}/stop", headers=headers)
        assert stopped.status == 200, await stopped.text()
        await asyncio.wait_for(owner.sessions["api-session"].task, 10)
        await async_wait_until(lambda: _settled(api, run_id), what="terminal run event")
        assert await _status(cli, run_id, headers) == "cancelled"
        assert (await cli.post(f"/v1/runs/{run_id}/resume", headers=headers)).status == 409


def _settled(adapter, run_id):
    return any(name in ("run.cancelled", "run.failed", "run.completed") for name in _event_names(adapter, run_id))


@pytest.mark.asyncio
async def test_authority_session_chat_stream_turn_pauses_and_resumes(api, owner):
    agent, adopted = PausableAgent(), asyncio.Event()
    owner.runner._handle_message = _adopting_turn(owner, agent, adopted)
    api._api_key = "request-scope-key-long-enough"
    headers = {"Authorization": "Bearer " + api._api_key}
    owner.db.create_session("stream-pause", source="api_server")
    async with TestClient(TestServer(_app(api))) as cli:
        stream = await cli.post("/api/sessions/stream-pause/chat/stream", json={"message": "go"}, headers=headers)
        await asyncio.wait_for(adopted.wait(), 10)
        (run_id,) = list(api._run_statuses)
        paused = await cli.post(f"/v1/runs/{run_id}/pause", headers=headers)
        assert paused.status == 200, await paused.text()
        agent.go.set()
        seen = await _read_until(stream, "event: run.paused")
        assert await _status(cli, run_id, headers) == "paused"
        assert (await cli.post(f"/v1/runs/{run_id}/steer", json={"input": "brief"}, headers=headers)).status == 200
        assert (await cli.post(f"/v1/runs/{run_id}/resume", headers=headers)).status == 200
        seen += await stream.text()
    assert seen.index("event: run.paused") < seen.index("event: run.resumed") < seen.index("event: run.completed")
    assert "steer=brief" in seen
