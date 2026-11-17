"""POST /v1/runs ``if_busy`` and POST /v1/runs/{id}/cancel against a session whose turn is running."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from hermes_state import SessionDB
from tests.waiting import async_wait_until
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _adapter, _runs_client


class Agent(InterruptControlMixin):
    def __init__(self):
        self._pending_steer = None
        self._interrupt_requested = False

    def interrupt(self, *args, **kwargs):
        self._interrupt_requested = True
        return True

    hard_interrupt = interrupt


async def _busy_session(tmp_path, monkeypatch):
    """A gateway whose first run on ``api-session`` holds the turn until ``release`` is set."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    agent, adopted, release = Agent(), asyncio.Event(), threading.Event()
    seen = {"texts": [], "steers": []}

    async def answer(event):
        from gateway.session_results import execution_result
        authority = runner.session_authority
        seen["texts"].append(event.text)
        if len(seen["texts"]) == 1:
            sid = next(iter(authority.sessions))
            generation = authority._handle(
                SimpleNamespace(profile_id=authority.profile_id, session_id=sid)).execution_generation
            await asyncio.to_thread(authority.adopt_agent, sid, generation, agent)
            adopted.set()
            await asyncio.to_thread(release.wait, 10)
            seen["steers"].append(agent._drain_pending_steer())
        result = {"final_response": "ACK_" + event.text, "messages": []}
        if agent._interrupt_requested and len(seen["texts"]) == 1:
            result = {"final_response": "", "interrupted": True, "completed": False, "messages": []}
        execution_result.get()["result"] = result
        return result["final_response"]

    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False, _handle_message=answer,
                             _cached_agent_for=lambda route: agent)
    await initialize_session_authority(runner, profile_id="default", instance_id="api-if-busy")
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter
    client = _runs_client(adapter)
    await client.start_server()
    head = await client.post("/v1/runs", json={"input": "long task", "session_id": "api-session"},
                             headers=_AUTH_HEADERS)
    assert head.status == 202, await head.text()
    await asyncio.wait_for(adopted.wait(), 5)

    async def close():
        release.set()
        for task in list(adapter._active_run_tasks.values()):
            task.cancel()
        await client.close()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        db.close()
        store.close_all_db_handles()

    return SimpleNamespace(client=client, adapter=adapter, agent=agent, release=release, seen=seen,
                           head_id=(await head.json())["run_id"], close=close)


async def _post(env, if_busy, text, **headers):
    body = {"input": text, "session_id": "api-session"}
    if if_busy is not None:
        body["if_busy"] = if_busy
    response = await env.client.post("/v1/runs", json=body, headers=_AUTH_HEADERS | headers)
    return response.status, await response.json()


async def _status(env, run_id, until):
    async def polled():
        current = await (await env.client.get(f"/v1/runs/{run_id}", headers=_AUTH_HEADERS)).json()
        return current if current["status"] in until else None
    return await async_wait_until(polled, timeout=10, what=f"run {run_id} reaching {sorted(until)}")


@pytest.mark.asyncio
async def test_if_busy_names_its_delivery_and_a_queued_run_can_be_withdrawn(tmp_path, monkeypatch):
    env = await _busy_session(tmp_path, monkeypatch)
    try:
        status, plain = await _post(env, None, "no policy")
        assert status == 202 and "delivery" not in plain, "an absent if_busy keeps today's 202 body"
        status, queued = await _post(env, "queue", "wait your turn")
        assert (status, queued["delivery"]) == (202, "queued")
        assert (await _status(env, queued["run_id"], {"queued"}))["status"] == "queued"

        cancel = await env.client.post(f"/v1/runs/{queued['run_id']}/cancel", headers=_AUTH_HEADERS)
        assert cancel.status == 200, await cancel.text()
        assert (await cancel.json())["status"] == "cancelled"
        again = await env.client.post(f"/v1/runs/{queued['run_id']}/cancel", headers=_AUTH_HEADERS)
        running = await env.client.post(f"/v1/runs/{env.head_id}/cancel", headers=_AUTH_HEADERS)
        for refused in (again, running):
            assert refused.status == 409
            assert (await refused.json())["error"]["code"] == "run_not_cancellable"

        status, steered = await _post(env, "steer", "also update the docs")
        assert (status, steered["delivery"]) == (202, "steered")
        env.release.set()
        assert (await _status(env, env.head_id, {"completed"}))["status"] == "completed"
        assert (await _status(env, steered["run_id"], {"completed"}))["status"] == "completed"
        await _status(env, plain["run_id"], {"completed"})
        assert env.seen["steers"] == ["also update the docs"], "the steer reached the running turn"
        assert "wait your turn" not in env.seen["texts"], "a withdrawn run never executes"
        assert "also update the docs" not in env.seen["texts"], "a steered run is not also its own turn"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_if_busy_interrupt_stops_the_running_turn_and_runs_next(tmp_path, monkeypatch):
    env = await _busy_session(tmp_path, monkeypatch)
    try:
        status, bad = await _post(env, "drop", "x")
        assert status == 400 and bad["error"]["code"] == "invalid_if_busy"
        status, taken = await _post(env, "interrupt", "do this instead", **{"Idempotency-Key": "k1"})
        assert (status, taken["delivery"]) == (202, "interrupted")
        assert env.agent._interrupt_requested
        status, replay = await _post(env, "interrupt", "do this instead", **{"Idempotency-Key": "k1"})
        assert (status, replay["run_id"], replay["delivery"]) == (202, taken["run_id"], "interrupted")
        status, conflict = await _post(env, "queue", "do this instead", **{"Idempotency-Key": "k1"})
        assert status == 409 and conflict["error"]["code"] == "idempotency_key_conflict"
        env.release.set()
        assert (await _status(env, env.head_id, {"cancelled", "completed"}))["status"] == "cancelled"
        done = await _status(env, taken["run_id"], {"completed"})
        assert done["status"] == "completed" and done["output"] == "ACK_do this instead"
    finally:
        await env.close()
