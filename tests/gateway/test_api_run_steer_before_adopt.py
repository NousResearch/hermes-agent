"""POST /v1/runs/{id}/steer accepted while the claimed turn is still building its agent reaches it."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from hermes_state import SessionDB
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _adapter, _runs_client


class Agent(InterruptControlMixin):
    def __init__(self):
        self._pending_steer = None
        self._interrupt_requested = False

    def interrupt(self, *args, **kwargs):
        self._interrupt_requested = True


@pytest.mark.asyncio
async def test_run_steer_before_adoption_reaches_the_turns_agent_not_the_cached_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    stale, fresh = Agent(), Agent()
    claimed, steered = asyncio.Event(), threading.Event()
    seen = {}

    async def answer(event):
        from gateway.session_results import execution_result
        authority = runner.session_authority
        sid = next(iter(authority.sessions))
        claimed.set()
        await asyncio.to_thread(steered.wait, 5)
        # The turn builds a fresh agent (the cached one is replaced) and the wiring adopts it.
        generation = authority._handle(SimpleNamespace(profile_id=authority.profile_id, session_id=sid)).execution_generation
        await asyncio.to_thread(authority.adopt_agent, sid, generation, fresh)
        seen["drained"] = fresh._drain_pending_steer()
        execution_result.get()["result"] = {"final_response": "ok", "messages": []}
        return "ok"

    # The previous turn's reusable agent is what the route resolves to before adoption.
    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False, _handle_message=answer,
                             _cached_agent_for=lambda route: stale)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="api-steer")
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter
    client = _runs_client(adapter)
    await client.start_server()
    try:
        started = await client.post("/v1/runs", json={"input": "build it", "session_id": "api-session"},
                                    headers=_AUTH_HEADERS | {"Idempotency-Key": "steer-run"})
        assert started.status == 202, await started.text()
        run_id = (await started.json())["run_id"]
        await asyncio.wait_for(claimed.wait(), 5)

        for text in ("use the staging db", "and skip tests"):
            response = await client.post(f"/v1/runs/{run_id}/steer", json={"input": text}, headers=_AUTH_HEADERS)
            assert response.status == 200, await response.text()
            assert (await response.json())["accepted"] is True
        steered.set()
        await asyncio.wait_for(authority.sessions["api-session"].task, 5)

        assert seen["drained"] == "use the staging db\nand skip tests", "the acknowledged steer was lost"
        assert stale._pending_steer is None, "the replaced cached agent must not swallow it"
        assert not authority.pending_steers
    finally:
        steered.set()
        for task in list(adapter._active_run_tasks.values()):
            task.cancel()
        await client.close()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        db.close()
        store.close_all_db_handles()
