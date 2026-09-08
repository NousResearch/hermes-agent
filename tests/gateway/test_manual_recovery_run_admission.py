"""Recovery fences new HTTP admissions, not existing receipts or Stop."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms as rooms
from gateway.platforms import api_server_room_grants as grants, api_server_runs as runs
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app, _use_idempotency_db
from tests.gateway.test_hosted_room_replica_ingress import HOME, TARGET, SECRET, grant, pair
from tests.gateway.test_manual_group_promotion_staging import saved, stage


@asynccontextmanager
async def listener(saved, tmp_path, monkeypatch, colocated):
    monkeypatch.setattr(rooms, "default_db_path", lambda: saved[1])
    monkeypatch.setattr(grants, "_effective_room_profile", lambda _: "reviewer")
    adapter = _make_adapter(api_key="synthetic-owner-key")
    _use_idempotency_db(adapter, saved[1] if colocated else tmp_path / "runs.db")
    monkeypatch.setattr(adapter, "_room_grant_secret", lambda: SECRET)
    _, catalog = grants._local_room_catalog(adapter, "reviewer", TARGET)
    token, _ = grant(saved[1], grant_id="dispatch-before-recovery",
        permissions=("dispatch", "status", "stop", "replicate", "work_records"),
        execution_policy_digest=catalog["execution_policy"]["policy_digest"])
    launched, release = [], asyncio.Event()

    async def inert_execution(_adapter, launch, **kwargs):
        launched.append(launch.run_id)
        await release.wait()

    monkeypatch.setattr(runs, "_execute_run", inert_execution)
    monkeypatch.setattr(adapter, "_create_agent", lambda **_: pytest.fail("No model calls allowed"))
    state = SimpleNamespace(adapter=adapter, catalog=catalog, token=token, launched=launched)
    try:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            state.client = client
            yield state
    finally:
        release.set()
        await asyncio.gather(*adapter._active_run_tasks.values(), return_exceptions=True)
        adapter._run_idempotency_store.close()


def request(state, task):
    prompt = "Synthetic admission probe; no model or tools."
    dispatch = {
        "protocol_version": 2, "room_id": "room", "home_install_id": HOME,
        "authority_gateway_id": HOME, "authority_epoch": 1, "member_id": "reviewer",
        "target_install_id": TARGET, "target_profile": "reviewer", "task_id": task,
        "execution_generation": 1, "source_event_seq": 1, "cancellation_scope_id": f"cancel-{task}",
        "prompt": prompt, "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": state.catalog["catalog_digest"],
        "execution_policy_digest": state.catalog["execution_policy"]["policy_digest"],
        "trace_id": f"trace-{task}",
    }
    return state.client.post("/v1/runs", json={"input": prompt, "hosted_room_dispatch": dispatch},
        headers={"Authorization": f"HermesRoom {state.token}", "Idempotency-Key": f"room:{task}:1"})


@pytest.mark.asyncio
@pytest.mark.parametrize("colocated", [False, True])
async def test_stage_rejects_new_work_but_preserves_receipt_status_and_stop(saved, tmp_path, monkeypatch, colocated):
    async with listener(saved, tmp_path, monkeypatch, colocated) as state:
        accepted = await request(state, "accepted")
        assert accepted.status == 202
        original = await accepted.json()
        factory = state.adapter._run_idempotency_store._conn.row_factory
        stage(saved)
        replay = await request(state, "accepted")
        assert replay.status == 202
        assert (await replay.json())["run_id"] == original["run_id"]
        assert replay.headers["Idempotency-Replayed"] == "true"
        refused = await request(state, "new")
        assert refused.status == 403
        assert (await refused.json())["error"]["code"] == "room_admission_fenced"
        headers = {"Authorization": f"HermesRoom {state.token}"}
        status = await state.client.get(f"/v1/runs/{original['run_id']}", headers=headers)
        assert status.status == 200
        stopped = await state.client.post(f"/v1/runs/{original['run_id']}/stop", headers=headers)
        assert stopped.status == 200
        assert (await stopped.json())["status"] == "stopping"
        assert state.launched == [original["run_id"]]
        assert set(state.adapter._run_statuses) == {original["run_id"]}
        assert state.adapter._run_idempotency_store._conn.row_factory is factory


@pytest.mark.asyncio
@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize("competing_admission", [False, True])
async def test_stage_between_lookup_and_reservation_distinguishes_new_work_from_replay(
        saved, tmp_path, monkeypatch, colocated, competing_admission):
    async with listener(saved, tmp_path, monkeypatch, colocated) as state:
        entered, release = asyncio.Event(), asyncio.Event()
        original_history = state.adapter._conversation_history_for_session

        async def paused_history(session_id):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return await original_history(session_id)

        monkeypatch.setattr(state.adapter, "_conversation_history_for_session", paused_history)
        delayed = asyncio.ensure_future(request(state, "racing"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            original = None
            if competing_admission:
                accepted = await request(state, "racing")
                assert accepted.status == 202
                original = await accepted.json()
            stage(saved)
            release.set()
            response = await asyncio.wait_for(delayed, 5)
            result = await response.json()
            if competing_admission:
                assert response.status == 202
                assert result["run_id"] == original["run_id"]
                assert result["replayed"] is True
                assert state.launched == [original["run_id"]]
            else:
                assert response.status == 403
                assert result["error"]["code"] == "room_admission_fenced"
                assert not state.launched
                assert not state.adapter._run_statuses
        finally:
            release.set()
            await asyncio.gather(delayed, return_exceptions=True)
