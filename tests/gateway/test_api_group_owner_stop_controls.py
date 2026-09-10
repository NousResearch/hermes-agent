"""Execution and control boundaries for a frozen participant, with real HTTP auth."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs as runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry
from tests.gateway.test_api_group_owner_stop import (
    KEY, OWNER, STOP, adapter as adapter, app_for, command, dispatch,
    invite, participation, seed, submit,
)
from tests.gateway.test_api_server_runs import _make_adapter, _make_slow_agent


@pytest.mark.asyncio
async def test_second_listener_freezes_queued_work_before_executor_starts(adapter, monkeypatch):
    release = asyncio.Event()
    execute = runs._execute_run

    async def delayed(*args, **kwargs):
        await release.wait()
        return await execute(*args, **kwargs)

    monkeypatch.setattr(runs, "_execute_run", delayed)
    create = MagicMock(side_effect=AssertionError("frozen work must not create an agent"))
    monkeypatch.setattr(adapter, "_create_agent", create)
    controller = _make_adapter(KEY)
    try:
        async with TestClient(TestServer(app_for(adapter))) as cli, TestClient(TestServer(app_for(controller))) as owner:
            invited = await invite(cli)
            payload = dispatch(invited)
            response = await submit(cli, invited, payload)
            assert response.status == 202, await response.json()
            run_id = (await response.json())["run_id"]
            tasks = list(adapter._active_run_tasks.values())
            stopped = await owner.post(STOP, headers=OWNER, json=command(participation(payload)))
            assert stopped.status == 200, await stopped.json()
            assert (await stopped.json())["work_state"] == "unresolved"
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            assert adapter._run_statuses[run_id]["status"] == "cancelled"
            create.assert_not_called()
            readback = await owner.get(STOP + "/owner-stop-1", headers=OWNER)
            assert (await readback.json())["work_state"] == "no_active_recorded_runs"
    finally:
        release.set()
        controller._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_second_listener_cannot_claim_local_stop_but_existing_sweeper_consumes_intent(adapter, monkeypatch):
    agent, ready, interrupted = _make_slow_agent()
    monkeypatch.setattr(adapter, "_create_agent", MagicMock(return_value=agent))
    controller = _make_adapter(KEY)
    try:
        async with TestClient(TestServer(app_for(adapter))) as cli, TestClient(TestServer(app_for(controller))) as owner:
            invited = await invite(cli)
            payload = dispatch(invited)
            accepted = await submit(cli, invited, payload)
            assert accepted.status == 202, await accepted.json()
            assert await asyncio.to_thread(ready.wait, 3)
            stopped = await owner.post(STOP, headers=OWNER, json=command(participation(payload)))
            body = await stopped.json()
            assert stopped.status == 200, body
            assert body["work_state"] == "unresolved"
            assert all(record["locally_managed"] is False for record in body["runs"])
            assert not interrupted.is_set()
            tasks = list(adapter._active_run_tasks.values())
            adapter._sweep_orphaned_runs_once()
            assert interrupted.is_set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            readback = await owner.get(STOP + "/owner-stop-1", headers=OWNER)
            assert (await readback.json())["work_state"] == "no_active_recorded_runs"
    finally:
        controller._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_freeze_refuses_real_grant_approval_but_still_allows_exact_deny(adapter):
    async with TestClient(TestServer(app_for(adapter))) as cli:
        invited = await invite(cli)
        participant = participation(dispatch(invited))
        run_id = "run-approval"
        seed(adapter, participant, run_id=run_id, status="waiting_for_approval", local=True)
        pending = _ApprovalEntry({"request_id": "approval-known", "command": "private command"})
        adapter._run_approval_sessions[run_id] = run_id
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]
        adapter._run_idempotency_store.freeze_room_scope(participant, "owner-stop-approval")
        headers = {"Authorization": f"HermesRoom {invited['grant']}"}
        refused = await cli.post(f"/v1/runs/{run_id}/approval", headers=headers,
            json={"choice": "once", "request_id": "approval-known"})
        body = await refused.json()
        assert refused.status == 409, body
        assert body["error"]["code"] == "group_work_frozen"
        assert not pending.event.is_set()
        denied = await cli.post(f"/v1/runs/{run_id}/approval", headers=headers,
            json={"choice": "deny", "request_id": "approval-known"})
        assert denied.status == 200, await denied.json()
        assert pending.result == "deny" and pending.event.is_set()


@pytest.mark.asyncio
async def test_scoped_work_never_borrows_regular_api_key_steer_permissions(adapter):
    async with TestClient(TestServer(app_for(adapter))) as cli:
        invited = await invite(cli)
        participant = participation(dispatch(invited))
        run_id = "run-steer"
        seed(adapter, participant, run_id=run_id, local=True)
        agent = MagicMock()
        adapter._active_run_agents[run_id] = agent
        for frozen in (False, True):
            if frozen:
                adapter._run_idempotency_store.freeze_room_scope(participant, "owner-stop-steer")
            for headers, expected in (({"Authorization": f"HermesRoom {invited['grant']}"}, 401), (OWNER, 404)):
                result = await cli.post(f"/v1/runs/{run_id}/steer", headers=headers, json={"input": "more work"})
                assert result.status == expected, await result.json()
        agent.steer.assert_not_called()


@pytest.mark.asyncio
async def test_memory_fallback_refuses_group_admission_but_keeps_regular_controls(adapter, monkeypatch):
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(":memory:")
    agent, ready, interrupted = _make_slow_agent()
    create = MagicMock(return_value=agent)
    monkeypatch.setattr(adapter, "_create_agent", create)
    async with TestClient(TestServer(app_for(adapter))) as cli:
        invited = await invite(cli)
        refused = await submit(cli, invited, dispatch(invited))
        body = await refused.json()
        assert refused.status == 503, body
        assert body["error"]["code"] == "group_stop_storage_unavailable"
        create.assert_not_called()
        accepted = await cli.post("/v1/runs", headers={**OWNER, "Idempotency-Key": "ordinary-run"}, json={"input": "ordinary task"})
        assert accepted.status == 202, await accepted.json()
        run_id = (await accepted.json())["run_id"]
        assert await asyncio.to_thread(ready.wait, 3)
        steer = await cli.post(f"/v1/runs/{run_id}/steer", headers=OWNER, json={"input": "ordinary steer"})
        assert steer.status == 200, await steer.json()
        agent.steer.assert_called_once_with("ordinary steer")
        pending = _ApprovalEntry({"request_id": "approval-ordinary", "command": "private command"})
        session = adapter._run_approval_sessions[run_id]
        with approval_mod._lock:
            approval_mod._gateway_queues[session] = [pending]
        allowed = await cli.post(f"/v1/runs/{run_id}/approval", headers=OWNER,
            json={"choice": "once", "request_id": "approval-ordinary"})
        assert allowed.status == 200, await allowed.json()
        assert pending.result == "once"
        stopped = await cli.post(f"/v1/runs/{run_id}/stop", headers=OWNER)
        assert stopped.status == 200, await stopped.json()
        assert interrupted.is_set()


@pytest.mark.asyncio
async def test_freeze_committed_while_in_executor_queue_prevents_conversation_start(adapter, monkeypatch):
    release = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    blocker = executor.submit(release.wait, 10)
    loop = asyncio.get_running_loop()
    schedule = loop.run_in_executor
    queued = asyncio.Event()
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "must not run"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    queue_next = False

    def create_agent(**_kwargs):
        nonlocal queue_next
        queue_next = True
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    controller = _make_adapter(KEY)

    def queue_execution(_executor, function, *args):
        nonlocal queue_next
        if queue_next:
            queue_next = False
            result = schedule(executor, function, *args)
            queued.set()
            return result
        return schedule(_executor, function, *args)

    try:
        async with TestClient(TestServer(app_for(adapter))) as cli, TestClient(TestServer(app_for(controller))) as owner:
            invited = await invite(cli)
            payload = dispatch(invited)
            with monkeypatch.context() as execution_patch:
                execution_patch.setattr(loop, "run_in_executor", queue_execution)
                accepted = await submit(cli, invited, payload)
                assert accepted.status == 202, await accepted.json()
                await asyncio.wait_for(queued.wait(), 3)
            tasks = list(adapter._active_run_tasks.values())
            stopped = await owner.post(STOP, headers=OWNER, json=command(participation(payload)))
            assert stopped.status == 200, await stopped.json()
            assert (await stopped.json())["admissions_frozen"] is True
            assert not blocker.done()
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            agent.run_conversation.assert_not_called()
            readback = await owner.get(STOP + "/owner-stop-1", headers=OWNER)
            assert (await readback.json())["work_state"] == "no_active_recorded_runs"
    finally:
        release.set()
        await asyncio.to_thread(executor.shutdown, wait=True)
        controller._run_idempotency_store.close()
