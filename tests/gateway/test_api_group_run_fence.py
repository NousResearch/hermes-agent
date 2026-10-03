"""A fenced room epoch at a real participant: refused new work, readable runs, successor control."""

import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_fence as fence
from gateway.platforms import api_server_runs as runs
from gateway.platforms.api_server_authority_runs import run_admission
from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry
from tests.gateway.test_api_group_owner_stop import (
    OWNER, PARTICIPANTS, STOP, adapter as adapter, app_for, command, dispatch, participation, seed,
)
from tests.gateway.test_api_group_owner_stop_canonical import canonical as canonical
from tests.gateway.test_api_server_runs import _make_slow_agent

ROOM = "room-owner-stop"
HOME = "install:unavailable-home"
SUCCESSOR = "install:successor"


async def invite(cli, authority=HOME, epoch=1, member_id="writer"):
    response = await cli.post("/v1/room-members/invitations", headers=OWNER, json={
        "room_id": ROOM, "home_install_id": authority, "authority_gateway_id": authority,
        "authority_epoch": epoch, "member_id": member_id})
    result = await response.json()
    assert response.status == 201, result
    return result


def scoped(invited, authority=HOME, epoch=1, task="task-1"):
    payload = dispatch(invited, task=task)
    return {**payload, "home_install_id": authority, "authority_gateway_id": authority, "authority_epoch": epoch}


def bearer(invited, **headers):
    return {"Authorization": f"HermesRoom {invited['grant']}", **headers}


async def submit(cli, invited, payload):
    return await cli.post("/v1/runs", json={"input": payload["prompt"], "hosted_room_dispatch": payload},
                          headers=bearer(invited, **{"Idempotency-Key": f"room:{payload['task_id']}:1"}))


def promise(adapter, candidate=SUCCESSOR, epoch=2):
    return fence.fence_and_promise(adapter._run_idempotency_store.path, room_id=ROOM, fence_epoch=epoch - 1,
                                   promise_epoch=epoch, candidate_install_id=candidate)


@pytest.mark.asyncio
async def test_fenced_epoch_refuses_new_work_while_its_runs_keep_running_and_pass_to_the_successor(
        adapter, monkeypatch):
    kept, kept_ready, kept_stopped = _make_slow_agent()
    other, other_ready, other_stopped = _make_slow_agent()
    create = MagicMock(side_effect=[kept, other])
    monkeypatch.setattr(adapter, "_create_agent", create)
    async with TestClient(TestServer(app_for(adapter))) as cli:
        old = await invite(cli)
        payload = scoped(old)
        run_ids = []
        for task, ready in ((payload, kept_ready), (scoped(old, task="task-0"), other_ready)):
            accepted = await submit(cli, old, task)
            assert accepted.status == 202, await accepted.json()
            run_ids.append((await accepted.json())["run_id"])
            assert await asyncio.to_thread(ready.wait, 3)
        run_id, old_run_id = run_ids
        promise(adapter)

        refused = await submit(cli, old, scoped(old, task="task-2"))
        body = await refused.json()
        assert refused.status == 409, body
        assert body["error"]["code"] == "room_authority_fenced"
        assert create.call_count == 2
        # Admitted runs are existing work: they replay, keep running and stay readable.
        replay = await submit(cli, old, payload)
        assert replay.status == 202 and (await replay.json())["run_id"] == run_id
        read = await cli.get(f"/v1/runs/{run_id}", headers=bearer(old))
        assert read.status == 200 and (await read.json())["status"] == "running"
        assert not kept_stopped.is_set()
        # Stop only reduces work, so the old epoch keeps it.
        assert (await cli.post(f"/v1/runs/{old_run_id}/stop", headers=bearer(old))).status == 200
        assert await asyncio.to_thread(other_stopped.wait, 3)

        # The promised successor reads and stops the other run with its own grant.
        successor = await invite(cli, SUCCESSOR, 2)
        status = await cli.get(f"/v1/runs/{run_id}", headers=bearer(successor))
        observed = await status.json()
        assert status.status == 200, observed
        assert observed["run_id"] == run_id and observed["status"] == "running"
        approval = await cli.post(f"/v1/runs/{run_id}/approval", headers=bearer(successor),
                                  json={"choice": "once", "request_id": "approval-1"})
        assert approval.status == 404
        assert not kept_stopped.is_set()
        stopped = await cli.post(f"/v1/runs/{run_id}/stop", headers=bearer(successor))
        assert stopped.status == 200, await stopped.json()
        assert await asyncio.to_thread(kept_stopped.wait, 3)
        await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values(), return_exceptions=True), 3)
        final = await cli.get(f"/v1/runs/{run_id}", headers=bearer(successor))
        assert (await final.json())["status"] in {"cancelled", "interrupted"}

        # A grant from an epoch this gateway never promised controls nothing here.
        stranger = await invite(cli, "install:stranger", 3)
        assert (await cli.get(f"/v1/runs/{run_id}", headers=bearer(stranger))).status == 404
        assert (await cli.post(f"/v1/runs/{run_id}/stop", headers=bearer(stranger))).status == 404


@pytest.mark.asyncio
async def test_fenced_epoch_controls_refuse_approval_but_keep_deny_and_stop(adapter):
    async with TestClient(TestServer(app_for(adapter))) as cli:
        old = await invite(cli)
        participant = participation(scoped(old))
        run_id = "run-approval"
        seed(adapter, participant, run_id=run_id, status="waiting_for_approval", local=True)
        adapter._run_idempotency_store.reserve(
            runs.room_run_scope_key(participant), "identity", "fp", "run-identity", {"status": "completed"},
            identity=participant)
        pending = _ApprovalEntry({"request_id": "approval-known", "command": "private command"})
        adapter._run_approval_sessions[run_id] = run_id
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]
        promise(adapter)
        refused = await cli.post(f"/v1/runs/{run_id}/approval", headers=bearer(old),
                                 json={"choice": "once", "request_id": "approval-known"})
        body = await refused.json()
        assert refused.status == 409, body
        assert body["error"]["code"] == "room_authority_fenced"
        assert not pending.event.is_set()
        denied = await cli.post(f"/v1/runs/{run_id}/approval", headers=bearer(old),
                                json={"choice": "deny", "request_id": "approval-known"})
        assert denied.status == 200, await denied.json()
        assert pending.result == "deny"
        read = await cli.get(f"/v1/runs/{run_id}", headers=bearer(old))
        assert read.status == 200, await read.json()
        stop = await cli.post(f"/v1/runs/{run_id}/stop", headers=bearer(old))
        assert stop.status == 409 and (await stop.json())["error"]["code"] == "run_not_active"


@pytest.mark.asyncio
async def test_owner_freeze_still_lists_and_freezes_a_fenced_participant(adapter):
    async with TestClient(TestServer(app_for(adapter))) as cli:
        old = await invite(cli)
        participant = participation(scoped(old))
        seed(adapter, participant, run_id="run-known")
        adapter._run_idempotency_store.reserve(
            runs.room_run_scope_key(participant), "identity", "fp", "run-identity", {"status": "queued"},
            identity=participant)
        promise(adapter)
        listed = await cli.get(PARTICIPANTS, headers=OWNER)
        assert [item["participant"] for item in (await listed.json())["data"]] == [participant]
        stopped = await cli.post(STOP, headers=OWNER, json=command(participant))
        reply = await stopped.json()
        assert stopped.status == 200, reply
        assert reply["admissions_frozen"] is True and reply["counts"]["total"] == 2
        refused = await submit(cli, old, scoped(old, task="task-2"))
        assert (await refused.json())["error"]["code"] == "group_work_frozen"
        assert fence.room_fence_state(adapter._run_idempotency_store.path, ROOM)["fenced_epoch"] == 1


@pytest.mark.asyncio
async def test_a_freeze_outlives_a_verified_move_and_refuses_the_new_hosts_work(adapter, monkeypatch):
    create = MagicMock()
    monkeypatch.setattr(adapter, "_create_agent", create)
    path = adapter._run_idempotency_store.path
    async with TestClient(TestServer(app_for(adapter))) as cli:
        old = await invite(cli)
        participant = participation(scoped(old))
        adapter._run_idempotency_store.reserve(
            runs.room_run_scope_key(participant), "identity", "fp", "run-identity", {"status": "running"},
            identity=participant)
        stopped = await cli.post(STOP, headers=OWNER, json=command(participant))
        assert stopped.status == 200, await stopped.json()
        # The group moves: this participant promised the next epoch to the successor, then learned it took over.
        promise(adapter)
        fence.learn_authority(path, room_id=ROOM, epoch=2, install_id=SUCCESSOR)
        successor = await invite(cli, SUCCESSOR, 2)
        refused = await submit(cli, successor, scoped(successor, SUCCESSOR, 2, task="task-after-move"))
        body = await refused.json()
        assert refused.status == 403, body
        assert body["error"]["code"] == "group_work_frozen"
        assert create.call_count == 0
        listed = await (await cli.get(PARTICIPANTS, headers=OWNER)).json()
        assert [item["admissions_frozen"] for item in listed["data"]] == [True]


@pytest.mark.asyncio
async def test_canonical_fenced_epoch_refuses_answers_and_passes_status_and_stop_to_the_successor(canonical):
    adapter, authority, runner = canonical
    entered, interrupted = asyncio.Event(), asyncio.Event()
    pending = _ApprovalEntry({'request_id': 'approval-exact', 'command': 'private command'})
    runner.agent.interrupt = lambda: (interrupted.set(), pending.event.set())

    async def service(event):
        from gateway.session_results import execution_result
        with authority.db._read_ctx() as conn:
            sid, generation = conn.execute(
                "SELECT target_session_id, generation FROM session_admissions WHERE principal_id='api'").fetchone()
        live = authority.sessions[sid]
        with approval_mod._lock:
            approval_mod._gateway_queues[live.route] = [pending]
        authority.register_approval(sid, generation, live.route, pending.data)
        entered.set()
        await asyncio.to_thread(pending.event.wait, 5)
        execution_result.get()['result'] = {'final_response': '', 'interrupted': True}
        return ''

    runner._handle_message = service
    try:
        async with TestClient(TestServer(app_for(adapter))) as cli:
            old = await invite(cli)
            accepted = await submit(cli, old, scoped(old))
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            exact = {'request_id': 'approval-exact', 'execution_generation': run_admission(adapter, run_id)[1]['generation']}
            promise(adapter)
            for kind, answer in (('approval', {'choice': 'once'}), ('clarify', {'answer': 'continue'})):
                refused = await cli.post(f'/v1/runs/{run_id}/{kind}', headers=bearer(old), json={**exact, **answer})
                assert refused.status == 409, await refused.text()
                assert (await refused.json())['error']['code'] == 'room_authority_fenced'
            assert not pending.event.is_set()
            successor = await invite(cli, SUCCESSOR, 2)
            status = await cli.get(f'/v1/runs/{run_id}', headers=bearer(successor))
            assert status.status == 200, await status.text()
            assert (await status.json())['status'] == 'running'
            stopped = await cli.post(f'/v1/runs/{run_id}/stop', headers=bearer(successor))
            assert stopped.status == 200, await stopped.text()
            assert (await stopped.json())['status'] == 'stopping' and interrupted.is_set()
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
            final = await cli.get(f'/v1/runs/{run_id}', headers=bearer(successor))
            assert (await final.json())['status'] == 'cancelled'
            assert pending.result is None
    finally:
        pending.event.set()
