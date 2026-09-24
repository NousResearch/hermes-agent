"""Owner freeze controls real API admissions, not just adapter-local tasks."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from gateway.platforms.api_server_authority_runs import run_admission
from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, get_session_admission
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_api_group_owner_stop import (
    KEY, OWNER, STOP, app_for, command, dispatch, invite, participation, seed, submit,
)
from tests.gateway.test_api_server_runs import _make_adapter


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    home = tmp_path / 'root'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    db = SessionDB(home / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(
        config=GatewayConfig(), sessions_dir=home / 'sessions'))
    authority = SessionAuthority(runner, profile_id='default', instance_id='owner', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    adapter = _make_adapter(KEY)
    adapter.gateway_runner = runner
    runner._adapter_for_source = lambda source: adapter
    runner._cached_agent_for = lambda route: runner.agent
    runner.agent = SimpleNamespace(interrupt=lambda: None)
    yield adapter, authority, runner
    assert not any(not task.done() for task in adapter._active_run_tasks.values())
    adapter._run_idempotency_store.close()
    adapter._response_store.close()
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('second_listener', [False, True])
async def test_started_canonical_service_wait_is_interrupted_by_owner_freeze(canonical, second_listener):
    adapter, authority, runner = canonical
    entered, released = asyncio.Event(), asyncio.Event()
    interrupted = asyncio.Event()
    runner.agent.interrupt = interrupted.set

    async def service(event):
        from gateway.session_results import execution_result
        entered.set()
        await released.wait()
        execution_result.get()['result'] = (
            {'interrupted': True, 'final_response': '', 'completed': False}
            if interrupted.is_set() else {'final_response': 'should not complete'})
        return ''
    runner._handle_message = service
    controller = _make_adapter(KEY) if second_listener else adapter
    try:
        async with TestClient(TestServer(app_for(adapter))) as client, TestClient(TestServer(app_for(controller))) as owner:
            invitation = await invite(client)
            payload = dispatch(invitation)
            response = await submit(client, invitation, payload)
            assert response.status == 202, await response.text()
            run_id = (await response.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            owned = run_admission(adapter, run_id)
            assert owned[1]['status'] == 'started'
            frozen = await owner.post(STOP, headers=OWNER, json=command(participation(payload)))
            assert frozen.status == 200, await frozen.text()
            if second_listener:
                assert not interrupted.is_set()
                await adapter._sweep_orphaned_runs_once()
            assert interrupted.is_set(), 'canonical generation did not receive Stop'
            released.set()
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
            assert adapter._run_statuses[run_id]['status'] == 'cancelled'
            assert get_session_admission(authority.db, admission_id=owned[1]['admission_id'])['status'] == 'terminal'
    finally:
        released.set()
        if second_listener:
            controller._run_idempotency_store.close()
            controller._response_store.close()


@pytest.mark.asyncio
async def test_queued_canonical_admission_is_cancelled_before_execution(canonical):
    adapter, authority, runner = canonical
    calls = []

    async def service(event):
        calls.append(event.text)
        return 'must not execute'
    runner._handle_message = service
    # Hold the real authority scheduler before its claim, not the admission
    # ledger or Stop API. The queued row remains durable and cancelable.
    authority._schedule = lambda ref: None
    async with TestClient(TestServer(app_for(adapter))) as client:
        invitation = await invite(client)
        payload = dispatch(invitation)
        accepted = await submit(client, invitation, payload)
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        owned = run_admission(adapter, run_id)
        assert owned[1]['status'] == 'queued'
        frozen = await client.post(STOP, headers=OWNER, json=command(participation(payload)))
        assert frozen.status == 200, await frozen.text()
        assert get_session_admission(authority.db, admission_id=owned[1]['admission_id'])['outcome'] == 'cancelled'
        await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
        assert calls == []
        assert adapter._run_statuses[run_id]['status'] == 'cancelled'


@pytest.mark.asyncio
@pytest.mark.parametrize('owner_stops', [False, True])
async def test_canonical_exact_approval_refuses_once_after_freeze_but_denies(canonical, owner_stops):
    adapter, authority, runner = canonical
    entered = asyncio.Event()
    interrupted = asyncio.Event()
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
        async with TestClient(TestServer(app_for(adapter))) as client:
            invitation = await invite(client)
            payload = dispatch(invitation)
            accepted = await submit(client, invitation, payload)
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            row = run_admission(adapter, run_id)[1]
            body = {'request_id': 'approval-exact', 'execution_generation': row['generation']}
            headers = {'Authorization': f"HermesRoom {invitation['grant']}"}
            if owner_stops:
                stopped = await client.post(STOP, headers=OWNER, json=command(participation(payload)))
                assert stopped.status == 200, await stopped.text()
                assert interrupted.is_set()
            else:
                adapter._run_idempotency_store.freeze_room_scope(participation(payload), 'freeze-approval')
            refused = await client.post(f'/v1/runs/{run_id}/approval', headers=headers,
                                        json={**body, 'choice': 'once'})
            assert refused.status == 409, await refused.text()
            assert (await refused.json())['error']['code'] == 'group_work_frozen'
            if not owner_stops:
                assert not pending.event.is_set()
                denied = await client.post(f'/v1/runs/{run_id}/approval', headers=headers,
                                           json={**body, 'choice': 'deny'})
                assert denied.status == 200, await denied.text()
                assert pending.event.is_set() and pending.result == 'deny'
            else:
                assert pending.result is None
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
            assert adapter._run_statuses[run_id]['status'] == 'cancelled'
    finally:
        pending.event.set()


@pytest.mark.asyncio
async def test_owner_receipt_cannot_interrupt_foreign_canonical_scope(canonical):
    adapter, authority, runner = canonical
    entered, release = asyncio.Event(), asyncio.Event()
    interrupted = asyncio.Event()
    runner.agent.interrupt = interrupted.set

    async def service(event):
        from gateway.session_results import execution_result
        entered.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'foreign completed'}
        return 'foreign completed'

    runner._handle_message = service
    try:
        async with TestClient(TestServer(app_for(adapter))) as client:
            invitation = await invite(client)
            target = participation(dispatch(invitation))
            accepted = await client.post('/v1/runs', headers=OWNER, json={'input': 'ordinary canonical work'})
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            admission = run_admission(adapter, run_id)[1]
            assert admission['status'] == 'started'
            target_scope = seed(adapter, target, run_id=run_id)
            # A stale/misassociated adapter mirror is not canonical authority.
            adapter._run_owners[run_id] = target_scope
            frozen = await client.post(STOP, headers=OWNER, json=command(target))
            assert frozen.status == 200, await frozen.text()
            assert not interrupted.is_set()
            assert get_session_admission(authority.db, admission_id=admission['admission_id'])['status'] == 'started'
            release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
            assert not interrupted.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_failed_canonical_interrupt_keeps_durable_intent_retryable(canonical, monkeypatch):
    adapter, authority, runner = canonical
    entered, release = asyncio.Event(), asyncio.Event()
    interrupted = asyncio.Event()
    runner.agent.interrupt = interrupted.set

    async def service(event):
        from gateway.session_results import execution_result
        entered.set()
        await release.wait()
        execution_result.get()['result'] = {'interrupted': interrupted.is_set(), 'final_response': ''}
        return ''

    runner._handle_message = service
    try:
        async with TestClient(TestServer(app_for(adapter))) as client:
            invitation = await invite(client)
            payload = dispatch(invitation)
            accepted = await submit(client, invitation, payload)
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            original = authority.interrupt
            attempts = []

            async def fail_once(*args):
                attempts.append(True)
                if len(attempts) == 1:
                    raise RuntimeStoreError('temporary_interrupt_failure')
                return await original(*args)

            monkeypatch.setattr(authority, 'interrupt', fail_once)
            body = command(participation(payload))
            failed = await client.post(STOP, headers=OWNER, json=body)
            assert failed.status == 503, await failed.text()
            assert run_id not in adapter._stopping_run_ids
            assert not interrupted.is_set()
            assert adapter._run_idempotency_store.is_scope_frozen(adapter._run_owners[run_id])
            await adapter._sweep_orphaned_runs_once()
            assert interrupted.is_set()
            assert len(attempts) == 2
            release.set()
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
            assert adapter._run_statuses[run_id]['status'] == 'cancelled'
    finally:
        release.set()


@pytest.mark.asyncio
async def test_canonical_allow_and_freeze_serialize_actual_approval_mutation(canonical, monkeypatch):
    adapter, authority, runner = canonical
    entered = asyncio.Event()
    pending = _ApprovalEntry({'request_id': 'approval-race', 'command': 'private command'})

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
    frozen = threading.Event()
    launched = threading.Event()
    workers = []
    try:
        async with TestClient(TestServer(app_for(adapter))) as client:
            invitation = await invite(client)
            payload = dispatch(invitation)
            accepted = await submit(client, invitation, payload)
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())['run_id']
            await asyncio.wait_for(entered.wait(), 3)
            row = run_admission(adapter, run_id)[1]
            live = authority.sessions[row['target_session_id']]
            original = live.controls.respond

            def interleave(*args, **kwargs):
                def freeze():
                    launched.set()
                    adapter._run_idempotency_store.freeze_room_scope(participation(payload), 'racing-freeze')
                    frozen.set()

                worker = threading.Thread(target=freeze, daemon=True)
                workers.append(worker)
                worker.start()
                assert launched.wait(1)
                # This executes inside the real authority mutation. A
                # check-then-act gate lets the competing freeze win here.
                assert not frozen.wait(0.05)
                return original(*args, **kwargs)

            monkeypatch.setattr(live.controls, 'respond', interleave)
            response = await client.post(f'/v1/runs/{run_id}/approval',
                headers={'Authorization': f"HermesRoom {invitation['grant']}"},
                json={'request_id': 'approval-race', 'execution_generation': row['generation'], 'choice': 'once'})
            assert response.status == 200, await response.text()
            assert pending.result == 'once'
            assert await asyncio.to_thread(frozen.wait, 2)
            assert adapter._run_idempotency_store.is_scope_frozen(adapter._run_owners[run_id])
            await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 3)
    finally:
        pending.event.set()
        for worker in workers:
            worker.join(timeout=2)
            assert not worker.is_alive()
