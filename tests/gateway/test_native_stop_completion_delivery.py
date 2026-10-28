"""Stop cancels completions in gateway batches and at native FIFO admission boundaries."""
import asyncio
import shlex
import sys
import threading

import pytest

from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
from tests.gateway.test_native_stop_reconciliation import caller, stop
from tests.waiting import async_wait_until


async def owner_with_completion(tmp_path, monkeypatch):
    from gateway.platforms.event import MessageEvent
    from tools import process_registry as module
    runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    receipt = await _admit_human(runner, authority, MessageEvent(text='seed', source=source, message_id='seed'))
    ref = receipt.ref
    actor = Principal('42', authority.profile_id, frozenset({'session:read', 'session:submit', 'session:control'}), 'fixture')
    await authority.cancel_queued(actor, ref, receipt.admission_id)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    route = authority.sessions[ref.session_id].route
    own = registry.spawn_local(shlex.join([sys.executable, '-c', 'print("retained stopped output")']),
        cwd=str(tmp_path), task_id=ref.session_id, owner_task_id=ref.session_id, session_key=route)
    own.notify_on_complete = True
    assert await asyncio.to_thread(own._completion_event.wait, 10)
    event = {'type': 'completion', 'session_id': own.id, 'session_key': route,
             'parent_session_id': ref.session_id, 'started_at': own.started_at, 'platform': 'discord',
             'chat_type': 'dm', 'chat_id': '42', 'user_id': '42', 'output': own.output_buffer,
             'command': own.command, 'exit_code': 0, 'completion_reason': 'exited'}
    return runner, adapter, authority, source, ref, actor, registry, own, event


@pytest.mark.asyncio
async def test_native_stop_filters_actual_completion_batch_before_admission(tmp_path, monkeypatch):
    runner, _adapter, authority, _source, ref, actor, registry, own, event = await owner_with_completion(tmp_path, monkeypatch)
    release = asyncio.Event()
    flush = runner._flush_process_completion_batch

    async def held_batch(key):
        await release.wait()
        await flush(key)

    monkeypatch.setattr(runner, '_flush_process_completion_batch', held_batch)
    task = asyncio.create_task(runner._enqueue_process_completion_notification('retained stopped output', event))
    try:
        await async_wait_until(lambda: bool(getattr(runner, '_completion_notification_batches', {})), timeout=10)
        assert 'result' in await stop(caller(authority, actor), ref)
        release.set()
        assert await asyncio.wait_for(task, 10) is None
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        assert len(rows) == 1 and rows[0]['request_id'] == 'seed'
        assert registry.is_completion_consumed(own.id) and own.exit_code == 0
        assert 'retained stopped output' in own.output_buffer
    finally:
        release.set()
        await runner._cancel_process_completion_batch_tasks()
        await task
        authority.db.close()


@pytest.mark.asyncio
async def test_stop_during_native_admission_write_refuses_consumed_completion(tmp_path, monkeypatch):
    from gateway import session_automation
    runner, _adapter, authority, _source, ref, actor, _registry, _own, event = await owner_with_completion(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    admit = session_automation.admit_session_input

    def held_write(*args, **kwargs):
        if kwargs['principal_id'].startswith('automation:'):
            entered.set()
            assert release.wait(10)
        return admit(*args, **kwargs)

    monkeypatch.setattr(session_automation, 'admit_session_input', held_write)
    task = asyncio.create_task(runner._deliver_completion_notification('retained stopped output', event))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert 'result' in await stop(caller(authority, actor), ref)
        release.set()
        assert await asyncio.wait_for(task, 10) is False
        assert len(list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)) == 1
    finally:
        release.set()
        await task
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('refuse_withdraw', [False, True], ids=['withdrawn', 'withdrawal-retry'])
async def test_stop_withdraws_already_queued_completion_and_runs_human_follower(tmp_path, monkeypatch, refuse_withdraw):
    from gateway import session_automation
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import SessionAuthority
    schedule = SessionAuthority._schedule
    runner, _adapter, authority, source, ref, actor, registry, own, event = await owner_with_completion(tmp_path, monkeypatch)
    executed = []

    async def execute(owner, target, row):
        executed.append(row['request_id'])
        return 'human answer'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        assert await runner._enqueue_process_completion_notification('retained stopped output', event) is True
        await _admit_human(runner, authority, MessageEvent(text='continue', source=source, message_id='human-follower'))
        withdraw = session_automation._withdraw
        kill, kill_calls = registry.kill_process, []
        if refuse_withdraw:
            async def refused(owner, row):
                return False
            monkeypatch.setattr(session_automation, '_withdraw', refused)
            def observed_kill(*args, **kwargs):
                kill_calls.append(args[0])
                return kill(*args, **kwargs)
            monkeypatch.setattr(registry, 'kill_process', observed_kill)
            response = await stop(caller(authority, actor), ref)
            assert response['error']['data']['reason'] == 'process_stop_unknown'
            assert registry.is_completion_consumed(own.id) and own.exited
            assert executed == []
            assert kill_calls == [own.id]
            kill_calls.clear()
            monkeypatch.setattr(session_automation, '_withdraw', withdraw)
        assert 'result' in await stop(caller(authority, actor), ref)
        assert kill_calls == [], 'withdrawal recovery must work without another physical kill'
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        completion = next(row for row in rows if row['principal_id'].startswith('automation:'))
        assert (completion['status'], completion['outcome']) == ('terminal', 'cancelled')
        monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
        authority._schedule(ref)
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 10)
        assert executed == ['human-follower']
        assert own.exit_code == 0 and 'retained stopped output' in own.output_buffer
        assert registry.is_completion_consumed(own.id)
    finally:
        await runner._cancel_process_completion_batch_tasks()
        authority.db.close()
