"""Independent review repro: rejected withdrawal must stay fenced after owner reload."""
import asyncio
from dataclasses import replace
import json
import pytest
from gateway.session_ancillary import process_stop_pending, retry_process_stop
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_native_stop_completion_delivery import owner_with_completion
from tests.gateway.test_native_stop_reconciliation import caller, stop


@pytest.mark.asyncio
async def test_withdrawal_refusal_survives_owner_reload(tmp_path, monkeypatch):
    from gateway import session_automation
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.platforms.event import MessageEvent
    from tests.gateway.test_automation_compression_media import _admit_human
    from tools import process_registry as module
    schedule = SessionAuthority._schedule
    runner, _adapter, authority, source, ref, actor, registry, own, event = await owner_with_completion(tmp_path, monkeypatch)
    try:
        assert await runner._enqueue_process_completion_notification('retained stopped output', event) is True
        await _admit_human(runner, authority, MessageEvent(text='continue', source=source, message_id='human-follower'))
        withdraw = session_automation._withdraw
        async def refused(owner, row):
            return False
        monkeypatch.setattr(session_automation, '_withdraw', refused)
        response = await stop(caller(authority, actor), ref)
        assert response['error']['data']['reason'] == 'process_stop_unknown'
        assert process_stop_pending(authority.db, ref.session_id) is not None
        assert registry.is_completion_consumed(own.id)
        monkeypatch.setattr(session_automation, '_withdraw', withdraw)
        # Recreate the native process-local registry and initialize a new authority epoch.
        monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
        old_epoch = authority.epoch
        authority = await initialize_session_authority(runner, profile_id=authority.profile_id,
                                                       instance_id='review-restarted-owner', db=authority.db)
        assert authority.epoch != old_epoch
        await authority.resolve(actor, ref)
        assert await retry_process_stop(authority, ref)
        assert process_stop_pending(authority.db, ref.session_id) is None
        executed = []
        async def execute(owner, target, row):
            executed.append(row['request_id'])
            return 'review fixture answer'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
        authority._schedule(ref)
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 10)
        rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
        completion = next(row for row in rows if row['principal_id'].startswith('automation:'))
        evidence = {'old_epoch': old_epoch, 'new_epoch': authority.epoch, 'executed': executed,
                    'completion_status': completion['status'], 'completion_outcome': completion['outcome'],
                    'stop_fence': process_stop_pending(authority.db, ref.session_id)}
        print(json.dumps(evidence))
        assert executed == ['human-follower'], evidence
        assert (completion['status'], completion['outcome']) == ('terminal', 'cancelled'), completion
    finally:
        await runner._cancel_process_completion_batch_tasks()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('refuse_result_write', [False, True], ids=['ready-completion', 'result-write-retry'])
async def test_ready_completion_stays_consumed_after_owner_reload(tmp_path, monkeypatch, refuse_result_write):
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from gateway.platforms.event import MessageEvent
    from tests.gateway.test_automation_compression_media import _admit_human
    from tools import process_registry as module, process_registry_results as results

    schedule = SessionAuthority._schedule
    runner, _adapter, authority, source, ref, actor, _registry, own, event = await owner_with_completion(tmp_path, monkeypatch)
    original_output, original_exit = own.output_buffer, own.exit_code
    write = results.atomic_json_write
    try:
        await _admit_human(runner, authority, MessageEvent(text='continue', source=source, message_id='human-follower'))
        if refuse_result_write:
            def refused(*args, **kwargs):
                raise PermissionError('fixture process result write refusal')
            monkeypatch.setattr(results, 'atomic_json_write', refused)
            response = await stop(caller(authority, actor), ref)
            assert response['error']['data']['reason'] == 'process_stop_unknown'
            assert process_stop_pending(authority.db, ref.session_id) is not None
            monkeypatch.setattr(results, 'atomic_json_write', write)
        else:
            assert 'result' in await stop(caller(authority, actor), ref)
        monkeypatch.setattr(module, 'process_registry', module.ProcessRegistry())
        previous_epoch = authority.epoch
        authority = await initialize_session_authority(runner, profile_id=authority.profile_id,
            instance_id='reloaded-ready-completion', db=authority.db)
        assert authority.epoch != previous_epoch
        await authority.resolve(actor, ref)
        assert await retry_process_stop(authority, ref)
        assert process_stop_pending(authority.db, ref.session_id) is None
        assert module.process_registry.is_completion_consumed(own.id)
        # A retiring reader may publish a snapshot captured before Stop. Its
        # normal native result write cannot unconsume the committed completion.
        assert results.save_completed_result(replace(own, completion_consumed=False))
        assert module.process_registry.is_completion_consumed(own.id)
        assert await runner._enqueue_process_completion_notification(original_output, event) is None
        assert not any(row['principal_id'].startswith('automation:') for row in
            list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False))
        retained = results.load_completed_results(owns_entry=lambda row: row['id'] == own.id, strict=True)[own.id]
        assert retained.output_buffer == original_output and retained.exit_code == original_exit
        executed = []
        async def execute(owner, target, row):
            executed.append(row['request_id'])
            return 'human answer'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        monkeypatch.setattr(SessionAuthority, '_schedule', schedule)
        authority._schedule(ref)
        await asyncio.wait_for(authority.sessions[ref.session_id].task, 10)
        assert executed == ['human-follower']
    finally:
        await runner._cancel_process_completion_batch_tasks()
        authority.db.close()
