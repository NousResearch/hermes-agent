"""Known turn results settle independently of native resource cleanup uncertainty."""
import asyncio
import shlex
import sys
import threading
import pytest
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['live', 'restart', 'unknown-execution', 'failed-settlement'])
async def test_active_stop_result_then_cleanup_wakes_human(tmp_path, monkeypatch, boundary):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    from tests.waiting import async_wait_until
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority.runner, '_cached_agent_for', lambda route: None, raising=False)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    reader_entered, release_reader = threading.Event(), threading.Event()
    ingest = registry._ingest_output
    def pause(session, text, **kwargs):
        if 'original output' in text:
            reader_entered.set()
            assert release_reader.wait(30)
        ingest(session, text, **kwargs)
    monkeypatch.setattr(registry, '_ingest_output', pause)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local(shlex.join([sys.executable, '-c',
            'print("original output",flush=True); raise SystemExit(7)']),
            cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    started, finish_turn = asyncio.Event(), asyncio.Event()
    executed=[]
    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        if row['request_id']=='original-human':
            started.set()
            await finish_turn.wait()
            owner.pending_results[row['admission_id']]={'result': {'final_response':'original stopped result', 'interrupted':True}, 'usage':{}}
            if boundary == 'unknown-execution':
                from gateway.session_managed_worker import ManagedExecutionUnknown
                from gateway.session_settlement_recovery import recover_failed_settlement
                await recover_failed_settlement(owner, row)
                raise ManagedExecutionUnknown('independent worker fate unknown')
            return 'original stopped result'
        return 'follower done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        assert await asyncio.to_thread(reader_entered.wait, 5)
        await asyncio.to_thread(own.process.wait, 5)
        assert job_quiescent(own)
        await _submit(authority,'original-human')
        await asyncio.wait_for(started.wait(),5)
        await _submit(authority,'queued-human')
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR,REF,authority._handle(REF).execution_generation)
        if boundary == 'failed-settlement':
            import sqlite3
            from gateway import session_results
            settle = session_results.settle_session_input
            def refuse(*args, **kwargs):
                raise sqlite3.OperationalError('finite original settlement store refusal')
            monkeypatch.setattr(session_results, 'settle_session_input', refuse)
        finish_turn.set()
        await asyncio.wait_for(authority.sessions['s'].task,10)
        print('AFTER_TURN',[(r['request_id'],r['status']) for r in list_session_admissions(db,session_id='s',pending_only=False)])
        original = next(row for row in list_session_admissions(db, session_id='s', pending_only=False)
                        if row['request_id'] == 'original-human')
        if boundary in {'unknown-execution', 'failed-settlement'}:
            assert original['status'] == 'unknown'
        else:
            from gateway.session_results import admission_result
            assert (original['status'], original['outcome']) == ('terminal', 'interrupted')
            saved = admission_result(db, original['admission_id'])
            assert saved['result']['final_response'] == 'original stopped result'
            assert saved['result']['interrupted'] is True
            assert original['admission_id'] not in authority.pending_results
        if boundary == 'restart':
            from gateway.session_authority import SessionAuthority, LiveSession
            from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
            epoch = begin_runtime_epoch(db, instance_id='after-known-result')
            recover_session_inputs(db, epoch=epoch)
            old = authority
            authority = SessionAuthority(old.runner, profile_id='owned', instance_id='after-known-result', db=db, epoch=epoch)
            authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
            recovered = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', recovered)
            assert recovered.recover_from_checkpoint() == 1
            current = recovered.get(own.id)
            authority._schedule(REF)
            await asyncio.wait_for(authority.sessions['s'].task, 10)
            # A cold reader can already collect the real log after physical quiescence;
            # its complete durable receipt, rather than the retired reader, gates work.
            assert current._completion_event.is_set() and job_quiescent(current)
            assert current.output_buffer.count('original output') == 1
            assert current._result_persisted_generation == current.ownership_generation
        else:
            current = own
        release_reader.set()
        assert await asyncio.to_thread(own._completion_event.wait,5)
        assert job_quiescent(own)
        assert own._result_persisted_generation==own.ownership_generation
        await async_wait_until(lambda: process_stop_pending(db,'s') is None,timeout=5)
        assert await asyncio.to_thread(current._completion_event.wait, 5)
        await async_wait_until(lambda: authority.sessions['s'].task.done(), timeout=5)
        if boundary in {'unknown-execution', 'failed-settlement'}:
            assert executed == ['original-human']
            original = next(row for row in list_session_admissions(db, session_id='s', pending_only=False)
                            if row['request_id'] == 'original-human')
            assert original['status'] == 'unknown'
            if boundary == 'failed-settlement':
                assert original['admission_id'] in authority.pending_results
                monkeypatch.setattr(session_results, 'settle_session_input', settle)
                await authority.resolve_unknown(ACTOR, REF, original['admission_id'], original['generation'])
                await async_wait_until(lambda: executed == ['original-human', 'queued-human'], timeout=5)
            return
        await async_wait_until(lambda: executed == ['original-human', 'queued-human'],timeout=5)
        assert process_stop_pending(db,'s') is None
    finally:
        finish_turn.set(); release_reader.set()
        await asyncio.to_thread(own._completion_event.wait,5)
        await asyncio.to_thread(registry.kill_process,own.id)
        db.close()
