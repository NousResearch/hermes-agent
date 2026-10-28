"""Unknown physical cleanup cannot rewrite an independently validated turn result."""
import asyncio
import json
import shlex
import sqlite3
import sys

import pytest

from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('restart', [False, True], ids=['live', 'cold-owner'])
async def test_known_answer_survives_actual_unreadable_process_ownership(tmp_path, monkeypatch, restart):
    from gateway.session_ancillary import process_stop_pending, record_process_stop
    from gateway.session_context import set_session_vars, clear_session_vars
    from gateway.session_results import admission_result
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    from utils import atomic_json_write
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        neighbor = registry.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(tmp_path), task_id='foreign-worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    effect = tmp_path / 'effect'
    payload = f'from pathlib import Path; Path({str(effect)!r}).write_text("once"); print("physical original output"); raise SystemExit(7)'
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local(shlex.join([sys.executable, '-c', payload]),
            cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    started, finish = asyncio.Event(), asyncio.Event()
    executed = []
    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        if row['request_id'] == 'original':
            started.set()
            await finish.wait()
            owner.pending_results[row['admission_id']] = {
                'result': {'final_response': 'KNOWN_COMPLETED_ANSWER', 'completed': True}, 'usage': {}}
            record_process_stop(owner, ref, row['generation'])
            owner.delivered_stops[ref.session_id] = row['generation']
            return 'KNOWN_COMPLETED_ANSWER'
        assert job_quiescent(own) and neighbor.process.poll() is None
        return 'human continued'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    path, saved = tmp_path / 'logs' / 'process-results' / (own.id + '.json'), None
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert job_quiescent(own) and own.exit_code == 7
        saved = json.loads(path.read_text())
        path.write_text('{unreadable native retained ownership')
        receipt = await _submit(authority, 'original')
        await asyncio.wait_for(started.wait(), 10)
        await _submit(authority, 'queued-human')
        finish.set()
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        rows = {r['request_id']: r for r in list_session_admissions(db, session_id='s', pending_only=False)}
        assert (rows['original']['status'], rows['original']['outcome']) == ('terminal', 'completed')
        answer = admission_result(db, receipt.admission_id)['result']
        assert answer['final_response'] == 'KNOWN_COMPLETED_ANSWER' and answer['completed'] is True
        assert answer.get('failed') is not True
        assert rows['queued-human']['status'] == 'queued' and executed == ['original']
        assert process_stop_pending(db, 's') and neighbor.process.poll() is None
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        if restart:
            from gateway.session_authority import SessionAuthority, LiveSession
            from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
            old = authority
            epoch = begin_runtime_epoch(db, instance_id='after-known-result')
            recover_session_inputs(db, epoch=epoch)
            authority = SessionAuthority(old.runner, profile_id='owned', instance_id='after-known-result', db=db, epoch=epoch)
            authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
            cold = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', cold)
            cold.recover_from_checkpoint()
            authority._schedule(REF)
            await asyncio.wait_for(authority.sessions['s'].task, 10)
            assert executed == ['original'] and process_stop_pending(db, 's')
            assert admission_result(db, receipt.admission_id)['result']['completed'] is True
        atomic_json_write(path, saved, mode=0o600)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await async_wait_until(lambda: executed == ['original', 'queued-human'] and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None and neighbor.process.poll() is None
        assert effect.read_text() == 'once'
        assert json.loads(path.read_text())['output'].count('physical original output') == 1
    finally:
        finish.set()
        if saved is not None:
            atomic_json_write(path, saved, mode=0o600)
        await asyncio.to_thread(registry.kill_process, neighbor.id)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['cleanup-unknown', 'storage-unavailable', 'stale-generation', 'sqlite', 'os-error', 'clear-write'])
async def test_retry_normalizes_only_typed_physical_cleanup_unknown(tmp_path, monkeypatch, fault):
    from gateway import session_ancillary as ancillary
    db, authority = _authority(tmp_path, monkeypatch)
    ancillary.record_process_stop(authority, REF, 0)
    error = {'cleanup-unknown': RuntimeStoreError('process_stop_unknown'),
             'storage-unavailable': RuntimeStoreError('storage_unavailable'),
             'stale-generation': RuntimeStoreError('stale_generation'),
             'sqlite': sqlite3.OperationalError('native store unavailable'),
             'os-error': OSError('native store unavailable'),
             'clear-write': RuntimeStoreError('process_stop_unknown')}[fault]
    async def interrupt(*args):
        if fault == 'clear-write':
            return True
        raise error
    monkeypatch.setattr(ancillary, 'interrupt_processes', interrupt)
    if fault == 'clear-write':
        def refuse_clear(*args):
            raise error
        monkeypatch.setattr(ancillary, 'clear_process_stop', refuse_clear)
    try:
        if fault == 'cleanup-unknown':
            assert await ancillary.retry_process_stop(authority, REF) is False
        else:
            with pytest.raises(type(error)) as raised:
                await ancillary.retry_process_stop(authority, REF)
            assert raised.value is error
        assert ancillary.process_stop_pending(db, 's')
    finally:
        db.close()


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('wake', ['owned', 'none', 'foreign'])
async def test_real_owned_resource_readiness_wake_during_refused_claim(tmp_path, monkeypatch, wake):
    from gateway import session_ancillary as ancillary
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    from utils import atomic_json_write
    db, authority = _authority(tmp_path, monkeypatch)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        neighbor = registry.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(tmp_path), task_id='foreign-worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        own = registry.spawn_local(shlex.join([sys.executable, '-c',
            'print("original physical output"); raise SystemExit(7)']), cwd=str(tmp_path), task_id='worker', session_key='route')
    finally:
        clear_session_vars(tokens)
    path, saved = tmp_path / 'logs' / 'process-results' / (own.id + '.json'), None
    retry, attempts, executed = ancillary.retry_process_stop, [], []
    async def ready_during_refusal(owner, ref, agent=None):
        verdict = await retry(owner, ref, agent)
        attempts.append(verdict)
        if len(attempts) == 1:
            assert verdict is False and ancillary.process_stop_pending(db, 's')
            assert job_quiescent(own) and own._completion_event.is_set()
            assert own._result_complete and own._result_persisted_generation == own.ownership_generation
            # The real old resource is collected. Restore its actual durable
            # ownership and deliver the existing native wake before returning
            # the already-observed refusal. No positive cleanup is mocked.
            atomic_json_write(path, saved, mode=0o600)
            if wake != 'none':
                owner.wake_after_worker('s' if wake == 'owned' else 'foreign-parent')
        return verdict
    async def execute(owner, ref, row):
        assert job_quiescent(own) and neighbor.process.poll() is None
        assert own.output_buffer.count('original physical output') == 1
        executed.append(row['request_id'])
        return 'human continued once'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(ancillary, 'retry_process_stop', ready_during_refusal)
    try:
        assert await asyncio.to_thread(own._completion_event.wait, 10)
        assert job_quiescent(own) and own.exit_code == 7
        saved = json.loads(path.read_text())
        path.write_text('{unreadable native retained ownership')
        ancillary.record_process_stop(authority, REF, 0)
        await _submit(authority, 'only-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        if wake != 'owned':
            assert attempts == [False] and executed == []
            assert ancillary.process_stop_pending(db, 's')
            rows = list_session_admissions(db, session_id='s', pending_only=False)
            assert [(r['request_id'], r['status']) for r in rows] == [('only-human', 'queued')]
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await async_wait_until(lambda: executed == ['only-human'] and authority.sessions['s'].task.done(), timeout=10)
        assert ancillary.process_stop_pending(db, 's') is None and neighbor.process.poll() is None
        assert json.loads(path.read_text())['output'].count('original physical output') == 1
    finally:
        if saved is not None:
            atomic_json_write(path, saved, mode=0o600)
        await asyncio.to_thread(registry.kill_process, neighbor.id)
        db.close()
