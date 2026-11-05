"""Lifecycle regressions; real finite children, synthetic owners and fault injection.
No network/model execution, signals, live gateway, or production-home access.
"""
import asyncio
import json
import shlex
import sqlite3
import sys
import threading

import pytest

from tests.gateway.test_process_work_results import owner, enrolled, frame
from tests.waiting import async_wait_until


def command(text, delay=0.15, linger=0):
    code = f'import time; time.sleep({delay}); print({text!r}, end="", flush=True); time.sleep({linger})'
    return shlex.quote(sys.executable) + ' -c ' + shlex.quote(code)


def spawn(registry, key, sid, text, **kwargs):
    from gateway.session_context import scoped_current_session_id
    with scoped_current_session_id(sid):
        return registry.spawn_local(command(text, **kwargs), session_key=key,
            completion_work=[{'work_id': 'build', 'incarnation': '1'}])


def finish(source):
    source.process.wait(timeout=5)
    source._reader_thread.join(timeout=5)
    assert not source._reader_thread.is_alive()


@pytest.mark.asyncio
async def test_public_still_running_result_under_actual_lifetime_cap(monkeypatch, tmp_path):
    from gateway.session_context import scoped_current_session_id
    from tools import process_registry as processes
    from tools.terminal_tool import _handle_terminal
    from tools.process_work_results import records, receipts
    runner, authority, key, sid = await owner(monkeypatch)
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, 'process_registry', registry)
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    # Synthetic clock, real unchanged scanner cap. No cap constants are relaxed.
    with scoped_current_session_id(sid):
        result = json.loads(await asyncio.to_thread(_handle_terminal, {
            'command': command(frame(), delay=1, linger=2), 'background': True,
            'workdir': str(tmp_path),
            'completion_work': [{'work_id': 'build', 'incarnation': '1'}],
        }, task_id=key))
    assert not result.get('error'), result
    source = registry.get(result['session_id'])
    try:
        source.watch_patterns = ['READY']
        now = 1000.0
        with monkeypatch.context() as patch:
            patch.setattr(processes.time, 'time', lambda: now)
            for _ in range(processes.WATCH_LIFETIME_MAX_HITS):
                registry._check_watch_patterns(source, 'READY\n')
                now += processes.WATCH_MIN_INTERVAL_SECONDS + 1
        assert source._watch_disabled
        while not registry.completion_queue.empty():
            registry.completion_queue.get_nowait()
        await async_wait_until(lambda: records(authority.db)[0]['disposition'] == 'pending',
                               timeout=5, what='durable result while the child runs')
        assert source.process.poll() is None
        assert registry.completion_queue.empty()
        await runner._drain_watch_notifications(registry.completion_queue)
        receipt = receipts(source)[0]
        assert receipt['disposition'] == 'admitted'
        assert receipt['acceptance'] == 'not_verified'
        assert source.process.poll() is None, 'must admit before exit'
    finally:
        finish(source)


@pytest.mark.asyncio
async def test_real_reader_rejects_malformed_evidence_then_accepts_valid(monkeypatch):
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    bad = frame(evidence={'not': 'text'}) + frame(evidence='x' * 4097)
    source = spawn(ProcessRegistry(), key, sid, bad + frame())
    finish(source)
    assert source.exited
    assert records(authority.db)[0]['disposition'] == 'pending'


@pytest.mark.asyncio
@pytest.mark.parametrize('text,expected', [(frame(), 'pending'), (frame() + frame(), 'pending'),
    (frame() + frame(evidence='conflict'), 'conflict')])
async def test_transient_storage_failure_preserves_single_frame_and_order(monkeypatch, text, expected):
    from hermes_state import SessionDB
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    original = SessionDB._execute_write
    injected = []
    failures = []
    def fail_once(db, callback, *args, **kwargs):
        if threading.current_thread().name.startswith('proc-reader-') and not injected:
            injected.append('sqlite OperationalError at durable result write')
            raise sqlite3.OperationalError('synthetic transient storage failure')
        return original(db, callback, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_execute_write', fail_once)
    monkeypatch.setattr(threading, 'excepthook', lambda args: failures.append(repr(args.exc_value)))
    source = spawn(ProcessRegistry(), key, sid, text)
    finish(source)
    assert injected
    observed = (source.exited, records(authority.db)[0]['disposition'], failures)
    assert observed == (True, expected, []), repr(observed)


@pytest.mark.asyncio
async def test_single_frame_storage_recovery_while_child_still_running(monkeypatch):
    from hermes_state import SessionDB
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import receipts
    runner, _, key, sid = await owner(monkeypatch)
    real = SessionDB._execute_write
    attempts = []
    def fault(db, callback, *args, **kwargs):
        if threading.current_thread().name.startswith('proc-reader-'):
            attempts.append(1)
            if len(attempts) == 1:
                raise sqlite3.OperationalError('one fault')
        return real(db, callback, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_execute_write', fault)
    registry = ProcessRegistry()
    source = spawn(registry, key, sid, frame(evidence='single live result'), linger=2)
    try:
        await async_wait_until(lambda: receipts(source)[0]['disposition'] == 'pending',
                               timeout=5, what='single frame storage recovery')
        await runner._drain_watch_notifications(registry.completion_queue)
        assert receipts(source)[0]['disposition'] == 'admitted'
        assert not source.exited and source.process.poll() is None
    finally:
        finish(source)


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', [False, True])
async def test_storage_exhaustion_finishes_process_not_work_and_retains_receipt(monkeypatch, stop):
    from pathlib import Path
    from hermes_state import SessionDB
    from gateway.session_context import scoped_current_session_id
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import receipts, records
    runner, authority, key, sid = await owner(monkeypatch)
    real = SessionDB._execute_write
    attempts, failures = [], []
    def fault(db, callback, *args, **kwargs):
        if threading.current_thread().name.startswith('proc-reader-'):
            attempts.append(1)
            raise sqlite3.OperationalError('persistent storage outage')
        return real(db, callback, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_execute_write', fault)
    monkeypatch.setattr(threading, 'excepthook', lambda args: failures.append(repr(args.exc_value)))
    registry = ProcessRegistry()
    source = spawn(registry, key, sid, frame(), linger=.2)
    finish(source)
    assert source.exited and source.exit_code == 0
    assert failures == []
    assert 1 < len(attempts) <= 4
    assert frame() in Path(source.output_log).read_text()
    assert registry.poll(source.id)['work_result_error'] == 'storage_retry_exhausted'
    if stop:
        authority.fence_route(key)
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == ('dropped' if stop else 'enrolled')
    with scoped_current_session_id(sid):
        reopened = ProcessRegistry()
        status = reopened.poll(source.id)
        assert status['work_result_error'] == 'storage_retry_exhausted'
        restored = reopened.get(source.id)
        assert restored.output_log == source.output_log
        assert receipts(restored)[0]['acceptance'] == 'not_verified'
        assert reopened.completion_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['ordinary completed text\n', frame().rstrip('\n')])
async def test_exit_without_frame_is_supervision_failure_not_task_result(monkeypatch, text):
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records, receipts
    from gateway.session_context import scoped_current_session_id
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    registry = ProcessRegistry()
    source = spawn(registry, key, sid, text)
    finish(source)
    receipt = receipts(source)[0]
    assert receipt['disposition'] == 'supervision_failed'
    assert receipt['error'] == 'missing_terminal_result'
    assert 'status' not in receipt
    assert records(authority.db)[0]['payload'] is None
    assert source.exit_code == 0
    with scoped_current_session_id(sid):
        status = ProcessRegistry().poll(source.id)
        assert status['work_results'][0] == receipt
    await runner._drain_watch_notifications(registry.completion_queue)
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 2
    assert json.loads(rows[-1]['request_id'])[0] == 'process_work_supervision'
    delivered = receipts(source)[0]
    assert delivered['disposition'] == 'supervision_failed_admitted'
    assert delivered['supervision'] == 'supervision_failed'
    assert delivered['error'] == receipt['error'] and 'status' not in delivered
    assert delivered['admission_status'] == 'queued'


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [False, True])
async def test_definite_os_spawn_failure_has_literal_lifecycle_receipt(monkeypatch, tmp_path, owned):
    from tools import process_registry as processes
    from tools.terminal_tool import _handle_terminal
    from gateway.session_context import scoped_current_session_id
    from hermes_state_runtime import list_session_admissions
    from tools.process_work_results import records
    import subprocess
    runner, authority, key, sid = await owner(monkeypatch)
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, 'process_registry', registry)
    monkeypatch.setattr(registry, '_owns_job', lambda _: owned)
    monkeypatch.setenv('TERMINAL_ENV', 'local')
    attempts = []
    real_popen = subprocess.Popen
    def cannot_spawn(*args, **kwargs):
        if 'must-not-execute' not in str(args):
            return real_popen(*args, **kwargs)
        attempts.append(1)
        raise OSError('synthetic definite OS spawn failure')
    monkeypatch.setattr(subprocess, 'Popen', cannot_spawn)
    work = {'work_id': ' literal work ', 'incarnation': ' 01 '}
    with scoped_current_session_id(sid):
        result = json.loads(await asyncio.to_thread(_handle_terminal, {
            'command': 'must-not-execute', 'workdir': str(tmp_path), 'background': True,
            'completion_work': [work]}, task_id=key))
    assert result.get('error'), result
    row = records(authority.db)[0]
    assert row['disposition'] == 'never_started'
    assert row['payload'] is None
    assert result['session_id'] == row['process_id']
    assert result['work_results'][0]['work_id'] == work['work_id']
    assert result['work_results'][0]['incarnation'] == work['incarnation']
    assert result['work_results'][0]['acceptance'] == 'not_verified'
    assert len(attempts) == 1
    authority.fence_route(key)
    with scoped_current_session_id(sid):
        status = processes.ProcessRegistry().poll(row['process_id'])
        assert status['work_results'][0]['disposition'] == 'dropped'
        assert status['work_results'][0]['supervision'] == 'never_started'
        assert status['work_results'][0]['error'] == 'os_spawn_failed'
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', [False, True])
async def test_retry_after_ambiguous_commit_is_idempotent_and_stop_wins(monkeypatch, boundary):
    from hermes_state import SessionDB
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    from hermes_state_runtime import list_session_admissions
    runner, authority, key, sid = await owner(monkeypatch)
    real = SessionDB._execute_write
    injected = []
    def committed_then_fault(db, callback, *args, **kwargs):
        result = real(db, callback, *args, **kwargs)
        if threading.current_thread().name.startswith('proc-reader-') and not injected:
            injected.append(1)
            if boundary:
                authority.fence_route(key)
            raise sqlite3.OperationalError('uncertain caller after commit')
        return result
    monkeypatch.setattr(SessionDB, '_execute_write', committed_then_fault)
    registry = ProcessRegistry()
    source = spawn(registry, key, sid, frame())
    finish(source)
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == ('dropped' if boundary else 'admitted')
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == (1 if boundary else 2)


@pytest.mark.asyncio
async def test_postspawn_ownership_error_is_not_never_started(monkeypatch):
    from hermes_cli import process_identity
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_owns_job', lambda _: True)
    children = []
    real = process_identity.subprocess.Popen
    def capture(*args, **kwargs):
        child = real(*args, **kwargs)
        children.append(child)
        return child
    def uncertain(*args, **kwargs):
        raise OSError('postspawn tracking uncertainty')
    monkeypatch.setattr(process_identity.subprocess, 'Popen', capture)
    monkeypatch.setattr(process_identity, 'own_process_groups', uncertain)
    try:
        with pytest.raises(OSError, match='postspawn tracking uncertainty') as caught:
            spawn(registry, key, sid, '', delay=0)
        assert not getattr(caught.value, '_hermes_child_not_started', False)
        assert records(authority.db)[0]['disposition'] == 'enrolled'
    finally:
        for child in children:
            child.wait(timeout=5)


@pytest.mark.asyncio
async def test_unreadable_transport_is_not_missing_evidence(monkeypatch):
    import builtins
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    real = builtins.open
    def unreadable(path, mode='r', *args, **kwargs):
        if threading.current_thread().name.startswith('proc-reader-') and mode == 'rb':
            raise OSError('log temporarily unreadable')
        return real(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', unreadable)
    registry = ProcessRegistry()
    source = spawn(registry, key, sid, frame())
    finish(source)
    assert source.exited
    assert records(authority.db)[0]['disposition'] == 'enrolled'
    assert registry.poll(source.id)['work_result_error'] == 'output_log_unreadable'


@pytest.mark.parametrize('owned', [False, True])
def test_definite_spawn_error_preserves_os_error_details(monkeypatch, owned):
    import subprocess
    from tools.process_registry import ProcessRegistry, ProcessSession
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, '_owns_job', lambda _: owned)
    def reject(*args, **kwargs):
        raise FileNotFoundError(2, 'synthetic missing executable', '/literal/missing')
    monkeypatch.setattr(subprocess, 'Popen', reject)
    with pytest.raises(FileNotFoundError) as caught:
        registry._spawn_job(ProcessSession(id='proc_no_os_child', command='test'), ['missing'], False)
    assert getattr(caught.value, '_hermes_child_not_started', False)
    assert caught.value.errno == 2
    assert caught.value.filename == '/literal/missing'


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', [False, True])
async def test_checkpoint_reopen_replays_single_frame_without_boundary_revival(monkeypatch, stop):
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    from gateway.session_context import scoped_current_session_id
    runner, authority, key, sid = await owner(monkeypatch)
    original = ProcessRegistry()
    def before_reader(session, *args, **kwargs):
        original._running[session.id] = session
        original._write_checkpoint()
    monkeypatch.setattr(original, '_track_started', before_reader)
    source = spawn(original, key, sid, frame())
    source.process.wait(timeout=5)
    assert records(authority.db)[0]['disposition'] == 'enrolled'
    if stop:
        authority.fence_route(key)
    with scoped_current_session_id(sid):
        reopened = ProcessRegistry()
        assert reopened.recover_from_checkpoint() == 1
        restored = reopened.get(source.id)
        assert restored is not None
        restored._reader_thread.join(timeout=5)
        assert not restored._reader_thread.is_alive()
        await runner._drain_watch_notifications(reopened.completion_queue)
        assert records(authority.db)[0]['disposition'] == ('dropped' if stop else 'admitted')
        again = ProcessRegistry()
        assert again.poll(source.id)['work_results'][0]['disposition'] == ('dropped' if stop else 'admitted')
        assert again.completion_queue.empty()


@pytest.mark.asyncio
async def test_fragmented_utf8_frame_retries_without_decoder_reset(monkeypatch):
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    from hermes_state import SessionDB
    from gateway.session_context import scoped_current_session_id
    _, authority, key, sid = await owner(monkeypatch)
    raw = frame(evidence='雪☃').replace('\\u96ea', '雪').replace('\\u2603', '☃').encode()
    cut = raw.index('雪'.encode()) + 1
    code = f'import os,time; os.write(1,{raw[:cut]!r}); time.sleep(.3); os.write(1,{raw[cut:]!r})'
    real = SessionDB._execute_write
    injected = []
    def fault(db, callback, *args, **kwargs):
        if threading.current_thread().name.startswith('proc-reader-') and not injected:
            injected.append(1)
            raise sqlite3.OperationalError('split UTF-8 frame storage fault')
        return real(db, callback, *args, **kwargs)
    monkeypatch.setattr(SessionDB, '_execute_write', fault)
    with scoped_current_session_id(sid):
        source = ProcessRegistry().spawn_local(shlex.quote(sys.executable) + ' -c ' + shlex.quote(code),
            session_key=key, completion_work=[{'work_id': 'build', 'incarnation': '1'}])
    finish(source)
    assert injected
    assert json.loads(records(authority.db)[0]['payload'])['evidence'] == '雪☃'


@pytest.mark.asyncio
async def test_spawn_failure_with_unwritable_disposition_keeps_identity_and_unresolved_error(monkeypatch):
    from hermes_state import SessionDB
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    from gateway.session_context import scoped_current_session_id
    _, authority, key, sid = await owner(monkeypatch)
    registry = ProcessRegistry()
    real = SessionDB._execute_write
    def fault(db, callback, *args, **kwargs):
        raise sqlite3.OperationalError('lifecycle write unavailable')
    def reject(*args, **kwargs):
        monkeypatch.setattr(SessionDB, '_execute_write', fault)
        error = OSError('definite spawn rejection')
        error._hermes_child_not_started = True  # type: ignore[attr-defined]
        raise error
    monkeypatch.setattr(registry, '_spawn_job', reject)
    with pytest.raises(OSError) as caught:
        spawn(registry, key, sid, '')
    monkeypatch.setattr(SessionDB, '_execute_write', real)
    receipt = caught.value.process_receipt
    assert receipt['session_id'] == records(authority.db)[0]['process_id']
    assert receipt['work_result_error'] == 'spawn_lifecycle_storage_unresolved'
    assert receipt['work_results'][0]['disposition'] == 'enrolled'
    with scoped_current_session_id(sid):
        status = ProcessRegistry().poll(receipt['session_id'])
        assert status['completion_reason'] == 'never_started'
        assert status['work_result_error'] == 'spawn_lifecycle_storage_unresolved'


@pytest.mark.asyncio
async def test_multiple_owner_incarnations_remain_separate_and_settlement_is_not_acceptance(monkeypatch):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input, settle_session_input, list_session_admissions
    from tools.process_work_results import receipts
    runner, authority, key, sid = await owner(monkeypatch)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42,43')
    adapter = runner.adapters[Platform.DISCORD]
    other = adapter.build_source(chat_id='43', chat_type='dm', user_id='43')
    event = MessageEvent(text='second synthetic owner', source=other, message_id='second-human')
    with native_callback(runner, event, get_hermes_home()):
        await authority.admit_native(event)
    other_key = runner.session_store._generate_session_key(other)
    other_sid = runner.session_store.peek_session_id(other_key)
    assert other_sid and other_sid != sid
    sources = []
    for index, (route, parent, incarnation) in enumerate([(key, sid, '1'), (key, sid, '2'), (other_key, other_sid, '1')]):
        registry, source = enrolled(route, parent, process=f'proc_owner_{index}',
            work=[{'work_id': 'build', 'incarnation': incarnation}])
        registry._ingest_output(source, frame(incarnation=incarnation, evidence=f'owner-{index}'))
        sources.append(source)
    await runner._drain_watch_notifications(registry.completion_queue)
    for parent, expected in [(sid, ['owner-0', 'owner-1']), (other_sid, ['owner-2'])]:
        rows = list_session_admissions(authority.db, session_id=parent, pending_only=False)
        assert len(rows) == len(expected) + 1
        assert all(token in rows[index + 1]['payload']['text'] for index, token in enumerate(expected))
        for _ in rows:
            claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=parent)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=claim['admission_id'],
                generation=claim['generation'], outcome='completed')
    for source in sources:
        receipt = receipts(source)[0]
        assert (receipt['admission_status'], receipt['admission_outcome']) == ('terminal', 'completed')
        assert receipt['acceptance'] == 'not_verified'
