"""Result framing, enrollment boundaries and owner isolation; no live services."""
import asyncio
import json
import shlex
import sys
import threading

import pytest

from tests.gateway.test_process_work_results import owner, enrolled, frame


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['work_id', 'incarnation', 'evidence'])
async def test_escaped_lone_surrogate_does_not_abort_following_result(monkeypatch, field):
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    value = dict(work_id='build', incarnation='1', status='completed', evidence='ok')
    value[field] = '\ud800'
    bad = 'HERMES_WORK_RESULT_V1 ' + json.dumps(value) + '\n'
    registry._ingest_output(source, bad + frame())
    # Evidence is opaque, JSON-escaped storage; a different later body correctly conflicts.
    assert records(authority.db)[0]['disposition'] == ('conflict' if field == 'evidence' else 'pending')


@pytest.mark.asyncio
async def test_actual_reader_survives_unknown_surrogate_binding(monkeypatch):
    from gateway.session_context import scoped_current_session_id
    from tools.process_registry import ProcessRegistry
    from tools.process_work_results import records
    _, authority, key, sid = await owner(monkeypatch)
    registry = ProcessRegistry()
    failures = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: failures.append(repr(args.exc_value)))
    code = 'print(' + repr(frame(work='\ud800') + frame()) + ', end="", flush=True)'
    with scoped_current_session_id(sid):
        source = registry.spawn_local(shlex.quote(sys.executable) + ' -c ' + shlex.quote(code),
            session_key=key, completion_work=[{'work_id': 'build', 'incarnation': '1'}])
    try:
        source.process.wait(timeout=5)
        source._reader_thread.join(timeout=5)
        row = records(authority.db)[0]
        observed = (row['disposition'], source.exited, failures)
        assert observed == ('pending', True, []), repr(observed)
    finally:
        if source.process.poll() is None:
            registry.kill_process(source.id)
        source.process.wait(timeout=5)


@pytest.mark.asyncio
async def test_stop_between_enrollment_parent_check_and_commit_fences_old_work(monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_runtime import list_session_admissions
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    original = SessionDB.get_session
    raced = []

    def check_then_stop(db, session_id, *args, **kwargs):
        result = original(db, session_id, *args, **kwargs)
        if session_id == sid and not raced:
            raced.append(True)
            authority.fence_route(key)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(SessionDB, 'get_session', check_then_stop)
        # Reject before spawning, rather than launch an already-cancelled producer.
        with pytest.raises(ValueError, match='enrollment crossed a user boundary'):
            enrolled(key, sid)
    assert records(authority.db) == []
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 1
    # Fresh work explicitly begun after Stop is not permanently disabled.
    registry, source = enrolled(key, sid)
    registry._ingest_output(source, frame())
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 2


@pytest.mark.asyncio
async def test_conflict_after_pending_snapshot_before_transaction_is_not_admitted(monkeypatch):
    from gateway import process_work_results as delivery
    from hermes_state_runtime import list_session_admissions
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    registry._ingest_output(source, frame())
    real = delivery.admit_automation

    async def conflict(*args, **kwargs):
        registry._ingest_output(source, frame(evidence='contradiction'))
        return await real(*args, **kwargs)

    monkeypatch.setattr(delivery, 'admit_automation', conflict)
    await runner._drain_watch_notifications(registry.completion_queue)
    assert records(authority.db)[0]['disposition'] == 'conflict'
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1


@pytest.mark.asyncio
async def test_unrelated_parent_cannot_inject_into_live_route(monkeypatch):
    from hermes_state_runtime import list_session_admissions
    from tools.process_work_results import records
    runner, authority, key, sid = await owner(monkeypatch)
    authority.db.create_session('unrelated', source='discord')
    registry, source = enrolled(key, 'unrelated')
    registry._ingest_output(source, frame())
    await runner._drain_watch_notifications(registry.completion_queue)
    assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1
    assert records(authority.db)[0]['disposition'] != 'admitted'


def test_receipt_before_any_native_admission_table_exists():
    from contextlib import closing
    from hermes_state import SessionDB
    from tools.process_work_results import receipts
    with closing(SessionDB()) as db:
        db.create_session('persisted-parent', source='discord')
        _, source = enrolled('agent:main:discord:dm:42', 'persisted-parent')
        assert receipts(source)[0]['admission_status'] is None


@pytest.mark.asyncio
async def test_existing_idle_consumer_admits_result_without_foreground_turn(monkeypatch):
    from hermes_state_runtime import list_session_admissions
    from tools import process_registry as processes
    runner, authority, key, sid = await owner(monkeypatch)
    registry, source = enrolled(key, sid)
    monkeypatch.setattr(processes, 'process_registry', registry)
    source._watch_disabled = True
    source.watch_patterns = ['fixture-result']
    registry._ingest_output(source, frame())
    assert registry.completion_queue.empty(), 'readiness queue is silent'
    scheduled = []
    runner._running = True

    def accept_schedule(ref):
        scheduled.append(ref)
        runner._running = False

    monkeypatch.setattr(authority, '_schedule', accept_schedule)
    try:
        # Exercise the existing consumer, not a manual result-drain call. The
        # real startup delay is retained; no user turn or new watcher is added.
        await asyncio.wait_for(runner._async_delegation_watcher(interval=.01), timeout=8)
    finally:
        runner._running = False
    assert len(scheduled) == 1
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 2
    assert 'fixture-result' in rows[-1]['payload']['text']
