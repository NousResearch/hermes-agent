"""Cron admissions retain exact job identity and stored-job model policy."""
import asyncio
from types import SimpleNamespace

import pytest


def test_cron_admission_uses_job_model_and_exact_retry(tmp_path, monkeypatch):
    from cron import jobs
    from gateway import run, session_cron
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'original', 'model': 'per-job-model'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    db = SessionDB(tmp_path / 'state.db')
    from gateway.config import GatewayConfig
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_draining=False, session_store=store, adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create', 'session:submit', 'session:control'}), 'viewer')

    async def probe():
        params = {'job_id': 'job', 'request_id': 'fire', 'extra_prompt': 'extra'}
        receipt = await session_cron.operation(authority, 'submit', params, actor)
        assert await session_cron.operation(authority, 'submit', params, actor) == receipt
        assert await session_cron.operation(authority, 'submit', params) == receipt
        with pytest.raises(RuntimeStoreError):
            await session_cron.operation(authority, 'submit', dict(params, extra_prompt='changed'), actor)
        policy = runner.adapters[next(iter(runner.adapters))].policies[receipt['session_id']]
        assert policy.model == 'per-job-model'
        assert policy.source == 'cron'

    try:
        asyncio.run(probe())
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cron_cancel_between_claim_and_execution_registration_stops_the_job(tmp_path, monkeypatch):
    """A started admission whose execute() has not registered its event yet still honours cancel."""
    from contextlib import nullcontext
    import json
    from cron import scheduler
    from gateway import run, session_cron
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='cron-owner'), 'route')
    actor = Principal('cron-owner', 'p', frozenset({'session:create', 'session:submit', 'session:control'}), 'ticker')
    ref = SessionRef('p', 's')
    monkeypatch.setattr(run, '_profile_runtime_scope', lambda home: nullcontext())
    seen = {}

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        seen['cancelled_at_start'] = cancel_event.is_set()
        return (False, '', '', 'cancelled')
    monkeypatch.setattr(scheduler, 'run_job', run_job)

    with db:
        admit_session_input(db, epoch=authority.epoch, principal_id='cron-owner', session_id='s',
                            request_id='cron:job:fire', payload={'text': ''})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        # The claim is committed and visible as `started`; execute() has not run yet.
        params = {'session_id': 's', 'admission_id': row['admission_id']}
        assert await session_cron.operation(authority, 'cancel', params, actor) == {'ok': True}
        policy = SimpleNamespace(request_json=json.dumps({
            'extra_prompt': None, 'request_id': 'cron:job:fire', 'cron_job': {'id': 'job'}}))
        with pytest.raises(RuntimeError):
            await session_cron.execute(authority, ref, row, policy)
    assert seen == {'cancelled_at_start': True}
    assert authority._cron_cancellations == {}


@pytest.mark.asyncio
async def test_owner_execution_binds_the_scheduler_cron_identity_inside_run_job(tmp_path, monkeypatch):
    """The owner runs the job on its own loop thread, which never saw the firing scheduler's
    ``enter_cron_execution``: ``ctx.current_cron_execution()`` must still name this fire, keyed on
    the scheduler's execution id (the admission's request id), never on the admission id."""
    from contextlib import nullcontext
    import json
    from cron import scheduler
    from cron.execution_identity import current_cron_execution
    from cron.executions import create_execution, mark_execution_running
    from gateway import run, session_cron
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    execution = create_execution('job', source='builtin', scheduled_instant='2026-10-08T09:00:00')
    record = mark_execution_running(execution['id'])
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', source='cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='cron-owner'), 'route')
    monkeypatch.setattr(run, '_profile_runtime_scope', lambda home: nullcontext())
    seen = {}

    def run_job(job, *, extra_prompt, execution_id, cancel_event):
        seen['execution_id'] = execution_id
        seen['identity'] = current_cron_execution()
        return (True, 'doc', 'answer', None)
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    request_id = 'cron:job:' + execution['id']
    with db:
        admit_session_input(db, epoch=authority.epoch, principal_id='cron-owner', session_id='s',
                            request_id=request_id, payload={'text': ''})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        policy = SimpleNamespace(request_json=json.dumps({
            'extra_prompt': None, 'request_id': request_id, 'cron_job': {'id': 'job', 'name': 'Job'}}))
        assert await session_cron.execute(authority, SessionRef('p', 's'), row, policy) == 'answer'
    ident = seen['identity']
    assert seen['execution_id'] == execution['id'] != row['admission_id']
    assert ident is not None and (ident.job_id, ident.job_name, ident.execution_id) == ('job', 'Job', execution['id'])
    assert (ident.source, ident.scheduled_instant, ident.started_at) == (
        'builtin', record['scheduled_instant'], record['started_at'])
    assert current_cron_execution() is None


def test_draining_owner_still_answers_status_and_cancel_of_admitted_cron_work(tmp_path, monkeypatch):
    """A stop/restart sets ``_draining`` while it waits for in-flight cron work. The in-gateway
    firer polls ``status`` until the run settles: refusing it made the firer raise
    CronExecutionUnknown and pause the job although the run completed. New fires stay fenced."""
    import threading
    from cron import jobs
    from cron.scheduler_authority import run_canonical_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import (RuntimeStoreError, begin_runtime_epoch, claim_session_input,
                                      settle_session_input)

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'p', 'model': 'm'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    admitted = threading.Event()
    monkeypatch.setattr(authority, '_schedule', lambda ref: admitted.set())
    polled_while_draining = threading.Event()
    real_operation = session_cron.operation

    async def observed_operation(owner, name, params, actor=None):
        draining = runner._draining
        result = await real_operation(owner, name, params, actor)
        if name == 'status' and draining:
            polled_while_draining.set()  # answered, not refused, during the drain
        return result
    monkeypatch.setattr(session_cron, 'operation', observed_operation)

    async def probe():
        session_cron.bind_owner(authority)
        try:
            firer = asyncio.create_task(asyncio.to_thread(run_canonical_job, {'id': 'job'}, execution_id='fire'))
            assert await asyncio.to_thread(admitted.wait, 10), firer
            runner._draining = True  # shutdown begins while the admitted run executes
            sid = next(iter(authority.sessions))
            row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
            assert await session_cron.operation(
                authority, 'cancel', {'session_id': sid, 'admission_id': row['admission_id']}) == {'ok': True}
            # the firer polls status at least once during the drain, and is answered
            assert await asyncio.to_thread(polled_while_draining.wait, 10), 'firer never polled status during the drain'
            settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed',
                                 result={'result': {'cron_result': [True, 'doc', 'answer', None]}, 'usage': {}})
            assert await asyncio.wait_for(firer, 10) == (True, 'doc', 'answer', None)
            with pytest.raises(RuntimeStoreError, match='runtime_draining'):
                await session_cron.operation(authority, 'submit',
                                             {'job_id': 'job', 'request_id': 'next', 'extra_prompt': None})
        finally:
            session_cron.unbind_owner(authority)

    try:
        asyncio.run(probe())
    finally:
        db.close()


async def _cron_owner(tmp_path, monkeypatch, run_job, *, schedule=True):
    """A real owner (authority, drain, settlement) whose fires run *run_job*; returns
    ``(authority, runner, native)``, *native* being the local operator transport Ink's picker uses."""
    from contextlib import nullcontext
    from cron import jobs, scheduler
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'name': jid, 'prompt': 'p', 'model': 'm'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_profile_runtime_scope', lambda home: nullcontext())
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _restored_source=lambda entry: entry.origin)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    if not schedule:
        authority._schedule = lambda ref: None  # admitted, never claimed: a paused queue
    session_cron.bind_owner(authority)
    native = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'native', 'profile_id': str(tmp_path), 'instance_id': 'owner'}, operator=True)
    return authority, runner, native


def _fire(authority, job_id):
    from gateway import session_cron
    return session_cron.operation(authority, 'submit', {'job_id': job_id, 'request_id': job_id + '-fire',
                                                        'extra_prompt': None})


def _row(authority, sid):
    from hermes_state_local import local_receipt
    return authority.db.get_session(local_receipt(authority.db, sid)['entry']['session_id'])


async def _listed(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return {row['id'] for row in reply['result']['sessions']}


def _gated_or_answered(job, *, extra_prompt, execution_id, cancel_event):
    """``run_job``'s two shapes: gated before an agent (no row write, no end) or answered by one."""
    if job['id'].startswith('answered'):
        from gateway.session_cron import current_execution
        db, sid = current_execution()[0].db, current_execution()[1]
        db.append_message(sid, 'user', 'p')
        db.append_message(sid, 'assistant', 'report')
        db.end_session(sid, 'cron_complete')  # _finalize_cron_session's booking
        return True, 'doc', 'report', None
    return True, 'doc', '[SILENT]', None


@pytest.mark.asyncio
async def test_a_gated_cron_fire_ends_its_row_and_is_neither_live_nor_resumable(tmp_path, monkeypatch):
    """A fire gated before any agent ran (monitor unchanged, ``wakeAgent=false``) never reaches
    ``run_job``'s ``end_session``: its row stayed open and read as a live, idle, untitled session in
    every picker (310 in one TUI switcher). The drain going idle ends it, and the stored feed (the
    switcher's "resumable" half) leaves it out; a run an agent answered keeps its own end and stays
    resumable, and a fire still running stays live."""
    import threading
    from gateway import session_cron
    started, release = threading.Event(), threading.Event()

    def run_job(job, **kwargs):
        if job['id'] == 'slow':
            started.set()
            release.wait(10)
        return _gated_or_answered(job, **kwargs)
    authority, _runner, native = await _cron_owner(tmp_path, monkeypatch, run_job)
    try:
        gated = (await _fire(authority, 'gated'))['session_id']
        answered = (await _fire(authority, 'answered'))['session_id']
        for sid in (gated, answered):
            await authority.sessions[sid].task
        assert (_row(authority, gated)['end_reason'], _row(authority, answered)['end_reason']) == (
            'cron_settled', 'cron_complete')
        stored = await _listed(native, 'session.list', scope='stored')
        assert answered in stored and gated not in stored, stored

        slow = (await _fire(authority, 'slow'))['session_id']
        assert await asyncio.to_thread(started.wait, 10)
        for method in ('session.list', 'session.active_list'):
            ids = await _listed(native, method)
            assert gated not in ids and slow in ids, (method, ids)
        release.set()
        await authority.sessions[slow].task
        for method in ('session.list', 'session.active_list'):
            assert slow not in await _listed(native, method), method
    finally:
        release.set()
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_queued_cron_fire_stays_live_until_it_runs(tmp_path, monkeypatch):
    """Guard, not a regression test for the fix: a fire admitted behind a paused queue reads ``idle``
    (no claim yet) with an open row. It is pending work and must stay listed live; no cron-specific
    listing rule may hide it as a finished run."""
    from gateway import session_cron
    authority, _runner, native = await _cron_owner(tmp_path, monkeypatch, _gated_or_answered, schedule=False)
    try:
        queued = (await _fire(authority, 'queued'))['session_id']
        for method in ('session.list', 'session.active_list'):
            assert queued in await _listed(native, method), method
        assert _row(authority, queued)['ended_at'] is None
    finally:
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('wrote_before_idle', [True, False])
async def test_a_worker_that_outlives_run_job_books_its_own_end(tmp_path, monkeypatch, wrote_before_idle):
    """An inactivity timeout returns from ``run_job`` while the agent thread still runs; its Future
    finalizes later (``scheduler_detached_worker``). The drain is idle by then: a row the agent already
    wrote to is left to it, and a stamp made before its first write yields to the run's own
    classification (#93820 ``cron_incomplete_no_output``), not first-reason-wins."""
    from cron import scheduler
    from gateway import session_cron
    late = {}

    def run_job(job, **kwargs):
        db = scheduler._open_cron_session_db(job)  # the store a real run holds until its teardown
        sid = session_cron.current_execution()[1]
        if wrote_before_idle:
            db.append_message(sid, 'user', 'p')
        late.update(db=db, sid=sid)
        return False, 'doc', '', 'TimeoutError: inactivity'  # the watchdog gave up; the worker runs on
    authority, _runner, _native = await _cron_owner(tmp_path, monkeypatch, run_job)
    try:
        sid = (await _fire(authority, 'late'))['session_id']
        await authority.sessions[sid].task
        row = _row(authority, sid)
        assert (row['ended_at'] is None, row['end_reason']) == (
            (True, None) if wrote_before_idle else (False, 'cron_settled'))
        if not wrote_before_idle:
            late['db'].append_message(late['sid'], 'user', 'p')  # its first write, after the stamp
        await asyncio.to_thread(scheduler._finalize_cron_session, late['db'], None, 'late', 'late', late['sid'])
        row = _row(authority, sid)
        assert (row['ended_at'] is not None, row['end_reason']) == (True, 'cron_incomplete_no_output')
    finally:
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_the_last_viewer_leaving_ends_a_gated_cron_fire(tmp_path, monkeypatch):
    """A viewer on the fire keeps its row open past the drain's idle point; its detach is then the
    only end the run gets."""
    import threading
    from gateway import session_cron
    started, release = threading.Event(), threading.Event()

    def run_job(job, **kwargs):
        started.set()
        release.wait(10)
        return _gated_or_answered(job, **kwargs)
    authority, _runner, native = await _cron_owner(tmp_path, monkeypatch, run_job)
    try:
        sid = (await _fire(authority, 'watched'))['session_id']
        assert await asyncio.to_thread(started.wait, 10)
        resumed = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'result' in resumed, resumed
        release.set()
        await authority.sessions[sid].task
        assert _row(authority, sid)['ended_at'] is None, 'a viewer keeps the run open'
        detached = await native.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': sid, 'subscription_id': resumed['result']['subscription_id']}})
        assert 'result' in detached, detached
        assert _row(authority, sid)['end_reason'] == 'cron_settled'
    finally:
        release.set()
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_startup_ends_stored_gated_cron_fires_once(tmp_path, monkeypatch):
    """Rows stored before the owner booked these ends stay open, and no drain ever visits them again
    (a restart only drains sessions with pending input). Startup ends each empty, idle one, once;
    a row an agent wrote to and a fire still queued keep their row open."""
    from gateway import session_cron
    from gateway.session_authority import initialize_session_authority
    from hermes_state_runtime import cancel_session_input, list_session_admissions
    authority, runner, native = await _cron_owner(tmp_path, monkeypatch, _gated_or_answered, schedule=False)
    db = authority.db

    async def stored_open(job_id, *, message=False):
        receipt = await _fire(authority, job_id)
        cancel_session_input(db, epoch=authority.epoch, admission_id=receipt['admission_id'])
        if message:
            db.append_message(receipt['session_id'], 'user', 'p')
        return receipt['session_id']
    try:
        empty, wrote = await stored_open('empty'), await stored_open('wrote', message=True)
        queued = (await _fire(authority, 'queued'))['session_id']
        created = await native.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'tui-1', 'source': 'tui', 'cwd': str(tmp_path), 'model': 'm'}})
        tui = created['result']['session_id']  # an empty chat, not a fire: never this repair's to end
        assert session_cron.end_settled_cron_fires(db, authority.epoch) == 1
        assert session_cron.end_settled_cron_fires(db, authority.epoch) == 0, 'idempotent'
        assert [_row(authority, s)['end_reason'] for s in (empty, wrote, queued)] == ['cron_settled', None, None]

        restored = await stored_open('restored')
        # The queued fire is cancelled too: once stored with nothing pending, startup ends it as well.
        [pending] = list_session_admissions(db, session_id=queued)
        cancel_session_input(db, epoch=authority.epoch, admission_id=pending['admission_id'])
        session_cron.unbind_owner(authority)
        restarted = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner-2')
        assert [_row(restarted, s)['end_reason'] for s in (restored, queued, wrote, tui)] == [
            'cron_settled', 'cron_settled', None, None]
        session_cron.unbind_owner(restarted)
    finally:
        session_cron.unbind_owner(authority)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_message_typed_into_a_finished_cron_run_leaves_it_finished(tmp_path, monkeypatch):
    """A person's message is refused read-only (``session_cron.execute``) and its card lands in the
    run's history; the run keeps the end its scheduler booked instead of reopening for good."""
    from gateway import session_cron
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input
    authority, _runner, _native = await _cron_owner(tmp_path, monkeypatch, _gated_or_answered)
    try:
        sid = (await _fire(authority, 'answered'))['session_id']
        await authority.sessions[sid].task
        admit_session_input(authority.db, epoch=authority.epoch, principal_id='human', session_id=sid,
                            request_id='typed-1', payload={'text': 'run it again', 'local_operator_v1': {
                                'profile_id': authority.profile_id, 'session_id': sid, 'principal_id': 'human'}})
        authority._schedule(SessionRef(authority.profile_id, sid))
        await authority.sessions[sid].task
        target = _row(authority, sid)
        assert (target['ended_at'] is not None, target['end_reason']) == (True, 'cron_complete')
        tail = authority.db.get_messages(target['id'])[-1]
        assert (tail['role'], tail['display_kind']) == ('assistant', 'failed_turn'), tail
    finally:
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_viewer_close_sent_during_a_fire_keeps_its_own_end_and_lifecycle(tmp_path, monkeypatch):
    """A close (``tui_close``) sent while the fire runs completes at the drain's idle point with the
    viewer's reason and the end-of-session lifecycle, as before cron fires were ended there: the
    owner's own stamp comes after it, so it does not pre-empt the close."""
    import threading
    from gateway import session_cron
    started, release = threading.Event(), threading.Event()

    def run_job(job, **kwargs):
        started.set()
        release.wait(10)
        return _gated_or_answered(job, **kwargs)
    authority, runner, native = await _cron_owner(tmp_path, monkeypatch, run_job)
    finalized = []

    async def finalize(**kwargs):
        finalized.append(kwargs['reason'])
        return []
    runner._finalize_session_off_loop = finalize
    try:
        sid = (await _fire(authority, 'closed'))['session_id']
        assert await asyncio.to_thread(started.wait, 10)
        resumed = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        detached = await native.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': sid, 'subscription_id': resumed['result']['subscription_id'], 'end_reason': 'tui_close'}})
        assert 'result' in detached, detached
        assert _row(authority, sid)['ended_at'] is None, 'the run is still going: the close waits'
        release.set()
        await authority.sessions[sid].task
        assert (_row(authority, sid)['end_reason'], finalized) == ('tui_close', ['tui_close'])
    finally:
        release.set()
        session_cron.unbind_owner(authority)
        authority.runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_idle_empty_chat_that_is_not_a_cron_fire_keeps_its_row_open(tmp_path, monkeypatch):
    """Guard: the owner's end is for cron fires only. An empty, unviewed TUI chat reaching the same
    points (drain idle, last viewer leaving) is not a run record and keeps its row open."""
    from gateway import session_cron
    authority, runner, native = await _cron_owner(tmp_path, monkeypatch, _gated_or_answered)
    try:
        created = await native.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'tui-1', 'source': 'tui', 'cwd': str(tmp_path), 'model': 'm'}})
        tui = created['result']['session_id']
        detached = await native.dispatch({'id': 2, 'method': 'session.detach', 'params': {
            'session_id': tui, 'subscription_id': created['result']['subscription_id']}})
        assert 'result' in detached and not authority.sessions[tui].subscribers, detached
        assert not session_cron.end_settled_cron_fire(authority, tui)  # the drain's idle point
        assert _row(authority, tui)['ended_at'] is None
    finally:
        session_cron.unbind_owner(authority)
        runner.session_store.close_all_db_handles()


def test_supersede_end_reason_replaces_only_the_provisional_stamp(tmp_path):
    """``supersede_end_reason`` re-books a ``cron_settled`` end and nothing else: a reason any other
    writer booked first keeps ``end_session``'s first-reason-wins."""
    from gateway.session_cron import CRON_SETTLED_END_REASON
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'state.db')
    try:
        for sid, first in (('settled', CRON_SETTLED_END_REASON), ('closed', 'tui_close'), ('open', None)):
            db.create_session(sid, 'cron')
            if first:
                db.end_session(sid, first)
        assert [db.supersede_end_reason(sid, CRON_SETTLED_END_REASON, 'cron_complete')
                for sid in ('settled', 'closed', 'open')] == [1, 0, 0]
        assert [db.get_session(sid)['end_reason'] for sid in ('settled', 'closed', 'open')] == [
            'cron_complete', 'tui_close', None]
    finally:
        db.close()
