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
            await asyncio.sleep(0.3)  # the firer polls status at least once during the drain
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
