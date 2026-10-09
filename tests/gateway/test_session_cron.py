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
