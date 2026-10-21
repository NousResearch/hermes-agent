"""Agent cron must enter the owner before importing any agent-side storage."""
import asyncio
from contextlib import asynccontextmanager

from cron import scheduler


def test_agent_cron_is_transport_only_and_refuses_without_owner(monkeypatch):
    from hermes_cli import gateway_client
    calls = []

    @asynccontextmanager
    async def unavailable():
        calls.append('connect')
        raise gateway_client.GatewayClientError('owner_unavailable')
        yield

    monkeypatch.setattr(gateway_client, 'connect_gateway', unavailable)
    monkeypatch.setattr(scheduler, '_prepare_job_prompt', lambda *a: (_ for _ in ()).throw(
        AssertionError('agent prompt preparation belongs to the owner')))
    result = scheduler.run_job({'id': 'job', 'prompt': 'test'}, execution_id='fire')
    assert calls == ['connect']
    assert result[0] is False and 'owner_unavailable' in result[3]


def test_owner_ticker_uses_direct_admission_without_loopback(monkeypatch):
    from gateway import session_cron
    from types import SimpleNamespace
    from hermes_constants import get_hermes_home
    import threading

    async def probe():
        authority = SimpleNamespace(db=SimpleNamespace(db_path=str(get_hermes_home() / 'state.db')))
        session_cron.bind_owner(authority)
        calls = []

        async def request(owner, operation, params, actor=None):
            assert owner is authority
            calls.append((operation, threading.get_ident()))
            if operation == 'submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

        monkeypatch.setattr(session_cron, 'operation', request)
        # A nested job must not inherit its parent's execution authorization.
        token = session_cron._execution.set((authority, 'parent-session', 'parent-job', 'parent-fire'))
        try:
            result = await asyncio.to_thread(scheduler.run_job, {'id': 'job'}, execution_id='fire')
        finally:
            session_cron._execution.reset(token)
        assert result == (True, 'document', 'answer', None)
        assert [op for op, _ in calls] == ['submit', 'status']
        assert all(tid == threading.get_ident() for _, tid in calls)
        session_cron.unbind_owner(authority)

    asyncio.run(probe())


def test_lost_cron_ack_retains_identity_instead_of_reporting_failure(tmp_path, monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    from hermes_cli import gateway_client
    from hermes_constants import get_hermes_home
    import json
    import pytest

    class Peer:
        async def rpc(self, method, **params):
            raise gateway_client.GatewayClientError('Gateway disconnected')

    @asynccontextmanager
    async def connected():
        yield Peer()

    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    records = list((get_hermes_home() / 'cron' / 'admissions').glob('*.json'))
    assert len(records) == 1
    assert json.loads(records[0].read_text())['params']['request_id'] == 'fire'


def _observe_with_status(monkeypatch, status, *, inactivity='600'):
    """Drive the observer against an owner whose ``cron.status`` answers are scripted by ``status``,
    on a fake clock (each timed-out poll costs the RPC budget) and without real backoff sleeps."""
    from types import SimpleNamespace
    from cron import scheduler_authority
    from hermes_cli import gateway_client

    now = [0.0]
    calls = []

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'cron.submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            return status(len([m for m in calls if m == 'cron.status']), now)

    @asynccontextmanager
    async def connected():
        yield Peer()

    real_sleep = asyncio.sleep

    async def no_wait(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setenv('HERMES_CRON_TIMEOUT', inactivity)
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    monkeypatch.setattr(scheduler_authority, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    return calls


def _timed_out(now):
    now[0] += 30
    raise TimeoutError()


def test_busy_owner_status_answer_past_rpc_budget_is_waited_out(monkeypatch):
    """#142414: an owner slowed by a delegated wave answers a poll after 30 s; its run is alive."""
    from cron.scheduler_authority import run_canonical_job

    def status(n, now):
        if n <= 2:
            _timed_out(now)
        return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

    calls = _observe_with_status(monkeypatch, status)
    assert run_canonical_job({'id': 'job'}, execution_id='fire') == (True, 'document', 'answer', None)
    assert calls.count('cron.status') == 3


def test_owner_silent_past_inactivity_budget_stays_unverified(monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    import pytest

    calls = _observe_with_status(monkeypatch, lambda n, now: _timed_out(now), inactivity='90')
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls.count('cron.status') == 3


def test_owner_gone_during_status_poll_is_not_retried(monkeypatch):
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    from hermes_cli import gateway_client
    import pytest

    def status(n, now):
        raise gateway_client.GatewayClientError('Gateway disconnected')

    calls = _observe_with_status(monkeypatch, status)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls.count('cron.status') == 1
