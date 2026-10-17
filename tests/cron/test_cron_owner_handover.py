"""A scoped restart-safe cron worker waits out an update's gateway handover instead of failing its fire.

An update installs the new checkout before the gateway is replaced, so a worker the old gateway
spawned in that window runs newer code than its parent: ``ensure`` reports the parent as
``incompatible``/``parent_outdated``, or the install as ``draining``/``update_paused``. Nothing was
submitted yet, so the worker that owns the fire waits (bounded) for the replacement and runs the fire
on it. Everyone else still fails fast with the owner's verdict.
"""
import threading
from contextlib import asynccontextmanager

import pytest

from cron import executions, jobs, scheduler_authority
from hermes_cli import gateway_client, gateway_runtime


def _owner(monkeypatch, verdicts):
    """``ensure`` answers each queued verdict through the real ``connect_gateway``; once they run
    out the replacement owner is up and the fire completes on it. Returns the call log."""
    calls = []
    real_connect = gateway_client.connect_gateway

    def ensure(home, **kwargs):
        state, reason = verdicts.pop(0)
        calls.append(f'ensure:{state}')
        return gateway_runtime.GatewayDiscovery(state, reason_code=reason)

    class Peer:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'cron.submit':
                return {'session_id': 'cron-fire', 'admission_id': 'admit'}
            return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}

    @asynccontextmanager
    async def connect():
        if verdicts:
            async with real_connect():
                raise AssertionError('a refused owner never yields a client')
        yield Peer()

    monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    return calls


def _worker(monkeypatch, fire, mode='scoped'):
    monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER', fire)
    monkeypatch.setenv('_HERMES_CRON_WORKER_DISPATCH', mode)


@pytest.mark.parametrize('verdicts', [
    [('incompatible', 'parent_outdated'), ('draining', 'replaced'), ('starting', 'deadline')],
    [('draining', 'update_paused'), ('draining', 'update_paused')],
])
def test_scoped_worker_runs_its_fire_on_the_replacement_gateway(tmp_path, monkeypatch, verdicts):
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        fire = executions.create_execution(job['id'], source='test')['id']
        assert executions.mark_execution_running(fire) is not None
        _worker(monkeypatch, fire)
        refusals = len(verdicts)
        calls = _owner(monkeypatch, verdicts)
        result = scheduler_authority.run_canonical_job(dict(job), execution_id=fire)
        assert result == (True, 'document', 'answer', None)
        assert calls[refusals:] == ['cron.submit', 'cron.status']
        # The worker kept its ledger row live while it waited.
        assert executions.get_execution(fire)['progress_at'] is not None


@pytest.mark.parametrize('mode, verdict', [
    ('degraded', ('incompatible', 'parent_outdated')),          # dies with the gateway's cgroup
    ('scoped', ('incompatible', 'runtime_protocol')),           # a newer, foreign or unverified owner
    ('scoped', ('incompatible', 'session_authority_unavailable')),
    ('scoped', ('starting', 'deadline')),                       # no handover behind it
    ('scoped', ('draining', None)),
])
def test_everything_but_a_confirmed_handover_fails_fast(monkeypatch, mode, verdict):
    _worker(monkeypatch, 'fire', mode)
    calls = _owner(monkeypatch, [verdict, verdict])
    result = scheduler_authority.run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls == [f'ensure:{verdict[0]}']
    assert result[0] is False and (verdict[1] or 'not_ready') in result[3]


def test_interactive_caller_fails_fast(monkeypatch):
    monkeypatch.delenv('_HERMES_CRON_EXTERNAL_WORKER', raising=False)
    calls = _owner(monkeypatch, [('incompatible', 'parent_outdated')] * 2)
    result = scheduler_authority.run_canonical_job({'id': 'job'}, execution_id='fire')
    assert calls == ['ensure:incompatible'] and result[0] is False


def test_cancel_during_the_wait_submits_nothing(monkeypatch):
    from cron import scheduler_owner_handover
    _worker(monkeypatch, 'fire')
    cancel = threading.Event()

    async def cancel_while_waiting(_seconds):
        cancel.set()  # e.g. the firer lost ownership while the worker slept

    monkeypatch.setattr(scheduler_owner_handover, '_pause', cancel_while_waiting)
    calls = _owner(monkeypatch, [('incompatible', 'parent_outdated')])
    result = scheduler_authority.run_canonical_job({'id': 'job'}, cancel_event=cancel, execution_id='fire')
    assert result[0] is False and 'parent_outdated' in result[3]
    assert 'cron.submit' not in calls


def test_the_wait_is_bounded(monkeypatch):
    from cron import scheduler_owner_handover
    _worker(monkeypatch, 'fire')

    async def no_pause(_seconds):
        return None

    monkeypatch.setattr(scheduler_owner_handover, '_pause', no_pause)
    calls = _owner(monkeypatch, [('incompatible', 'parent_outdated')] * 50)
    # Each refused dial costs 400 s, so a 900 s budget fits the first refusal and two more.
    monkeypatch.setattr(scheduler_owner_handover, '_clock', lambda: 400.0 * len(calls))
    result = scheduler_authority.run_canonical_job({'id': 'job'}, execution_id='fire')
    assert result[0] is False and 'parent_outdated' in result[3]
    assert calls == ['ensure:incompatible'] * 4