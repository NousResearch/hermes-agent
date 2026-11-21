"""A transient gateway drop after admission re-attaches the observer via ``cron.recover``."""
from contextlib import asynccontextmanager

import pytest


def _patch(monkeypatch, peers, grace=None):
    from hermes_cli import gateway_client
    import cron.scheduler_authority as sa
    seq = iter(peers)

    @asynccontextmanager
    async def connected():
        peer = next(seq)
        if isinstance(peer, Exception):
            raise peer
        yield peer
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)
    monkeypatch.setattr(sa.asyncio, 'sleep', _nosleep)
    if grace is not None:
        monkeypatch.setattr('cron.jobs._cron_config_number',
                            lambda key, default, cast: cast(grace) if key == 'observer_reconnect_grace_seconds' else cast(default))


async def _nosleep(_):
    return None


def test_drop_after_admission_reconnects_and_settles_without_resubmit(monkeypatch):
    from hermes_cli import gateway_client
    from cron.scheduler_authority import run_canonical_job
    calls = []

    class First:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'cron.submit':
                return {'session_id': 's', 'admission_id': 'a'}
            raise gateway_client.GatewayClientError('Gateway disconnected')

    class Second:
        async def rpc(self, method, **params):
            calls.append(method)
            if method == 'cron.recover':
                return {'status': 'started', 'session_id': 's', 'admission_id': 'a'}
            return {'status': 'terminal', 'result': [True, 'doc', '[SILENT]', None]}

    _patch(monkeypatch, [First(), gateway_client.GatewayClientError('Gateway connection failed'), Second()])
    assert run_canonical_job({'id': 'job'}, execution_id='fire') == (True, 'doc', '[SILENT]', None)
    assert calls.count('cron.submit') == 1
    assert calls[-2:] == ['cron.recover', 'cron.status']


def test_grace_zero_keeps_fail_closed_unknown(monkeypatch):
    from hermes_cli import gateway_client
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown

    class Peer:
        async def rpc(self, method, **params):
            if method == 'cron.submit':
                return {'session_id': 's', 'admission_id': 'a'}
            raise gateway_client.GatewayClientError('Gateway disconnected')

    _patch(monkeypatch, [Peer()], grace=0)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')


def test_recover_missing_after_drop_stays_unknown(monkeypatch):
    from hermes_cli import gateway_client
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown

    class First:
        async def rpc(self, method, **params):
            if method == 'cron.submit':
                return {'session_id': 's', 'admission_id': 'a'}
            raise gateway_client.GatewayClientError('Gateway disconnected')

    class Second:
        async def rpc(self, method, **params):
            assert method == 'cron.recover'
            return {'status': 'missing', 'result': None}

    _patch(monkeypatch, [First(), Second()])
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')


def test_lost_submit_answer_is_not_retried(monkeypatch):
    from hermes_cli import gateway_client
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    seen = []

    class Peer:
        async def rpc(self, method, **params):
            seen.append(method)
            raise gateway_client.GatewayClientError('Gateway disconnected')

    _patch(monkeypatch, [Peer()])
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert seen == ['cron.submit']


def test_definite_owner_refusal_is_not_retried(monkeypatch):
    from hermes_cli import gateway_client
    from cron.scheduler_authority import run_canonical_job, CronExecutionUnknown
    seen = []

    class Peer:
        async def rpc(self, method, **params):
            seen.append(method)
            if method == 'cron.submit':
                return {'session_id': 's', 'admission_id': 'a'}
            raise gateway_client.GatewayRPCError('permission_denied')

    _patch(monkeypatch, [Peer()])
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert seen == ['cron.submit', 'cron.status']
