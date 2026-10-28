"""Monitor-aware admission negotiates only before a definitively unsupported call."""
from contextlib import asynccontextmanager
import json

import pytest

from cron.scheduler_authority import CronExecutionUnknown, journal_path, run_canonical_job
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError


def _peer(monkeypatch, request):
    from hermes_cli import gateway_client
    class Peer:
        async def rpc(self, method, **params):
            return await request(method, params)
    @asynccontextmanager
    async def connected():
        yield Peer()
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)


def _job():
    return {'id': 'monitor', 'prompt': 'Report', 'monitor_url': 'https://example.invalid'}


def test_old_gateway_fallback_is_persisted_before_legacy_submit(monkeypatch):
    calls = []
    async def request(method, params):
        calls.append(method)
        record = json.loads(journal_path('monitor', 'fire').read_text())
        if method == 'cron.submit_v2':
            assert record['protocol'] == 2
            raise GatewayRPCError('unknown_method', code=-32601)
        assert record['protocol'] == 1
        if method == 'cron.submit':
            return {'session_id': 's', 'admission_id': 'a'}
        return {'status': 'terminal', 'result': [True, 'doc', 'answer', None]}
    _peer(monkeypatch, request)
    assert run_canonical_job(_job(), execution_id='fire')[2] == 'answer'
    assert calls == ['cron.submit_v2', 'cron.submit', 'cron.status']


@pytest.mark.parametrize('error', [GatewayClientError('disconnected'),
                                  GatewayRPCError('invalid_params', code=4001)])
def test_ambiguous_or_refused_v2_never_falls_back(monkeypatch, error):
    calls = []
    async def request(method, params):
        calls.append(method)
        raise error
    _peer(monkeypatch, request)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job(_job(), execution_id='fire')
    assert calls == ['cron.submit_v2']
    assert json.loads(journal_path('monitor', 'fire').read_text())['protocol'] == 2
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job(_job(), execution_id='fire')
    assert calls == ['cron.submit_v2', 'cron.recover_v2']


def test_v2_journal_does_not_downgrade_on_reconnect(monkeypatch):
    from utils import atomic_json_write
    params = {'job_id': 'monitor', 'request_id': 'fire', 'extra_prompt': None}
    path = journal_path('monitor', 'fire')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {'params': params, 'protocol': 2, 'receipt': None})
    calls = []
    async def request(method, params):
        calls.append(method)
        raise GatewayRPCError('unknown_method', code=-32601)
    _peer(monkeypatch, request)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job(_job(), execution_id='fire')
    assert calls == ['cron.recover_v2']


def test_v2_journal_fails_closed_for_a_legacy_reader(monkeypatch):
    """Old readers ignore protocol; their unchanged params checks must reject v2."""
    from gateway.session_cron import _create
    params = {'job_id': 'monitor', 'request_id': 'fire', 'extra_prompt': None}
    async def request(method, wire):
        assert wire == params
        raise GatewayClientError('lost response')
    _peer(monkeypatch, request)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job(_job(), execution_id='fire')
    record = json.loads(journal_path('monitor', 'fire').read_text())
    # Legacy direct execution refuses a changed params object before sending.
    assert record['params'] != params
    # Legacy recovery forwards persisted params. The existing exact-field
    # validator refuses these before reaching any job/session side effect.
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        _create(None, None, record['params'])


def test_old_journal_keeps_legacy_protocol(monkeypatch):
    from utils import atomic_json_write
    params = {'job_id': 'monitor', 'request_id': 'fire', 'extra_prompt': None}
    path = journal_path('monitor', 'fire')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {'params': params, 'receipt': None})
    calls = []
    async def request(method, params):
        calls.append(method)
        if method == 'cron.submit':
            return {'session_id': 's', 'admission_id': 'a'}
        return {'status': 'terminal', 'result': [True, 'doc', 'answer', None]}
    _peer(monkeypatch, request)
    assert run_canonical_job(_job(), execution_id='fire')[0]
    assert calls == ['cron.submit', 'cron.status']


def test_prepared_context_does_not_skip_nested_job_monitor(monkeypatch):
    from cron import monitor, scheduler
    from gateway import session_cron, session_cron_monitor
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(monitor, '_run_monitor_source', lambda job: (calls.append(job['id']) or True, 'stable'))
    nested = dict(_job(), id='nested', monitor_state={'last_output_hash': monitor.hash_monitor_output('stable')})
    authority = SimpleNamespace(profile_id='parent-profile')
    execution = session_cron.bind_execution(authority, 's', 'parent', 'parent-fire')
    prepared = session_cron_monitor.bind_context({'context': 'parent observation', 'job_id': 'parent',
                                                'request_id': 'parent-fire', 'profile_id': 'parent-profile'})
    try:
        early, extra, context = scheduler._apply_monitor_gate(nested, 'nested', 'Nested', 'user input')
        assert early is not None and early[2] == scheduler.SILENT_MARKER
        assert context is None and extra == 'user input'
        assert calls == ['nested']
    finally:
        session_cron_monitor.reset_context(prepared)
        session_cron._execution.reset(execution)


def test_cancel_during_monitor_preparation_is_forwarded(monkeypatch):
    import threading
    cancel = threading.Event()
    cancel.set()
    calls = []
    async def request(method, params):
        calls.append(method)
        if method == 'cron.submit_v2':
            return {'status': 'preparing', 'result': None}
        if method == 'cron.cancel_v2':
            return {'ok': True}
        return {'status': 'terminal', 'result': [False, 'doc', '', 'cancelled']}
    _peer(monkeypatch, request)
    assert run_canonical_job(_job(), execution_id='fire', cancel_event=cancel)[0] is False
    assert calls == ['cron.submit_v2', 'cron.cancel_v2', 'cron.recover_v2']
