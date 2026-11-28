"""A firer's gateway attach that fails before anything is sent is retried; nothing after it is."""
from contextlib import asynccontextmanager

import pytest

from cron import scheduler_authority
from cron.scheduler_authority import CronExecutionUnknown, run_canonical_job


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(scheduler_authority, 'ATTACH_RETRY_DELAYS', (0.0, 0.0), raising=False)


class Peer:
    """A gateway that admits the run and reports it terminal on the first status poll."""

    def __init__(self, submit_error=None):
        self.calls = []
        self.submit_error = submit_error

    async def rpc(self, method, **params):
        self.calls.append(method)
        if method == 'cron.submit':
            if self.submit_error is not None:
                raise self.submit_error
            return {'session_id': 'cron-fire', 'admission_id': 'admit'}
        return {'status': 'terminal', 'result': [True, 'document', 'answer', None]}


def gateway(monkeypatch, failures, peer):
    """``connect_gateway`` that raises each of ``failures`` on successive attaches, then yields ``peer``."""
    from hermes_cli import gateway_client
    attaches = []
    pending = list(failures)

    @asynccontextmanager
    async def connect():
        attaches.append('attach')
        if pending:
            raise pending.pop(0)
        yield peer

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    return attaches


def connection_failed():
    """What ``connect_gateway`` raises when the WebSocket open times out."""
    from hermes_cli.gateway_client import GatewayClientError
    try:
        raise GatewayClientError('Gateway connection failed; no local fallback') from TimeoutError()
    except GatewayClientError as exc:
        return exc


def test_ticket_timeouts_are_retried_and_the_run_is_submitted_once(monkeypatch):
    peer = Peer()
    attaches = gateway(monkeypatch, [TimeoutError('timed out'), TimeoutError('timed out')], peer)
    assert run_canonical_job({'id': 'job'}, execution_id='fire') == (True, 'document', 'answer', None)
    assert len(attaches) == 3
    assert peer.calls == ['cron.submit', 'cron.status']


def test_websocket_open_failure_is_retried(monkeypatch):
    peer = Peer()
    attaches = gateway(monkeypatch, [connection_failed()], peer)
    assert run_canonical_job({'id': 'job'}, execution_id='fire')[0] is True
    assert len(attaches) == 2
    assert peer.calls.count('cron.submit') == 1


def test_exhausted_retries_fail_without_blaming_the_model(monkeypatch):
    from cron.scheduler import _summarize_cron_failure_for_delivery
    from cron.scheduler_failure_copy import classify_cron_failure_reason
    from hermes_constants import get_hermes_home
    peer = Peer()
    attaches = gateway(monkeypatch, [TimeoutError('timed out')] * 3, peer)
    ok, _output, _response, error = run_canonical_job({'id': 'job'}, execution_id='fire')
    assert ok is False and len(attaches) == 3 and peer.calls == []
    assert error.startswith('Gateway did not answer before the run was sent')
    # Nothing was journaled: the run was never sent, so there is nothing to reconcile.
    assert not list((get_hermes_home() / 'cron' / 'admissions').glob('*.json'))
    assert classify_cron_failure_reason(error) != 'timeout'
    notice = _summarize_cron_failure_for_delivery({'id': 'job', 'name': 'report'}, error)
    assert 'AI model service' not in notice and 'hermes fallback' not in notice


def test_a_definite_gateway_verdict_is_not_retried(monkeypatch):
    from hermes_cli.gateway_client import GatewayUnavailableError
    attaches = gateway(monkeypatch, [GatewayUnavailableError('Gateway draining: restart', 'draining', 'restart')],
                       Peer())
    ok, _output, _response, error = run_canonical_job({'id': 'job'}, execution_id='fire')
    assert ok is False and len(attaches) == 1
    assert 'draining' in error


def test_a_failure_after_submit_was_sent_is_never_retried(monkeypatch):
    peer = Peer(submit_error=TimeoutError())
    attaches = gateway(monkeypatch, [], peer)
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job({'id': 'job'}, execution_id='fire')
    assert len(attaches) == 1
    assert peer.calls == ['cron.submit']
