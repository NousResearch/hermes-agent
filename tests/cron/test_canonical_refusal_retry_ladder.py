"""A fire the canonical (gateway-dispatched) cron path never got admitted must be flagged for the
bounded re-run ladder (``cron/unreachable_retry.py``), exactly like the inline path's own handler
does (``cron/scheduler.py``, the ``except (Exception, ConfigBackendUnavailable)`` block, #151585).

``run_canonical_job`` returns a FAILED tuple instead of raising, so ``run_job``'s handler never
runs on that path. Without the flag the bookkeeping tail books a hard failure, and a job lost to a
gateway drain (an update holding the install) waits for its full next occurrence — hours for a
daily job — although nothing ran and nothing was spent.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from cron.scheduler_authority import CronExecutionUnknown, run_canonical_job
from cron.unreachable_retry import is_model_unreachable_failure
from hermes_cli import gateway_client
from hermes_constants import get_hermes_home


def _refused_dial(exc):
    """``connect_gateway()`` that raises before yielding a client: nothing was submitted."""

    class _Dial:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *exc_info):
            return False

    return _Dial()


def _dialed(peer):
    class _Dial:
        async def __aenter__(self):
            return peer

        async def __aexit__(self, *exc_info):
            return False

    return _Dial()


class _Peer:
    """A dialed gateway whose ``submit`` fails and whose ``recover`` reports ``state``."""

    def __init__(self, submit_exc, recover_state):
        self.submit_exc = submit_exc
        self.recover_state = recover_state
        self.calls = []

    async def rpc(self, method, **params):
        self.calls.append(method)
        if method == 'cron.submit':
            raise self.submit_exc
        if method == 'cron.recover':
            return self.recover_state
        raise AssertionError(f'unexpected rpc {method}')


def test_a_draining_gateway_flags_the_bounded_re_run(monkeypatch):
    """The reported shape: `hermes update` holds the install, the dial refuses, nothing ran."""
    monkeypatch.setattr(
        gateway_client, 'connect_gateway',
        lambda: _refused_dial(gateway_client.GatewayUnavailableError(
            'Gateway draining: update_paused', state='draining')))

    job = {'id': 'upkeep'}
    success, _output, _answer, error = run_canonical_job(job, execution_id='fire-drain')

    assert success is False
    assert 'GatewayUnavailableError' in error
    assert job.get('_model_unreachable') is True


def test_a_transient_dial_failure_flags_the_bounded_re_run(monkeypatch):
    monkeypatch.setattr(
        gateway_client, 'connect_gateway',
        lambda: _refused_dial(httpx.ConnectError('connection refused')))

    job = {'id': 'upkeep'}
    success, _output, _answer, _error = run_canonical_job(job, execution_id='fire-dial')

    assert success is False
    assert job.get('_model_unreachable') is True


def test_an_owner_that_recovered_no_admission_flags_the_bounded_re_run(monkeypatch):
    """In-process owner: ``submit`` fails transiently, ``recover`` proves nothing was admitted."""
    from gateway import session_cron

    async def probe():
        authority = SimpleNamespace(db=SimpleNamespace(db_path=str(get_hermes_home() / 'state.db')))
        session_cron.bind_owner(authority)
        try:
            async def request(owner, operation, params, actor=None):
                assert owner is authority
                if operation == 'submit':
                    raise httpx.ConnectError('connection refused')
                if operation == 'recover':
                    return {'status': 'missing'}
                raise AssertionError(f'unexpected operation {operation}')

            monkeypatch.setattr(session_cron, 'operation', request)
            job = {'id': 'upkeep'}
            result = await asyncio.to_thread(run_canonical_job, job, execution_id='fire-owner')
        finally:
            session_cron.unbind_owner(authority)
        return job, result

    job, result = asyncio.run(probe())

    assert result[0] is False
    assert job.get('_model_unreachable') is True


def test_an_unverified_admission_is_never_flagged(monkeypatch):
    """The fire may have run: re-firing it could double a side effect, so no ladder flag."""
    peer = _Peer(gateway_client.GatewayRPCError('request_failed'), {'status': 'unknown'})
    monkeypatch.setattr(gateway_client, 'connect_gateway', lambda: _dialed(peer))

    job = {'id': 'upkeep'}
    with pytest.raises(CronExecutionUnknown):
        run_canonical_job(job, execution_id='fire-unknown')

    assert '_model_unreachable' not in job
    assert peer.calls == ['cron.submit', 'cron.recover']


def test_a_gateway_refusal_counts_as_never_reached_the_model():
    exc = gateway_client.GatewayUnavailableError('Gateway draining: update_paused', state='draining')
    assert is_model_unreachable_failure(exc) is True


def test_a_wrapped_gateway_refusal_is_recognised_through_its_cause():
    try:
        try:
            raise gateway_client.GatewayUnavailableError('no ready owner', state='starting')
        except gateway_client.GatewayUnavailableError as inner:
            raise RuntimeError('cron dispatch failed') from inner
    except RuntimeError as outer:
        assert is_model_unreachable_failure(outer) is True


def test_a_run_that_reached_the_model_is_never_retried():
    exc = gateway_client.GatewayUnavailableError('Gateway draining', state='draining')
    assert is_model_unreachable_failure(exc, SimpleNamespace(session_api_calls=1)) is False
