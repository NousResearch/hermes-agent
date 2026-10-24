"""A pre-submit dial failure is the host's condition: redial it inside the owner-wait budget
instead of spending the card's failure budget on it.

A worker whose gateway dial dies before anything is submitted — the owner thrashing at the host's
RAM ceiling refuses or stalls the websocket handshake ("Gateway connection failed; no local
fallback", or a refused connection) — exits 1 and the dispatcher books a crash against the card;
two of those give up a card whose content never failed. The dial is retried inside the same
OWNER_WAIT_S window the draining/starting wait already uses: nothing has been submitted, so a
redial can never run the claim twice. Once a connection attempt has yielded a client the claim can
be on the wire, so a failure past the dial raises instead — the owner may already hold the work
("execution unknown; retained by owner") and it must never be sent again.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_client, kanban_worker_client


class _Clock:
    """The retry loop's clock: sleeps advance it, so the budget test is deterministic."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def _connect(script, submitted, rpc_error=None):
    """connect_gateway driven by ``script``: an Exception entry raises at the dial, a ``None``
    entry yields a client whose ``rpc`` answers ``kanban.run`` (or raises ``rpc_error``)."""

    @asynccontextmanager
    async def connect():
        step = script.pop(0)
        attempts.append(step)
        if isinstance(step, BaseException):
            raise step

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                submitted.append(method)  # the request is on the wire before any answer fails
                if rpc_error is not None:
                    raise rpc_error
                return {'session_id': 's1', 'receipt': {'status': 'terminal', 'admission_id': 'a1'}}

        yield _Client()

    attempts = []
    return connect, attempts


def _setup(monkeypatch, tmp_path, script, *, rpc_error=None, owner_wait=None):
    submitted = []
    clock = _Clock()
    connect, attempts = _connect(script, submitted, rpc_error)

    async def _sleep(seconds):
        clock.tick(seconds)

    monkeypatch.setattr(gateway_client, 'connect_gateway', connect)
    monkeypatch.setattr(kanban_worker_client, 'time', SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(kanban_worker_client.asyncio, 'sleep', _sleep)
    if owner_wait is not None:
        monkeypatch.setattr(kanban_worker_client, 'OWNER_WAIT_S', owner_wait)
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result',
                        lambda *_: {'exit_code': 0, 'last_output': 'card closed'})
    return submitted, attempts, clock


@pytest.mark.asyncio
async def test_a_failed_dial_is_redialed_and_the_claim_is_sent_once(monkeypatch, tmp_path):
    """The transport-class dial failure is retried inside the budget; the claim goes out once."""
    submitted, attempts, clock = _setup(monkeypatch, tmp_path,
        [gateway_client.GatewayClientError('Gateway connection failed; no local fallback'), None])
    assert await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db')) == 0
    assert len(attempts) == 2
    assert submitted == ['kanban.run']
    assert clock.now == 1001.0  # one 1 s retry sleep between the two dials


@pytest.mark.asyncio
async def test_a_refused_connection_is_redialed(monkeypatch, tmp_path):
    """A raw OSError from the dial (connection refused) is the same host condition."""
    submitted, attempts, clock = _setup(monkeypatch, tmp_path,
        [ConnectionRefusedError(111, 'Connection refused'), None])
    assert await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db')) == 0
    assert len(attempts) == 2
    assert submitted == ['kanban.run']
    assert clock.now == 1001.0  # one 1 s retry sleep between the two dials


@pytest.mark.asyncio
async def test_a_non_transport_refusal_still_fails_at_once(monkeypatch, tmp_path):
    """A permanent client error is not a host transport condition: no redial, no wait."""
    submitted, attempts, clock = _setup(monkeypatch, tmp_path,
        [gateway_client.GatewayClientError('Gateway protocol mismatch; update/restart required')])
    with pytest.raises(gateway_client.GatewayClientError, match='protocol mismatch'):
        await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db'))
    assert len(attempts) == 1
    assert submitted == []  # a refused connection never carries a claim
    assert clock.now == 1000.0  # not a single retry sleep


@pytest.mark.asyncio
async def test_a_post_submit_failure_never_resends_the_claim(monkeypatch, tmp_path):
    """Once the connection exists the claim can be on the wire: a failure past the dial (the
    admission answer budget expiring raises TimeoutError, an OSError) must not redial."""
    submitted, attempts, clock = _setup(monkeypatch, tmp_path, [None],
        rpc_error=TimeoutError('admission answer budget expired'))
    with pytest.raises(TimeoutError):
        await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db'))
    assert len(attempts) == 1  # never redialed
    assert submitted == ['kanban.run']  # sent exactly once
    assert clock.now == 1000.0  # the failure never reached a retry sleep


@pytest.mark.asyncio
async def test_the_redial_stops_at_the_owner_wait_budget(monkeypatch, tmp_path):
    """The dial is retried only inside OWNER_WAIT_S; on the deadline the failure raises."""
    dial = gateway_client.GatewayClientError('Gateway connection failed; no local fallback')
    submitted, attempts, clock = _setup(monkeypatch, tmp_path, [dial] * 5, owner_wait=2.0)
    with pytest.raises(gateway_client.GatewayClientError, match='connection failed'):
        await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db'))
    assert len(attempts) == 3  # dials at t=0 and t=1 inside the 2 s budget, the third lands on it
    assert submitted == []
    assert clock.now == 1002.0  # two 1 s retry sleeps before the deadline raises
