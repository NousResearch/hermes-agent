"""A re-adopted worker's first dial can time out while the new owner still imports its socket
handler. Nothing was sent, so the detached store retries that write; the plain store does not."""

import json
import queue
import threading
from types import SimpleNamespace

import pytest

import websockets.sync.client as ws_client

from agent import runtime_session_store
from agent.managed_worker_detach import channel as managed_worker_detach
from hermes_cli import gateway_client, gateway_runtime, gateway_runtime_discovery
from agent.managed_worker_detach.channel import DetachedRPC
from agent.runtime_session_store import WorkerRPC

PARAMS = dict(execution_id='fixture', epoch=7, generation=3, sequence=37, operation='transcript.append',
              payload={'messages': [{'role': 'assistant', 'content': 'fixture'}]})


class _Socket:
    def __init__(self, sent):
        self.sent, self.replies = sent, queue.SimpleQueue()

    def send(self, raw):
        frame = json.loads(raw)
        self.sent.append(frame)
        self.replies.put(json.dumps({'jsonrpc': '2.0', 'id': frame['id'], 'result': {'count': 1}}))

    def recv(self):
        return self.replies.get()

    def close(self):
        pass


@pytest.fixture
def owner(monkeypatch, tmp_path):
    """Real WorkerRPC._connection with discovery, identify, ticket and the dial controlled."""
    dials, sent = [], []
    endpoint = SimpleNamespace(name='fixture')
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
    monkeypatch.setattr(gateway_runtime, 'control_home_for', lambda home, ep: home)
    monkeypatch.setattr(gateway_runtime_discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, ep, purpose: f'ticket-{len(dials)}')
    monkeypatch.setattr(gateway_client, 'gateway_ws_target', lambda ep, ticket: ('ws://fixture/api/ws', ['p']))
    failures = []

    def connect(url, **kwargs):
        dials.append(kwargs['open_timeout'])
        if failures:
            raise failures.pop(0)
        return _Socket(sent)

    monkeypatch.setattr(ws_client, 'connect', connect)
    monkeypatch.setattr(managed_worker_detach.time, 'sleep', lambda seconds: None)
    return SimpleNamespace(home=tmp_path, dials=dials, sent=sent, failures=failures)


def _detached(owner, *, reattached_at):
    channel = SimpleNamespace(lock=threading.RLock(), epoch=8, attachments=1, orphaned=False,
                              reattached_at=reattached_at, await_reattach=lambda seen: False)
    return DetachedRPC(WorkerRPC(owner.home), channel)


@pytest.mark.parametrize('error', [TimeoutError('timed out while waiting for handshake response'),
                                   ConnectionRefusedError(111, 'refused')])
def test_pre_send_dial_failure_after_reattach_is_retried_once_sent(owner, error):
    owner.failures.append(error)
    detached = _detached(owner, reattached_at=managed_worker_detach.time.monotonic())
    assert detached('worker.persist', **PARAMS) == {'count': 1}
    assert len(owner.dials) == 2
    assert [frame['params'] for frame in owner.sent] == [PARAMS | {'epoch': 8}]


def test_pre_send_dial_failure_without_reattach_window_stays_terminal(owner):
    owner.failures.append(TimeoutError('timed out while waiting for handshake response'))
    detached = _detached(owner, reattached_at=None)
    with pytest.raises(runtime_session_store.WorkerOwnerUnreachable):
        detached('worker.persist', **PARAMS)
    assert len(owner.dials) == 1 and owner.sent == []


def test_plain_store_keeps_the_original_dial_error(owner):
    owner.failures.append(TimeoutError('timed out while waiting for handshake response'))
    with pytest.raises(TimeoutError, match='handshake response'):
        WorkerRPC(owner.home)('worker.persist', **PARAMS)
    assert owner.sent == []
