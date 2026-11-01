"""A worker whose owner stalls past the receipt budget resends the same journal entry instead of failing.

Memory pressure can freeze the gateway's event loop for longer than one 20 s receipt wait. The owner's
receipts are idempotent per (execution, sequence, digest), so a timed-out save is redialled and resent
until the owner has been silent for ``OWNER_WAIT_S``; before this, the first timeout stuck as the
store's failure and every later save, lease refresh and close failed with it.
"""
import json
import threading
from types import SimpleNamespace

import pytest

from agent.runtime_session_store import (
    RECEIPT_RECOVERY_S, RuntimeSessionStore, WorkerPersistenceError, WorkerReceiptTimeout, WorkerRPC)

SCOPE = {'session_id': 's', 'execution_id': 'w', 'epoch': 1}


class _Owner:
    """An owner with idempotent receipts that goes silent for its first ``silent`` calls."""

    def __init__(self, silent=0, clock=None, step=20.0, fail_with=None):
        self.silent, self.clock, self.step, self.fail_with = silent, clock, step, fail_with
        self.calls = []
        self.receipts = {}

    def __call__(self, method, **params):
        self.calls.append(params)
        if self.clock is not None:
            self.clock.now += self.step  # each silent call spends its receipt wait
        if len(self.calls) <= self.silent:
            if self.fail_with is not None:
                raise self.fail_with(len(self.calls))
            raise WorkerReceiptTimeout('worker_receipt_timeout')
        # The first send may have committed before the owner stalled: a resend returns the stored result.
        return self.receipts.setdefault(params['sequence'], {'count': 1, 'annotations': [{}], 'value': True})


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _store(tmp_path, owner, clock=None):
    clock = clock or _Clock()
    owner.clock = owner.clock or clock
    return RuntimeSessionStore(owner, SCOPE, tmp_path / 'private', clock=clock, sleep=lambda s: None), clock


def test_receipt_timeout_is_resent_and_the_save_succeeds(tmp_path):
    owner = _Owner(silent=2)
    store, _ = _store(tmp_path, owner)
    try:
        assert store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}]) == 1
        assert len(owner.calls) == 3
        assert {json.dumps(call, sort_keys=True) for call in owner.calls} == {json.dumps(owner.calls[0], sort_keys=True)}
        assert (store.failure, store.journal['pending']) == (None, [])
    finally:
        store.close()


def test_gives_up_when_the_owner_stays_silent_for_the_owner_wait(tmp_path):
    from hermes_cli.kanban_worker_client import OWNER_WAIT_S
    assert RECEIPT_RECOVERY_S == OWNER_WAIT_S == 180.0
    owner = _Owner(silent=10 ** 6)
    store, _ = _store(tmp_path, owner)
    try:
        with pytest.raises(WorkerReceiptTimeout):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}])
        # 20 s per silent attempt: the ninth reaches the 180 s mark and stands.
        assert len(owner.calls) == 9
        assert store.failure == 'worker_receipt_timeout'
        assert len(store.journal['pending']) == 1  # kept for the next explicit retry or adoption
    finally:
        with pytest.raises(WorkerPersistenceError, match='worker_receipt_timeout'):
            store.close()


def test_save_lease_refresh_and_close_work_after_recovery(tmp_path):
    owner = _Owner(silent=1)
    store, _ = _store(tmp_path, owner)
    assert store.append_messages_batch('s', [{'role': 'user', 'content': 'first'}]) == 1
    assert store.refresh_session_turn_lease('s', 'holder') is True
    assert store.append_messages_batch('s', [{'role': 'assistant', 'content': 'second'}]) == 1
    store.close()  # flush_token_counts: no sticky failure, nothing pending
    assert [call['operation'] for call in owner.calls] == [
        'transcript.append', 'transcript.append', 'turn.renew', 'transcript.append']
    assert [call['sequence'] for call in owner.calls] == [1, 1, 2, 3]
    assert json.loads(store.path.read_text())['pending'] == []


def test_a_later_stall_gets_its_own_budget(tmp_path):
    """The window is per entry, from its first send: a recovered save does not shorten the next one."""
    owner = _Owner(silent=8)
    store, _ = _store(tmp_path, owner)
    try:
        store.append_messages_batch('s', [{'role': 'user', 'content': 'first'}])  # 160 s of silence, recovered
        owner.silent = len(owner.calls) + 8
        store.append_messages_batch('s', [{'role': 'user', 'content': 'second'}])
        assert store.failure is None
    finally:
        store.close()


@pytest.mark.parametrize('redial_error', [
    lambda n: TimeoutError('open handshake timed out'),
    lambda n: ConnectionRefusedError(),
    lambda n: WorkerPersistenceError('owner_unavailable'),
])
def test_a_redial_that_also_stalls_stays_inside_the_same_window(tmp_path, redial_error):
    calls = []

    def rpc(method, **params):
        calls.append(params)
        if len(calls) == 1:
            raise WorkerReceiptTimeout('worker_receipt_timeout')
        if len(calls) < 4:
            raise redial_error(len(calls))
        return {'count': 1, 'annotations': [{}]}
    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', sleep=lambda s: None)
    try:
        assert store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}]) == 1
        assert len(calls) == 4 and store.failure is None
    finally:
        store.close()


@pytest.mark.parametrize('error', [
    TimeoutError('offline'),                      # not the receipt wait: nothing says the owner went silent
    ConnectionRefusedError(),
    WorkerPersistenceError('owner_unavailable'),
])
def test_errors_before_the_owner_went_silent_are_not_retried(tmp_path, error):
    calls = []

    def rpc(method, **params):
        calls.append(params)
        raise error
    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', sleep=lambda s: None)
    try:
        with pytest.raises(type(error)):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}])
        assert len(calls) == 1
    finally:
        with pytest.raises(WorkerPersistenceError):
            store.close()


def test_an_owner_refusal_after_a_timeout_is_final(tmp_path):
    calls = []

    def rpc(method, **params):
        calls.append(params)
        if len(calls) == 1:
            raise WorkerReceiptTimeout('worker_receipt_timeout')
        raise WorkerPersistenceError('stale_generation')
    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', sleep=lambda s: None)
    try:
        with pytest.raises(WorkerPersistenceError, match='stale_generation'):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}])
        assert len(calls) == 2
    finally:
        with pytest.raises(WorkerPersistenceError, match='stale_generation'):
            store.close()


@pytest.fixture
def stalling_owner(monkeypatch):
    """A real loopback owner behind real ``WorkerRPC``: it swallows the first request it reads (never
    replies, like a frozen loop) and answers a resend with the receipt it already stored."""
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery
    import agent.runtime_session_store as module

    state = SimpleNamespace(connections=0, committed=[], receipts={}, swallow=1)

    def handler(ws):
        state.connections += 1
        for raw in ws:
            request = json.loads(raw)
            if state.swallow:
                state.swallow -= 1
                continue
            sequence = request['params']['sequence']
            if sequence not in state.receipts:
                state.committed.append(sequence)
                state.receipts[sequence] = {'count': 1, 'annotations': [{}], 'value': True}
            ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': state.receipts[sequence]}))

    with serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'ticket')
        monkeypatch.setattr(module, 'RECEIPT_WAIT_S', 0.5)
        try:
            yield state
        finally:
            server.shutdown()


def test_a_real_connection_that_stalls_once_is_redialled_and_commits_once(stalling_owner, tmp_path):
    rpc = WorkerRPC(tmp_path)
    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private')
    try:
        assert store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}]) == 1
        assert store.refresh_session_turn_lease('s', 'holder') is not None
        assert (stalling_owner.connections, stalling_owner.committed) == (2, [1, 2])
    finally:
        store.close()
        rpc.close()
