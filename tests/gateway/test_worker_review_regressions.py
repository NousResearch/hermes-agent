"""Concrete predecessor review findings on the same successor lineage."""
import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_worker_receipt_reconciliation import owner, committed
from tests.waiting import wait_until


@pytest.mark.parametrize('flag', [False, None])
def test_optional_receipt_default_matches_canonical_dispatch(owner, flag):
    from gateway.session_controls import AuthorityConnection
    connection = object.__new__(AuthorityConnection)
    connection.authority, connection.actor = owner.authority, owner.connection.actor
    connection._operator = False
    owner.authority.profile_name = None
    params = dict(owner.scope, sequence=1, operation='session.activity',
                  payload={'ts':1234567890.0,'description':None,'provenance':None}, receipt_only=flag)
    response = asyncio.run_coroutine_threadsafe(connection.dispatch(
        {'jsonrpc':'2.0','id':1,'method':'worker.persist','params':params}), owner.loop).result(5)
    assert response.get('result') == {'value': None}, response
    assert len(committed(owner)[1]) == 1
    response = asyncio.run_coroutine_threadsafe(connection.dispatch(
        {'jsonrpc':'2.0','id':2,'method':'worker.persist','params':dict(params,receipt_only=True)}), owner.loop).result(5)
    assert response.get('result') == {'value': None} and len(committed(owner)[1]) == 1


def test_close_settles_unlimited_clarify_and_refuses_new_waiters(monkeypatch):
    from agent import managed_worker as mw
    from tools import clarify_gateway
    rd, wr = os.pipe()
    reader = os.fdopen(rd, 'rb', buffering=0)
    monkeypatch.setattr(sys, 'stdin', reader)
    monkeypatch.setattr(clarify_gateway, 'get_clarify_timeout', lambda: 0)
    frames, answers = [], []
    channel = SimpleNamespace(send=lambda kind, **kw: frames.append(kind))
    controls = mw.WorkerControls(channel, 'synthetic')
    waiter = threading.Thread(target=lambda: answers.append(controls._ask('question', [], False)))
    try:
        waiter.start()
        wait_until(lambda: 'clarify' in frames, timeout=2)
        controls.close()
        waiter.join(2)
        assert not waiter.is_alive(), 'closed reader left an unlimited clarification waiter'
        assert answers == [mw._INTERRUPTED] and not controls.clarifications
        assert controls._ask('after close', [], False) is mw._INTERRUPTED
        controls.close()
    finally:
        controls.stop()
        waiter.join(3)
        controls.close()
        os.close(wr); reader.close()


def test_actual_sendall_error_uses_receipt_not_second_mutation(owner, monkeypatch):
    """Native socket actually transmits before the supplier reports send failure."""
    real = owner.rpc._connection
    injected = []
    class SentThenError:
        def __init__(self, sock): self.sock = sock
        def __getattr__(self, name): return getattr(self.sock, name)
        def sendall(self, data, *a, **kw):
            self.sock.sendall(data, *a, **kw)
            if not injected:
                injected.append(True)
                assert owner.returned.wait(3), 'native owner did not physically commit'
                raise OSError('synthetic error after real sendall and commit')
    def connection():
        link = real()
        if not injected and not isinstance(link.ws.socket, SentThenError):
            link.ws.socket = SentThenError(link.ws.socket)
        return link
    monkeypatch.setattr(owner.rpc, '_connection', connection)
    rows = [{'role': 'assistant', 'content': 'native sendall effect'}]
    assert owner.store.append_messages_batch('owned', rows) == 1
    assert injected and rows[0]['_row_id']
    assert len(owner.requests) == 2
    assert owner.requests[1] == dict(owner.requests[0], receipt_only=True)
    assert committed(owner)[0] == len(committed(owner)[1]) == 1
