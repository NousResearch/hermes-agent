"""One send per confirmed owner; native fence refusals are terminal."""

import json
import queue
import threading
from types import SimpleNamespace

import pytest

from agent.managed_worker_detach.channel import DetachedRPC
from agent.runtime_session_store import WorkerPersistenceError, WorkerRPC, _OwnerLink


@pytest.mark.parametrize('mode,expected_epochs', [
    ('attached', [8]), ('ambiguous', [8]), ('stale_epoch', [8]), ('reattached', [8, 9]),
])
def test_detached_rpc_send_epochs(mode, expected_epochs):
    sent = []
    channel = SimpleNamespace(lock=threading.RLock(), epoch=8, attachments=1, orphaned=False, reattached_at=0)
    rpc = WorkerRPC.__new__(WorkerRPC)
    rpc.lock, rpc.request_id, rpc.link = threading.Lock(), 0, None

    def advance():
        channel.attachments, channel.epoch = 2, 9

    def wait_for_attachment(seen):
        if channel.attachments != seen:
            return True
        if mode == 'reattached' and seen == 1:
            advance()
            return True
        return False

    class Replies:
        def get(self, *, timeout):
            if len(sent) == 1:
                if mode in ('ambiguous', 'reattached'):
                    raise queue.Empty
                if mode == 'stale_epoch':
                    advance()  # a genuine new attachment cannot turn a refusal into a retry
                    return {'error': {'message': 'stale_epoch'}}
            return {'result': {'count': 1}}

    def connect():
        link = _OwnerLink.__new__(_OwnerLink)
        link.ws = SimpleNamespace(send=lambda raw: sent.append(json.loads(raw)), close=lambda: None)
        link.attachment, link.awaiting, link.replies = channel.attachments, None, Replies()
        rpc.link = link
        return link

    channel.await_reattach, rpc._connection = wait_for_attachment, connect
    detached = DetachedRPC(rpc, channel)
    params = dict(execution_id='fixture', epoch=7, generation=3, sequence=4,
                  operation='transcript.append', payload={'messages': [{'role': 'assistant', 'content': 'fixture'}]})
    if mode in ('ambiguous', 'stale_epoch'):
        error = 'stale_epoch' if mode == 'stale_epoch' else 'worker_receipt_timeout'
        with pytest.raises((WorkerPersistenceError, TimeoutError), match=error):
            detached('worker.persist', **params)
    else:
        assert detached('worker.persist', **params) == {'count': 1}
    assert [frame['params'] for frame in sent] == [params | {'epoch': epoch} for epoch in expected_epochs]
