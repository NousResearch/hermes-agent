"""One send per confirmed owner; native fence refusals are terminal."""

import json
import queue
import threading
from types import SimpleNamespace

import pytest

from agent.managed_worker_detach.channel import DetachedRPC
from agent.runtime_session_store import WorkerPersistenceError, WorkerRPC, _OwnerLink


@pytest.mark.parametrize('mode,operation,expected_epochs', [
    ('attached', 'transcript.append', [8]),
    ('ambiguous', 'transcript.append', [8]),
    ('stale_epoch', 'transcript.append', [8]),
    ('reattached', 'transcript.append', [8, 9]),
    ('no_handoff', 'execution.finish', [8]),
    ('same_epoch', 'execution.finish', [8]),
    ('repeated_stale', 'execution.finish', [8, 9]),
    ('ambiguous', 'execution.finish', [8]),
    ('permission_denied', 'execution.finish', [8]),
    ('stale_generation', 'execution.finish', [8]),
])
def test_detached_rpc_send_epochs(mode, operation, expected_epochs):
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
            if mode == 'repeated_stale':
                channel.attachments += 1
                channel.epoch += 1
                return {'error': {'message': 'stale_epoch'}}
            if len(sent) == 1:
                if mode in ('ambiguous', 'reattached'):
                    raise queue.Empty
                if mode in ('stale_epoch', 'permission_denied', 'stale_generation'):
                    advance()  # These refusals stay terminal even after a handoff.
                    return {'error': {'message': mode}}
                if mode == 'no_handoff':
                    return {'error': {'message': 'stale_epoch'}}
                if mode == 'same_epoch':
                    channel.attachments += 1
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
                  operation=operation, payload={} if operation == 'execution.finish' else {
                      'messages': [{'role': 'assistant', 'content': 'fixture'}]})
    if mode not in ('attached', 'reattached'):
        error = 'worker_receipt_timeout' if mode == 'ambiguous' else mode
        if mode in ('no_handoff', 'same_epoch', 'repeated_stale'):
            error = 'stale_epoch'
        with pytest.raises((WorkerPersistenceError, TimeoutError), match=error):
            detached('worker.persist', **params)
    else:
        assert detached('worker.persist', **params) == {'count': 1}
    assert [frame['params'] for frame in sent] == [params | {'epoch': epoch} for epoch in expected_epochs]
