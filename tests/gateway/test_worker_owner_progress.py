"""Connected owner progress notification contract, not a mutation receipt."""
import asyncio
from types import SimpleNamespace

from tests.gateway.test_worker_receipt_reconciliation import owner, committed
from tests.waiting import wait_until


def test_owner_emits_identity_bound_progress_while_physical_write_is_held(owner, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_runtime_workers import mutation_tasks
    frames = []
    connection = object.__new__(AuthorityConnection)
    connection.authority, connection.actor = owner.authority, owner.connection.actor
    connection._operator = False
    connection.transport = SimpleNamespace(write=lambda frame: frames.append(frame) or True)
    owner.authority.profile_name = None
    def hold():
        owner.entered.set()
        assert owner.release.wait(5)
    monkeypatch.setattr(owner.db, '_try_wal_checkpoint', hold)
    owner.db._write_count = owner.db._CHECKPOINT_EVERY_N_WRITES - 1
    params = dict(owner.scope, sequence=1, operation='session.activity',
                  payload={'ts':1234567890.0,'description':None,'provenance':None})
    future = asyncio.run_coroutine_threadsafe(connection.dispatch(
        {'jsonrpc':'2.0','id':1,'method':'worker.persist','params':params}), owner.loop)
    try:
        assert owner.entered.wait(3)
        wait_until(lambda: bool(frames), timeout=2)
        progress = frames[-1]
        assert progress['method'] == 'worker.persist.progress'
        identity = progress['params']
        assert identity['execution_id'] == 'worker' and identity['sequence'] == 1
        assert identity['operation'] == 'session.activity' and len(identity['payload_digest']) == 64
        assert identity['phase'] == 'storage_pending'
        assert 'payload' not in identity and 'secret' not in identity
        assert mutation_tasks(owner.authority) and len(committed(owner)[1]) == 1
    finally:
        owner.release.set()
        future.result(5)


def test_owner_progress_is_observed_but_cannot_satisfy_receipt(monkeypatch):
    import json, queue, threading
    from agent.runtime_session_store import _OwnerLink
    incoming = queue.Queue()
    class Socket:
        def recv(self):
            frame = incoming.get(timeout=3)
            if isinstance(frame, Exception): raise frame
            return json.dumps(frame)
        def send(self, message): pass
    link = _OwnerLink(Socket())
    params = {'execution_id':'e','sequence':1,'operation':'session.activity','payload':{'ts':1}}
    link.send(7,json.dumps({'id':7,'params':params}))
    from gateway.session_admission import admission_fingerprint
    identity = dict(execution_id='e',sequence=1,operation='session.activity',
                    payload_digest=admission_fingerprint(canonical_target='',payload={'operation':params['operation'],'payload':params['payload']}),
                    phase='storage_pending')
    try:
        incoming.put({'method':'worker.persist.progress','params':identity})
        wait_until(lambda: getattr(link,'progress',None) is not None, timeout=2)
        assert link.progress == identity
        try:
            link.receive(.05)
        except TimeoutError:
            pass
        else:
            raise AssertionError('a progress frame was treated as a receipt')
        incoming.put({'id':7,'result':{'value':None}})
        assert link.receive(2) == {'id':7,'result':{'value':None}}
    finally:
        incoming.put(OSError('fixture close'));link.reader.join(3)
        assert not link.reader.is_alive()
