"""Exact worker receipts reconcile a committed write whose WebSocket ACK is delayed."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from websockets.sync.server import serve


@pytest.fixture
def receipt_owner(monkeypatch, tmp_path):
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, register_worker_execution

    db = SessionDB(tmp_path / 'state.db')
    epoch = begin_runtime_epoch(db, instance_id='test-owner')
    scopes = []
    for index in range(6):
        session_id = f'session-{index}'
        execution_id = f'execution-{index}'
        db.create_session(session_id, source='cli')
        scope = {'epoch': epoch, 'execution_id': execution_id,
                 'session_id': session_id, 'generation': 0}
        register_worker_execution(db, **scope, kind='compute', adoption_secret=f'secret-{index}')
        scopes.append(scope)

    lock = threading.Lock()
    seen = set()
    retry_seen = {}
    for scope in scopes:
        retry_seen[(scope['execution_id'], 1)] = threading.Event()
    state = SimpleNamespace(requests=[], committed=set(), retry_seen=retry_seen, lock=lock, drop_replies=False)

    def handler(ws):
        for raw in ws:
            request = json.loads(raw)
            key = (request['params'].get('execution_id'), request['params'].get('sequence'))
            with lock:
                state.requests.append(request)
                first = key not in seen
                seen.add(key)
            try:
                from hermes_state_runtime import mutate_worker_execution
                result = mutate_worker_execution(db, **request['params'])
            except Exception as exc:
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                    'error': {'code': -32000, 'message': str(exc)}}))
                continue
            if first:
                state.committed.add(key)
                # Hold the first ACK after the real SQLite transaction commits. The second
                # identical request must resolve to the same receipt before this gate opens.
                state.retry_seen[key].wait(2)
            else:
                state.retry_seen[key].set()
            if state.drop_replies:
                continue
            try:
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}))
            except Exception as exc:
                from websockets.exceptions import ConnectionClosed
                if not isinstance(exc, ConnectionClosed):
                    raise  # only the expected lost-ACK close is consumed by this fixture

    server = serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1'])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}', control_home=None)
    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                        lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
    monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
    monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'test-ticket')
    try:
        yield state, db, scopes
    finally:
        for event in retry_seen.values():
            event.set()
        server.shutdown()
        db.close()


def _fast_receipts(monkeypatch):
    """Keep the real bounded receive/error path but scale its deadline for this controlled test."""
    from agent.runtime_session_store import _OwnerLink
    receive = _OwnerLink.receive
    monkeypatch.setattr(_OwnerLink, 'receive', lambda self, timeout: receive(self, min(timeout, 0.2)))


def test_six_workers_reconcile_lost_ack_without_duplicate_usage_or_transcript(
        receipt_owner, monkeypatch, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC

    state, db, scopes = receipt_owner
    _fast_receipts(monkeypatch)
    stores = [RuntimeSessionStore(WorkerRPC(tmp_path), scope, tmp_path / f'outbox-{index}')
              for index, scope in enumerate(scopes)]

    def write(index):
        store = stores[index]
        session_id = scopes[index]['session_id']
        if index % 2:
            store.queue_token_counts(session_id, input_tokens=5, output_tokens=2, api_call_count=1)
            return 'usage'
        store.append_messages_batch(session_id, [{'role': 'user', 'content': f'once-{index}'}])
        return 'transcript'

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(write, range(6)))
        assert outcomes == ['transcript', 'usage', 'transcript', 'usage', 'transcript', 'usage']
        assert len(state.committed) == 6
        assert len(state.requests) == 12  # each durable sequence was applied, then exact-receipt read
        assert all(not store.journal['pending'] and store.failure is None for store in stores)
        for index in range(6):
            row = db.get_session(scopes[index]['session_id'])
            if index % 2:
                assert (row['input_tokens'], row['output_tokens'], row['api_call_count']) == (5, 2, 1)
            else:
                messages = db.get_messages(scopes[index]['session_id'])
                assert [message['content'] for message in messages] == [f'once-{index}']
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 6
    finally:
        for store in stores:
            store._outbox_owner.close()
            store.rpc.close()


def test_second_lost_ack_keeps_outcome_unknown_and_the_exact_entry_pending(
        receipt_owner, monkeypatch, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC

    state, db, scopes = receipt_owner
    state.drop_replies = True
    _fast_receipts(monkeypatch)
    scope = scopes[0]
    store = RuntimeSessionStore(WorkerRPC(tmp_path), scope, tmp_path / 'pending-outbox')
    try:
        with pytest.raises(TimeoutError, match='worker_receipt_timeout'):
            store.queue_token_counts(scope['session_id'], input_tokens=5)
        assert len(state.requests) == 2  # one initial write and one exact receipt reconciliation only
        assert state.requests[0]['params'] == state.requests[1]['params']
        assert store.failure == 'worker_receipt_timeout'
        assert store.journal['pending'] == [{'sequence': 1, 'operation': 'usage.main',
                                             'payload': {'input_tokens': 5}}]
        assert db.get_session(scope['session_id'])['input_tokens'] == 5
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 1
    finally:
        store._outbox_owner.close()
        store.rpc.close()


@pytest.mark.parametrize(('scope_index', 'field', 'value', 'reason'), [
    (0, 'epoch', 0, 'stale_epoch'),
    (1, 'generation', 1, 'stale_generation'),
])
def test_owner_fences_do_not_become_ambiguous_receipt_retries(
        receipt_owner, monkeypatch, tmp_path, scope_index, field, value, reason):
    from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError, WorkerRPC

    state, db, scopes = receipt_owner
    _fast_receipts(monkeypatch)
    scope = dict(scopes[scope_index], **{field: value})
    store = RuntimeSessionStore(WorkerRPC(tmp_path), scope, tmp_path / 'stale-outbox')
    try:
        with pytest.raises(WorkerPersistenceError, match=reason):
            store.queue_token_counts(scope['session_id'], input_tokens=5)
        assert len(state.requests) == 1
        assert len(store.journal['pending']) == 1
        assert db.get_session(scope['session_id'])['input_tokens'] == 0
    finally:
        store._outbox_owner.close()
        store.rpc.close()
