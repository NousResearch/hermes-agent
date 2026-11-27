"""Registry/publication boundaries and real owner batch integration."""
import threading
import asyncio
import pytest
from types import SimpleNamespace

from tests.gateway.fixtures.canonical_batch_owner import Peer, QUESTIONS, make_owner, opening, rpc

from gateway.session_events import SessionEvents
from gateway import session_server_requests as requests


def test_open_observer_can_snapshot_and_settle_without_registry_deadlock():
    events = SessionEvents()
    peer = SimpleNamespace(write=lambda frame: True, canonical_clarify_projection=True)
    events.fanout.attach(peer)
    authority = SimpleNamespace(profile_id='owner', epoch=1, sessions={'s': SimpleNamespace(event_stream=events)},
                                executing={'s': 7}, _handle=lambda ref: SimpleNamespace(execution_generation=7, execution_state='running'))
    observed, output, errors = threading.Event(), {}, []

    def observer(frame):
        if frame['params']['type'] != 'clarify.batch.request':
            return
        projection = requests.canonical_batches(events, 's')[0]
        assert requests.resolve({'id': projection['request_id'], 'result': {
            'answers': {'q0': ''}, 'outcome': 'submitted'}}, identity=projection['identity'])
        observed.set()

    events.observers.add(observer)

    def run():
        try:
            output.update(requests.ask_viewer_result(events, 's', 'clarify', {'questions': [
                {'qid': 'q0', 'question': 'First?'}]}, qids=['q0'], timeout=.25, batch_owner=(authority, 7)))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert observed.wait(1), 'synchronous observer deadlocked while reading the registry'
    worker.join(1)
    assert not worker.is_alive()
    assert not errors
    assert output == {'answers': {'q0': ''}, 'outcome': 'submitted'}


def test_settlement_and_snapshot_wait_for_execution_publication_fence():
    events = SessionEvents()
    authority = SimpleNamespace(profile_id='owner', epoch=1, sessions={'s': SimpleNamespace(event_stream=events)},
                                executing={'s': 7}, _handle=lambda ref: SimpleNamespace(execution_generation=7, execution_state='running'))
    pending = requests._Pending('s', ['q0'], events, 'clarify', {'questions': []}, batch_owner=(authority, 7))
    for operation in ('resolve', 'lock', 'snapshot'):
        rid = 'srq-fence-' + operation
        with requests._lock:
            requests._open[rid] = pending
        entered, done, output = threading.Event(), threading.Event(), []
        def act():
            entered.set()
            if operation == 'resolve':
                output.append(requests.resolve({'id': rid, 'result': {'answers': {'q0': 'late'}}}))
            elif operation == 'lock':
                output.append(requests.lock_answer(rid, 'q0', 'late', session_ok=lambda *args: True))
            else:
                output.append(requests.canonical_batches(events, 's'))
            done.set()
        authority.executing['s'] = 7
        with events.lock:
            worker = threading.Thread(target=act, daemon=True)
            worker.start()
            assert entered.wait(1)
            assert not done.wait(2), operation + ' escaped the stream publication fence'
            authority.executing['s'] = 8
        assert done.wait(1)
        worker.join(1)
        assert output == [False if operation == 'resolve' else None if operation == 'lock' else []]
        assert pending.locked == {}
        with requests._lock:
            requests._open.pop(rid, None)


@pytest.mark.asyncio
async def test_real_admission_producer_negotiation_snapshot_and_exact_terminal(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_controls import AuthorityConnection
    from gateway.run_turn_runner import TurnRunner
    seen = []
    async def answer(event):
        generation = authority.executing['s']
        turn = SimpleNamespace(_approval_owner=(authority, 's', generation), _ctx=SimpleNamespace(
            inbound_message_id=event.message_id, source=SimpleNamespace(platform=Platform.LOCAL),
            agent_holder=[SimpleNamespace(_interrupt_requested=False)]))
        result = await asyncio.to_thread(TurnRunner._clarify_callback_sync, turn, QUESTIONS)
        seen.append(result)
        return 'ok'
    authority = await make_owner(tmp_path, monkeypatch, answer)
    peer, legacy = Peer(), Peer()
    connection = AuthorityConnection(authority, peer, {'user_id': 'owner'})
    try:
        # Native dispatcher, not a mock capability or an overridden client getter.
        supported = await rpc(connection, 'client.capabilities', server_requests=True)
        assert supported.get('result', {}).get('canonical_clarify_projection_v1') is True, supported
        selected = await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1=True)
        assert selected['result']['canonical_clarify_projection_v1'] is True
        assert (await rpc(connection, 'session.resume', session_id='s')).get('result')
        stream = authority.sessions['s'].event_stream
        stream.fanout.attach(legacy)
        submitted = await rpc(connection, 'prompt.submit', session_id='s', submission_id='batch-turn', text='ask',
                              clarify_batch=True, clarify_projection='canonical_batch_v1')
        assert submitted['result']['status'] == 'queued', submitted
        projection = await opening(peer)
        rid, identity = projection['request_id'], projection['identity']
        assert identity == {'profile_id': 'p', 'authority_epoch': authority.epoch,
                            'stream_id': stream._key, 'execution_generation': authority.executing['s']}
        assert projection['questions'] == QUESTIONS
        assert not any(f.get('method') == 'clarify' for f in peer.frames)
        assert any(f.get('id') == rid and f.get('method') == 'clarify' for f in legacy.frames)
        # A negotiated consumer must use the exact-identity routes, not fall back to scalar/raw mutation.
        old_lock = await rpc(connection, 'clarify.lock', request_id=rid, question_id='q0', answer='unfenced')
        assert old_lock.get('result') == {'status': 'expired'}, old_lock
        await connection.dispatch({'id': rid, 'result': {'answers': {'q0': 'unfenced'}}})
        assert requests.canonical_batches(stream, 's') == [projection]
        for field, bad in [('profile_id', 'foreign'), ('authority_epoch', identity['authority_epoch'] + 1),
                           ('stream_id', 'replaced-stream'), ('execution_generation', identity['execution_generation'] + 1)]:
            wrong = await rpc(connection, 'clarify.batch.lock', session_id='s', request_id=rid,
                              identity={**identity, field: bad}, question_id='q0', answer='untrusted')
            assert wrong.get('result') == {'status': 'expired'}, wrong
        malformed = await rpc(connection, 'clarify.batch.lock', session_id='s', request_id=rid,
                              identity={**identity, 'execution_generation': True}, question_id='q0', answer='untrusted')
        assert malformed.get('error', {}).get('message') == 'invalid_params', malformed
        first = await rpc(connection, 'clarify.batch.lock', session_id='s', request_id=rid, identity=identity,
                          question_id='q0', answer=['red', 'blue'])
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        # Read permission and negotiation do not grant answer-bearing recovery or mutation.
        for negotiated in (False, True):
            reader_peer = Peer()
            reader = AuthorityConnection(authority, reader_peer, {'user_id': 'owner', 'capabilities': ['session:read']})
            await rpc(reader, 'client.capabilities')
            if negotiated:
                await rpc(reader, 'client.capabilities', canonical_clarify_projection_v1=True)
            await rpc(reader, 'session.resume', session_id='s')
            try:
                read = await rpc(reader, 'session.events.since', session_id='s', replay_epoch=stream.epoch, last_sequence=0)
                assert read['result']['open_requests'] == [], read
                assert read['result'].get('clarify_batches', []) == []
                refused = await rpc(reader, 'clarify.batch.respond', session_id='s', request_id=rid, identity=identity,
                                    result={'answers': {}, 'outcome': 'cancelled'})
                assert 'error' in refused, refused
            finally:
                await reader.close()
        unattached = AuthorityConnection(authority, Peer(), {'user_id': 'owner'})
        await rpc(unattached, 'client.capabilities')
        await rpc(unattached, 'client.capabilities', canonical_clarify_projection_v1=True)
        try:
            read = await rpc(unattached, 'session.events.since', session_id='s', replay_epoch=stream.epoch, last_sequence=0)
            assert read['result']['clarify_batches'] == []
            refused = await rpc(unattached, 'clarify.batch.lock', session_id='s', request_id=rid, identity=identity,
                                question_id='q1', answer='not-attached')
            assert refused.get('error', {}).get('message') == 'permission_denied', refused
        finally:
            await unattached.close()
        snapshot = await rpc(connection, 'session.events.since', session_id='s', replay_epoch=stream.epoch, last_sequence=0)
        recovered = snapshot['result']['clarify_batches']
        assert recovered == [{**projection, 'answers': {'q0': '["red", "blue"]'}, 'replayed': True}]
        assert snapshot['result']['open_requests'] == []
        assert not any(f['type'] == 'clarify.batch.request' for f in snapshot['result']['events'])
        terminal = await rpc(connection, 'clarify.batch.respond', session_id='s', request_id=rid, identity=identity,
                             result={'answers': {'q1': None}, 'outcome': 'submitted'})
        assert terminal.get('result') == {'status': 'ok'}, terminal
        await authority.sessions['s'].task
        assert seen == [{'answers': {'q0': '["red", "blue"]', 'q1': None}, 'outcome': 'submitted'}]
        from tests.waiting import async_wait_until
        await async_wait_until(lambda: any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames))
        withdrawal = next(f['params']['payload'] for f in peer.frames if f.get('params', {}).get('type') == 'request.cancel')
        assert withdrawal == {'id': rid, 'method': 'clarify', 'reason': 'resolved', 'identity': identity}
        replay = await rpc(connection, 'session.events.since', session_id='s', replay_epoch=stream.epoch, last_sequence=0)
        assert replay['result']['clarify_batches'] == []
        assert not any(f['type'] == 'clarify.batch.request' for f in replay['result']['events'])
        row = next(r for r in authority.persistence.admissions.list(session_id='s', pending_only=False) if r.request_id == 'batch-turn')
        assert row.payload['cli_turn_v1'] == {'clarify_batch': True, 'clarify_projection': 'canonical_batch_v1'}
        # Capability is not turn selection. A later legacy admission stays raw, and its genuine
        # unsupported-method error still wins native settlement (no global -32601 suppression).
        legacy_submit = await rpc(connection, 'prompt.submit', session_id='s', submission_id='legacy-turn', text='legacy',
                                  clarify_batch=True)
        assert legacy_submit['result']['status'] == 'queued'
        await async_wait_until(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        raw = next(f for f in peer.frames if f.get('method') == 'clarify')
        await connection.dispatch({'id': raw['id'], 'error': {'code': -32601, 'message': 'unsupported'}})
        await authority.sessions['s'].task
        assert seen[-1] == {'answers': {}, 'outcome': 'cancelled'}
        assert requests.canonical_batches(stream, 's') == []
    finally:
        requests.withdraw(authority.sessions['s'].event_stream, 's', 'clarify')
        if authority.sessions['s'].task:
            await authority.sessions['s'].task
        await connection.close()


@pytest.mark.asyncio
async def test_capability_selection_requires_ack_and_is_transactional(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    authority = await make_owner(tmp_path, monkeypatch, lambda event: None)
    peer = Peer()
    connection = AuthorityConnection(authority, peer, {'user_id': 'owner'})
    try:
        unoffered = await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1=True)
        assert unoffered.get('error', {}).get('message') == 'invalid_params'
        assert not getattr(peer, 'canonical_clarify_projection', False)
        await rpc(connection, 'client.capabilities')
        bad = await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1=True,
                        client={'name': 'Not A Name'})
        assert 'error' in bad
        assert not getattr(peer, 'canonical_clarify_projection', False)
        bad_type = await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1='true')
        assert bad_type.get('error', {}).get('message') == 'invalid_params'
    finally:
        await connection.close()
