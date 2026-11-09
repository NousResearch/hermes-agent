"""A remote Desktop window answers a decision box on its shared socket without naming the profile.

A remote app's window holds one host-scoped WebSocket per gateway, homed at the gateway's launch
profile. A request for another profile's chat names that profile and reaches it through a sibling
connection on the same socket, which is the connection that attaches the chat and receives its batch
``clarify`` request. A window opened for that other profile (``?profile=``) takes it for the socket's
own, so the card's ``clarify.lock`` names no profile and reaches the launch connection. The lock must
settle the request the sibling attached, as the closing response frame for the same request id does
(``AuthorityConnection._resolve_response``). It was answered ``expired`` while the turn kept waiting,
and the card took that for an answer.
"""
import asyncio
import itertools
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
from tests.gateway.test_ticket_profile_scope import _Transport

QUESTIONS = [{'qid': 'q0', 'question': 'colour?', 'choices': ['red', 'blue'], 'multi_select': False},
             {'qid': 'q1', 'question': 'size?', 'choices': None, 'multi_select': False}]


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


async def _wait_for(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline
        await asyncio.sleep(0.01)


def _host_connection(runner, launch):
    """A remote window's socket: a host-scope ticket for the launch profile, as the Desktop app gets."""
    from gateway.session_authorities import owner_scope
    from gateway.session_controls import AuthorityConnection
    store = runner.session_ticket_store
    ticket = store.mint(profile_id=launch.profile_id, subject='uid:desk', purpose='interactive', scope='host')
    grant = store.redeem(ticket, profile_id=None, purpose='interactive')
    identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                'native_bootstrap': True, 'profile_scope': 'host'}
    with owner_scope(launch):
        return AuthorityConnection(launch, _Transport(), identity, operator=True)


def _start_batch(authority, session_id):
    from gateway.session_clarify_batch import ask_clarify_batch
    peer = _Peer()
    authority.sessions[session_id].event_stream.fanout.attach(peer)
    turn = SimpleNamespace(_approval_owner=(authority, session_id, 1), _ctx=SimpleNamespace(
        source=SimpleNamespace(platform=Platform.LOCAL), agent_holder=[SimpleNamespace(_interrupt_requested=False)]))
    out = {}
    worker = threading.Thread(target=lambda: out.update(ask_clarify_batch(turn, QUESTIONS)), daemon=True)
    worker.start()
    return peer, worker, out


@pytest.mark.asyncio
async def test_an_unnamed_lock_on_a_shared_socket_settles_the_request_its_sibling_attached(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_server_requests import _open, resolve
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    beta_home = homes[2][1]
    connections, runner, rid = [], None, None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        launch = authority_for_profile_id(runner, str(root.resolve()))
        window = _host_connection(runner, launch)
        other = _host_connection(runner, launch)  # the same user's other window, which never opened the chat
        connections += [window, other]
        ids = itertools.count(1)

        async def send(connection, method, params):
            with owner_scope(launch):
                return await connection.dispatch({'id': next(ids), 'method': method, 'params': params})

        created = await send(window, 'session.create', {'profile': 'beta', 'request_id': 'c1', 'source': 'cli',
                                                        'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
        sid = created['result']['session_id']
        assert 'result' in await send(window, 'session.resume', {'profile': 'beta', 'session_id': sid})
        assert 'result' in await send(other, 'session.list', {'profile': 'beta'})  # it has a beta sibling too
        beta = runner.session_authorities.for_home(beta_home)
        peer, worker, out = _start_batch(beta, sid)
        await _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')

        def lock(qid, answer):
            return {'request_id': rid, 'question_id': qid, 'answer': answer}

        refused = await send(other, 'clarify.lock', lock('q0', 'OTHER'))
        assert refused.get('result') == {'status': 'expired'}, refused
        assert rid in _open and not _open[rid].locked

        first = await send(window, 'clarify.lock', lock('q0', 'red'))
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        last = await send(window, 'clarify.lock', lock('q1', 'big'))
        assert last.get('result') == {'status': 'ok', 'remaining': []}, last
        await asyncio.to_thread(worker.join, 5)
        assert out == {'answers': {'q0': 'red', 'q1': 'big'}, 'outcome': 'submitted'}, out
        rid = None
    finally:
        if rid is not None:
            resolve({'jsonrpc': '2.0', 'id': rid, 'result': {}})  # a failed assertion must not leave the wait open
        for connection in connections:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in getattr(runner, 'session_authorities', None) or []:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
