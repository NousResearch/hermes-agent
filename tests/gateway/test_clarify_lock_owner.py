"""#140400: the batch ``clarify`` request a local viewer gets is answered with ``clarify.lock``.

Ink's batch panel and Desktop's batch card lock each answer with ``clarify.lock {request_id,
question_id, answer}``. The owner's ``AuthorityConnection.dispatch`` had no such verb (``-32601
unknown method``), so every answer was refused and the turn sat out ``clarify_timeout``. Driven
through the real dispatcher against a real ``ask_clarify_batch`` wait.
"""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.gateway.test_surface_admission import _authority

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


def _start_batch(authority):
    from gateway.session_clarify_batch import ask_clarify_batch
    peer = _Peer()
    authority.sessions['s'].event_stream.fanout.attach(peer)
    turn = SimpleNamespace(_approval_owner=(authority, 's', 1), _ctx=SimpleNamespace(
        source=SimpleNamespace(platform=Platform.LOCAL), agent_holder=[SimpleNamespace(_interrupt_requested=False)]))
    out = {}
    worker = threading.Thread(target=lambda: out.update(ask_clarify_batch(turn, QUESTIONS)), daemon=True)
    worker.start()
    return peer, worker, out


async def _lock(connection, rid, qid, answer, n):
    return await connection.dispatch({'id': n, 'method': 'clarify.lock', 'params': {
        'request_id': rid, 'question_id': qid, 'answer': answer}})


@pytest.mark.asyncio
async def test_clarify_lock_answers_the_owner_batch_request(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')

        first = await _lock(connection, rid, 'q0', 'blue', 2)
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        # Revisit before the last lock: the newer answer wins.
        assert (await _lock(connection, rid, 'q0', 'red', 3))['result']['remaining'] == ['q1']
        unknown = await _lock(connection, rid, 'zz', 'x', 4)
        assert unknown['error']['message'] == 'invalid_params', unknown
        last = await _lock(connection, rid, 'q1', None, 5)
        assert last.get('result') == {'status': 'ok', 'remaining': []}, last
        await asyncio.to_thread(worker.join, 5)
        assert out == {'answers': {'q0': 'red', 'q1': None}, 'outcome': 'submitted'}
        # Settled: every other viewer's card (and a classic CLI parked in its panel) is withdrawn.
        await _wait_for(lambda: any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames))
        cancel = next(f['params'] for f in peer.frames if f.get('params', {}).get('type') == 'request.cancel')
        assert cancel['payload'] == {'id': rid, 'method': 'clarify', 'reason': 'resolved'}
        assert 'red' not in repr(cancel)  # no answer rides the withdrawal
        late = await _lock(connection, rid, 'q1', 'late', 6)
        assert late.get('result') == {'status': 'expired'}, late
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_a_viewer_without_the_session_cannot_lock_its_answers(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_server_requests import resolve

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    outsider = AuthorityConnection(authority, object(), {'user_id': 'someone-else'})
    try:
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')
        refused = await _lock(outsider, rid, 'q0', 'red', 2)
        assert refused.get('result') == {'status': 'expired'}, refused
        resolve({'jsonrpc': '2.0', 'id': rid, 'result': {}})  # the session's own viewer cancels
        await asyncio.to_thread(worker.join, 5)
        assert out['answers'] == {}
    finally:
        await outsider.close()


def test_cli_turn_batch_answered_by_clarify_lock_from_a_second_viewer_end_to_end(tmp_path):
    """Real gateway process path: the locked answers reach the real model's tool result."""
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    repo = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures' / 'authority_clarify_lock_peer.py')],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (state / 'lock-passed').read_text() == 'both locked answers reached the real model'


@pytest.mark.asyncio
async def test_another_profiles_viewer_of_the_same_session_id_cannot_answer(tmp_path, monkeypatch):
    """The pending-request table is process-wide and session ids are only unique per profile. Two real
    authorities (separate homes and SQLite stores) both mount ``s``: profile B's own, attached viewer
    with ``session:respond`` must not lock or answer profile A's request, through ``clarify.lock`` or a
    raw response frame."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_server_requests import _open

    async def answer(event):
        return 'ok'
    a = await _authority(tmp_path / 'a', monkeypatch, answer)
    b = await _authority(tmp_path / 'b', monkeypatch, answer)
    b.profile_id = str(tmp_path / 'b')
    own = AuthorityConnection(a, object(), {'user_id': 'owner'})
    foreign = AuthorityConnection(b, object(), {'user_id': 'owner'})
    try:
        for connection in (own, foreign):
            resumed = await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
            assert 'result' in resumed, resumed
        peer, worker, out = _start_batch(a)
        await _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')
        refused = await _lock(foreign, rid, 'q0', 'FROM_PROFILE_B', 2)
        assert refused.get('result') == {'status': 'expired'}, refused
        foreign._resolve_response({'jsonrpc': '2.0', 'id': rid, 'result': {'answers': {'q0': 'B', 'q1': 'B'}}})
        assert rid in _open and not _open[rid].locked
        assert (await _lock(own, rid, 'q0', 'red', 3))['result'] == {'status': 'ok', 'remaining': ['q1']}
        assert (await _lock(own, rid, 'q1', 'big', 4))['result'] == {'status': 'ok', 'remaining': []}
        await asyncio.to_thread(worker.join, 5)
        assert out == {'answers': {'q0': 'red', 'q1': 'big'}, 'outcome': 'submitted'}
    finally:
        await own.close()
        await foreign.close()


@pytest.mark.asyncio
async def test_clarify_lock_params_are_checked_by_the_declared_canonical_contract(tmp_path, monkeypatch):
    """``clarify.lock`` is a declared ``CANONICAL_METHODS`` verb: the dispatcher refuses an unknown key or a
    mistyped field from ``ClarifyLockParams`` (``4001`` naming the field) before the handler runs."""
    from gateway.session_controls import AuthorityConnection
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    from tui_gateway.contracts.prompt_voice import ClarifyLockParams, ClarifyLockResult

    assert CANONICAL_METHODS['clarify.lock'].params is ClarifyLockParams
    assert CANONICAL_METHODS['clarify.lock'].result is ClarifyLockResult

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        extra = await connection.dispatch({'id': 1, 'method': 'clarify.lock', 'params': {
            'request_id': 'srq-x', 'question_id': 'q0', 'answer': 'a', 'bogus': 1}})
        assert extra['error']['data'] == {
            'reason': 'invalid_params', 'field': 'bogus', 'detail': 'does not match the session contract',
            'fields': ['bogus']}, extra
        typed = await connection.dispatch({'id': 2, 'method': 'clarify.lock', 'params': {
            'request_id': 7, 'question_id': 'q0'}})
        assert typed['error']['data'] == {
            'reason': 'invalid_params', 'field': 'request_id', 'detail': 'does not match the session contract',
            'fields': ['request_id']}, typed
        # A well-formed lock for a request nobody waits on is still the handler's own ``expired``.
        late = await connection.dispatch({'id': 3, 'method': 'clarify.lock', 'params': {
            'request_id': 'srq-x', 'question_id': 'q0', 'answer': None}})
        assert late['result'] == {'status': 'expired'}, late
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_an_unattached_connection_cannot_settle_the_batch_with_a_raw_response_frame(tmp_path, monkeypatch):
    """``clarify.lock`` and the closing response frame are two ways to settle the same request id: both go
    through one entitlement rule (``session_server_requests.may_settle``), so a same-profile connection that
    never attached the session can no more close the batch with a frame than lock one of its answers."""
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    outsider = AuthorityConnection(authority, object(), {'user_id': 'owner'})  # same actor, never attached
    try:
        await owner.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(f.get('method') == 'clarify' for f in peer.frames))
        rid = next(f['id'] for f in peer.frames if f.get('method') == 'clarify')
        from gateway.session_server_requests import _open
        # ``dispatch`` settles a response frame synchronously: still open afterwards means refused.
        await outsider.dispatch({'jsonrpc': '2.0', 'id': rid, 'result': {'answers': {'q0': 'X', 'q1': 'X'}}})
        assert rid in _open and not _open[rid].event.is_set() and out == {}, out
        await owner.dispatch({'jsonrpc': '2.0', 'id': rid, 'result': {'answers': {'q0': 'red', 'q1': 'big'}}})
        await asyncio.to_thread(worker.join, 5)
        assert out == {'answers': {'q0': 'red', 'q1': 'big'}, 'outcome': 'submitted'}, out
    finally:
        await outsider.close()
        await owner.close()
