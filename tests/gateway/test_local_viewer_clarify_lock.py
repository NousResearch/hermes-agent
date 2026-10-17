"""A Desktop or Ink viewer answers a classic CLI batch ``clarify`` with ``clarify.lock``.

The owner sends one ``clarify`` server request to every viewer on the session. Those viewers lock
each question through ``clarify.lock``; the last lock resolves the batch with the whole answer set.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from tests.waiting import async_wait_until


class _Peer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


QUESTIONS = [{'qid': 'q0', 'question': 'colour?', 'choices': ['red', 'blue'], 'multi_select': False},
             {'qid': 'q1', 'question': 'size?', 'choices': None, 'multi_select': False}]


async def _lock(connection, request_id, question_id, answer, rpc_id='lock'):
    params = {'request_id': request_id, 'question_id': question_id, 'answer': answer}
    return await connection.dispatch({
        'jsonrpc': '2.0', 'id': rpc_id, 'method': 'clarify.lock', 'params': params})


async def _open_batch(tmp_path, monkeypatch, *, timeout=3, operator=False):
    from gateway.session_clarify_batch import ask_clarify_batch
    from gateway.session_controls import AuthorityConnection
    import tools.clarify_gateway as cg
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(cg, 'get_clarify_timeout', lambda: timeout)
    viewer = _Peer()
    authority.sessions['s'].event_stream.fanout.attach(viewer)
    agent = SimpleNamespace(_interrupt_requested=False)
    turn = SimpleNamespace(_approval_owner=(authority, 's', 1), _ctx=SimpleNamespace(
        source=SimpleNamespace(platform=Platform.LOCAL), agent_holder=[agent]))
    out = {}
    worker = threading.Thread(target=lambda: out.update(ask_clarify_batch(turn, QUESTIONS)))
    worker.start()
    frame = await async_wait_until(
        lambda: next((f for f in viewer.frames if f.get('method') == 'clarify'), None),
        timeout=5, what='the clarify server request')
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'}, operator=operator)
    return SimpleNamespace(authority=authority, connection=connection, request_id=frame['id'],
                           worker=worker, out=out, agent=agent, viewer=viewer)


async def _join(batch):
    await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)


def _cancels(viewer):
    return [frame for frame in viewer.frames
            if frame.get('method') == 'event' and (frame.get('params') or {}).get('type') == 'request.cancel']


@pytest.mark.asyncio
async def test_viewer_clarify_lock_answers_the_cli_batch(tmp_path, monkeypatch):
    """Desktop's card and Ink's overlay lock each question; the last lock submits the batch."""
    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        second = await _lock(batch.connection, batch.request_id, 'q1', 'large', 'lock-q1')
        await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        assert second.get('result') == {'status': 'ok', 'remaining': []}, second
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_a_panel_answer_after_the_last_lock_does_not_change_the_batch(tmp_path, monkeypatch):
    """The classic CLI panel can still answer once Desktop locked every question; that answer is late."""
    from gateway import session_sudo_prompt

    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        await _lock(batch.connection, batch.request_id, 'q1', 'large', 'lock-q1')
        await _join(batch)
        late = await batch.connection.dispatch({
            'jsonrpc': '2.0', 'id': batch.request_id,
            'result': {'answers': {'q0': 'blue', 'q1': 'small'}, 'outcome': 'submitted'}})
        assert late is None
        assert batch.request_id not in session_sudo_prompt._open
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_clarify_lock_rejects_a_missing_id_or_unknown_question(tmp_path, monkeypatch):
    """A missing id or a question the batch did not ask is ``invalid_params``; the batch stays open."""
    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        unknown = await _lock(batch.connection, batch.request_id, 'nope', 'red', 'lock-unknown')
        assert unknown['error']['message'] == 'invalid_params', unknown
        for params, rpc_id in (
            ({'question_id': 'q0', 'answer': 'red'}, 'missing-request'),
            ({'request_id': batch.request_id, 'answer': 'red'}, 'missing-question'),
            ({}, 'missing-both'),
        ):
            reply = await batch.connection.dispatch({
                'jsonrpc': '2.0', 'id': rpc_id, 'method': 'clarify.lock', 'params': params})
            assert reply['error']['message'] == 'invalid_params', reply
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        second = await _lock(batch.connection, batch.request_id, 'q1', 'large', 'lock-q1')
        await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        assert first.get('result', {}).get('status') == 'ok', first
        assert second.get('result') == {'status': 'ok', 'remaining': []}, second
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_viewer_without_session_respond_cannot_lock_the_batch(tmp_path, monkeypatch):
    """``session:respond`` is required on the batch's own session; a refusal leaves the batch open."""
    from gateway.session_controls import AuthorityConnection
    from gateway import session_sudo_prompt

    batch = await _open_batch(tmp_path, monkeypatch)
    guest = AuthorityConnection(batch.authority, object(), {
        'user_id': 'guest', 'capabilities': ['session:read']})
    try:
        refused = await _lock(guest, batch.request_id, 'q0', 'stolen', 'lock-guest')
        assert refused.get('result') == {'status': 'expired'}, refused
        assert session_sudo_prompt._open[batch.request_id].locked == {}
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        second = await _lock(batch.connection, batch.request_id, 'q1', 'large', 'lock-q1')
        await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        assert second.get('result') == {'status': 'ok', 'remaining': []}, second
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        await guest.close()
        await batch.connection.close()


@pytest.mark.asyncio
async def test_relock_keeps_the_latest_answer_and_a_late_lock_expires(tmp_path, monkeypatch):
    """A question stays editable until every one is locked; a lock after that is ``expired``."""
    from gateway import session_sudo_prompt

    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        encoded = await _lock(batch.connection, batch.request_id, 'q0', ['red'], 'lock-json')
        assert encoded.get('result') == {'status': 'ok', 'remaining': ['q1']}, encoded
        assert session_sudo_prompt._open[batch.request_id].locked['q0'] == '["red"]'
        revised = await _lock(batch.connection, batch.request_id, 'q0', 'blue', 'lock-redo')
        assert revised.get('result') == {'status': 'ok', 'remaining': ['q1']}, revised
        skipped = await _lock(batch.connection, batch.request_id, 'q1', None, 'lock-skip')
        assert skipped.get('result') == {'status': 'ok', 'remaining': []}, skipped
        late = await _lock(batch.connection, batch.request_id, 'q0', 'too-late', 'lock-late')
        await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        assert late.get('result') == {'status': 'expired'}, late
        assert batch.out == {'answers': {'q0': 'blue', 'q1': None}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await asyncio.get_running_loop().run_in_executor(None, batch.worker.join, 6)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_timeout_keeps_the_answers_already_locked(tmp_path, monkeypatch):
    """The clarify deadline ends the wait and keeps every question locked before it."""
    batch = await _open_batch(tmp_path, monkeypatch, timeout=1)
    try:
        first = await _lock(batch.connection, batch.request_id, 'q0', 'kept', 'lock-q0')
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        await _join(batch)
        assert batch.out == {'answers': {'q0': 'kept'}, 'outcome': 'timed_out'}, batch.out
        cancels = _cancels(batch.viewer)
        assert [frame['params']['payload']['id'] for frame in cancels] == [batch.request_id]
        assert cancels[0]['params']['payload']['reason'] == 'timeout'
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_interrupt_keeps_the_answers_already_locked(tmp_path, monkeypatch):
    """Stopping the turn keeps the answers locked so far and reports the batch cancelled."""
    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        first = await _lock(batch.connection, batch.request_id, 'q0', 'kept', 'lock-q0')
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        batch.agent._interrupt_requested = True
        await _join(batch)
        assert batch.out == {'answers': {'q0': 'kept'}, 'outcome': 'cancelled'}, batch.out
        assert _cancels(batch.viewer)[-1]['params']['payload']['reason'] == 'interrupted'
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_cancel_all_keeps_the_answers_already_locked(tmp_path, monkeypatch):
    """Ink Esc and Desktop Skip send an empty result; locks already accepted still stand."""
    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        reply = await batch.connection.dispatch({
            'jsonrpc': '2.0', 'id': batch.request_id, 'result': {}})
        assert reply is None
        await _join(batch)
        assert batch.out == {'answers': {'q0': 'red'}, 'outcome': 'cancelled'}, batch.out
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_a_partial_response_frame_keeps_the_answers_already_locked(tmp_path, monkeypatch):
    """A closing frame overlays the locks. A question it omits stays, and its outcome is kept."""
    batch = await _open_batch(tmp_path, monkeypatch)
    try:
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        await batch.connection.dispatch({
            'jsonrpc': '2.0', 'id': batch.request_id,
            'result': {'answers': {'q1': 'large'}, 'outcome': 'timed_out'}})
        await _join(batch)
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'timed_out'}, batch.out
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()


@pytest.mark.asyncio
async def test_clarify_lock_is_served_during_restart_wait(tmp_path, monkeypatch):
    """An operator socket admitted during the restart's after-turn wait may lock the parked batch."""
    batch = await _open_batch(tmp_path, monkeypatch, operator=True)
    batch.authority.runner._draining = True
    batch.authority.runner._running = True
    batch.authority.runner.session_runtime_descriptor = {
        'state': 'draining', 'drain_reason': 'restart_wait'}
    try:
        refused = await batch.connection.dispatch({
            'jsonrpc': '2.0', 'id': 'bad-respond', 'method': 'clarify.respond', 'params': {}})
        assert refused['error']['message'] == 'invalid_params', refused
        first = await _lock(batch.connection, batch.request_id, 'q0', 'red', 'lock-q0')
        assert first.get('result') == {'status': 'ok', 'remaining': ['q1']}, first
        second = await _lock(batch.connection, batch.request_id, 'q1', 'large', 'lock-q1')
        await _join(batch)
        assert second.get('result') == {'status': 'ok', 'remaining': []}, second
        assert batch.out == {'answers': {'q0': 'red', 'q1': 'large'}, 'outcome': 'submitted'}, batch.out
    finally:
        if batch.worker.is_alive():
            await _join(batch)
        await batch.connection.close()
