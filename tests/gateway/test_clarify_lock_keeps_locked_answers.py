"""A batch answer already locked with ``clarify.lock`` survives every way the wait can end, as on main's
sidecar (``tui_gateway/server_requests.py``: "locked answers survive a timeout"): a timeout returns
``{answers: <locked so far>, outcome: timed_out}``, an interrupt returns them with ``cancelled``, and a
cancel-all closing frame (Ink Esc / Desktop Skip: no ``answers``) keeps them and reports ``cancelled``,
never ``submitted``. Driven through the real dispatcher against a real ``ask_clarify_batch`` wait.
"""
import asyncio

import pytest

from tests.gateway.test_clarify_lock_owner import _lock, _start_batch, _wait_for
from tests.gateway.test_surface_admission import _authority


async def _locked_then(tmp_path, monkeypatch, end):
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
        assert (await _lock(connection, rid, 'q0', 'blue', 2))['result'] == {'status': 'ok', 'remaining': ['q1']}
        end(connection, rid, worker)
        await asyncio.to_thread(worker.join, 10)
        assert not worker.is_alive()
        return out, peer
    finally:
        await connection.close()


def _frame_of(worker):
    import sys
    frame = sys._current_frames()[worker.ident]
    while frame is not None and 'turn' not in frame.f_locals:
        frame = frame.f_back
    return frame.f_locals['turn']


@pytest.mark.asyncio
async def test_a_timeout_keeps_the_answer_already_locked(tmp_path, monkeypatch):
    from tools import clarify_gateway
    monkeypatch.setattr(clarify_gateway, 'get_clarify_timeout', lambda: 1)
    out, peer = await _locked_then(tmp_path, monkeypatch, lambda *a: None)
    assert out == {'answers': {'q0': 'blue'}, 'outcome': 'timed_out'}, out
    # The open cards still close.
    assert any(f.get('params', {}).get('type') == 'request.cancel' for f in peer.frames)


@pytest.mark.asyncio
async def test_an_interrupt_keeps_the_answer_already_locked(tmp_path, monkeypatch):
    def interrupt(connection, rid, worker):
        _frame_of(worker)._ctx.agent_holder[0]._interrupt_requested = True
    out, _peer = await _locked_then(tmp_path, monkeypatch, interrupt)
    assert out == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}, out


@pytest.mark.asyncio
async def test_a_cancel_all_after_a_lock_keeps_the_answer_and_reports_cancelled(tmp_path, monkeypatch):
    def skip(connection, rid, worker):
        connection._resolve_response({'jsonrpc': '2.0', 'id': rid, 'result': {}})
    out, _peer = await _locked_then(tmp_path, monkeypatch, skip)
    assert out == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}, out


@pytest.mark.asyncio
async def test_a_partial_closing_frame_after_a_lock_is_submitted_with_both(tmp_path, monkeypatch):
    def submit_tail(connection, rid, worker):
        connection._resolve_response({'jsonrpc': '2.0', 'id': rid, 'result': {'answers': {'q1': 'large'}}})
    out, _peer = await _locked_then(tmp_path, monkeypatch, submit_tail)
    assert out == {'answers': {'q0': 'blue', 'q1': 'large'}, 'outcome': 'submitted'}, out


@pytest.mark.asyncio
async def test_an_error_frame_after_a_lock_keeps_the_answer(tmp_path, monkeypatch):
    def fail(connection, rid, worker):
        connection._resolve_response({'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32603, 'message': 'x'}})
    out, _peer = await _locked_then(tmp_path, monkeypatch, fail)
    assert out == {'answers': {'q0': 'blue'}, 'outcome': 'cancelled'}, out
