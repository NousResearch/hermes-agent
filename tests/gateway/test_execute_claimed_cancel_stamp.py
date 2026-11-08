"""A drain cancelled while its turn waits on a cold ``/refine`` review's cancel leaves no executing stamp.

``_stamp_claim`` sets ``executing[sid]`` before ``_execute_claimed`` runs; session retirement skips an
executing session, so a stamp left behind pins the session in memory forever. The real review-run
handshake and canceller run; only the admitted turn is stubbed.
"""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_a_drain_cancelled_during_the_review_cancel_clears_the_executing_stamp(monkeypatch):
    from agent.background_review import finish_background_review_run, prepare_background_review_run
    from gateway import session_finite
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    ran = []

    async def turn(authority, ref, row):
        ran.append(row['admission_id'])
        return 'ok'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', turn)
    ref = SessionRef('p', 'review-cancel-stamp')
    live = LiveSession(SimpleNamespace(chat_id=ref.session_id), 'route')
    run = prepare_background_review_run(live.review)  # a review still in its request phase
    owner = SimpleNamespace(executing={ref.session_id: 7})  # as _stamp_claim left it
    task = asyncio.ensure_future(SessionAuthority._execute_claimed(owner, ref, live, {'admission_id': 'a'}))
    try:
        # The canceller has fenced the review and now waits (off-loop) for its acknowledgement.
        assert await asyncio.to_thread(run.cancel_requested.wait, 10), 'the turn never cancelled the review'
        task.cancel()
        await asyncio.wait({task}, timeout=10)
    finally:
        finish_background_review_run(live.review, run)  # releases the canceller's bounded wait
    assert task.cancelled() and ran == [], (task, ran)
    assert ref.session_id not in owner.executing, owner.executing
