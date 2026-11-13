"""A cold ``/refine`` review belongs to the session, not to its throwaway side agent (T7).

The resident agent's review is fenced by one per-agent guard: a second ``/refine`` is refused while
it runs, and the session's next turn cancels it. The cold path builds a fresh side agent per call,
so its guard must live on the session (``LiveSession.review``) for both to hold. The real refine body,
review-run handshake, canceller and ``SessionAuthority._execute_claimed`` run; agent construction,
review inference and the admitted turn are stubbed.
"""
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until


def _stub_refine(monkeypatch, refine, live, sid, review):
    from gateway import session_tui_side, session_policy, session_local_recovery
    from agent import background_review
    from run_agent import AIAgent
    authority = SimpleNamespace(profile_id='p', sessions={sid: live},
        db=SimpleNamespace(get_messages_as_conversation=lambda *a, **k: [{'role': 'user', 'content': 'h'}]))
    monkeypatch.setattr(refine, '_local_session', lambda r, s: (authority, live))
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda r, s: object())
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda a, r: sid)
    monkeypatch.setattr(session_tui_side, '_side_scope', lambda *a: nullcontext())

    def side_agent(*args, **kwargs):
        agent = object.__new__(AIAgent)
        agent.session_id = kwargs.get('session_id')
        agent.valid_tool_names = set()
        return agent
    monkeypatch.setattr(session_tui_side, '_side_agent', side_agent)
    monkeypatch.setattr(background_review, 'spawn_background_review_thread',
                        lambda agent, *a, review_run=None, **k: (lambda: review(agent, review_run), ''))
    monkeypatch.setattr(background_review, '_interrupt_background_review', lambda *a, **k: None)
    threads = []

    def spawn(a, r, t, e, body, **kw):
        thread = threading.Thread(target=body, daemon=True)
        threads.append(thread)
        thread.start()
    monkeypatch.setattr(session_tui_side, '_spawn', spawn)
    return threads


@pytest.mark.asyncio
async def test_second_cold_refine_while_one_runs_is_refused(monkeypatch):
    refine = pytest.importorskip('gateway.session_refine')
    from agent.background_review import finish_background_review_run
    from gateway.session_authority import LiveSession
    sid = 'refine-guard-dup'
    live = LiveSession(SimpleNamespace(chat_id=sid), 'route')
    release, started = threading.Event(), []

    def review(agent, run):
        started.append(run)
        release.wait(10)
        finish_background_review_run(agent, run)
    threads = _stub_refine(monkeypatch, refine, live, sid, review)
    try:
        first = await refine.refine_local_session(object(), live.source, None)
        wait_until(lambda: started, what='the first review started')
        second = await refine.refine_local_session(object(), live.source, None)
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    assert 'Reviewing this conversation' in first, first
    # Deduped, as the resident agent's guard does (``_spawn_background_review`` returns early).
    assert len(started) == 1, ('a second /refine started another review', second)


@pytest.mark.asyncio
async def test_new_turn_cancels_the_cold_refine_review(monkeypatch):
    refine = pytest.importorskip('gateway.session_refine')
    from agent.background_review import finish_background_review_run
    from gateway import session_finite
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    sid = 'refine-guard-turn'
    live = LiveSession(SimpleNamespace(chat_id=sid), 'route')
    started, outcome = [], {}

    def review(agent, run):
        assert run.begin_request(agent)
        started.append(run)
        outcome['cancelled'] = run.cancel_requested.wait(5)
        finish_background_review_run(agent, run)
    threads = _stub_refine(monkeypatch, refine, live, sid, review)

    async def turn(authority, ref, row):
        outcome['turn_saw_review_done'] = started[0].request_done.is_set()
        return 'ok'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', turn)
    try:
        await refine.refine_local_session(object(), live.source, None)
        wait_until(lambda: started, what='the review started')
        owner = SimpleNamespace(executing={})
        result = await SessionAuthority._execute_claimed(owner, SessionRef('p', sid), live, {'admission_id': 'a'})
    finally:
        for thread in threads:
            thread.join(10)
    assert result == ('ok', 'completed'), result
    assert outcome.get('cancelled') is True, 'the new turn left the cold /refine review running'
    assert outcome['turn_saw_review_done'], 'the turn started before the review acknowledged'
