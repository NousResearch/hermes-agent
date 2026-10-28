"""A cold ``/refine`` is guarded per session, never per profile.

``LiveSession.review`` is a bare ``SimpleNamespace`` (``gateway/session_refine.py`` builds a side
agent per call), so the review run it holds carries no session id of its own. Published under the
session's transcript target, two local sessions of one profile run their cold reviews side by
side, a second ``/refine`` on the same session still joins the running one, and the registry
answers that session's id with the cold run (so an automatic review on it is excluded in-process,
not only by the durable lease). Stubs as in ``test_local_refine_review_guard``.
"""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agent import review_admission
from tests.waiting import wait_until


def _stub_refine(monkeypatch, refine, lives, review):
    from gateway import session_tui_side, session_policy, session_local_recovery
    from agent import background_review
    from run_agent import AIAgent

    authority = SimpleNamespace(
        profile_id="p",
        sessions=dict(lives),
        db=SimpleNamespace(
            get_messages_as_conversation=lambda *a, **k: [
                {"role": "user", "content": "h"}
            ]
        ),
    )
    monkeypatch.setattr(
        refine, "_local_session", lambda r, s: (authority, lives[s.chat_id])
    )
    monkeypatch.setattr(session_policy, "policy_for_source", lambda r, s: object())
    monkeypatch.setattr(
        session_local_recovery, "transcript_target", lambda a, r: r.session_id
    )
    monkeypatch.setattr(session_tui_side, "_side_scope", lambda *a: nullcontext())

    def side_agent(*args, **kwargs):
        agent = object.__new__(AIAgent)
        agent.session_id = kwargs.get("session_id")
        agent.valid_tool_names = set()
        return agent

    monkeypatch.setattr(session_tui_side, "_side_agent", side_agent)
    monkeypatch.setattr(
        background_review,
        "spawn_background_review_thread",
        lambda agent, *a, review_run=None, **k: (lambda: review(agent, review_run), ""),
    )
    monkeypatch.setattr(
        background_review, "_interrupt_background_review", lambda *a, **k: None
    )
    threads = []

    def spawn(a, r, t, e, body, **kw):
        thread = threading.Thread(target=body, daemon=True)
        threads.append(thread)
        thread.start()

    monkeypatch.setattr(session_tui_side, "_spawn", spawn)
    return threads


@pytest.mark.asyncio
async def test_two_sessions_cold_refines_both_run_and_the_same_session_still_joins(
    monkeypatch,
):
    refine = pytest.importorskip("gateway.session_refine")
    from agent.background_review import finish_background_review_run
    from gateway.session_authority import LiveSession

    first_id, second_id = "refine-guard-session-a", "refine-guard-session-b"
    lives = {
        sid: LiveSession(SimpleNamespace(chat_id=sid), "route-" + sid)
        for sid in (first_id, second_id)
    }
    release, started = threading.Event(), []

    def review(agent, run):
        started.append(agent.session_id)
        release.wait(10)
        finish_background_review_run(agent, run)

    threads = _stub_refine(monkeypatch, refine, lives, review)
    try:
        first = await refine.refine_local_session(
            object(), lives[first_id].source, None
        )
        wait_until(lambda: started, what="the first review started")
        second = await refine.refine_local_session(
            object(), lives[second_id].source, None
        )
        wait_until(
            lambda: len(started) == 2, what="the second session's review started"
        )
        joined = await refine.refine_local_session(
            object(), lives[first_id].source, None
        )
        assert sorted(started) == [first_id, second_id], started
        # Each run is the registry's answer for ITS session: an automatic review there is excluded.
        assert (
            review_admission.current_review_run(first_id)
            is lives[first_id].review._background_review_run
        )
        assert (
            review_admission.current_review_run(second_id)
            is lives[second_id].review._background_review_run
        )
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    for reply in (first, second, joined):
        assert "Reviewing this conversation" in reply, reply
    assert len(started) == 2, (
        "a /refine on the same session started another review",
        joined,
    )
