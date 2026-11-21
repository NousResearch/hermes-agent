"""A delegated child's own transcript, written by an async delegation a claimed turn dispatched.

The child runner carries its parent turn's identity as work that outlives the turn (the owner-epoch half of the fence)
but writes its OWN session's rows, through the same turn-path flush (``agent.session_persistence._db_flush_write``) and
the same store. While the owner holds the epoch the child's rows land, also after the parent session's next claim;
once another owner takes the store over, the child's flush is refused and its transcript stays empty.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import tools.async_delegation as ad
from persistence import SessionId, TurnFenceSuperseded
from persistence.conversation.sessions import NewSession, SessionLineageKind
from tests.persistence.hosted._turn_rows import flush
from tests.persistence.hosted.test_turn_fence_lifetime import S, _next_claim, _take_over, _turn

CHILD = SessionId("delegated-child")


@pytest.mark.parametrize(("supersession", "outcome"), [(None, "written"), ("next_claim", "written"), ("takeover", "refused")])
def test_a_delegated_childs_transcript_keeps_the_owner_half_of_the_fence(claimed, monkeypatch, supersession, outcome):
    store, epoch, row = claimed
    store.sessions.create(NewSession(session_id=CHILD, source="subagent", parent_session_id=S,
                                     lineage_kind=SessionLineageKind.DELEGATE))
    ready, go, done = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    def runner():
        ready.set()
        assert go.wait(10)
        child = SimpleNamespace(persistence=store, _session_db=store, session_id=CHILD)
        try:
            flush(child, "delegated transcript")
            outcomes.append("written")
        except TurnFenceSuperseded:
            outcomes.append("refused")
        finally:
            done.set()
        return {"status": "completed"}

    monkeypatch.setattr(ad, "_persist_dispatch", lambda record: None)
    monkeypatch.setattr(ad, "_finalize", lambda *_a, **_k: None)
    with _turn(store, row):
        handle = ad.dispatch_async_delegation(goal="probe", context=None, toolsets=None, role="leaf", model=None,
                                              session_key="", parent_session_id=S, runner=runner)
    assert handle["status"] == "dispatched"
    assert ready.wait(10)
    if supersession == "next_claim":
        _next_claim(store, epoch, row)
    elif supersession == "takeover":
        _take_over(store)
    go.set()
    assert done.wait(10)
    assert outcomes == [outcome]
    assert [m.content for m in store.messages.read_transcript(CHILD)] == (["delegated transcript"] if outcome == "written" else [])
