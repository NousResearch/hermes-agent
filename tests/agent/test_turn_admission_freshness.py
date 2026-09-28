"""A free turn lease must not admit a stale cross-surface snapshot."""
from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tests.agent.test_cross_process_turn_lease import _agent_with_db


@pytest.mark.parametrize("change", ["unchanged", "append", "rotate", "compact"])
@pytest.mark.parametrize("contended", [False, True])
def test_admission_uses_current_durable_context(tmp_path, monkeypatch, change, contended):
    db = SessionDB(tmp_path / "state.db")
    other = SessionDB(tmp_path / "state.db")
    sid, tip, holder = "original", "continuation", "fixture-writer"
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="first input")
    db.append_message(sid, role="assistant", content="first answer")
    history = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    assert other.try_acquire_session_turn_lease(sid, holder)
    target = sid
    if change == "append":
        other.append_message(sid, role="user", content="external input")
        other.append_message(sid, role="assistant", content="external answer")
    elif change == "rotate":
        other.end_session(sid, "compression")
        other.create_session(tip, source="cli", parent_session_id=sid)
        other.append_message(tip, role="user", content="current compacted context")
        target = tip
    elif change == "compact":
        other.replace_messages(sid, [{"role": "user", "content": "current compacted context"}])
    if not contended:
        other.release_session_turn_lease(sid, holder)
    expected = other.get_messages_as_conversation(target, repair_alternation=True, include_row_ids=True)
    agent = _agent_with_db(db, session_id=sid)
    waited = []

    def status(kind, text=None):
        if text and "waiting for it to finish" in text:
            waited.append(True)
            other.release_session_turn_lease(sid, holder)

    def load(actual_sid):
        assert actual_sid == target
        assert not other.try_acquire_session_turn_lease(sid, "competing-writer")
        current = db.get_messages_as_conversation(actual_sid, repair_alternation=True, include_row_ids=True)
        return history if current == history else current

    def loop(actual_agent, message, system, actual_history, *args, **kwargs):
        assert not other.try_acquire_session_turn_lease(sid, "competing-writer")
        assert message == "not-yet-persisted input"
        assert actual_agent.session_id == target
        assert actual_history == expected
        if change == "unchanged" and not contended:
            assert actual_history is history
        return {"messages": actual_history, "final_response": "ok"}

    agent.status_callback = status
    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 3)
    try:
        assert agent.run_conversation("not-yet-persisted input", conversation_history=history, conversation_history_loader=load)["final_response"] == "ok"
        assert bool(waited) == contended
        assert other.try_acquire_session_turn_lease(sid, "after")
        other.release_session_turn_lease(sid, "after")
    finally:
        other.release_session_turn_lease(sid, holder)
        other.release_session_turn_lease(sid, "competing-writer")
        other.close()
        db.close()


@pytest.mark.parametrize("external_append", [False, True])
def test_unpersisted_tail_survives_reconciliation(tmp_path, monkeypatch, external_append):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="cli")
    db.append_message("shared", role="user", content="first")
    db.append_message("shared", role="assistant", content="answer")
    history = db.get_messages_as_conversation("shared", include_row_ids=True)
    pending = {"role": "user", "content": "unsaved caller note"}
    history.append(pending)
    if external_append:
        db.append_message("shared", role="user", content="external")
        db.append_message("shared", role="assistant", content="external answer")
    expected = db.get_messages_as_conversation("shared", include_row_ids=True) + [pending]
    agent = _agent_with_db(db, session_id="shared")

    def loop(_agent, message, system, actual, *args, **kwargs):
        assert actual == expected
        assert actual[-1] is pending
        if not external_append:
            assert actual is history
        return {"messages": actual, "final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        agent.run_conversation("next", conversation_history=history,
                               conversation_history_loader=lambda sid: history if not external_append else expected)
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["exception", "invalid-result"])
def test_loader_failure_releases_the_real_lease(tmp_path, monkeypatch, failure):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="cli")
    agent = _agent_with_db(db, session_id="shared")

    def load(sid):
        if failure == "exception":
            raise RuntimeError("snapshot unavailable")
        return None

    monkeypatch.setattr("agent.conversation_loop.run_conversation", lambda *a, **k: pytest.fail("model ran"))
    try:
        with pytest.raises(RuntimeError if failure == "exception" else TypeError):
            agent.run_conversation("next", conversation_history_loader=load)
        assert db.try_acquire_session_turn_lease("shared", "after-failure")
        db.release_session_turn_lease("shared", "after-failure")
    finally:
        db.close()


@pytest.mark.parametrize("supplied", [True, False])
def test_explicit_history_is_authoritative_but_omitted_history_loads_db(tmp_path, monkeypatch, supplied):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="cli")
    db.append_message("shared", role="user", content="durable")
    explicit = [{"role": "user", "content": "client-controlled context", "_db_persisted": True}]
    agent = _agent_with_db(db, session_id="shared")

    def loop(_agent, message, system, history, *args, **kwargs):
        if supplied:
            assert history is explicit
        else:
            assert [m["content"] for m in history] == ["durable"]
        return {"messages": history, "final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        kw = {} if supplied else {"conversation_history_loader": lambda sid: db.get_messages_as_conversation(sid)}
        agent.run_conversation("next", conversation_history=explicit if supplied else None, **kw)
    finally:
        db.close()
