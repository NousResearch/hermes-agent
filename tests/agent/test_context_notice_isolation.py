"""Isolation and failure-safe replay across actual session/profile databases."""
import sqlite3

import pytest

from hermes_state import SessionDB
from tests.agent.test_context_notices import _agent, _outcome
from tui_gateway import server


@pytest.mark.parametrize("scope", ["session", "profile"])
def test_notice_identity_and_recovery_are_scoped(monkeypatch, tmp_path, scope):
    homes = [tmp_path / "one", tmp_path / ("two" if scope == "profile" else "one")]
    ids = ["same-id", "same-id" if scope == "profile" else "other-id"]
    events, keys = [], []
    monkeypatch.setattr(server, "_sessions", {})
    for index, (home, session_id) in enumerate(zip(homes, ids)):
        db = SessionDB(db_path=home / "state.db")
        db.create_session(session_id, source="gui")
        agent = _agent(db, monkeypatch, events, session_id)
        _outcome(agent, "first")
        _outcome(agent, "second")
        keys.append(events[-1][2]["key"])
        db.close()
        server._sessions[f"runtime-{index}"] = {"session_key": session_id, "profile_home": str(home), "agent": None}
    assert keys[0] != keys[1]
    events.clear()
    for index in range(2):
        server._methods["session.events.since"](1, {"session_id": f"runtime-{index}"})
    assert [e[2]["key"] for e in events] == keys
    db = SessionDB(db_path=homes[0] / "state.db")
    try:
        agent = _agent(db, monkeypatch, events, ids[0])
        _outcome(agent, "healthy", failure=None, committed=True)
        events.clear()
        for index in range(2):
            server._methods["session.events.since"](1, {"session_id": f"runtime-{index}"})
        assert events[0][:2] == ("notification.clear", "runtime-0")
        assert events[0][2] == {"key": keys[0], "state_key": keys[0],
                                "state_revision": db.get_context_notice_state(ids[0])["revision"]}
        assert events[1][0] == "notification.show" and events[1][2]["key"] == keys[1]
        events.clear()
        with monkeypatch.context() as broken:
            def unreadable(*args, **kwargs):
                raise sqlite3.OperationalError("unavailable")
            broken.setattr(SessionDB, "get_context_notice_state", unreadable)
            server._methods["session.events.since"](1, {"session_id": "runtime-1"})
        assert events == [], "an unreadable snapshot is unknown, not recovery"
    finally:
        db.close()
