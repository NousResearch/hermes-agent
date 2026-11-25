"""Hidden rows participate in the SQLite session counts, but not in default listings."""

from persistence.conversation.sessions import NewSession, SessionCountQuery, SessionQuery, SessionWindow


def test_counts_include_hidden_sessions(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    for sid in ("visible", "hidden"):
        sessions.create(NewSession(sid, "cli"))
    sessions.set_hidden("hidden", True)
    assert sessions.count_sessions(SessionCountQuery()) == 2
    assert sessions.count_by_source() == {"cli": 2}
    assert sessions.count_at_least(2)
    assert not sessions.count_at_least(3)
    page = sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
    assert [row.session.session_id for row in page] == ["visible"]
