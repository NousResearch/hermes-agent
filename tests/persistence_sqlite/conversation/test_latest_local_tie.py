"""The roster's latest local conversation does not rank tied rows by their identifiers."""

from unittest.mock import patch

from persistence.conversation.sessions import NewSession


def test_latest_local_keeps_the_first_equal_timestamp_row(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    with patch("time.time", return_value=1700000000):
        sessions.create(NewSession("sess-lls-1-old", "cli", chat_id="local-old", user_id="owner"))
        sessions.create(NewSession("sess-lls-2-new", "cli", chat_id="local-new", user_id="owner"))
    assert sessions.roster_sessions(owner_user_id="owner", owner=None).last_session.id == "sess-lls-1-old"
    sessions.touch_activity("sess-lls-2-new", 1700000100)
    assert sessions.roster_sessions(owner_user_id="owner", owner=None).last_session.id == "sess-lls-2-new"
