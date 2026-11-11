"""The SQLite listing reads carry the corrupt ``model_config`` flag of the stored cell, as ``get`` does."""

import sqlite3

from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from tests.persistence_sqlite.conftest import store_for


def test_listing_rows_flag_a_corrupt_cell(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("bad", "cli"))
    sessions.create(NewSession("good", "cli", model_config={"model": "m"}))
    with sqlite3.connect(store_for(scope_a)) as connection:
        connection.execute("UPDATE sessions SET model_config = '{oops' WHERE id = 'bad'")
    listed = {e.session.session_id: e.session for e in sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))}
    assert listed["bad"].model_config_corrupt is True and dict(listed["bad"].model_config) == {}
    assert listed["good"].model_config_corrupt is False and dict(listed["good"].model_config) == {"model": "m"}
    assert sessions.summary("bad").session.model_config_corrupt is True
    assert [e.session.model_config_corrupt for e in sessions.search_sessions() if e.session.session_id == "bad"] == [True]
