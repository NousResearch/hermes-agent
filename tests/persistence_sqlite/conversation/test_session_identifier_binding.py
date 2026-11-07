"""Session identifiers reach the SQLite store without stringifying unsupported values."""

import sqlite3

import pytest

from persistence import PermanentStoreError
from persistence.conversation.sessions import NewSession, SessionCountQuery


@pytest.mark.parametrize("identifier", [123, True])
def test_numeric_session_create_keeps_the_native_refusal(backend, scope_a, identifier):
    with pytest.raises(TypeError, match="can only concatenate str"):
        backend.open(scope_a).sessions.create(NewSession(identifier, "cli"))


@pytest.mark.parametrize("identifier", [["bad"], {"bad": 1}])
def test_container_session_ids_keep_the_binding_error(backend, scope_a, identifier):
    sessions = backend.open(scope_a).sessions
    with pytest.raises(PermanentStoreError) as refusal:
        sessions.get(identifier)
    assert isinstance(refusal.value.__cause__, sqlite3.ProgrammingError)
    with pytest.raises(PermanentStoreError) as refusal:
        sessions.create(NewSession(identifier, "cli"))
    assert isinstance(refusal.value.__cause__, sqlite3.ProgrammingError)
    assert sessions.count_sessions(SessionCountQuery(include_archived=True)) == 0
