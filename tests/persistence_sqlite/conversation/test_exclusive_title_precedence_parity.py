"""Exclusive session creation on SQLite checks identity before title validation."""

import pytest

from persistence.conversation.sessions import SessionExists


def test_existing_id_precedes_oversized_title(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create_exclusive("same", source="cli", title="original")
    before = sessions.get("same")
    with pytest.raises(SessionExists):
        sessions.create_exclusive("same", source="cli", title="x" * 1000)
    assert sessions.get("same") == before
    with pytest.raises(ValueError, match="Title too long"):
        sessions.create_exclusive("fresh", source="cli", title="x" * 1000)
    assert sessions.get("fresh") is None
