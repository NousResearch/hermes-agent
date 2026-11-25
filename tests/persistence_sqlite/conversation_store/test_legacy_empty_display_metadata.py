"""A JSONL-converted row stores ``display_metadata = ''``, and ``json_extract('')`` raises
``malformed JSON``. SQLite evaluates the lineage match per row before comparing, so one such row in
a session's compression lineage failed every completion note / delegation delivery for that chain
(``Admitted turn ... failed``, 272-384/day). The guard reads those rows as "no match" instead.
"""

from __future__ import annotations

import pytest

from persistence.sqlite.conversation._store.detached_results import append_completion_note, append_delegation_delivery
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session


@pytest.fixture
def backend_store(backend, scope_a):
    return backend.open(scope_a)


def _legacy_session(backend_store) -> tuple[object, str]:
    db = backend_store.holder()
    session_id = "legacy-empty-metadata"
    create_session(db, session_id, "cli")
    append_message(db, session_id, "user", "a turn imported from JSONL")

    def _blank_metadata(conn):
        conn.execute("UPDATE messages SET display_metadata = '' WHERE session_id = ?", (session_id,))

    db._execute_write(_blank_metadata)
    return db, session_id


def test_completion_note_survives_a_blank_display_metadata_row(backend_store):
    db, session_id = _legacy_session(backend_store)

    note_id = append_completion_note(db, session_id, "admission-1", "done", "internal_notification", {})

    assert note_id > 0
    # The row it just wrote has valid JSON, so the idempotent replay finds it.
    assert append_completion_note(db, session_id, "admission-1", "done", "internal_notification", {}) == note_id


def test_delegation_delivery_survives_a_blank_display_metadata_row(backend_store):
    db, session_id = _legacy_session(backend_store)
    metadata = {"delegation_id": "delegation-1", "delivery_notice": "delivered"}

    delivery_id = append_delegation_delivery(db, session_id, "the delegated answer", metadata)

    assert delivery_id > 0
    assert append_delegation_delivery(db, session_id, "the delegated answer", metadata) == delivery_id
