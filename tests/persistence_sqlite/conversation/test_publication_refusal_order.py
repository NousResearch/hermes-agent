"""Compression publication on SQLite keeps the legacy store's refusal precedence and causes."""

import sqlite3
import time

import pytest

from persistence import CompressionBusy, CompressionPublishRefused
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tests.persistence_sqlite.conftest import store_for


def _lock_holder(scope, session_id):
    """The live compression lock holder, read as the legacy store's diagnostic reads it."""
    with sqlite3.connect(store_for(scope)) as connection:
        row = connection.execute(
            "SELECT holder FROM compression_locks WHERE session_id = ? AND expires_at >= ?", (session_id, time.time())
        ).fetchone()
    return None if row is None else row[0]


@pytest.mark.parametrize("parent, handoff, leased, refusal, message", [
    (False, False, False, CompressionPublishRefused, "Compression parent not found: parent"),
    (False, True, True, CompressionBusy, "Compression lease lost before publication: parent"),
    (False, False, True, CompressionBusy, "Compression lease lost before publication: parent"),
    (True, False, True, CompressionBusy, "Compression lease lost before publication: parent"),
    (True, False, False, CompressionPublishRefused, "Compression child handoff must not be empty"),
])
def test_publication_preserves_native_refusal(backend, scope_a, parent, handoff, leased, refusal, message):
    store = backend.open(scope_a)
    if parent:
        store.sessions.create(NewSession("parent", "cli"))
    with pytest.raises(refusal) as raised:
        store.lineage.publish_compression_child(
            "parent", CompressionChild("child", "cli"),
            [NewMessage(role="user", content="summary", timestamp=1672531200.0)] if handoff else [],
            compression_lock_holder="lost" if leased else None, require_compression_lease=leased,
        )
    assert str(raised.value.__cause__) == message
    assert store.sessions.get("child") is None
    assert _lock_holder(scope_a, "parent") is None
    if parent:
        assert store.sessions.get("parent").end_reason is None
