"""Legacy SQLite BLOB message cells decode like their TEXT counterparts (#159863)."""

from __future__ import annotations

import json

import pytest

from persistence.conversation.messages import CONTENT_JSON_PREFIX
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session
from tests.fixtures.storage_holder import raw_holder


@pytest.mark.parametrize("stored, expected", [
    ("Xin chào".encode(), "Xin chào"),
    (b"", ""),
    ((CONTENT_JSON_PREFIX + json.dumps([{"type": "text", "text": "hello"}])).encode(),
     [{"type": "text", "text": "hello"}]),
    ((CONTENT_JSON_PREFIX + json.dumps(CONTENT_JSON_PREFIX + "[]")).encode(), CONTENT_JSON_PREFIX + "[]"),
    ((CONTENT_JSON_PREFIX + "{broken").encode(), CONTENT_JSON_PREFIX + "{broken"),
    (b"legacy\xfftext", "legacy\ufffdtext"),
])
def test_legacy_blob_message_is_text_or_decoded_structure(backend, scope_a, stored, expected):
    store = backend.open(scope_a)
    holder = raw_holder(store)
    create_session(holder, "blob-content", "cli")
    append_message(holder, "blob-content", "user", "placeholder")
    # Simulate old writers binding bytes: SQLite keeps BLOB values even in TEXT columns.
    holder._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET content = ? WHERE session_id = ?", (stored, "blob-content")))
    assert holder._read_one("SELECT typeof(content) FROM messages WHERE session_id = 'blob-content'")[0] == "blob"

    assert get_messages(holder, "blob-content")[0]["content"] == expected
