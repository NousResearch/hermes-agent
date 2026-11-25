"""Session delete and id resolution through the conversation store's functions, and export and import through the SQLite maintenance repository: a missing session exports as nothing, and an oversized import payload is refused whole."""

import pytest

from persistence.housekeeping.maintenance import IMPORT_MAX_MESSAGES_PER_SESSION, IMPORT_MAX_SESSION_BYTES
from persistence.sqlite.conversation._store.messages import append_message, message_count
from persistence.sqlite.conversation._store.sessions import create_session, delete_session, get_session, resolve_session_id


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


@pytest.fixture
def media_root(tmp_path):
    return str(tmp_path / "media")


# =========================================================================
# Delete and export
# =========================================================================

class TestDeleteAndExport:
    def test_delete_session(self, db, media_root):
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="Hello")

        assert "s1" in delete_session(db, "s1", media_root=media_root)
        assert get_session(db, "s1") is None
        assert message_count(db, session_id="s1") == 0

    def test_resolve_session_id_ambiguous_prefix_returns_none(self, db):
        create_session(db, session_id="20260315_092437_c9a6aa", source="cli")
        create_session(db, session_id="20260315_092437_c9a6bb", source="cli")
        assert resolve_session_id(db, "20260315_092437_c9a6") is None

    def test_export_nonexistent(self, store):
        assert store.maintenance.export_session("nope") is None

    def test_import_sessions_rejects_oversized_payloads_atomically(self, db, store):
        oversized = "x" * (IMPORT_MAX_SESSION_BYTES + 1)
        result = store.maintenance.import_sessions(
            [{"id": "oversized", "messages": [{"role": "user", "content": oversized}]}]
        )

        assert result.ok is False
        assert result.errors[0]["error"] == "session exceeds the import size limit"
        assert get_session(db, "oversized") is None

        result = store.maintenance.import_sessions(
            [
                {
                    "id": "too-many-messages",
                    "messages": [
                        {"role": "user", "content": "x"}
                    ]
                    * (IMPORT_MAX_MESSAGES_PER_SESSION + 1),
                }
            ]
        )

        assert result.ok is False
        assert result.errors[0]["error"] == "messages exceeds the per-session import limit"
        assert get_session(db, "too-many-messages") is None
