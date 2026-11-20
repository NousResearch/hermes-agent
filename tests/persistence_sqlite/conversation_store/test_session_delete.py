"""The conversation store's delete functions on a backend store's StoreHolder: single and bulk delete orphan children instead of cascading, the empty-session sweep touches only ended, non-archived empty rows, and the removed ids' transcript files go with them through ``session_retention.after_sessions_removed``."""

import pytest

from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    count_empty_sessions,
    create_session,
    delete_empty_sessions,
    delete_session,
    delete_sessions,
    end_session,
    get_session,
)
from session_retention import after_sessions_removed


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


class TestDeleteSessionOrphansChildren:
    def test_delete_orphans_children(self, db, media_root):
        """Deleting a parent session orphans its children."""
        create_session(db, session_id="parent", source="cli")
        create_session(db, session_id="child", source="cli", parent_session_id="parent")
        create_session(db, session_id="grandchild", source="cli", parent_session_id="child")

        # Should not raise IntegrityError
        result = delete_session(db, "parent", media_root=media_root)
        assert "parent" in result
        assert get_session(db, "parent") is None
        # Child is orphaned, not deleted
        child = get_session(db, "child")
        assert child is not None
        assert child["parent_session_id"] is None
        # Grandchild is untouched
        grandchild = get_session(db, "grandchild")
        assert grandchild is not None
        assert grandchild["parent_session_id"] == "child"


class TestBulkDeleteSessions:
    """``delete_sessions(ids)`` — the bulk-delete primitive backing the
    sessions-page "Delete N selected" button. Per-row contract matches
    :func:`delete_session` (children orphaned, not cascade-
    deleted), but applied across the whole list in one transaction.

    Invariants this class locks in:

    1. Returns the real deleted count (existing intersection), not
       just ``len(session_ids)`` — selection state in the UI can race
       against another tab's delete.
    2. Unknown IDs are silently skipped, never raise.
    3. ``message_count > 0`` sessions are deleted too — unlike
       ``delete_empty_sessions``, the user explicitly picked them, so
       we trust the selection.
    4. Live (un-ended) and archived sessions ARE deleted on explicit
       selection (no bulk-sweep safety guards apply when the user
       hand-picks the row).
    5. Children of any deleted parent are orphaned, even when the
       parent is mid-list.
    6. ``[]`` / ``None``-laden lists are safe no-ops.
    """

    def test_deletes_listed_sessions(self, db, media_root):
        create_session(db, session_id="a", source="cli")
        append_message(db, "a", role="user", content="hi")
        create_session(db, session_id="b", source="cli")
        create_session(db, session_id="c", source="cli")

        deleted, _ = delete_sessions(db, ["a", "b"], media_root=media_root)
        assert deleted == 2
        assert get_session(db, "a") is None
        assert get_session(db, "b") is None
        # Unlisted survives.
        assert get_session(db, "c") is not None

    def test_orphans_children_of_deleted_parents(self, db, media_root):
        """Bulk-deleting a parent leaves its children alive but
        re-parented to NULL. Same contract as the single-session
        :meth:`delete_session` path."""
        create_session(db, session_id="parent", source="cli")
        create_session(db,
            session_id="child", source="cli", parent_session_id="parent"
        )

        deleted, _ = delete_sessions(db, ["parent"], media_root=media_root)
        assert deleted == 1
        child = get_session(db, "child")
        assert child is not None
        assert child["parent_session_id"] is None

    def test_cleans_up_transcript_files(self, db, store, tmp_path, media_root):
        """The removed ids' on-disk transcripts are swept after the bulk
        delete commits, as after a per-row delete, so the bulk-delete
        CLI / web flows don't leak files."""
        create_session(db, session_id="s1", source="cli")
        create_session(db, session_id="s2", source="cli")
        (tmp_path / "s1.jsonl").write_text("")
        (tmp_path / "s2.json").write_text("{}")

        deleted, removed = delete_sessions(db, ["s1", "s2"], media_root=media_root)
        after_sessions_removed(store, store.holder().db_path.parent, removed, reason="delete", sessions_dir=tmp_path)
        assert deleted == 2
        assert not (tmp_path / "s1.jsonl").exists()
        assert not (tmp_path / "s2.json").exists()


class TestDeleteEmptySessions:
    """``delete_empty_sessions`` sweeps every ended, non-archived session
    whose ``message_count`` is 0. Backs the dashboard's "Delete empty"
    button — see ``SessionsPage.tsx`` + ``DELETE /api/sessions/empty``
    in ``hermes_cli/web_server.py``.

    Invariants this class locks in:

    1. Only ``message_count = 0`` rows are touched.
    2. Active (un-ended) sessions are skipped even if they're empty —
       the agent might be mid-handshake, and yanking the row would
       race the live runtime.
    3. Archived sessions are skipped — the user already filed them away.
    4. Children of a deleted parent are orphaned (parent_session_id →
       NULL) rather than cascade-deleted, matching the
       ``delete_session`` / ``prune_sessions`` contract.
    5. The pre-DB count matches the post-DB delete return value.
    """

    def test_count_and_delete_empties_only(self, db, media_root):
        # Two empty + ended sessions → both should be in the kill list.
        create_session(db, session_id="empty1", source="cli")
        end_session(db, "empty1", end_reason="done")
        create_session(db, session_id="empty2", source="cli")
        end_session(db, "empty2", end_reason="done")

        # One non-empty + ended session → must survive.
        create_session(db, session_id="hasmsg", source="cli")
        append_message(db, "hasmsg", role="user", content="Hello")
        end_session(db, "hasmsg", end_reason="done")

        assert count_empty_sessions(db) == 2

        deleted = len(delete_empty_sessions(db, media_root=media_root))
        assert deleted == 2
        assert get_session(db, "empty1") is None
        assert get_session(db, "empty2") is None
        assert get_session(db, "hasmsg") is not None
        assert count_empty_sessions(db) == 0

    def test_cleans_up_on_disk_transcript_files(self, db, store, tmp_path, media_root):
        """With a ``sessions_dir``, transcript files left
        behind by a crashed gateway (``request_dump_*.json``) are swept
        too. Empty sessions rarely have ``{id}.json`` / ``.jsonl``
        transcripts, but the request-dump path is real — the gateway
        writes one before the first reply lands, so a crash mid-reply
        produces an empty session with a non-empty dump file."""
        create_session(db, session_id="empty_with_dump", source="cli")
        end_session(db, "empty_with_dump", end_reason="done")

        dump = tmp_path / "request_dump_empty_with_dump_0.json"
        dump.write_text("{}")
        transcript = tmp_path / "empty_with_dump.jsonl"
        transcript.write_text("")

        removed = delete_empty_sessions(db, media_root=media_root)
        after_sessions_removed(store, store.holder().db_path.parent, removed, reason="empty_sweep", sessions_dir=tmp_path)
        deleted = len(removed)
        assert deleted == 1
        assert not dump.exists()
        assert not transcript.exists()
