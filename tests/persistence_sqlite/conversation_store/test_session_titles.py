"""The conversation store's title functions on a real StoreHolder: setting, clearing and resolving titles, conflicts and the ended-empty ghost that yields, the title-index repair on open, title transfer along a compression lineage, sanitising, and LIKE wildcards in titles."""

import contextlib
import sqlite3

import pytest

from persistence.sqlite import _schema
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    end_session,
    get_session,
    list_sessions_rich,
    set_session_hidden,
)
from persistence.sqlite.conversation._store.titles import (
    get_next_title_in_lineage,
    resolve_session_by_title,
    sanitize_title,
    set_session_title,
)


@pytest.fixture()
def db(tmp_path):
    """Create a StoreHolder with a temp database file."""
    db_path = tmp_path / "test_state.db"
    holder = StoreHolder(db_path=db_path)
    yield holder
    holder.close()


# =========================================================================
# Session title
# =========================================================================

class TestSessionTitle:
    def test_set_and_get_title(self, db):
        create_session(db, session_id="s1", source="cli")
        assert set_session_title(db, "s1", "My Session") is True

        session = get_session(db, "s1")
        assert session["title"] == "My Session"

    def test_title_empty_string_normalized_to_none(self, db):
        """Empty strings are normalized to None (clearing the title)."""
        create_session(db, session_id="s1", source="cli")
        set_session_title(db, "s1", "My Title")
        # Setting to empty string should clear the title (normalize to None)
        set_session_title(db, "s1", "")

        session = get_session(db, "s1")
        assert session["title"] is None

    @pytest.mark.parametrize("holder_has_message,holder_hidden,holder_ended,yields", [
        (False, False, True, True),    # ended empty visible ghost yields its title (#81888)
        (False, False, False, False),  # live empty session (/title before its first turn) keeps it
        (False, True, True, False),    # hidden empty row (fresh canonical Bot Chat) keeps it
        (True, False, True, False),    # a real conversation keeps it
    ])
    def test_title_conflict_yields_only_to_ended_empty_visible_holder(self, db, holder_has_message, holder_hidden,
                                                                      holder_ended, yields):
        """A title holder yields only when it is ended, empty, visible and unarchived (#81888);
        every other holder still conflicts. The ghost stays writable afterwards (no partial
        unique index to trip on its first append_message)."""
        create_session(db, "ghost", "desktop")
        set_session_title(db, "ghost", "Canada")
        if holder_has_message:
            append_message(db, "ghost", "user", "m0")
        set_session_hidden(db, "ghost", holder_hidden)
        if holder_ended:
            end_session(db, "ghost", "user_exit")
        create_session(db, "real", "desktop")
        append_message(db, "real", "user", "Hello")
        if not yields:
            with pytest.raises(ValueError, match="already in use"):
                set_session_title(db, "real", "Canada")
            assert get_session(db, "ghost")["title"] == "Canada"
            return
        assert set_session_title(db, "real", "Canada")
        assert get_session(db, "ghost")["title"] is None
        assert resolve_session_by_title(db, "Canada") == "real"
        append_message(db, "ghost", "user", "late first message")
        assert get_session(db, "ghost")["message_count"] == 1


class TestSessionTitleIndexRepair:
    @staticmethod
    def _seed_legacy_database(tmp_path, *, duplicate_titles):
        db_path = tmp_path / "legacy_titles.db"
        session_db = StoreHolder(db_path=db_path)
        create_session(session_db, "older", "cli")
        append_message(session_db, "older", role="user", content="keep older message")
        create_session(session_db, "newer", "cli")
        append_message(session_db,
            "newer", role="assistant", content="keep newer message"
        )
        create_session(session_db, "unique", "cli")
        set_session_title(session_db, "unique", "unique-title")
        session_db.close()

        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP INDEX idx_sessions_title_unique")
            if duplicate_titles:
                conn.execute(
                    "UPDATE sessions SET title = 'shared-title' "
                    "WHERE id IN ('older', 'newer')"
                )

        return db_path

    def test_duplicate_titles_are_repaired_without_deleting_sessions(self, tmp_path):
        db_path = self._seed_legacy_database(tmp_path, duplicate_titles=True)

        reopened = StoreHolder(db_path=db_path)
        try:
            conn = reopened._conn
            assert conn is not None
            rows = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, title FROM sessions ORDER BY rowid"
                ).fetchall()
            }
            assert set(rows) == {"older", "newer", "unique"}
            # NULL title_source ranks as user: neither is dropped; the newer gets "#2" so the
            # "#N"-preferring title lookup still opens the newest session.
            assert rows["older"]["title"] == "shared-title"
            assert rows["newer"]["title"] == "shared-title #2"
            assert resolve_session_by_title(reopened, "shared-title") == "newer"
            assert rows["unique"]["title"] == "unique-title"
            assert get_messages(reopened, "older")[0]["content"] == "keep older message"
            assert get_messages(reopened, "newer")[0]["content"] == "keep newer message"
            index = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'index' AND name = 'idx_sessions_title_unique'"
            ).fetchone()
            assert index is not None
        finally:
            reopened.close()

    @pytest.mark.parametrize("broken_rename", [False, True])
    def test_repair_keeps_highest_ranked_newest_title(self, tmp_path, monkeypatch, broken_rename):
        # #126764: rank (user > llm > derived) beats recency; within an auto rank the newest
        # started_at wins even when it has the lower rowid; user rows are renamed oldest-first.
        # broken_rename: the index still fails after rows were cleared -> the repair is rolled
        # back whole (no half-repaired store committed without the index).
        db_path = tmp_path / "ranked_titles.db"
        db = StoreHolder(db_path=db_path)
        for sid in "abcde":
            create_session(db, sid, "cli")
        db.close()
        seed = {"a": ("Trip", "user", 100), "b": ("Trip", "llm", 300), "c": ("Note", "derived", 200),
                "d": ("Note", "derived", 100), "e": ("Trip", "user", 200)}
        with contextlib.closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute("DROP INDEX idx_sessions_title_unique")
            conn.executemany(
                "UPDATE sessions SET title = ?, title_source = ?, started_at = ? WHERE id = ?",
                [(*row, sid) for sid, row in seed.items()],
            )
        if broken_rename:
            monkeypatch.setattr(_schema, "next_title_in_lineage", lambda conn, title: title)
        reopened = StoreHolder(db_path=db_path)
        try:
            titles = dict(reopened._conn.execute("SELECT id, title FROM sessions").fetchall())
            expected = ({sid: row[0] for sid, row in seed.items()} if broken_rename else
                        {"a": "Trip", "b": None, "c": "Note", "d": None, "e": "Trip #2"})
            assert titles == expected
            assert bool(reopened._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'idx_sessions_title_unique'"
            ).fetchone()) is not broken_rename
        finally:
            reopened.close()


class TestSessionTitleLineage:
    """Renaming a compression continuation back to its base title must succeed
    by transferring the title off the ended, hidden predecessor.

    After a context compaction the original session is ended and projected
    behind its live tip in the session list (list_sessions_rich), so the user
    cannot see or free it. Without lineage-aware handling, renaming the visible
    tip back to the base name dead-ends with "already in use by <session they
    can't find>".
    """

    def _make_compression_chain(self, db, t0, *, root="root", tip="tip"):
        create_session(db, root, "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, root))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
            (t0 + 100, root),
        )
        create_session(db, tip, "cli", parent_session_id=root)
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 200, tip))
        db._conn.commit()

    def test_rename_continuation_back_to_base_transfers_title(self, db):
        import time as _time
        self._make_compression_chain(db, _time.time() - 3600)
        set_session_title(db, "root", "fingerprint-scanner")
        set_session_title(db, "tip", "fingerprint-scanner #2")

        # User renames the visible tip back to the base name — must succeed.
        assert set_session_title(db, "tip", "fingerprint-scanner") is True
        assert get_session(db, "tip")["title"] == "fingerprint-scanner"
        # Title transferred off the hidden ancestor — no duplicate titles.
        assert get_session(db, "root")["title"] is None

    def test_unrelated_session_still_conflicts(self, db):
        create_session(db, "a", "cli")
        create_session(db, "b", "cli")
        set_session_title(db, "a", "shared")
        with pytest.raises(ValueError, match="already in use"):
            set_session_title(db, "b", "shared")
        # The unrelated holder keeps its title.
        assert get_session(db, "a")["title"] == "shared"

    def test_projected_tip_inherits_root_title_when_untitled(self, db):
        """A rotation that ended the root before the title carry ran leaves the name on the
        root only; the projected lineage row must still surface it (exact-title lookups such as
        `hermes peer dm` -> canonical "Bot Chat", #106165). A titled tip keeps its own title."""
        import time as _time
        self._make_compression_chain(db, _time.time() - 3600)
        set_session_title(db, "root", "Bot Chat")

        rows = list_sessions_rich(db, limit=50, order_by_last_active=True, search_query="Bot Chat")
        assert [(r["id"], r["title"], r["_lineage_root_id"]) for r in rows] == [("tip", "Bot Chat", "root")]

        set_session_title(db, "tip", "renamed tip")
        rows = list_sessions_rich(db, limit=50, order_by_last_active=True)
        assert [(r["id"], r["title"]) for r in rows] == [("tip", "renamed tip")]


class TestSanitizeTitle:
    """Tests for sanitize_title() validation and cleaning."""

    def test_normal_title_unchanged(self):
        assert sanitize_title("My Project") == "My Project"

    def test_control_chars_stripped(self):
        # Null byte, bell, backspace, etc.
        assert sanitize_title("hello\x00world") == "helloworld"
        assert sanitize_title("\x07\x08test\x1b") == "test"

    def test_exceeds_max_length_raises(self):
        title = "A" * 101
        with pytest.raises(ValueError, match="too long"):
            sanitize_title(title)


class TestTitleLineage:
    """Tests for title lineage resolution and auto-numbering."""

    def test_resolve_exact_title(self, db):
        create_session(db, "s1", "cli")
        set_session_title(db, "s1", "my project")
        assert resolve_session_by_title(db, "my project") == "s1"

    def test_resolve_nonexistent_title(self, db):
        assert resolve_session_by_title(db, "nonexistent") is None

    def test_next_title_no_existing(self, db):
        """With no existing sessions, base title is returned as-is."""
        assert get_next_title_in_lineage(db, "my project") == "my project"


class TestTitleSqlWildcards:
    """Titles containing SQL LIKE wildcards (%, _) must not cause false matches."""

    def test_resolve_title_with_underscore(self, db):
        """A title like 'test_project' should not match 'testXproject #2'."""
        create_session(db, "s1", "cli")
        set_session_title(db, "s1", "test_project")
        create_session(db, "s2", "cli")
        set_session_title(db, "s2", "testXproject #2")
        # Resolving "test_project" should return s1 (exact), not s2
        assert resolve_session_by_title(db, "test_project") == "s1"
