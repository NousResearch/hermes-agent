"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import pytest

from hermes_cli.session_listing import (
    parse_session_listing_args,
    query_session_listing,
)


class TestParseSessionListingArgs:
    def test_plain_listing(self):
        assert parse_session_listing_args("") == (False, False, "", None)




class TestQuerySessionListingSearch:
    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_an94", "telegram", user_id="1", chat_id="2")
        db.set_session_title("sess_an94", "AN-94 Prestige Barrel Build #2")
        db.create_session("sess_winton", "whatsapp", user_id="1", chat_id="2")
        db.set_session_title("sess_winton", "Winton Email Sheet Update #3")
        db.create_session("sess_untitled", "telegram", user_id="1", chat_id="2")
        yield db
        db.close()

    def _ids(self, db, **kw):
        return [r["id"] for r in query_session_listing(db, **kw)]



    def test_source_scoping(self, db):
        assert self._ids(db, source="telegram", search_query="winton") == []
        assert self._ids(db, source="whatsapp", search_query="winton") == ["sess_winton"]


    def test_search_matches_compression_root_title(self, tmp_path):
        """Searching an old (compressed-away) title surfaces the live tip."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "chain.db")
        db.create_session("root_1", "telegram", user_id="1", chat_id="2")
        db.set_session_title("root_1", "Old Chat")
        db.end_session("root_1", end_reason="compression")
        db.create_session(
            "tip_1", "telegram", user_id="1", chat_id="2", parent_session_id="root_1"
        )
        db.set_session_title("tip_1", "AN-94 Build")
        try:
            for query in ("old chat", "root_1", "an94"):
                rows = query_session_listing(db, source="telegram", search_query=query)
                assert [r["id"] for r in rows] == ["tip_1"], query
        finally:
            db.close()

    def test_punctuation_normalized_match_apostrophe(self, tmp_path):
        # The compact needle strips all punctuation (re.sub(r"[\W_]+", ...)),
        # so the title side must strip the same alphabet — "bobs" has to match
        # "Bob's Chat" even though the apostrophe is not a - _ . separator.
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "bob.db")
        # Id is deliberately chosen NOT to contain "bobs" so the match is
        # forced through the compact-title path, not the id-substring fallback.
        db.create_session("sess_x1", "telegram", user_id="1", chat_id="2")
        db.set_session_title("sess_x1", "Bob's Chat")
        try:
            assert self._ids(db, source="telegram", search_query="bobs") == ["sess_x1"]
            # The literal (non-compacted) fallback still matches the apostrophe.
            assert self._ids(db, source="telegram", search_query="bob's") == ["sess_x1"]
        finally:
            db.close()

    def test_punctuation_normalized_match_ampersand(self, tmp_path):
        # "rd" must match "R&D Notes" via the compact path (ampersand stripped).
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "rd.db")
        # Id avoids the "rd" substring so the id-fallback can't mask a broken
        # compact-title path.
        db.create_session("sess_x2", "telegram", user_id="1", chat_id="2")
        db.set_session_title("sess_x2", "R&D Notes")
        try:
            assert self._ids(db, source="telegram", search_query="rd") == ["sess_x2"]
        finally:
            db.close()

    def test_old_sqlite_deterministic_unsupported_falls_back(self, tmp_path, monkeypatch):
        # sqlite3.create_function(..., deterministic=True) raises
        # sqlite3.NotSupportedError (NOT TypeError) when the module knows the
        # kwarg but the linked SQLite is < 3.8.3. SessionDB.__init__ must
        # survive that by re-registering compact_punct without deterministic —
        # never crash the whole DB open on an old-SQLite runtime.
        import functools
        import sqlite3

        from hermes_state import SessionDB

        class _OldSqliteConnection(sqlite3.Connection):
            # Mimic a Python linked against SQLite < 3.8.3: the deterministic
            # kwarg is recognised by the module but rejected by the engine.
            def create_function(self, name, narg, func, *args, **kwargs):
                if kwargs.get("deterministic"):
                    raise sqlite3.NotSupportedError(
                        "deterministic=True requires SQLite 3.8.3 or higher"
                    )
                return super().create_function(name, narg, func, *args, **kwargs)

        real_connect = sqlite3.connect
        monkeypatch.setattr(
            sqlite3,
            "connect",
            functools.partial(real_connect, factory=_OldSqliteConnection),
        )

        db = SessionDB(db_path=tmp_path / "old_sqlite.db")
        try:
            # __init__ succeeded; the fallback (non-deterministic) registration
            # produced a working compact_punct — the compact-title path still
            # matches, proving we fell back rather than skipping the function.
            db.create_session("sess_x3", "telegram", user_id="1", chat_id="2")
            db.set_session_title("sess_x3", "R&D Notes")
            assert self._ids(db, source="telegram", search_query="rd") == ["sess_x3"]
        finally:
            db.close()

    def test_compact_search_works_on_the_wal_read_path(self, tmp_path):
        # SQLite scalar functions are registered PER CONNECTION. Under WAL the
        # search runs on the per-thread read-only connection from
        # `_get_read_conn`, not the writer, so compact_punct has to be
        # registered there too — otherwise `/sessions search` raises
        # "no such function: compact_punct" on every WAL install. This test
        # forces the read path on because runtimes with a WAL-unsafe SQLite
        # fall back to journal_mode=DELETE and would never exercise it.
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "walread.db")
        db.create_session("sess_x4", "telegram", user_id="1", chat_id="2")
        db.set_session_title("sess_x4", "R&D Notes")
        db._wal_active = True
        try:
            assert db._get_read_conn() is not None
            assert self._ids(db, source="telegram", search_query="rd") == ["sess_x4"]
        finally:
            db.close()


class TestQuerySessionListingLaneScope:
    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        lane_key = "agent:main:telegram:dm:lane"
        db.create_session(
            "lane_current", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.set_session_title("lane_current", "Current lane")
        db.create_session(
            "lane_named", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.set_session_title("lane_named", "Needle lane")
        db.create_session(
            "lane_unnamed", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        for i in range(60):
            db.create_session(
                f"foreign_{i}", "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.set_session_title(f"foreign_{i}", f"Needle foreign {i}")
        yield db, lane_key
        db.close()

    def test_exact_lane_precedes_limit_and_current_session_exclusion(self, db):
        session_db, lane_key = db

        rows = query_session_listing(
            session_db,
            source="telegram",
            session_key=lane_key,
            current_session_id="lane_current",
            limit=1,
        )

        assert [row["id"] for row in rows] == ["lane_named"]

    def test_exact_lane_preserves_full_and_search_modes(self, db):
        session_db, lane_key = db

        full_rows = query_session_listing(
            session_db,
            source="telegram",
            session_key=lane_key,
            include_unnamed=True,
            limit=10,
        )
        search_rows = query_session_listing(
            session_db,
            source="telegram",
            session_key=lane_key,
            search_query="needle",
            limit=10,
        )

        assert {row["id"] for row in full_rows} == {
            "lane_current", "lane_named", "lane_unnamed",
        }
        assert [row["id"] for row in search_rows] == ["lane_named"]

    def test_omitted_session_key_keeps_source_scope(self, db):
        session_db, _lane_key = db

        rows = query_session_listing(
            session_db,
            source="telegram",
            search_query="needle foreign 59",
            limit=10,
        )

        assert [row["id"] for row in rows] == ["foreign_59"]
