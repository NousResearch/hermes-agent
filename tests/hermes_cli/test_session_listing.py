"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import pytest

from hermes_cli.session_listing import (
    format_gateway_session_listing,
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

    def test_plain_listing_still_hides_unnamed(self, db):
        assert self._ids(db, source="telegram") == ["sess_an94"]

    def test_current_session_is_hidden_by_default(self, db):
        rows = query_session_listing(db, source="telegram", current_session_id="sess_an94")
        assert [r["id"] for r in rows] == []

    def test_current_session_can_be_listed_with_marker(self, db):
        rows = query_session_listing(
            db,
            source="telegram",
            current_session_id="sess_an94",
            include_current_session=True,
        )

        assert [r["id"] for r in rows] == ["sess_an94"]
        assert rows[0]["is_current_session"] is True


class TestFormatGatewaySessionListing:
    def test_marks_current_session(self):
        listing = format_gateway_session_listing(
            [
                {
                    "id": "sess_an94",
                    "title": "AN-94 Prestige Barrel Build #2",
                    "is_current_session": True,
                }
            ]
        )

        assert "**AN-94 Prestige Barrel Build #2** (current)" in listing

    def test_notice_appears_above_footer(self):
        listing = format_gateway_session_listing(
            [{"id": "sess_an94", "title": "AN-94"}],
            notice="_Note: `all` requires admin._",
        )
        lines = listing.splitlines()
        notice_idx = lines.index("_Note: `all` requires admin._")
        footer_idx = next(i for i, l in enumerate(lines) if l.startswith("Resume:"))
        assert notice_idx < footer_idx

    def test_notice_on_empty_listing(self):
        listing = format_gateway_session_listing([], notice="_scoped_")
        assert "No sessions found." in listing
        assert "_scoped_" in listing

    def test_no_notice_by_default(self):
        listing = format_gateway_session_listing([{"id": "x", "title": "T"}])
        assert "Note:" not in listing


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


class TestQuerySessionListingAdaptiveWidening:
    """The Python-side visibility filter runs after the DB fetch, so a single
    fixed window can hide an older eligible session behind newer hidden rows.
    The fetch widens (4× → 8× → 16×, always from row 0) until enough displayable
    rows survive or the DB is exhausted."""

    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        yield db
        db.close()

    def _make(self, db, n_unnamed):
        """One NAMED cli session (oldest) followed by *n_unnamed* unnamed cli
        sessions (newer), with strictly increasing ``started_at`` so the
        newest-first ordering is deterministic."""
        db.create_session("named_old", "cli", user_id="u", chat_id="c")
        db.set_session_title("named_old", "Keep Me")
        db._write_sql(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            ("2026-01-01 00:00:00", "named_old"),
        )
        for i in range(n_unnamed):
            sid = f"unnamed_{i:03d}"
            db.create_session(sid, "cli", user_id="u", chat_id="c")
            db._write_sql(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (f"2026-01-01 01:{(i + 1) // 60:02d}:{(i + 1) % 60:02d}", sid),
            )

    def test_limit_zero_or_negative_returns_empty_without_querying(self):
        class _ExplodingDB:
            def list_sessions_rich(self, *args, **kwargs):
                raise AssertionError(
                    "list_sessions_rich must not be called when limit <= 0"
                )

        assert query_session_listing(_ExplodingDB(), source="cli", limit=0) == []
        assert query_session_listing(_ExplodingDB(), source="cli", limit=-3) == []

    def test_older_named_session_survives_many_newer_unnamed(self, db):
        # 10 unnamed rows sit above the named one — well past the first 4×limit window.
        self._make(db, n_unnamed=10)
        rows = query_session_listing(db, source="cli", limit=1)
        assert [row["id"] for row in rows] == ["named_old"]

    def test_widens_more_than_once(self, db, monkeypatch):
        # 12 unnamed rows: the 4× window (4) and the 8× window (8) are both all
        # unnamed; only the 16× window reaches the named row.
        self._make(db, n_unnamed=12)
        real = db.list_sessions_rich
        fetched: list[int] = []

        def spy(*args, **kwargs):
            fetched.append(kwargs["limit"])
            return real(*args, **kwargs)

        monkeypatch.setattr(db, "list_sessions_rich", spy)
        rows = query_session_listing(db, source="cli", limit=1)

        assert len(fetched) >= 2, "should widen past the first 4x window"
        assert fetched == [4, 8, 16][: len(fetched)]
        assert [row["id"] for row in rows] == ["named_old"]

    def test_current_session_filter_cannot_starve_the_result(self, db):
        # Newest row is the current session (hidden by default); an older named
        # session must still surface rather than the list coming back empty.
        self._make(db, n_unnamed=6)
        db.create_session("cur", "cli", user_id="u", chat_id="c")
        db.set_session_title("cur", "Current")
        db._write_sql(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            ("2026-01-01 09:00:00", "cur"),
        )
        rows = query_session_listing(
            db, source="cli", current_session_id="cur", limit=1
        )
        assert [row["id"] for row in rows] == ["named_old"]

    def test_stops_early_when_first_window_is_enough(self, db, monkeypatch):
        # 2 unnamed only: the 4× window already yields the one displayable row,
        # so exactly one fetch happens (no needless widening).
        self._make(db, n_unnamed=2)
        real = db.list_sessions_rich
        fetched: list[int] = []

        def spy(*args, **kwargs):
            fetched.append(kwargs["limit"])
            return real(*args, **kwargs)

        monkeypatch.setattr(db, "list_sessions_rich", spy)
        rows = query_session_listing(db, source="cli", limit=1)

        assert fetched == [4]
        assert [row["id"] for row in rows] == ["named_old"]
