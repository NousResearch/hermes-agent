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

    @pytest.mark.parametrize("include_current", [False, True])
    def test_plain_listing_paginates_past_unnamed_rows(self, tmp_path, include_current):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "paging.db")
        db.create_session("named_old", "telegram")
        db.set_session_title("named_old", "Older Work")
        for i in range(65):
            db.create_session(f"unnamed_{i}", "telegram")
        db.create_session("current", "telegram")
        try:
            rows = query_session_listing(
                db, source="telegram", current_session_id="current",
                include_current_session=include_current, limit=2,
            )
            assert [r["id"] for r in rows] == (["current", "named_old"] if include_current else ["named_old"])
            if include_current:
                assert rows[0]["is_current_session"] is True
        finally:
            db.close()

    def test_paging_stops_after_enough_rows_or_exhaustion(self):
        class FiniteDB:
            def __init__(self, rows):
                self.rows = rows
                self.offsets = []

            def list_sessions_rich(self, *, limit, offset=0, **kwargs):
                assert offset not in self.offsets, "paging must advance"
                self.offsets.append(offset)
                return self.rows[offset:offset + limit]

        invisible = [{"id": f"unnamed_{i}"} for i in range(65)]
        exhausted = FiniteDB(invisible)
        assert query_session_listing(exhausted, source="telegram", limit=1) == []
        assert len(exhausted.offsets) > 1
        enough = FiniteDB([{"id": "first", "title": "First"}, *invisible])
        assert [row["id"] for row in query_session_listing(enough, source="telegram", limit=1)] == ["first"]
        assert enough.offsets == [0]

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
