"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import pytest

from hermes_cli.session_listing import (
    format_gateway_session_listing,
    query_session_listing,
)






class TestQuerySessionListingSearch:
    @pytest.fixture
    def db(self, tmp_path):
        from persistence_assembly import open_persistence, profile_scope
        from tests.fixtures.storage_seed import seed_session
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "sess_an94", "telegram", user_id="1", chat_id="2")
        db.sessions.set_title("sess_an94", "AN-94 Prestige Barrel Build #2")
        seed_session(db, "sess_winton", "whatsapp", user_id="1", chat_id="2")
        db.sessions.set_title("sess_winton", "Winton Email Sheet Update #3")
        seed_session(db, "sess_untitled", "telegram", user_id="1", chat_id="2")
        yield db
        db.close()

    def _ids(self, db, **kw):
        return [r["id"] for r in query_session_listing(None, persistence=db, **kw)]



    def test_source_scoping(self, db):
        assert self._ids(db, source="telegram", search_query="winton") == []
        assert self._ids(db, source="whatsapp", search_query="winton") == ["sess_winton"]


    def test_search_matches_compression_root_title(self, tmp_path):
        """Searching an old (compressed-away) title surfaces the live tip."""
        from persistence_assembly import open_persistence, profile_scope
        from tests.fixtures.storage_seed import seed_session
        db = open_persistence(profile_scope(home=tmp_path / "chain"))
        seed_session(db, "root_1", "telegram", user_id="1", chat_id="2")
        db.sessions.set_title("root_1", "Old Chat")
        db.sessions.end("root_1", "compression")
        seed_session(db,
            "tip_1", "telegram", user_id="1", chat_id="2", parent_session_id="root_1"
        )
        db.sessions.set_title("tip_1", "AN-94 Build")
        try:
            for query in ("old chat", "root_1", "an94"):
                rows = query_session_listing(None, persistence=db, source="telegram", search_query=query)
                assert [r["id"] for r in rows] == ["tip_1"], query
        finally:
            db.close()

    def test_plain_listing_still_hides_unnamed(self, db):
        assert self._ids(db, source="telegram") == ["sess_an94"]

    def test_current_session_is_hidden_by_default(self, db):
        rows = query_session_listing(None, persistence=db, source="telegram", current_session_id="sess_an94")
        assert [r["id"] for r in rows] == []

    def test_current_session_can_be_listed_with_marker(self, db):
        rows = query_session_listing(
            None, persistence=db,
            source="telegram",
            current_session_id="sess_an94",
            include_current_session=True,
        )

        assert [r["id"] for r in rows] == ["sess_an94"]
        assert rows[0]["is_current_session"] is True


class TestFormatGatewaySessionListing:


    def test_notice_on_empty_listing(self):
        listing = format_gateway_session_listing([], notice="_scoped_")
        assert "_scoped_" in listing



class TestQuerySessionListingLaneScope:
    @pytest.fixture
    def db(self, tmp_path):
        from persistence_assembly import open_persistence, profile_scope
        from tests.fixtures.storage_seed import seed_session

        db = open_persistence(profile_scope(home=tmp_path))
        lane_key = "agent:main:telegram:dm:lane"
        seed_session(db,
            "lane_current", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.sessions.set_title("lane_current", "Current lane")
        seed_session(db,
            "lane_named", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.sessions.set_title("lane_named", "Needle lane")
        seed_session(db,
            "lane_unnamed", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        for i in range(60):
            seed_session(db,
                f"foreign_{i}", "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
            db.sessions.set_title(f"foreign_{i}", f"Needle foreign {i}")
        yield db, lane_key
        db.close()

    def test_exact_lane_precedes_limit_and_current_session_exclusion(self, db):
        session_db, lane_key = db

        rows = query_session_listing(
            None, persistence=session_db,
            source="telegram",
            session_key=lane_key,
            current_session_id="lane_current",
            limit=1,
        )

        assert [row["id"] for row in rows] == ["lane_named"]

    def test_exact_lane_preserves_full_and_search_modes(self, db):
        session_db, lane_key = db

        full_rows = query_session_listing(
            None, persistence=session_db,
            source="telegram",
            session_key=lane_key,
            include_unnamed=True,
            limit=10,
        )
        search_rows = query_session_listing(
            None, persistence=session_db,
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
            None, persistence=session_db,
            source="telegram",
            search_query="needle foreign 59",
            limit=10,
        )

        assert [row["id"] for row in rows] == ["foreign_59"]
