"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import pytest

from hermes_cli.session_listing import (
    AUTOMATION_SOURCES,
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


class TestAutomationSourcesDenyList:
    """The picker denylist of internal/automation sources (#47214, #15745)."""

    def test_constant_is_the_canonical_internal_set(self):
        assert AUTOMATION_SOURCES == frozenset(
            {"cron", "tool", "kanban", "subagent"}
        )

    @pytest.mark.parametrize(
        "human",
        ["cli", "tui", "webui", "telegram", "discord", "signal", "slack",
         "whatsapp", "acp", "webhook", "custom"],
    )
    def test_human_surfaces_are_not_denied(self, human):
        # A denylist must never contain a human conversation surface, so new
        # gateway platforms surface in the picker automatically. ACP adapter
        # sessions, webhook sessions, and custom HERMES_SESSION_SOURCE values
        # are user-facing per the TUI picker contract (methods_session.py).
        assert human not in AUTOMATION_SOURCES

    @pytest.fixture
    def mixed_db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "deny.db")
        # Human surfaces, every one must be visible.
        for sid, src in [
            ("h_cli", "cli"),
            ("h_tui", "tui"),
            ("h_webui", "webui"),
            ("h_telegram", "telegram"),
            ("h_acp", "acp"),
            ("h_webhook", "webhook"),
            ("h_custom", "custom"),
        ]:
            db.create_session(sid, src)
            db.set_session_title(sid, f"Title {sid}")
        # Automation/internal, every one must be hidden.
        for sid, src in [
            ("a_cron", "cron"),
            ("a_tool", "tool"),
            ("a_kanban", "kanban"),
            ("a_sub", "subagent"),
        ]:
            db.create_session(sid, src)
            db.set_session_title(sid, f"Title {sid}")
        yield db
        db.close()

    def test_picker_shows_all_human_sources_hides_automation(self, mixed_db):
        # This is the exact call shape cli.py::_list_recent_sessions makes.
        rows = query_session_listing(
            mixed_db,
            source=None,
            include_all_sources=True,
            include_unnamed=True,
            exclude_sources=sorted(AUTOMATION_SOURCES),
            limit=20,
        )
        ids = {r["id"] for r in rows}
        assert {
            "h_cli", "h_tui", "h_webui", "h_telegram",
            "h_acp", "h_webhook", "h_custom",
        } <= ids
        assert ids.isdisjoint(
            {"a_cron", "a_tool", "a_kanban", "a_sub"}
        )
