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


class TestSessionVisibilityPolicy:
    """One shared automation deny-list across every resume surface.

    The local classic CLI caller passes source="cli" with no gateway lane
    (see cli_session_mixin._list_recent_sessions). Shared policy must treat
    that local ``cli`` label as provenance: the listing becomes cross-source
    and denies the reviewed automation set (cron/tool/kanban/subagent)
    without hiding ACP/webhook/custom, while a gateway caller constrained by
    source + session_key is never widened.
    """

    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "local.db")
        human_sources = ("cli", "tui", "webui", "acp", "webhook", "custom-human")
        for source in human_sources:
            sid = f"human_{source}"
            db.create_session(sid, source)
            db.set_session_title(sid, f"Human {source}")
        automation_sources = ("cron", "tool", "kanban", "subagent")
        for source in automation_sources:
            sid = f"automation_{source}"
            db.create_session(sid, source)
            db.set_session_title(sid, f"Automation {source}")
        yield db
        db.close()

    def test_classic_cli_is_cross_source_but_denies_automation(self, db):
        """The CLISessionMixin call shape (source="cli", no lane): the shared
        policy widens the local cli label to provenance and hides every
        automation source, while ACP/webhook/custom stay visible. Caller
        exclusions are additive — a narrower legacy list still ends up denying
        the full automation set."""
        from hermes_cli.session_listing import AUTOMATION_SOURCES

        rows = query_session_listing(
            db,
            source="cli",
            current_session_id="human_cli",
            include_all_sources=False,
            include_unnamed=True,
            limit=50,
            exclude_sources=["kanban", "tool"],
        )
        ids = {row["id"] for row in rows}

        assert "human_cli" not in ids  # current session excluded
        assert {
            "human_tui",
            "human_webui",
            "human_acp",
            "human_webhook",
            "human_custom-human",
        }.issubset(ids)
        assert ids.isdisjoint({f"automation_{source}" for source in AUTOMATION_SOURCES})

    def test_gateway_lane_scope_is_not_widened(self, db):
        """Cross-source local discovery must not become gateway authority."""
        db.create_session(
            "telegram_lane",
            "telegram",
            session_key="agent:main:telegram:dm:1",
            user_id="u",
            chat_id="1",
        )
        db.set_session_title("telegram_lane", "Telegram lane")

        rows = query_session_listing(
            db,
            source="telegram",
            session_key="agent:main:telegram:dm:1",
            include_unnamed=True,
            limit=50,
        )

        assert [row["id"] for row in rows] == ["telegram_lane"]

    def test_gateway_all_sources_path_denies_automation(self, db):
        """The gateway admin ``/sessions all`` shape (include_all_sources)
        passes the shared constant as its exclusion, so automation sources are
        hidden there too."""
        from hermes_cli.session_listing import AUTOMATION_SOURCES

        rows = query_session_listing(
            db,
            source="telegram",
            session_key=None,
            include_all_sources=True,
            include_unnamed=True,
            limit=50,
            exclude_sources=sorted(AUTOMATION_SOURCES),
        )
        ids = {row["id"] for row in rows}

        assert "human_cli" in ids
        assert "human_acp" in ids
        assert "human_webhook" in ids
        assert ids.isdisjoint({f"automation_{source}" for source in AUTOMATION_SOURCES})


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
