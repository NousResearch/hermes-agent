"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import pytest

# Warm the lazy import chain SessionDB.append_message hits at call time
# (hermes_state_messages -> context_compressor -> agent.conversation_loop, whose
# module-level dependency activation reads a payload manifest next to the real
# hermes home). Importing here, before the per-test HomeIOGuard arms, caches the
# module so the guard never sees that read.
import agent.conversation_loop  # noqa: E402,F401

from hermes_cli.session_listing import (
    format_gateway_session_listing,
    is_automated_session,
    is_ping_probe_session,
    parse_session_listing_args,
    query_session_listing,
)






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


    def test_notice_on_empty_listing(self):
        listing = format_gateway_session_listing([], notice="_scoped_")
        assert "_scoped_" in listing



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
        db.append_message("lane_unnamed", "user", "untitled but real conversation")
        # One extra empty shell: probe-filter browsing still surfaces unnamed
        # lanes in full mode, but zero-message debris stays hidden.
        db.create_session(
            "lane_empty_shell", "telegram", session_key=lane_key,
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
        assert "lane_empty_shell" not in {row["id"] for row in full_rows}
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


class TestQuerySessionListingHidesProbeDebris:
    """Blank-title probe/debris sessions stay out of browse listings.

    Shapes (all hit the live profile's state.db today):
    - ``ping, reply with one word`` probes from `hermes chat -q`
    - other one-shot "say hello / what model are you" formulas
    - zero-message empty shells (launched, never typed into)
    Search mode keeps them: an id match may be the only handle.
    """

    @pytest.fixture
    def db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("probe", "cli")
        db.append_message("probe", "user", "ping, reply with one word")
        db.create_session("shell", "cli")
        db.create_session("hello", "cli")
        db.append_message("hello", "user", "Say hello in one word")
        db.create_session("real_untitled", "cli")
        db.append_message("real_untitled", "user", "fix the login bug in auth.py")
        db.create_session("titled", "cli")
        db.append_message("titled", "user", "ping, reply with one word")
        # Auto-titled probe openers stay hidden when they're a single
        # exchange, so give this one a real continuation past the probe.
        db.append_message("titled", "assistant", "pong")
        db.append_message("titled", "user", "now diagnose why the fallback chain timed out")
        db.append_message("titled", "assistant", "Looking at the gateway logs…")
        db.set_session_title("titled", "Named chat — keep me")
        yield db
        db.close()

    def _ids(self, db, **kw):
        return [r["id"] for r in query_session_listing(db, source="cli", **kw)]

    def test_browse_hides_probes_but_keeps_real_untitled(self, db):
        ids = self._ids(db, include_unnamed=True, limit=50)
        assert "probe" not in ids
        assert "shell" not in ids
        assert "hello" not in ids
        assert "real_untitled" in ids
        assert "titled" in ids

    def test_search_still_finds_probe_by_id(self, db):
        assert self._ids(db, include_unnamed=True, search_query="probe", limit=10) == ["probe"]

    def test_explicit_automation_provenance_wins_over_prompt_shape(self):
        assert is_automated_session({"source": "jev", "title": "Smoke check", "preview": "normal work"})
        assert is_automated_session({"source": "cli", "model_config": '{"_session_origin": "automation"}'})
        assert is_automated_session({"source": "cli", "origin_json": '{"kind": "scheduled"}'})
        assert is_automated_session({"source": "cli", "model_config": '{"_automation": true}'})
        assert not is_automated_session({"source": "cli", "title": "Normal chat", "preview": "hello"})

    def test_is_ping_probe_session_shapes(self):
        assert is_ping_probe_session({"title": None, "preview": "ping, reply with one word"})
        assert is_ping_probe_session({"title": "Reply with exactly: pong", "preview": "Reply with exactly: pong"})
        assert is_ping_probe_session({"title": "reply with just: ok", "preview": "reply with just: ok"})
        assert is_ping_probe_session({"title": "", "preview": ""})
        # Auto-generated titles paraphrasing a probe: single exchange -> hide.
        assert is_ping_probe_session({"title": "Ping connectivity test", "preview": "ping, reply with one word", "message_count": 2})
        assert is_ping_probe_session({"title": "One-word ping reply", "preview": "ping, reply with one word", "message_count": 2})
        # Same probe opening but the conversation continued -> keep.
        assert not is_ping_probe_session({"title": "Ping connectivity test", "preview": "ping, reply with one word", "message_count": 40})
        assert not is_ping_probe_session({"title": "Chat", "preview": "ping, reply with one word", "message_count": 12})
        assert is_ping_probe_session({"title": None, "preview": "Say hello in one word"})
        assert is_ping_probe_session({"title": None, "preview": "Reply with exactly: pong"})
        assert is_ping_probe_session({"title": None, "preview": "只回复四个字：配置成功"})
        assert is_ping_probe_session({"title": "请只回复：JEV_SMOKE_OK", "preview": "请只回复：JEV_SMOKE_OK", "message_count": 2})
        assert is_ping_probe_session({"title": None, "preview": "请只回复：JEV_SMOKE_OK"})
        assert is_ping_probe_session({"title": None, "preview": "reply with just: ok"})
        assert is_ping_probe_session({"title": None, "preview": "ping, reply with exactly: pong"})
        assert is_ping_probe_session({"title": "hi", "preview": "hi", "message_count": 2})
        assert is_ping_probe_session({"title": "Friendly greeting #2", "preview": "hi", "message_count": 2})
        assert is_ping_probe_session({"title": "Ping one-word reply #2", "preview": "ping, reply with one word only", "message_count": 2})
        assert is_ping_probe_session({"title": "List mcp_kitesurf tool names", "preview": "List all your tools whose names start with mcp_kitesurf", "message_count": 4})
        assert is_ping_probe_session({"title": "List browser automation tools", "preview": "what tools do you have for browser automation? name them", "message_count": 5})
        assert not is_ping_probe_session({"title": "List menu items", "preview": "List the menu items for our restaurant", "message_count": 4})
        assert not is_ping_probe_session({"title": "List tools for my project", "preview": "List the tools we should use for this project", "message_count": 20})
        assert not is_ping_probe_session({"title": "Chat", "preview": "hi", "message_count": 8})
        assert not is_ping_probe_session({"title": "Chat", "preview": "ping, reply with one word", "message_count": 8})
        assert not is_ping_probe_session({"title": None, "preview": "fix the login bug"})

    def test_query_session_listing_hides_cron_source(self, db):
        """Cron watchdog/monitor runs flood recents; hidden unless searched."""
        db.create_session("cron_run", "cron")
        db.append_message("cron_run", "user", "/proxy-network-adaptive")
        db.create_session("jev_run", "jev", model_config='{"_session_origin":"jev"}')
        db.append_message("jev_run", "user", "normal-looking automation task")

        rows = query_session_listing(db, source=None, include_unnamed=True, limit=50)
        assert "cron_run" not in [r["id"] for r in rows]
        assert "jev_run" not in [r["id"] for r in rows]
        # Search keeps cron rows reachable (id may be the only handle).
        searched = query_session_listing(
            db, source=None, include_unnamed=True, search_query="cron_run", limit=50
        )
        assert [r["id"] for r in searched] == ["cron_run"]
