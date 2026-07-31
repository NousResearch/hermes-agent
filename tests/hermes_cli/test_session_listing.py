"""Tests for the shared session-listing helpers (hermes_cli/session_listing.py)."""

import inspect

import pytest

from hermes_cli.session_listing import (
    _compression_root,
    dedup_compression_chains,
    last_active_of,
    parse_session_listing_args,
    query_session_listing,
    search_session_listing,
    session_rank,
    session_rank_lookup,
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


def test_hop_caps_unified():
    """Every lineage walker shares one hop cap (F13)."""
    from hermes_state import COMPRESSION_CHAIN_MAX_HOPS

    assert COMPRESSION_CHAIN_MAX_HOPS == 100
    sig = inspect.signature(_compression_root)
    assert sig.parameters["max_hops"].default == COMPRESSION_CHAIN_MAX_HOPS


class TestSearchSessionListing:
    """The shared search pipeline (F9): both `hermes sessions search` and
    `/sessions search` must produce identical canonical rows — chain-aware
    rank, root Created, chain-total Tok, root-ancestor previews.
    """

    def test_chain_aware_rows(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "search.db")
        conn = db._conn
        db.create_session("sr_root", "cli")
        db.append_message("sr_root", "user", "needle alpha", timestamp=1.0)
        db.end_session("sr_root", end_reason="compression")
        db.create_session("sr_mid", "cli", parent_session_id="sr_root")
        db.append_message("sr_mid", "user", "needle beta", timestamp=2.0)
        db.end_session("sr_mid", end_reason="compression")
        db.create_session("sr_tip", "cli", parent_session_id="sr_mid")
        db.append_message("sr_tip", "user", "needle gamma", timestamp=3.0)
        db.create_session("sr_stand", "cli")
        db.append_message("sr_stand", "user", "needle standalone", timestamp=4.0)
        for sid, ts in [
            ("sr_root", 100.0),
            ("sr_mid", 200.0),
            ("sr_tip", 300.0),
            ("sr_stand", 400.0),
        ]:
            conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid))
        conn.execute(
            "UPDATE sessions SET input_tokens=100, output_tokens=10 WHERE id='sr_root'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=200, output_tokens=20 WHERE id='sr_mid'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=300, output_tokens=30 WHERE id='sr_tip'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=50, output_tokens=5 WHERE id='sr_stand'"
        )
        conn.commit()
        try:
            rows, previews = search_session_listing(db, "needle")
            # Recency order (last_active), chain deduped to its tip.
            assert [r["id"] for r in rows] == ["sr_stand", "sr_tip"]
            assert set(previews) == {"sr_stand", "sr_tip"}
            tip = next(r for r in rows if r["id"] == "sr_tip")
            assert tip["rank"] == 2  # canonical position of the chain
            assert tip["started_at"] == 100.0  # root's Created, not tip's
            assert (tip["input_tokens"], tip["output_tokens"]) == (600, 60)  # chain total
            assert previews["sr_tip"] == "needle alpha"  # root's first user msg
            stand = next(r for r in rows if r["id"] == "sr_stand")
            assert stand["rank"] == 1
            assert stand["started_at"] == 400.0
        finally:
            db.close()

    def test_exclude_session_id_and_limit(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "search2.db")
        for i, sid in enumerate(["sx_a", "sx_b", "sx_c"]):
            db.create_session(sid, "cli")
            db.append_message(sid, "user", f"needle {sid}", timestamp=float(i + 1))
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (float(i + 1), sid)
            )
        db._conn.commit()
        try:
            # Limit applies LAST, after recency sort.
            rows, _ = search_session_listing(db, "needle", limit=2)
            assert [r["id"] for r in rows] == ["sx_c", "sx_b"]
            rows, _ = search_session_listing(db, "needle", exclude_session_id="sx_c")
            assert {r["id"] for r in rows} == {"sx_b", "sx_a"}
        finally:
            db.close()

    def test_ancestor_only_hit_projects_to_tip(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "search_ancestor.db")
        conn = db._conn
        # Chain pa_root → pa_mid → pa_tip; only the ROOT's message matches
        # FTS. The surfaced row must be the tip (same generation as the #
        # rank and Last column resolve), with the root's Created + opener.
        db.create_session("pa_root", "cli")
        db.append_message("pa_root", "user", "needle only in the root", timestamp=1.0)
        db.end_session("pa_root", end_reason="compression")
        db.create_session("pa_mid", "cli", parent_session_id="pa_root")
        db.append_message("pa_mid", "user", "unrelated mid", timestamp=2.0)
        db.end_session("pa_mid", end_reason="compression")
        db.create_session("pa_tip", "cli", parent_session_id="pa_mid")
        db.append_message("pa_tip", "user", "unrelated tip", timestamp=3.0)
        for sid, ts in [("pa_root", 100.0), ("pa_mid", 200.0), ("pa_tip", 300.0)]:
            conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid))
        conn.commit()
        try:
            rows, previews = search_session_listing(db, "needle")
            assert [r["id"] for r in rows] == ["pa_tip"]  # projected, not pa_root
            assert rows[0]["rank"] == 1  # tip's canonical position
            assert rows[0]["started_at"] == 100.0  # root's Created
            assert previews["pa_tip"] == "needle only in the root"  # root's opener
        finally:
            db.close()

    def test_branch_child_preview_does_not_cross_into_parent(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "search_branch.db")
        conn = db._conn
        # Parent conversation + child with a parent_session_id but NO
        # compression end on the parent (branch-like split). The child's
        # preview must be its OWN opener — branches are separate
        # conversations and must not inherit the parent's opener.
        db.create_session("br_parent", "cli")
        db.append_message("br_parent", "user", "parent opener words", timestamp=1.0)
        db.create_session("br_child", "cli", parent_session_id="br_parent")
        db.append_message("br_child", "user", "needle child opener", timestamp=2.0)
        for sid, ts in [("br_parent", 100.0), ("br_child", 200.0)]:
            conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid))
        conn.commit()
        try:
            rows, previews = search_session_listing(db, "needle")
            assert [r["id"] for r in rows] == ["br_child"]
            assert previews["br_child"] == "needle child opener"
        finally:
            db.close()

    def test_no_matches_returns_empty(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "search3.db")
        db.create_session("sx_none", "cli")
        db.append_message("sx_none", "user", "unrelated words", timestamp=1.0)
        db._conn.commit()
        try:
            rows, previews = search_session_listing(db, "needle")
            assert rows == [] and previews == {}
        finally:
            db.close()


class TestLastActiveOf:
    """`last_active_of` must share the listing's Last-column definition:
    the latest message timestamp, never a later `ended_at`.

    Regression coverage for the search/list drift where search rendered
    `COALESCE(ended_at, started_at)` — a session that idled for hours
    after its final message showed a Last in the future relative to its
    own listing row, and the search sort inherited the same wrong key.
    """

    def test_uses_latest_message_timestamp_not_ended_at(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "la.db")
        db.create_session("sess_old", "cli")
        db.append_message("sess_old", "user", "first", timestamp=1_000_000.0)
        db.append_message("sess_old", "assistant", "second", timestamp=1_000_500.0)
        # Closed 2h after its final message: ended_at > last message ts.
        db.end_session("sess_old", end_reason="cli_close")
        try:
            last = last_active_of(db, ["sess_old"])["sess_old"]
            assert last == 1_000_500.0
            # Agrees with the canonical listing column.
            rows = db.list_sessions_rich(source="cli", include_children=False)
            assert {r["id"]: r["last_active"] for r in rows}["sess_old"] == 1_000_500.0
        finally:
            db.close()

    def test_falls_back_to_started_at_when_no_messages(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "empty.db")
        db.create_session("sess_empty", "cli")
        try:
            meta = db.get_session("sess_empty")
            assert last_active_of(db, ["sess_empty"])["sess_empty"] == meta["started_at"]
        finally:
            db.close()

    def test_search_order_matches_listing_order(self, tmp_path):
        """Sessions ordered by `last_active_of` sort identically to the
        canonical listing's last-active ordering — the contract search
        promises when it says it shows the same numbers/order as the list.
        """
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "order.db")
        # A: old messages, closed late (ended_at in the future relative to
        # its last message). B: newer messages, still open.
        db.create_session("sess_a", "cli")
        db.append_message("sess_a", "user", "a1", timestamp=1_000_000.0)
        db.end_session("sess_a", end_reason="cli_close")
        db.create_session("sess_b", "cli")
        db.append_message("sess_b", "user", "b1", timestamp=1_100_000.0)
        try:
            sid_latest = last_active_of(db, ["sess_a", "sess_b"])
            search_order = sorted(
                ["sess_a", "sess_b"],
                key=lambda s: sid_latest.get(s, 0),
                reverse=True,
            )
            listing_order = [
                r["id"]
                for r in db.list_sessions_rich(
                    source="cli", include_children=False, order_by_last_active=True
                )
            ]
            # Both surfaces agree on which session is "most recent" even
            # though sess_a's ended_at is later than sess_b's last message.
            assert search_order == listing_order == ["sess_b", "sess_a"]
        finally:
            db.close()


class TestChainTokenTotals:
    """Tok(ΣIn/ΣOut) shows the compression-chain total, not the root's or
    the tip's single-generation counts.

    Regression coverage for the projection surfacing the root's historical
    token counts on a projected tip row — for a long conversation the root
    figure can differ from the live tip by an order of magnitude.
    """

    def test_listing_shows_chain_total_not_root_or_tip(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "tok.db")
        db.create_session("tok_root", "cli")
        db.end_session("tok_root", end_reason="compression")
        db.create_session("tok_mid", "cli", parent_session_id="tok_root")
        db.end_session("tok_mid", end_reason="compression")
        db.create_session("tok_tip", "cli", parent_session_id="tok_mid")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET input_tokens=100, output_tokens=10 WHERE id='tok_root'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=200, output_tokens=20 WHERE id='tok_mid'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=300, output_tokens=30 WHERE id='tok_tip'"
        )
        conn.commit()
        try:
            rows = db.list_sessions_rich(source="cli", include_children=False)
            assert [r["id"] for r in rows] == ["tok_tip"]
            row = rows[0]
            assert row["input_tokens"] == 600, row["input_tokens"]
            assert row["output_tokens"] == 60, row["output_tokens"]
            # chain_token_totals resolves any generation of the chain.
            assert db.chain_token_totals(["tok_root", "tok_mid", "tok_tip"]) == {
                "tok_root": (600, 60),
                "tok_mid": (600, 60),
                "tok_tip": (600, 60),
            }
        finally:
            db.close()

    def test_branch_delegate_tool_children_not_counted(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "tok_excl.db")
        db.create_session("tok_root", "cli")
        db.end_session("tok_root", end_reason="compression")
        db.create_session("tok_tip", "cli", parent_session_id="tok_root")
        db.create_session("tok_branch", "cli", parent_session_id="tok_root")
        db.create_session("tok_delegate", "cli", parent_session_id="tok_root")
        db.create_session("tok_tool", "tool", parent_session_id="tok_root")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET input_tokens=100, output_tokens=10 WHERE id='tok_root'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=300, output_tokens=30 WHERE id='tok_tip'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=999, output_tokens=99 WHERE id='tok_branch'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=888, output_tokens=88 WHERE id='tok_delegate'"
        )
        conn.execute(
            "UPDATE sessions SET input_tokens=777, output_tokens=77 WHERE id='tok_tool'"
        )
        conn.execute(
            "UPDATE sessions SET model_config='{\"_branched_from\": \"tok_root\"}' "
            "WHERE id='tok_branch'"
        )
        conn.execute(
            "UPDATE sessions SET model_config='{\"_delegate_from\": \"tok_root\"}' "
            "WHERE id='tok_delegate'"
        )
        conn.commit()
        try:
            rows = db.list_sessions_rich(source="cli", include_children=False)
            # Branch children stay visible as their own rows; the projected
            # chain entry is the tip, summed over root + tip only.
            ids = [r["id"] for r in rows]
            assert "tok_tip" in ids and "tok_branch" in ids
            tip_row = next(r for r in rows if r["id"] == "tok_tip")
            assert (tip_row["input_tokens"], tip_row["output_tokens"]) == (400, 40)
            assert db.chain_token_totals(["tok_tip"]) == {"tok_tip": (400, 40)}
        finally:
            db.close()

    def test_standalone_session_keeps_own_tokens(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "tok_standalone.db")
        db.create_session("tok_standalone", "cli")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET input_tokens=50, output_tokens=5 WHERE id='tok_standalone'"
        )
        conn.commit()
        try:
            rows = db.list_sessions_rich(source="cli", include_children=False)
            assert [r["id"] for r in rows] == ["tok_standalone"]
            assert rows[0]["input_tokens"] == 50
            assert rows[0]["output_tokens"] == 5
            assert db.chain_token_totals(["tok_standalone"]) == {
                "tok_standalone": (50, 5)
            }
        finally:
            db.close()


class TestRenderPreviewLineage:
    """Projected compression rows must preview the chain root's first user
    message — never the tip's own first message, which is the compaction
    banner on a continuation with no real user turns.

    Regression coverage for the plain listing leaking
    `[CONTEXT COMPACTION — REFERENCE ONLY]` into the Preview column: the
    projection keeps the root's parent_session_id (None), so the renderer's
    parent-walk fallback never fired and the tip's banner preview won.
    """

    def _render(self, db, rows):
        lines = []
        from hermes_cli.session_listing import render_sessions_table

        render_sessions_table(rows, out=lines.append, db=db)
        return "\n".join(lines)

    def test_projected_tip_previews_root_first_user_message(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "pv.db")
        db.create_session("pv_root", "cli")
        db.append_message("pv_root", "user", "Original conversation opener", timestamp=1.0)
        db.end_session("pv_root", end_reason="compression")
        db.create_session("pv_tip", "cli", parent_session_id="pv_root")
        db.append_message(
            "pv_tip", "user", "[CONTEXT COMPACTION — REFERENCE ONLY]", timestamp=2.0
        )
        try:
            rows = db.list_sessions_rich(source="cli", include_children=False)
            assert [r["id"] for r in rows] == ["pv_tip"]
            output = self._render(db, rows)
            assert "Original conversation opener" in output
            assert "COMPACTION" not in output
        finally:
            db.close()

    def test_standalone_row_previews_own_first_user_message(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "pv2.db")
        db.create_session("pv_standalone", "cli")
        db.append_message("pv_standalone", "user", "Standalone opener", timestamp=1.0)
        try:
            rows = db.list_sessions_rich(source="cli", include_children=False)
            output = self._render(db, rows)
            assert "Standalone opener" in output
        finally:
            db.close()


class TestSessionRank:
    """`session_rank` must resolve a mid-chain search hit through the
    chain's live tip using the same canonical edge definition as the
    listing projection — never through a branch/delegate/tool child that
    happens to have the latest started_at.
    """

    def test_delegate_child_does_not_hijack_rank_walk(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "rank.db")
        db.create_session("rk_root", "cli")
        db.end_session("rk_root", end_reason="compression")
        db.create_session("rk_tip", "cli", parent_session_id="rk_root")
        # A delegate child that started LATER than the tip: the unfiltered
        # started_at-first walk would follow it and lose the chain.
        db.create_session("rk_delegate", "cli", parent_session_id="rk_root")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET model_config='{\"_delegate_from\": \"rk_root\"}' "
            "WHERE id='rk_delegate'"
        )
        conn.execute("UPDATE sessions SET started_at=9999999999 WHERE id='rk_delegate'")
        conn.commit()
        try:
            rank_of = session_rank_lookup(db)
            assert set(rank_of) == {"rk_tip"}
            assert session_rank(db, "rk_root", rank_of) == 1
            assert session_rank(db, "rk_tip", rank_of) == 1
        finally:
            db.close()

    def test_rank_identical_across_generations(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "rank2.db")
        db.create_session("rg_root", "cli")
        db.end_session("rg_root", end_reason="compression")
        db.create_session("rg_mid", "cli", parent_session_id="rg_root")
        db.end_session("rg_mid", end_reason="compression")
        db.create_session("rg_tip", "cli", parent_session_id="rg_mid")
        try:
            rank_of = session_rank_lookup(db)
            assert rank_of == {"rg_tip": 1}
            for sid in ("rg_root", "rg_mid", "rg_tip"):
                assert session_rank(db, sid, rank_of) == 1
        finally:
            db.close()

    def test_branch_child_has_own_rank(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "rank3.db")
        db.create_session("rb_root", "cli")
        db.end_session("rb_root", end_reason="compression")
        db.create_session("rb_tip", "cli", parent_session_id="rb_root")
        db.create_session("rb_branch", "cli", parent_session_id="rb_root")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET model_config='{\"_branched_from\": \"rb_root\"}' "
            "WHERE id='rb_branch'"
        )
        conn.commit()
        try:
            rank_of = session_rank_lookup(db)
            # Branch children surface as their own canonical rows.
            assert set(rank_of) == {"rb_tip", "rb_branch"}
            # The root resolves through the tip, never the branch.
            assert session_rank(db, "rb_root", rank_of) == rank_of["rb_tip"]
            assert session_rank(db, "rb_branch", rank_of) == rank_of["rb_branch"]
        finally:
            db.close()


class TestCompressionRoot:
    """`_compression_root` must stop at branch/delegate/tool children —
    they are their own conversations and must never be collapsed into a
    source chain by dedup.
    """

    def test_delegate_child_is_own_root(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "cr.db")
        db.create_session("cr_root", "cli")
        db.end_session("cr_root", end_reason="compression")
        db.create_session("cr_tip", "cli", parent_session_id="cr_root")
        db.create_session("cr_delegate", "cli", parent_session_id="cr_root")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET model_config='{\"_delegate_from\": \"cr_root\"}' "
            "WHERE id='cr_delegate'"
        )
        conn.execute("UPDATE sessions SET source='subagent' WHERE id='cr_delegate'")
        conn.commit()
        try:
            assert _compression_root(db, "cr_delegate") == "cr_delegate"
            assert _compression_root(db, "cr_tip") == "cr_root"
            assert _compression_root(db, "cr_root") == "cr_root"
        finally:
            db.close()

    def test_tool_child_is_own_root(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "cr2.db")
        db.create_session("ct_root", "cli")
        db.end_session("ct_root", end_reason="compression")
        db.create_session("ct_tool", "tool", parent_session_id="ct_root")
        try:
            assert _compression_root(db, "ct_tool") == "ct_tool"
            assert _compression_root(db, "ct_root") == "ct_root"
        finally:
            db.close()

    def test_dedup_does_not_collapse_delegate_into_chain(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "cr3.db")
        db.create_session("cd_root", "cli")
        db.end_session("cd_root", end_reason="compression")
        db.create_session("cd_tip", "cli", parent_session_id="cd_root")
        db.create_session("cd_delegate", "cli", parent_session_id="cd_root")
        conn = db._conn
        conn.execute(
            "UPDATE sessions SET model_config='{\"_delegate_from\": \"cd_root\"}' "
            "WHERE id='cd_delegate'"
        )
        conn.commit()
        try:
            kept = dedup_compression_chains(db, ["cd_root", "cd_tip", "cd_delegate"])
            assert kept == {"cd_tip", "cd_delegate"}
        finally:
            db.close()
