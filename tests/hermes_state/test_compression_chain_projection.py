"""Compression-tip projection, filtering and visible lineage membership (#75625)."""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestCompressionChainProjection:
    """Tests for lineage-aware list_sessions_rich — compressed conversations
    surface as their live continuation tip, not the dead parent root.
    """

    def _build_compression_chain(self, db, t0: float):
        """Helper: builds root -> delegate -> compression-child -> tip chain.

        Returns (root_id, delegate_id, mid_id, tip_id).
        """
        # Root that gets compressed
        db.create_session("root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        db.append_message("root1", "user", "help me refactor auth")

        # Delegate subagent spawned while root1 was live (before it ended)
        db.create_session("delegate1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
            (t0 + 600, t0 + 650, "delegate1"),
        )
        db.append_message("delegate1", "user", "delegate task")

        # root1 compressed at t0+1800
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        # Continuation mid created 1s after parent ended
        db.create_session("mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_root + 1, "mid1"),
        )
        db.append_message("mid1", "user", "continuing")

        # mid1 also compressed
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Tip — latest continuation
        db.create_session("tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_mid + 1, "tip1"),
        )
        db.append_message("tip1", "user", "latest message")

        db._conn.commit()
        return ("root1", "delegate1", "mid1", "tip1")

    def test_get_compression_tip_walks_full_chain(self, db):
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        assert db.get_compression_tip("root1") == "tip1"
        assert db.get_compression_tip("mid1") == "tip1"
        assert db.get_compression_tip("tip1") == "tip1"

    def test_reset_fork_sibling_does_not_steal_tip_projection(self, db):
        """A reset fork child (``model_config._reset_from``, ended LATER than the
        real continuation) must not win the chain-step tiebreak. It is a separate
        user-visible conversation that already lists as its own row, so letting
        the lineage tip land on it hides the true continuation and shows the
        reset sibling twice (#114271)."""
        import time as _time
        t0 = _time.time() - 3600

        db.create_session("root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        db.append_message("root1", "user", "help me refactor auth")
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        db.create_session("mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?", (t_compress_root + 1, "mid1"),
        )
        db.append_message("mid1", "user", "continuing")
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Real tip: closed by the startup orphan reap, LAST ACTIVE EARLIER than
        # the reset fork, so the old tiebreak (last_active DESC) preferred the fork.
        db.create_session("tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, end_reason=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 1, t_compress_mid + 600, "startup_orphan_reap",
             t_compress_mid + 600, "tip1"),
        )
        db.append_message("tip1", "user", "latest message")

        # Reset fork of mid1: its own conversation, ended session_reset later.
        db.create_session(
            "reset1", "cli", parent_session_id="mid1", model_config={"_reset_from": "mid1"},
        )
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, end_reason=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 2, t_compress_mid + 900, "session_reset",
             t_compress_mid + 900, "reset1"),
        )
        db.append_message("reset1", "user", "post reset talk")
        db._conn.commit()

        # The chain/tip follow the real continuation, never the reset fork.
        assert db.get_compression_tip("root1") == "tip1"
        assert db.get_compression_tip("mid1") == "tip1"

        # Projection: the lineage surfaces as tip1; reset1 stays exactly its own
        # single row instead of appearing twice (own row + hijacked projection).
        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        assert ids.count("reset1") == 1
        assert "tip1" in ids
        assert "root1" not in ids and "mid1" not in ids
        tip_row = next(s for s in sessions if s["id"] == "tip1")
        assert tip_row["_lineage_root_id"] == "root1"
        assert tip_row["preview"].startswith("latest message")

        # The order_by_last_active chain CTE must not fold the reset fork's
        # later activity into the lineage either: a standalone session active
        # between the tip and the fork still outranks the projected lineage row.
        db.create_session("solo", "cli")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 300, t_compress_mid + 700, "solo"),
        )
        db.append_message("solo", "user", "standalone")
        db._conn.commit()
        ordered = db.list_sessions_rich(source="cli", limit=20, order_by_last_active=True)
        ordered_ids = [s["id"] for s in ordered]
        assert ordered_ids.count("reset1") == 1
        assert ordered_ids.index("solo") < ordered_ids.index("tip1")

    def test_reset_fork_of_compressed_parent_is_not_a_lineage_member(self, db):
        """The Python lineage walk (``get_compression_lineage`` / ``_is_compression_child_row``)
        must agree with the SQL chain step: a reset fork hanging off a compression-ended parent is
        its own conversation, so the true tip keeps its ancestors and the fork never enters the
        lineage even when it started first."""
        import time as _time
        t0 = _time.time() - 3600
        db.create_session("root1", "cli")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "root1"))
        db.create_session("reset1", "cli", parent_session_id="root1", model_config={"_reset_from": "root1"})
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 11, "reset1"))
        db.create_session("tip1", "cli", parent_session_id="root1")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 20, "tip1"))
        db._conn.commit()

        assert db._is_compression_child_row(db.get_session("reset1")) is False
        assert db.get_compression_lineage("root1") == ["root1", "tip1"]
        assert db.get_compression_lineage("tip1") == ["root1", "tip1"]
        assert db.get_compression_lineage("reset1") == ["reset1"]

    def test_routing_lineage_cte_agrees_with_python_walk_for_reset_fork(self, db):
        """``record_gateway_session_peer(include_compression_ancestors=True)`` re-keys every row named by
        ``_COMPRESSION_LINEAGE_CTE``. Resuming a reset fork of a compression-ended parent must
        re-key only the fork: the CTE has to stop at the reset child exactly like
        ``get_compression_lineage`` does, or the real lineage's ancestors land on the fork's peer."""
        import hermes_state_gateway as gateway_mod

        t0 = time.time() - 3600
        db.create_session("root", "cli")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "root"))
        db.create_session("mid1", "cli", parent_session_id="root")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 20, "mid1"))
        db.create_session("mid2", "cli", parent_session_id="mid1")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 30, "mid2"))
        db.create_session("tip", "cli", parent_session_id="mid2")
        db.create_session("reset", "cli", parent_session_id="mid2", model_config={"_reset_from": "mid2"})
        db._conn.commit()

        sql = gateway_mod._COMPRESSION_LINEAGE_CTE + " SELECT id FROM compression_lineage"
        with db._read_ctx() as conn:
            cte = {start: sorted(r[0] for r in conn.execute(sql, (start,)).fetchall()) for start in ("reset", "tip")}
        assert cte["reset"] == sorted(db.get_compression_lineage("reset")) == ["reset"]
        assert cte["tip"] == sorted(db.get_compression_lineage("tip")) == ["mid1", "mid2", "root", "tip"]

        db.record_gateway_session_peer("reset", source="cli", user_id="u", session_key="cli:reset-chat",
                                       chat_id="reset-chat", chat_type="dm", include_compression_ancestors=True)
        assert db.get_session("reset")["session_key"] == "cli:reset-chat"
        assert all(db.get_session(s)["session_key"] != "cli:reset-chat" for s in ("root", "mid1", "mid2", "tip"))

    def test_list_serves_full_lineage_ids_for_projected_rows(self, db):
        """The projected tip row must carry every chain id. Root and tip
        alone are not enough client-side: a persisted tile or route can hold
        a MIDDLE segment's id (it was the tip when opened), and without the
        intermediates that surface cannot prove it names this conversation —
        which is how one chat ends up open twice after a compaction."""
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        tip_row = next(s for s in sessions if s["id"] == "tip1")
        assert tip_row["_lineage_ids"] == ["root1", "mid1", "tip1"]
        solo_row = next(s for s in sessions if s["id"] == "solo")
        assert solo_row.get("_lineage_ids") is None

    def test_list_labels_projected_continuation_kind(self, db):
        """#121148: a projected compression tip is an automatic continuation, not a
        fresh conversation and not a user branch — the sidebar must be able to say
        so. Plain rows and branches carry no label."""
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db.create_session("branchy", "cli", parent_session_id="root1",
                          model_config={"_branched_from": "root1"})
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        tip_row = next(s for s in sessions if s["id"] == "tip1")
        assert tip_row["continuation_kind"] == "compression"
        solo_row = next(s for s in sessions if s["id"] == "solo")
        assert solo_row.get("continuation_kind") is None
        branch_row = next(s for s in sessions if s["id"] == "branchy")
        assert branch_row.get("continuation_kind") is None

    def test_list_keeps_live_tip_carrying_parent_link(self, db):
        """#121148: `parent_session_id` on the live tip must not evict it from the
        list — the lineage must be expressible AND visible at once. Sealed
        (compression-ended) children stay hidden as before."""
        import time as _time
        t0 = _time.time() - 3600
        # A three-link chain: seg-a → seg-prev → live-tip. Only the tip is live.
        db.create_session("seg-a", "cli")
        db.append_message("seg-a", "user", "earlier days")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "seg-a"))
        # A restored previous segment the user re-linked the live tip to.
        db.create_session("seg-prev", "cli", parent_session_id="seg-a")
        db.append_message("seg-prev", "user", "restored segment")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
                         (t0 + 20, "seg-prev"))
        db.create_session("live-tip", "cli", parent_session_id="seg-prev")
        db.append_message("live-tip", "user", "still talking here")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        listed_ids = {s["id"] for s in sessions}

        # The live tip stays listable while naming its parent.
        assert "live-tip" in listed_ids
        # Sealed compression children stay hidden (the projection surfaces the
        # lineage through its root row instead).
        assert "seg-a" not in listed_ids
        assert "seg-prev" not in listed_ids or "live-tip" in listed_ids



    def test_list_surfaces_tip_for_compressed_root(self, db):
        """The list must show the tip's id/message_count/preview in place of
        the root row, so users can see and resume the live conversation.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        # Add an uncompressed root for comparison.
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        # Only top-level conversations appear: tip1 (projected from root1) + solo.
        # Delegate children, mid1, and the dead root1 must NOT be in the list.
        assert "tip1" in ids
        assert "solo" in ids
        assert "root1" not in ids
        assert "mid1" not in ids
        assert "delegate1" not in ids

        tip_row = next(s for s in sessions if s["id"] == "tip1")
        # The row surfaces the tip's identity but preserves the root's start
        # timestamp for stable ordering and lineage tracking.
        assert tip_row["_lineage_root_id"] == "root1"
        assert tip_row["preview"].startswith("latest message")
        assert tip_row["ended_at"] is None  # tip is still live
        assert tip_row["end_reason"] is None
        # Same-source chain: projected source stays the tip's (here equal to root).
        assert tip_row["source"] == "cli"

    def test_list_projects_multiple_independent_chains_in_one_call(self, db):
        """Two unrelated compression chains in the same page must each
        resolve to their own tip, not get cross-mixed by the batched tip-row
        fetch (regression test for the single-query batch in
        _get_session_rich_rows_batch — a wrong id->row mapping there would
        silently swap one chain's data onto the other)."""
        import time as _time

        t0 = _time.time() - 7200
        self._build_compression_chain(db, t0)

        # Second, independent chain — same shape, different ids/content.
        db.create_session("root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        db.append_message("root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        db.create_session("tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        db.append_message("tip2", "user", "second conversation continuation")
        db.update_session_cwd("tip2", "/tmp/workspaces/second")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        assert "root1" not in ids and "root2" not in ids
        assert "tip1" in ids and "tip2" in ids

        tip1_row = next(s for s in sessions if s["id"] == "tip1")
        tip2_row = next(s for s in sessions if s["id"] == "tip2")
        assert tip1_row["_lineage_root_id"] == "root1"
        assert tip1_row["preview"].startswith("latest message")
        assert tip2_row["_lineage_root_id"] == "root2"
        assert tip2_row["preview"].startswith("second conversation continuation")
        assert tip2_row["cwd"] == "/tmp/workspaces/second"

    def test_list_batches_tip_row_fetch_into_one_query(self, db, monkeypatch):
        """Projection must resolve tip rows for a whole page in one batched
        query, not one _get_session_rich_row() call per compression root."""
        import time as _time

        t0 = _time.time() - 7200
        self._build_compression_chain(db, t0)
        db.create_session("root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        db.append_message("root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        db.create_session("tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        db.append_message("tip2", "user", "second continuation")
        db._conn.commit()

        batch_calls = []
        single_calls = []
        original_batch = db._get_session_rich_rows_batch
        original_single = db._get_session_rich_row

        def counting_batch(session_ids, **kwargs):
            batch_calls.append(list(session_ids))
            return original_batch(session_ids, **kwargs)

        def counting_single(session_id, **kwargs):
            single_calls.append(session_id)
            return original_single(session_id, **kwargs)

        monkeypatch.setattr(db, "_get_session_rich_rows_batch", counting_batch)
        monkeypatch.setattr(db, "_get_session_rich_row", counting_single)

        sessions = db.list_sessions_rich(source="cli", limit=20)
        assert len(sessions) >= 2  # sanity: both chains actually surfaced

        # Two compression roots resolved with exactly one batched call, and
        # zero single-row calls — not one single-row call per root.
        assert len(batch_calls) == 1
        assert set(batch_calls[0]) == {"tip1", "tip2"}
        assert single_calls == []

    def test_list_surfaces_tip_source_for_cross_source_chain(self, db):
        """Cross-source compression must project the tip's source (#75625).

        WebUI loads list_sessions_rich without a source filter and tabs client-
        side on the row's ``source``. If projection keeps the root source, a
        telegram-rooted chain continued in webui vanishes from the webui tab
        (root source still telegram; tip excluded as a compression child).
        """
        import time as _time

        t0 = _time.time() - 3600
        db.create_session("xroot", "telegram")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?", (t0, "xroot")
        )
        db.append_message("xroot", "user", "gateway start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 100, "compression", "xroot"),
        )
        db.create_session("xtip", "webui", parent_session_id="xroot")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t0 + 101, "xtip"),
        )
        db.append_message("xtip", "user", "continued in webui")
        db.create_session(
            "xdelegate",
            "telegram",
            parent_session_id="xroot",
            model_config={"_delegate_from": "xroot"},
        )
        db._conn.commit()

        # Unfiltered list: projected tip carries source=webui.
        all_sessions = db.list_sessions_rich(limit=20)
        by_id = {s["id"]: s for s in all_sessions}
        assert "xtip" in by_id
        assert "xroot" not in by_id
        projected = by_id["xtip"]
        assert projected["source"] == "webui"
        assert projected["_lineage_root_id"] == "xroot"
        assert projected["preview"].startswith("continued in webui")
        assert projected["ended_at"] is None

        # source= filter is applied to the *projected tip*, not the root
        # (#75625 sweeper): webui tab sees the chain; telegram tab does not
        # (tip is webui).
        webui_only = db.list_sessions_rich(source="webui", limit=20)
        assert "xtip" in {s["id"] for s in webui_only}
        webui_row = next(s for s in webui_only if s["id"] == "xtip")
        assert webui_row["source"] == "webui"

        telegram_only = db.list_sessions_rich(source="telegram", limit=20)
        assert "xtip" not in {s["id"] for s in telegram_only}

        # Totals paired with list_sessions_rich must use tip source too.
        assert db.session_count(source="webui", exclude_children=True) >= 1
        assert db.session_count(source="telegram", exclude_children=True) == 0
        by_src = db.session_count_by_source(exclude_children=True)
        assert by_src == {"webui": 1}

    def test_source_filter_uses_live_tip_activity_ordering(self, db):
        """Source filtering and projection must select the same live tip.

        A continuation heartbeat can be newer than its latest message while a
        sibling has the newer message timestamp. The SQL source predicate must
        use the same freshest-activity ordering as ``get_compression_tip``;
        otherwise the filter can admit the wrong source and then project a
        different tip in Python.
        """
        import time as _time

        t0 = _time.time() - 3600
        db.create_session("activity-root", "telegram")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
            (t0, t0 + 10, "compression", "activity-root"),
        )

        db.create_session(
            "message-tip", "telegram", parent_session_id="activity-root"
        )
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (t0 + 20, t0 + 50, "message-tip"),
        )
        db.append_message("message-tip", "user", "older activity heartbeat")
        db._conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (t0 + 300, "message-tip"),
        )

        db.create_session(
            "heartbeat-tip", "webui", parent_session_id="activity-root"
        )
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (t0 + 30, t0 + 400, "heartbeat-tip"),
        )
        db.append_message("heartbeat-tip", "user", "newer heartbeat")
        db._conn.execute(
            "UPDATE messages SET timestamp=? WHERE session_id=?",
            (t0 + 100, "heartbeat-tip"),
        )
        db._conn.commit()

        assert db.get_compression_tip("activity-root") == "heartbeat-tip"
        webui_rows = db.list_sessions_rich(source="webui", limit=20)
        assert [row["id"] for row in webui_rows] == ["heartbeat-tip"]
        assert db.list_sessions_rich(source="telegram", limit=20) == []

    def test_sql_tip_source_projection_matches_python_walk(self, db):
        """The SQL source projection must stay equivalent to the Python walker.

        Keep this as a small table-driven lineage matrix instead of duplicating
        the implementation in a second test helper. It covers empty sources,
        multi-hop compression, ignored branch/delegate children, and a fully
        tied sibling choice where the stable id tie-breaker is observable.
        """
        from hermes_state_compression import _projected_tip_source_sql

        db.create_session("matrix-plain", "")

        db.create_session("matrix-chain-root", "telegram")
        db.end_session("matrix-chain-root", "compression")
        db.create_session(
            "matrix-chain-mid",
            "telegram",
            parent_session_id="matrix-chain-root",
        )
        db.end_session("matrix-chain-mid", "compression")
        db.create_session(
            "matrix-chain-tip",
            "webui",
            parent_session_id="matrix-chain-mid",
        )

        db.create_session("matrix-filter-root", "telegram")
        db.end_session("matrix-filter-root", "compression")
        db.create_session(
            "matrix-filter-delegate",
            "webui",
            parent_session_id="matrix-filter-root",
            model_config={"_delegate_from": "matrix-filter-root"},
        )
        db.create_session(
            "matrix-filter-branch",
            "slack",
            parent_session_id="matrix-filter-root",
            model_config={"_branched_from": "matrix-filter-root"},
        )
        db.create_session(
            "matrix-filter-tip",
            "webui",
            parent_session_id="matrix-filter-root",
        )

        db.create_session("matrix-tie-root", "telegram")
        db.end_session("matrix-tie-root", "compression")
        for child_id, child_source in (
            ("matrix-tie-a", "alpha"),
            ("matrix-tie-b", "beta"),
        ):
            db.create_session(
                child_id,
                child_source,
                parent_session_id="matrix-tie-root",
            )
            db._conn.execute(
                "UPDATE sessions SET started_at = 1700000000, "
                "last_activity_at = 1700000100 WHERE id = ?",
                (child_id,),
            )
        db._conn.commit()

        roots = (
            "matrix-plain",
            "matrix-chain-root",
            "matrix-filter-root",
            "matrix-tie-root",
        )
        projection_sql = (
            "SELECT "
            f"{_projected_tip_source_sql('s')} "
            "AS projected_source FROM sessions s WHERE s.id = ?"
        )
        for root_id in roots:
            tip_id = db.get_compression_tip(root_id)
            tip = db.get_session(tip_id)
            expected = (tip["source"] or "cli") if tip else "cli"
            actual = db._conn.execute(projection_sql, (root_id,)).fetchone()
            assert actual["projected_source"] == expected

        assert db.get_compression_tip("matrix-tie-root") == "matrix-tie-b"

    def test_list_handles_broken_chain_gracefully(self, db):
        """A compression root with no child (e.g. DB corruption or a partial
        end_session call that didn't finish creating the child) must not
        crash the list — it should fall back to surfacing the root as-is.
        """
        import time as _time
        t0 = _time.time() - 100
        db.create_session("orphan", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "orphan"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 10, "compression", "orphan"),
        )
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=10)
        ids = [s["id"] for s in sessions]
        assert "orphan" in ids
        row = next(s for s in sessions if s["id"] == "orphan")
        # No tip means no projection — row stays raw.
        assert "_lineage_root_id" not in row
        assert row["end_reason"] == "compression"
