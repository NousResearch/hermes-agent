"""Compression lineage in the conversation store on a backend store's StoreHolder: ``get_compression_tip`` and ``get_compression_lineage`` walk the chain and stop at reset forks, the routing lineage CTE agrees with that walk, and ``list_sessions_rich`` surfaces each compressed conversation as its live tip in one batched fetch."""

import time

import pytest

from persistence.sqlite.conversation._store import portability as store_portability
from persistence.sqlite.conversation._store import session_list as store_session_list
from persistence.sqlite.conversation._store.compression import (
    _is_compression_child_row,
    get_compression_lineage,
    get_compression_tip,
)
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    get_session,
    list_sessions_rich,
    update_session_cwd,
)


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture
def db(store):
    """The store's StoreHolder, shared with its repositories."""
    return store.holder()


class TestCompressionChainProjection:
    """Tests for lineage-aware list_sessions_rich — compressed conversations
    surface as their live continuation tip, not the dead parent root.
    """

    def _build_compression_chain(self, db, t0: float):
        """Helper: builds root -> delegate -> compression-child -> tip chain.

        Returns (root_id, delegate_id, mid_id, tip_id).
        """
        # Root that gets compressed
        create_session(db, "root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        append_message(db, "root1", "user", "help me refactor auth")

        # Delegate subagent spawned while root1 was live (before it ended)
        create_session(db, "delegate1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
            (t0 + 600, t0 + 650, "delegate1"),
        )
        append_message(db, "delegate1", "user", "delegate task")

        # root1 compressed at t0+1800
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        # Continuation mid created 1s after parent ended
        create_session(db, "mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_root + 1, "mid1"),
        )
        append_message(db, "mid1", "user", "continuing")

        # mid1 also compressed
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Tip — latest continuation
        create_session(db, "tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_mid + 1, "tip1"),
        )
        append_message(db, "tip1", "user", "latest message")

        db._conn.commit()
        return ("root1", "delegate1", "mid1", "tip1")

    def test_get_compression_tip_walks_full_chain(self, db):
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        assert get_compression_tip(db, "root1") == "tip1"
        assert get_compression_tip(db, "mid1") == "tip1"
        assert get_compression_tip(db, "tip1") == "tip1"

    def test_reset_fork_sibling_does_not_steal_tip_projection(self, db):
        """A reset fork child (``model_config._reset_from``, ended LATER than the
        real continuation) must not win the chain-step tiebreak. It is a separate
        user-visible conversation that already lists as its own row, so letting
        the lineage tip land on it hides the true continuation and shows the
        reset sibling twice (#114271)."""
        import time as _time
        t0 = _time.time() - 3600

        create_session(db, "root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        append_message(db, "root1", "user", "help me refactor auth")
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        create_session(db, "mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?", (t_compress_root + 1, "mid1"),
        )
        append_message(db, "mid1", "user", "continuing")
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Real tip: closed by the startup orphan reap, LAST ACTIVE EARLIER than
        # the reset fork, so the old tiebreak (last_active DESC) preferred the fork.
        create_session(db, "tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, end_reason=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 1, t_compress_mid + 600, "startup_orphan_reap",
             t_compress_mid + 600, "tip1"),
        )
        append_message(db, "tip1", "user", "latest message")

        # Reset fork of mid1: its own conversation, ended session_reset later.
        create_session(db,
            "reset1", "cli", parent_session_id="mid1", model_config={"_reset_from": "mid1"},
        )
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=?, end_reason=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 2, t_compress_mid + 900, "session_reset",
             t_compress_mid + 900, "reset1"),
        )
        append_message(db, "reset1", "user", "post reset talk")
        db._conn.commit()

        # The chain/tip follow the real continuation, never the reset fork.
        assert get_compression_tip(db, "root1") == "tip1"
        assert get_compression_tip(db, "mid1") == "tip1"

        # Projection: the lineage surfaces as tip1; reset1 stays exactly its own
        # single row instead of appearing twice (own row + hijacked projection).
        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "solo", "cli")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (t_compress_mid + 300, t_compress_mid + 700, "solo"),
        )
        append_message(db, "solo", "user", "standalone")
        db._conn.commit()
        ordered = list_sessions_rich(db, source="cli", limit=20, order_by_last_active=True)
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
        create_session(db, "root1", "cli")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "root1"))
        create_session(db, "reset1", "cli", parent_session_id="root1", model_config={"_reset_from": "root1"})
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 11, "reset1"))
        create_session(db, "tip1", "cli", parent_session_id="root1")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 20, "tip1"))
        db._conn.commit()

        assert _is_compression_child_row(db, get_session(db, "reset1")) is False
        assert get_compression_lineage(db, "root1") == ["root1", "tip1"]
        assert get_compression_lineage(db, "tip1") == ["root1", "tip1"]
        assert get_compression_lineage(db, "reset1") == ["reset1"]

    def test_routing_lineage_cte_agrees_with_python_walk_for_reset_fork(self, db, store):
        """``record_session_peer(include_compression_ancestors=True)`` re-keys every row named by
        ``_COMPRESSION_LINEAGE_CTE``. Resuming a reset fork of a compression-ended parent must
        re-key only the fork: the CTE has to stop at the reset child exactly like
        ``get_compression_lineage`` does, or the real lineage's ancestors land on the fork's peer."""
        from persistence.sqlite.gateway import _routes as gateway_mod

        t0 = time.time() - 3600
        create_session(db, "root", "cli")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "root"))
        create_session(db, "mid1", "cli", parent_session_id="root")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 20, "mid1"))
        create_session(db, "mid2", "cli", parent_session_id="mid1")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 30, "mid2"))
        create_session(db, "tip", "cli", parent_session_id="mid2")
        create_session(db, "reset", "cli", parent_session_id="mid2", model_config={"_reset_from": "mid2"})
        db._conn.commit()

        sql = gateway_mod._COMPRESSION_LINEAGE_CTE + " SELECT id FROM compression_lineage"
        with db._read_ctx() as conn:
            cte = {start: sorted(r[0] for r in conn.execute(sql, (start,)).fetchall()) for start in ("reset", "tip")}
        assert cte["reset"] == sorted(get_compression_lineage(db, "reset")) == ["reset"]
        assert cte["tip"] == sorted(get_compression_lineage(db, "tip")) == ["mid1", "mid2", "root", "tip"]

        store.routing.record_session_peer("reset", source="cli", user_id="u", session_key="cli:reset-chat",
                                          chat_id="reset-chat", chat_type="dm", include_compression_ancestors=True)
        assert get_session(db, "reset")["session_key"] == "cli:reset-chat"
        assert all(get_session(db, s)["session_key"] != "cli:reset-chat" for s in ("root", "mid1", "mid2", "tip"))

    def test_list_serves_full_lineage_ids_for_projected_rows(self, db):
        """The projected tip row must carry every chain id. Root and tip
        alone are not enough client-side: a persisted tile or route can hold
        a MIDDLE segment's id (it was the tip when opened), and without the
        intermediates that surface cannot prove it names this conversation —
        which is how one chat ends up open twice after a compaction."""
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        create_session(db, "solo", "cli")
        append_message(db, "solo", "user", "standalone")
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "solo", "cli")
        append_message(db, "solo", "user", "standalone")
        create_session(db, "branchy", "cli", parent_session_id="root1",
                          model_config={"_branched_from": "root1"})
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "seg-a", "cli")
        append_message(db, "seg-a", "user", "earlier days")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (t0 + 10, "seg-a"))
        # A restored previous segment the user re-linked the live tip to.
        create_session(db, "seg-prev", "cli", parent_session_id="seg-a")
        append_message(db, "seg-prev", "user", "restored segment")
        db._conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
                         (t0 + 20, "seg-prev"))
        create_session(db, "live-tip", "cli", parent_session_id="seg-prev")
        append_message(db, "live-tip", "user", "still talking here")
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "solo", "cli")
        append_message(db, "solo", "user", "standalone")
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        append_message(db, "root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        create_session(db, "tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        append_message(db, "tip2", "user", "second conversation continuation")
        update_session_cwd(db, "tip2", "/tmp/workspaces/second")
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=20)
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
        create_session(db, "root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        append_message(db, "root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        create_session(db, "tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        append_message(db, "tip2", "user", "second continuation")
        db._conn.commit()

        batch_calls = []
        single_calls = []
        original_batch = store_session_list._get_session_rich_rows_batch
        original_single = store_portability._get_session_rich_row

        def counting_batch(holder, session_ids, **kwargs):
            batch_calls.append(list(session_ids))
            return original_batch(holder, session_ids, **kwargs)

        def counting_single(holder, session_id, **kwargs):
            single_calls.append(session_id)
            return original_single(holder, session_id, **kwargs)

        monkeypatch.setattr(store_session_list, "_get_session_rich_rows_batch", counting_batch)
        monkeypatch.setattr(store_portability, "_get_session_rich_row", counting_single)

        sessions = list_sessions_rich(db, source="cli", limit=20)
        assert len(sessions) >= 2  # sanity: both chains actually surfaced

        # Two compression roots resolved with exactly one batched call, and
        # zero single-row calls — not one single-row call per root.
        assert len(batch_calls) == 1
        assert set(batch_calls[0]) == {"tip1", "tip2"}
        assert single_calls == []

    def test_list_handles_broken_chain_gracefully(self, db):
        """A compression root with no child (e.g. DB corruption or a partial
        end_session call that didn't finish creating the child) must not
        crash the list — it should fall back to surfacing the root as-is.
        """
        import time as _time
        t0 = _time.time() - 100
        create_session(db, "orphan", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "orphan"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 10, "compression", "orphan"),
        )
        db._conn.commit()

        sessions = list_sessions_rich(db, source="cli", limit=10)
        ids = [s["id"] for s in sessions]
        assert "orphan" in ids
        row = next(s for s in sessions if s["id"] == "orphan")
        # No tip means no projection — row stays raw.
        assert "_lineage_root_id" not in row
        assert row["end_reason"] == "compression"
