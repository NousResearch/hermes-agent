"""Regression tests for compression-aware session-search lineage (#129).

Reconstructs the accepted #68 compression-root semantics on the current
upstream seam: ``SessionDB.resolve_lineage_winners`` consumes the ranked
raw-hit set from ``search_messages`` and replaces generic-parent dedup with
positive compression-continuation roots — one query-local memo, one coherent
read snapshot, one work budget, fail-closed missing-parent/cycle outcomes,
early-K stop, and explicit bound-hit truncation.  The tool layer composes
exact-title / current-session exclusion through the same root meaning and
surfaces B exhaustion as an explicit truncation warning.
"""
import json
import threading
import time

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import _order_for_recall, session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _message(db, session_id, content, role="user"):
    return db.append_message(session_id, role=role, content=content)


def _create(db, session_id, source="cli", parent=None):
    db.create_session(session_id, source=source)
    if parent is not None:
        _set_parent(db, session_id, parent)


def _set_parent(db, session_id, parent):
    """Point *session_id* at *parent* (FK-safe for dangling parents)."""
    db._conn.execute("PRAGMA foreign_keys = OFF")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        (parent, session_id),
    )
    db._conn.commit()
    db._conn.execute("PRAGMA foreign_keys = ON")


def _set_marker(db, session_id, **markers):
    """Write ``_branched_from`` / ``_delegate_from`` into model_config."""
    row = db.get_session(session_id) or {}
    config = row.get("model_config")
    if isinstance(config, str):
        config = json.loads(config) if config else {}
    config = dict(config or {})
    config.update(markers)
    db._conn.execute(
        "UPDATE sessions SET model_config = ? WHERE id = ?",
        (json.dumps(config), session_id),
    )
    db._conn.commit()


def _chain_sessions(db, prefix, n, source="cli"):
    ids = [f"{prefix}-{i}" for i in range(n)]
    for sid in ids:
        db.create_session(sid, source=source)
    return ids


def _link_positive_chain(db, ids):
    """Link ``ids[i] -> ids[i+1]`` with positive compression edges.

    Each parent ends by ``'compression'``, so resolving ``ids[0]`` walks
    ``len(ids)`` successful uncached point lookups to the root ``ids[-1]``.
    Call AFTER appending messages (append_message refuses compression-ended
    sessions).
    """
    for i in range(len(ids) - 1):
        child, parent = ids[i], ids[i + 1]
        db._conn.execute("PRAGMA foreign_keys = OFF")
        db._conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
            (parent, child),
        )
        db._conn.execute(
            "UPDATE sessions SET end_reason = 'compression' WHERE id = ?",
            (parent,),
        )
        db._conn.commit()
        db._conn.execute("PRAGMA foreign_keys = ON")


def _candidate(db, session_id, role="user"):
    """A minimal ranked candidate row for *session_id*'s first message."""
    msgs = db.get_messages(session_id)
    return {
        "id": msgs[0]["id"],
        "session_id": session_id,
        "role": role,
        "snippet": "needle",
        "source": "cli",
        "model": None,
        "session_started": None,
    }


def _ranked(db, query, limit=300, sort=None):
    """Ranked raw-hit candidates from the real FTS lane."""
    return _order_for_recall(
        db.search_messages(
            query,
            role_filter=["user"],
            exclude_sources=["subagent", "tool"],
            limit=limit,
            sort=sort,
        )
    )


# ---------------------------------------------------------------------------
# Compression-continuation semantics
# ---------------------------------------------------------------------------


def test_lineage_winners_dedupe_by_compression_root_and_preserve_anchor(db):
    _create(db, "root", source="cli")
    root_id = _message(db, "root", "needle root")
    db.end_session("root", "compression")
    _create(db, "child", source="cli", parent="root")
    child_id = _message(db, "child", "needle child")
    _message(db, "child", "needle child second")
    _create(db, "other", source="cli")
    other_id = _message(db, "other", "needle other")

    result = db.resolve_lineage_winners(
        [
            _candidate(db, "root"),
            _candidate(db, "child"),
            _candidate(db, "other"),
        ],
        result_limit=2,
    )

    winners = result["winners"]
    assert len(winners) == 2
    assert {row["lineage_root_id"] for row in winners} == {"root", "other"}
    assert {row["session_id"] for row in winners} == {"root", "other"}
    # The compression child collapses into the root; the root's first hit is
    # the anchor (highest-ranked raw hit of the lineage).
    assert child_id not in {row["id"] for row in winners}
    assert root_id in {row["id"] for row in winners}
    assert other_id in {row["id"] for row in winners}
    assert result["stats"]["candidate_count"] == 3
    assert result["stats"]["candidate_unique_sessions"] == 3
    assert all("content" not in row for row in winners)
    assert all("context" not in row for row in winners)


def test_lineage_winners_generic_parentage_not_lineage(db):
    # A generic parent link (parent did NOT end by compression) keeps the
    # child a distinct root: parentage alone never collapses conversations.
    _create(db, "parent", source="cli")
    _message(db, "parent", "needle generic")
    _create(db, "child", source="cli", parent="parent")
    _message(db, "child", "needle generic")

    result = db.resolve_lineage_winners(
        [_candidate(db, "parent"), _candidate(db, "child")],
        result_limit=10,
    )

    assert {row["lineage_root_id"] for row in result["winners"]} == {
        "parent",
        "child",
    }


def test_lineage_winners_branch_marker_to_parent_stays_distinct(db):
    _create(db, "broot", source="cli")
    _message(db, "broot", "needle branch")
    db.end_session("broot", "compression")
    _create(db, "bchild", source="cli", parent="broot")
    _message(db, "bchild", "needle branch")
    _set_marker(db, "bchild", _branched_from="broot")

    result = db.resolve_lineage_winners(
        [_candidate(db, "bchild")], result_limit=10
    )

    assert result["winners"][0]["lineage_root_id"] == "bchild"


def test_lineage_winners_delegate_marker_to_parent_stays_distinct(db):
    _create(db, "droot", source="cli")
    _message(db, "droot", "needle delegate")
    db.end_session("droot", "compression")
    _create(db, "dchild", source="cli", parent="droot")
    _message(db, "dchild", "needle delegate")
    _set_marker(db, "dchild", _delegate_from="droot")

    result = db.resolve_lineage_winners(
        [_candidate(db, "dchild")], result_limit=10
    )

    assert result["winners"][0]["lineage_root_id"] == "dchild"


def test_lineage_winners_tool_child_stays_distinct(db):
    _create(db, "troot", source="cli")
    _message(db, "troot", "needle tool")
    db.end_session("troot", "compression")
    _create(db, "tchild", source="tool", parent="troot")
    _message(db, "tchild", "needle tool")

    result = db.resolve_lineage_winners(
        [_candidate(db, "tchild")], result_limit=10
    )

    assert result["winners"][0]["lineage_root_id"] == "tchild"


def test_lineage_winners_foreign_marker_does_not_block_continuation(db):
    # A stale/foreign branch or delegate marker pointing somewhere OTHER than
    # the parent must not disqualify a legitimate compression continuation.
    _create(db, "froot", source="cli")
    _message(db, "froot", "needle foreign")
    db.end_session("froot", "compression")
    _create(db, "fchild", source="cli", parent="froot")
    _message(db, "fchild", "needle foreign")
    _set_marker(db, "fchild", _branched_from="elsewhere")

    result = db.resolve_lineage_winners(
        [_candidate(db, "fchild")], result_limit=10
    )

    assert result["winners"][0]["lineage_root_id"] == "froot"


def test_lineage_winners_missing_parent_fails_closed_and_memo_reuses(db):
    _create(db, "m-bad", source="cli")
    _message(db, "m-bad", "needle missing")
    db.end_session("m-bad", "compression")
    _set_parent(db, "m-bad", "m-missing")
    _create(db, "m-bad-child", source="cli", parent="m-bad")
    _message(db, "m-bad-child", "needle missing")
    _create(db, "m-ok", source="cli")
    _message(db, "m-ok", "needle missing")

    result = db.resolve_lineage_winners(
        [
            _candidate(db, "m-bad"),
            _candidate(db, "m-bad-child"),
            _candidate(db, "m-ok"),
        ],
        result_limit=10,
    )

    assert [row["session_id"] for row in result["winners"]] == ["m-ok"]
    # m-bad resolves with one lookup; m-bad-child walks one lookup then hits
    # m-bad's proven-unresolved memo at zero additional work.
    assert result["stats"]["lineage_work"] == 3
    assert result["stats"]["lineage_memo_hits"] >= 1


def test_lineage_winners_two_node_cycle_fails_closed(db):
    _create(db, "c-a", source="cli")
    _create(db, "c-b", source="cli")
    _message(db, "c-a", "needle cycle")
    _message(db, "c-b", "needle cycle")
    db.end_session("c-a", "compression")
    db.end_session("c-b", "compression")
    _set_parent(db, "c-a", "c-b")
    _set_parent(db, "c-b", "c-a")

    result = db.resolve_lineage_winners(
        [_candidate(db, "c-a"), _candidate(db, "c-b")], result_limit=10
    )

    assert result["winners"] == []


def test_lineage_winners_long_cycle_fails_closed(db):
    for sid in ("l-a", "l-b", "l-c"):
        _create(db, sid, source="cli")
        _message(db, sid, "needle cycle")
        db.end_session(sid, "compression")
    _set_parent(db, "l-a", "l-b")
    _set_parent(db, "l-b", "l-c")
    _set_parent(db, "l-c", "l-a")

    result = db.resolve_lineage_winners(
        [_candidate(db, "l-a"), _candidate(db, "l-b"), _candidate(db, "l-c")],
        result_limit=10,
    )

    assert result["winners"] == []


def test_lineage_winners_tail_entering_cycle_fails_closed(db):
    _create(db, "t-a", source="cli")
    _create(db, "t-b", source="cli")
    _create(db, "t-c", source="cli")
    _message(db, "t-a", "needle cycle")
    _message(db, "t-b", "needle cycle")
    _message(db, "t-c", "needle cycle")
    db.end_session("t-b", "compression")
    db.end_session("t-c", "compression")
    _set_parent(db, "t-a", "t-b")
    _set_parent(db, "t-b", "t-c")
    _set_parent(db, "t-c", "t-b")

    result = db.resolve_lineage_winners(
        [_candidate(db, "t-a"), _candidate(db, "t-b")], result_limit=10
    )

    assert result["winners"] == []


def test_lineage_winners_positive_lineage_memo_reuse(db):
    # A 15-session positive lineage (observed depth-14/size-15 tail): every
    # candidate in the lineage resolves to the same root, and after the
    # first candidate all later ones are memo hits (zero extra lookups).
    ids = _chain_sessions(db, "memo", 15)
    for sid in ids:
        _message(db, sid, "needle memo")
    _link_positive_chain(db, ids)

    result = db.resolve_lineage_winners(
        [_candidate(db, sid) for sid in ids], result_limit=10
    )

    winners = result["winners"]
    assert len(winners) == 1
    assert winners[0]["lineage_root_id"] == ids[-1]
    assert result["stats"]["lineage_work"] == 15
    assert result["stats"]["lineage_memo_hits"] == 14
    assert result["stats"]["lineage_memo_entries"] == 15


def test_lineage_winners_owner_dedupe_preserves_best_anchor(db):
    # Repeated raw hits from the same owner do not create repeated resolver
    # candidates; the highest-ranked (first) anchor survives.
    _create(db, "o", source="cli")
    first = _message(db, "o", "needle first")
    _message(db, "o", "needle second")
    _create(db, "p", source="cli")
    _message(db, "p", "needle other")

    result = db.resolve_lineage_winners(
        [
            _candidate(db, "o"),
            _candidate(db, "o"),
            _candidate(db, "p"),
        ],
        result_limit=10,
    )

    assert len(result["winners"]) == 2
    anchors = [row["id"] for row in result["winners"] if row["session_id"] == "o"]
    assert anchors == [first]


def test_lineage_winners_early_k_stops(db):
    _create(db, "k0", source="cli")
    db.end_session("k0", "compression")
    for i in range(12):
        _create(db, f"k0-child-{i}", source="cli", parent="k0")
        _message(db, f"k0-child-{i}", "needle k")
    for i in range(1, 15):
        _create(db, f"k{i}", source="cli")
        _message(db, f"k{i}", "needle k")

    candidates = [
        _candidate(db, f"k0-child-{i}") for i in range(12)
    ] + [_candidate(db, f"k{i}") for i in range(1, 15)]
    for k in (1, 3, 10):
        result = db.resolve_lineage_winners(candidates, result_limit=k)
        assert len(result["winners"]) == k
        assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_candidates_inspected_counts_distinct_owners(db):
    _create(db, "m1", source="cli")
    _message(db, "m1", "needle first")
    _message(db, "m1", "needle second")
    _create(db, "m2", source="cli")
    _message(db, "m2", "needle other")

    result = db.resolve_lineage_winners(
        [_candidate(db, "m1"), _candidate(db, "m1"), _candidate(db, "m2")],
        result_limit=5,
    )

    assert result["stats"]["candidate_count"] == 3  # raw hits
    assert result["stats"]["candidate_unique_sessions"] == 2  # distinct owners
    assert result["stats"]["lineage_candidates_inspected"] == 2  # not 3
    assert len(result["winners"]) == 2


# ---------------------------------------------------------------------------
# Work budget B=2000: exact accounting, fail-closed, bound-hit
# ---------------------------------------------------------------------------


def test_lineage_winners_cycle_at_work_bound_is_cycle_not_bound_hit(db):
    # A cycle provable from the traversal-local seen-set at work == B is
    # classified as cycle, NOT bound-hit (no further DB lookup needed).
    for sid in ("cb-a", "cb-b", "cb-c"):
        _create(db, sid, source="cli")
        _message(db, sid, "needle cycle")
        db.end_session(sid, "compression")
    _set_parent(db, "cb-a", "cb-b")
    _set_parent(db, "cb-b", "cb-c")
    _set_parent(db, "cb-c", "cb-a")

    result = db.resolve_lineage_winners(
        [_candidate(db, "cb-a")], result_limit=10, work_budget=3
    )

    assert result["winners"] == []
    assert result["stats"]["lineage_bound_hit"] is False
    assert result["stats"]["lineage_work"] == 3


def test_lineage_winners_bound_hit_before_cycle_proof(db):
    # Proving the cycle would need one MORE uncached lookup after the budget
    # is exhausted -> bound-hit, and the partial path must not be memoized as
    # unresolved (a fresh query resolves it).
    ids = _chain_sessions(db, "bp", 5)
    for sid in ids:
        _message(db, sid, "needle cycle")
    _link_positive_chain(db, ids)
    # Close the 5-node chain into a cycle: bp-4 -> bp-0 (positive edge).
    db._conn.execute(
        "UPDATE sessions SET end_reason = 'compression' WHERE id = ?",
        (ids[0],),
    )
    db._conn.commit()
    _set_parent(db, ids[-1], ids[0])

    result = db.resolve_lineage_winners(
        [_candidate(db, ids[0])], result_limit=10, work_budget=4
    )

    assert result["winners"] == []
    assert result["stats"]["lineage_bound_hit"] is True
    assert result["stats"]["lineage_work"] == 4


def test_lineage_winners_root_resolved_exactly_at_work_bound(db):
    # A root resolved on successful lookup number B succeeds (the boundary is
    # inclusive of legitimate completed work).
    ids = _chain_sessions(db, "b5", 5)
    _message(db, ids[0], "needle exact")
    _link_positive_chain(db, ids)

    result = db.resolve_lineage_winners(
        [_candidate(db, ids[0])], result_limit=10, work_budget=5
    )

    assert result["winners"][0]["lineage_root_id"] == ids[-1]
    assert result["stats"]["lineage_work"] == 5
    assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_lookup_beyond_work_bound_truncates(db):
    # A resolution that requires lookup B+1 stops BEFORE that lookup, flags
    # bound-hit, and returns only already-proven safe winners.  The bound is
    # reached on the first (higher-ranked) candidate, so the scan stops
    # rather than accepting the later safe candidate (ranking preserved).
    ids = _chain_sessions(db, "b6", 6)
    _message(db, ids[0], "needle trunc")
    _link_positive_chain(db, ids)
    _create(db, "trunc-safe", source="cli")
    _message(db, "trunc-safe", "needle trunc")

    result = db.resolve_lineage_winners(
        [_candidate(db, ids[0]), _candidate(db, "trunc-safe")],
        result_limit=10,
        work_budget=5,
    )

    assert result["stats"]["lineage_bound_hit"] is True
    assert result["winners"] == []


def test_lineage_winners_bound_exhaustion_does_not_poison_memo(db):
    ids = _chain_sessions(db, "poison", 5)
    _message(db, ids[0], "needle poison")
    _link_positive_chain(db, ids)

    first = db.resolve_lineage_winners(
        [_candidate(db, ids[0])], result_limit=10, work_budget=3
    )
    assert first["stats"]["lineage_bound_hit"] is True

    # A fresh query (new memo + restored budget) resolves the same chain
    # normally, proving the exhausted partial path was never memoized.
    second = db.resolve_lineage_winners(
        [_candidate(db, ids[0])], result_limit=10, work_budget=2000
    )
    assert second["stats"]["lineage_bound_hit"] is False
    assert second["winners"][0]["lineage_root_id"] == ids[-1]


def test_lineage_winners_stats_report_work_and_bound(db):
    _create(db, "st-root", source="cli")
    _message(db, "st-root", "needle stats")
    db.end_session("st-root", "compression")
    _create(db, "st-child", source="cli", parent="st-root")
    _message(db, "st-child", "needle stats")
    _create(db, "st-other", source="cli")
    _message(db, "st-other", "needle stats")

    result = db.resolve_lineage_winners(
        [_candidate(db, "st-root"), _candidate(db, "st-child"), _candidate(db, "st-other")],
        result_limit=5,
    )

    stats = result["stats"]
    assert stats["candidate_count"] == 3
    assert stats["candidate_unique_sessions"] == 3
    assert stats["lineage_count"] == 2
    assert stats["winner_count"] == 2
    assert stats["lineage_work"] >= 1
    assert stats["lineage_candidates_inspected"] == 3
    assert stats["lineage_bound_hit"] is False


def test_lineage_winners_match_resolve_compression_lineage_oracle(db):
    # The real FTS lane (search_messages) + the same compression-lineage
    # implementation used by the winner seam produce identical winners: this
    # validates the candidate ranking / owner dedupe / early-K integration,
    # not a second, generic-parent definition of lineage.
    _create(db, "oracle-root", source="telegram")
    _message(db, "oracle-root", "oracle needle root")
    db.end_session("oracle-root", "compression")
    _create(db, "oracle-child", source="cron", parent="oracle-root")
    _create(db, "oracle-other", source="cli")
    _create(db, "oracle-cron", source="cron")
    _message(db, "oracle-child", "oracle needle child")
    _message(db, "oracle-other", "oracle needle other")
    _message(db, "oracle-cron", "oracle needle cron")

    for sort in (None, "newest", "oldest"):
        raw = db.search_messages(
            "oracle",
            role_filter=["user"],
            exclude_sources=["subagent", "tool"],
            limit=300,
            sort=sort,
        )
        expected = []
        seen = set()
        for hit in _order_for_recall(raw):
            root = db.resolve_compression_lineage(hit["session_id"])
            if root is None:
                continue
            if root in seen:
                continue
            seen.add(root)
            expected.append(
                (
                    hit["session_id"],
                    root,
                    hit["id"],
                    hit["role"],
                    hit["source"],
                )
            )
            if len(expected) == 3:
                break

        actual = db.resolve_lineage_winners(
            _order_for_recall(raw), result_limit=3
        )["winners"]
        actual = [
            (
                row["session_id"],
                row["lineage_root_id"],
                row["id"],
                row["role"],
                row["source"],
            )
            for row in actual
        ]
        assert actual == expected, f"sort={sort!r}"


def test_lineage_winners_lineage_lookups_in_one_read_transaction(
    db, monkeypatch
):
    # All lineage point lookups for one logical search run inside one explicit
    # read transaction (one coherent logical snapshot).
    original = SessionDB._resolve_compression_lineage_on_conn
    observed = []

    def spy(self, conn, session_id, state):
        observed.append(conn.in_transaction)
        return original(self, conn, session_id, state)

    monkeypatch.setattr(SessionDB, "_resolve_compression_lineage_on_conn", spy)
    _create(db, "root", source="cli")
    _message(db, "root", "snap needle")
    db.end_session("root", "compression")
    _create(db, "child", source="cli", parent="root")
    _message(db, "child", "snap needle")

    db.resolve_lineage_winners(
        [_candidate(db, "root"), _candidate(db, "child")], result_limit=5
    )

    assert observed
    assert all(observed)


# ---------------------------------------------------------------------------
# Exclusion parity: current session + exact title share compression-root
# ---------------------------------------------------------------------------


def test_lineage_winners_current_exclusion_uses_same_root_semantics(db):
    # The live continuation child is current-excluded; its compression-ended
    # parent's archived content surfaces; a generic child of the current root
    # keeps its own root and is NOT swallowed by current exclusion.
    _create(db, "s_parent", source="cli")
    _message(db, "s_parent", "filter needle")
    db.end_session("s_parent", "compression")
    _create(db, "s_current", source="cli", parent="s_parent")
    _message(db, "s_current", "filter needle")
    _create(db, "gen-child", source="cli", parent="s_current")
    _message(db, "gen-child", "filter needle")
    _create(db, "kept", source="cli")
    _message(db, "kept", "filter needle")

    result = db.resolve_lineage_winners(
        [
            _candidate(db, "s_current"),
            _candidate(db, "s_parent"),
            _candidate(db, "gen-child"),
            _candidate(db, "kept"),
        ],
        result_limit=10,
        current_session_id="s_current",
    )

    by_root = {row["lineage_root_id"] for row in result["winners"]}
    # s_parent is in the current compression lineage but its content is
    # archived (compression-ended) -> discoverable; gen-child is a distinct
    # root; the current session itself never surfaces.
    assert by_root == {"s_parent", "gen-child", "kept"}
    assert all(row["session_id"] != "s_current" for row in result["winners"])


def test_lineage_winners_fresh_reset_predecessor_stays_discoverable(db):
    # A /new-style predecessor (fresh-reset end reason) of the current session
    # is NOT in the current compression lineage, but it IS a live-context
    # generic ancestor.  Its content left live context (fresh reset), so it
    # must stay discoverable (#85756) — the base's reset boundary is
    # preserved on top of compression-root semantics.
    _create(db, "predecessor", source="cli")
    _message(db, "predecessor", "reset needle")
    db.end_session("predecessor", "idle")
    _create(db, "current", source="cli", parent="predecessor")
    _message(db, "current", "reset needle")

    result = db.resolve_lineage_winners(
        [_candidate(db, "predecessor"), _candidate(db, "current")],
        result_limit=10,
        current_session_id="current",
    )

    roots = [row["lineage_root_id"] for row in result["winners"]]
    assert "predecessor" in roots
    assert "current" not in roots


def test_lineage_winners_exact_title_exclusion_uses_same_root_semantics(db):
    # Exact-title exclusion arrives as the title's resolved root in
    # excluded_lineage_roots: the whole compression lineage is fully excluded
    # from content winners.
    _create(db, "troot", source="cli")
    _message(db, "troot", "needle title root")
    db.end_session("troot", "compression")
    _create(db, "tchild", source="cli", parent="troot")
    _message(db, "tchild", "needle title child")
    _create(db, "other", source="cli")
    _message(db, "other", "needle title other")

    result = db.resolve_lineage_winners(
        [_candidate(db, "troot"), _candidate(db, "tchild"), _candidate(db, "other")],
        result_limit=5,
        excluded_lineage_roots=("troot",),
    )

    assert [row["session_id"] for row in result["winners"]] == ["other"]
    assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_compacted_history_anchor_fallback(db):
    # In the current lineage, the newest hit is live (current-excluded), but
    # the older compacted hit of the same owner must still surface AS THE
    # ANCHOR — per-owner dedupe must never erase compacted history.
    db.create_session("cur", source="cli")
    compacted_id = _message(db, "cur", "old compacted needle")
    db.archive_and_compact("cur", [{"role": "assistant", "content": "summary"}])
    _message(db, "cur", "new live needle")

    result = db.resolve_lineage_winners(
        [
            _candidate(db, "cur"),
            {"id": compacted_id, "session_id": "cur", "role": "user",
             "snippet": "needle", "source": "cli", "model": None,
             "session_started": None},
        ],
        result_limit=5,
        current_session_id="cur",
    )

    winners = result["winners"]
    assert len(winners) == 1
    assert winners[0]["session_id"] == "cur"
    assert winners[0]["id"] == compacted_id


def test_lineage_winners_displayable_anchor_ordering_not_live_rank(db):
    # Winner ordering follows the rank of each owner's FIRST DISPLAYABLE
    # anchor.  With K=1 and 'cur-live(#1) -> other(#2) -> cur-compacted(#3)',
    # the higher-ranked displayable 'other' must win — cur's live #1 hit is
    # current-excluded and must NOT let cur's later compacted anchor jump
    # ahead of a genuinely higher-ranked winner.
    import time

    db.create_session("cur", source="cli")
    compacted_id = _message(db, "cur", "old compacted needle")
    db.archive_and_compact("cur", [{"role": "assistant", "content": "summary"}])
    live_id = _message(db, "cur", "new live needle")
    _create(db, "other-s", source="cli")
    other_id = _message(db, "other-s", "needle other")
    now = time.time()
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE id = ?", (now - 100, live_id)
    )
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE id = ?", (now - 200, other_id)
    )
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE id = ?", (now - 300, compacted_id)
    )
    db._conn.commit()

    candidates = _ranked(db, "needle", sort="newest")
    result = db.resolve_lineage_winners(
        candidates, result_limit=1, current_session_id="cur"
    )

    winners = result["winners"]
    assert len(winners) == 1
    assert winners[0]["session_id"] == "other-s"
    assert winners[0]["id"] == other_id


# ---------------------------------------------------------------------------
# Tool layer: dedupe composition, exact-title slot, truncation signal
# ---------------------------------------------------------------------------


def _bulk_positive_chain(db, prefix, n, source="cli"):
    """Bulk-build a positive compression chain of *n* sessions, fast.

    SessionDB's writer connection is autocommit, so per-row inserts would
    fsync every row (~40x slower).  One explicit BEGIN/COMMIT keeps the
    fixture fast enough for the real B=2000-boundary and 10k-chain tests.
    """
    base = prefix
    now = time.time()
    db._conn.execute("BEGIN")
    db._conn.executemany(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        [(f"{base}-{i}", source, now) for i in range(n)],
    )
    for i in range(n - 1):
        child, parent = f"{base}-{i}", f"{base}-{i + 1}"
        db._conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
            (parent, child),
        )
        db._conn.execute(
            "UPDATE sessions SET end_reason = 'compression' WHERE id = ?",
            (parent,),
        )
    db._conn.execute("COMMIT")
    return [f"{base}-{i}" for i in range(n)]


def _seed_bound_chain(db, n=6):
    """A positive compression chain longer than the monkeypatched budget."""
    ids = _chain_sessions(db, "bhit", n)
    _link_positive_chain(db, ids)
    db.append_message("bhit-0", role="user", content="bound needle chain")


def test_discover_bound_hit_returns_truncated_safe_prefix(db, monkeypatch):
    import hermes_state_search

    monkeypatch.setattr(hermes_state_search, "_LINEAGE_WORK_BUDGET", 5)
    db.create_session("safe", source="cli")
    db.append_message("safe", role="user", content="bound needle safe")
    _seed_bound_chain(db)

    result = json.loads(session_search(query="bound needle", db=db))

    assert result["success"] is True
    assert result.get("truncated") is True
    assert "warning" in result
    assert result["count"] == 1
    assert [r["session_id"] for r in result["results"]] == ["safe"]


def test_discover_bound_hit_with_zero_winners_is_incomplete_not_no_match(
    db, monkeypatch
):
    import hermes_state_search

    monkeypatch.setattr(hermes_state_search, "_LINEAGE_WORK_BUDGET", 5)
    _seed_bound_chain(db)

    result = json.loads(session_search(query="bound needle", db=db))

    assert result["success"] is True
    assert result.get("truncated") is True
    assert result["count"] == 0
    assert "No matching sessions found" not in (result.get("message") or "")


def test_discover_title_only_limit_one_surfaces_bound_exhaustion(
    db, monkeypatch
):
    import hermes_state_search

    monkeypatch.setattr(hermes_state_search, "_LINEAGE_WORK_BUDGET", 4)
    # current session is the tip of a compression chain deeper than B
    for i in range(6):
        db.create_session(f"cur-{i}", source="cli")
    for i in range(5):
        child, parent = f"cur-{i}", f"cur-{i + 1}"
        db._conn.execute("PRAGMA foreign_keys = OFF")
        db._conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
            (parent, child),
        )
        db._conn.execute(
            "UPDATE sessions SET end_reason = 'compression' WHERE id = ?",
            (parent,),
        )
        db._conn.commit()
        db._conn.execute("PRAGMA foreign_keys = ON")
    db.create_session("title-session", source="cli")
    db.set_session_title("title-session", "needle-title")
    db.append_message("title-session", role="user", content="title content filler")

    result = json.loads(
        session_search(
            query="needle-title", db=db, limit=1, current_session_id="cur-0"
        )
    )

    assert result["success"] is True
    assert result["count"] == 1
    assert [r["session_id"] for r in result["results"]] == ["title-session"]
    assert result.get("truncated") is True
    assert "warning" in result


def test_discover_compression_dedupe_and_exact_title_slot(db):
    # Exact-title result occupies its slot; its compression-lineage members
    # are excluded from the content lane.
    _create(db, "t_parent", source="cli")
    _message(db, "t_parent", "title content")
    db.end_session("t_parent", "compression")
    _create(db, "t_child", source="cli", parent="t_parent")
    _message(db, "t_child", "title content")
    db.set_session_title("t_parent", "needle-title")

    result = json.loads(
        session_search(query="needle-title", db=db, limit=1)
    )

    assert result["success"] is True
    sids = [r["session_id"] for r in result["results"]]
    # Title slot: t_parent. Content lane: t_child is the same compression
    # lineage -> excluded, so the title is the only result.
    assert sids == ["t_parent"]
    assert result.get("truncated") is False


def test_lineage_winners_default_budget_1999_succeeds(db):
    # A root resolved on successful lookup 1999 succeeds with the default B.
    ids = _bulk_positive_chain(db, "b1999", 1999)
    _message(db, ids[0], "needle 1999")

    result = db.resolve_lineage_winners([_candidate(db, ids[0])], result_limit=5)
    assert result["winners"][0]["lineage_root_id"] == ids[-1]
    assert result["stats"]["lineage_work"] == 1999
    assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_default_budget_2000_succeeds(db):
    # A root resolved on successful lookup 2000 (the boundary) succeeds with
    # the default B.
    ids = _bulk_positive_chain(db, "b2000", 2000)
    _message(db, ids[0], "needle 2000")

    result = db.resolve_lineage_winners([_candidate(db, ids[0])], result_limit=5)
    assert result["winners"][0]["lineage_root_id"] == ids[-1]
    assert result["stats"]["lineage_work"] == 2000
    assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_default_budget_2001_truncates(db):
    # A resolution that would need lookup 2001 stops BEFORE it with the
    # default B, flags bound-hit, and returns no fabricated winner.
    ids = _bulk_positive_chain(db, "b2001", 2001)
    _message(db, ids[0], "needle 2001")

    result = db.resolve_lineage_winners([_candidate(db, ids[0])], result_limit=5)
    assert result["stats"]["lineage_bound_hit"] is True
    assert result["winners"] == []


def test_lineage_winners_10k_chain_cut_by_default_budget(db):
    # A real 10k-node acyclic chain is cut by the default B=2000 work budget,
    # not by any semantic depth cap.
    ids = _bulk_positive_chain(db, "path10k", 10000)
    _message(db, ids[0], "needle path")

    result = db.resolve_lineage_winners([_candidate(db, ids[0])], result_limit=5)
    assert result["stats"]["lineage_bound_hit"] is True
    assert result["stats"]["lineage_work"] == 2000
    assert result["winners"] == []


def test_lineage_winners_internal_k_stress(db):
    # Larger internal K (beyond the public K<=10 tool path) is a defensive
    # DB contract, not the normal product path.
    for i in range(60):
        _create(db, f"ik-{i}", source="cli")
        _message(db, f"ik-{i}", "needle internal k")

    candidates = [_candidate(db, f"ik-{i}") for i in range(60)]
    result = db.resolve_lineage_winners(candidates, result_limit=60)
    assert len(result["winners"]) == 60
    assert result["stats"]["lineage_bound_hit"] is False


def test_lineage_winners_lineage_reads_coherent_under_concurrent_writer(
    db, monkeypatch,
):
    # A writer hammering session creation during a winner search cannot tear
    # the resolver's lineage snapshot: lineage lookups observe one coherent
    # root set from one logical moment.
    original = SessionDB._resolve_compression_lineage_on_conn

    def slow(self, conn, session_id, state):
        time.sleep(0.02)
        return original(self, conn, session_id, state)

    monkeypatch.setattr(SessionDB, "_resolve_compression_lineage_on_conn", slow)
    _create(db, "root", source="cli")
    _message(db, "root", "snap needle")
    db.end_session("root", "compression")
    _create(db, "child", source="cli", parent="root")
    _message(db, "child", "snap needle")
    _create(db, "other", source="cli")
    _message(db, "other", "snap needle")

    candidates = [
        _candidate(db, "root"),
        _candidate(db, "child"),
        _candidate(db, "other"),
    ]

    def writer():
        for i in range(10):
            db.create_session(f"cw-{i}", source="cli")

    t = threading.Thread(target=writer)
    t.start()
    result = db.resolve_lineage_winners(candidates, result_limit=5)
    t.join()

    roots = {row["lineage_root_id"] for row in result["winners"]}
    assert roots == {"root", "other"}
    assert result["stats"]["lineage_bound_hit"] is False
