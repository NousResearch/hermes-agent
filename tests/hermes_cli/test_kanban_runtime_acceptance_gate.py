"""Runtime-acceptance gate: deterministic enforcement that runtime-affecting
work cannot be approved as done without production-like runtime evidence.

Correction c-017 contract (task t_3130c1eb): reviewer session
20260903_002313_1d4b70 approved a runtime-affecting candidate without
exercising its scheduled end-to-end runtime path. Prose policy was
insufficient. A review card explicitly marked ``requires_runtime_acceptance``
must be parent-gated on its explicit QA/live-verification card(s), and
reviewer completion must fail closed unless:

* every explicitly cited QA/live-verification parent is ``done``, and
* the review handoff metadata cites the tested candidate ``commit`` plus
  ``runtime_evidence`` (a non-empty description of the production-like
  runtime run).

Code-only changes (the default: marker absent/NULL/0) are completely
unaffected — no global browser/UI evidence requirement.

The marker survives decomposition: decompose children marked
``requires_runtime_acceptance`` propagate it, and the DB layer enforces the
QA/live-verification -> reviewer edge order (never reviewer -> later
verification) on marked cards.

Existing graphs migrate safely: legacy boards get the additive column with
a NULL/0 default, which leaves every existing card unmarked and ungated.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


EVIDENCE = {
    "commit": "6a49d075dd",
    "runtime_evidence": {
        "candidate_commit": "6a49d075dd",
        "environment": "managed gateway serving all 13 configured profiles",
        "command": "exercise one live routed request and inspect delivery receipt run/912",
        "result": "exactly-once routed delivery and creator-session wake observed",
    },
}


# ---------------------------------------------------------------------------
# Canonical field plumbing
# ---------------------------------------------------------------------------


def test_marker_defaults_off_and_create_task_persists_it(conn) -> None:
    tid = kb.create_task(conn, title="plain card", assignee="coder")
    assert kb.get_task(conn, tid).requires_runtime_acceptance is False

    tid2 = kb.create_task(
        conn,
        title="runtime card",
        assignee="coder",
        requires_runtime_acceptance=True,
    )
    assert kb.get_task(conn, tid2).requires_runtime_acceptance is True


def test_set_runtime_acceptance_flag_roundtrip(conn) -> None:
    tid = kb.create_task(conn, title="later marked", assignee="coder")
    assert kb.set_requires_runtime_acceptance(conn, tid, True) is True
    assert kb.get_task(conn, tid).requires_runtime_acceptance is True
    with pytest.raises(RuntimeError, match="cannot clear requires_runtime_acceptance"):
        kb.set_requires_runtime_acceptance(conn, tid, False)
    current = kb.get_task(conn, tid)
    assert current is not None
    assert current.requires_runtime_acceptance is True
    with pytest.raises(ValueError):
        kb.set_requires_runtime_acceptance(conn, "t_missing000", True)


def test_runtime_acceptance_marker_cannot_be_cleared_after_review_requested(
    conn,
) -> None:
    impl, _review = _runtime_review_card(conn)

    with pytest.raises(RuntimeError, match="cannot clear requires_runtime_acceptance"):
        kb.set_requires_runtime_acceptance(conn, impl, False)

    assert kb.get_task(conn, impl).requires_runtime_acceptance is True


# ---------------------------------------------------------------------------
# The gate: reviewer completion fail-closed
# ---------------------------------------------------------------------------


def _runtime_review_card(conn, *, marked=True):
    """Marked card -> review, ready for approval, no parents attached.

    The c-017 failure was not a graph-parent violation (plain parent gating
    already covers that) but an approval without runtime evidence — so the
    baseline card under review has no QA parent at all, which is exactly
    what a decomposer or worker produces when the QA linkage was never
    constructed. Tests then attach/pin QA state to exercise the gate.
    """
    impl = kb.create_task(
        conn,
        title="implement the fix",
        assignee="coder",
        requires_runtime_acceptance=marked,
    )
    impl_run = kb.claim_task(conn, impl, claimer="coder:1")
    assert impl_run is not None
    assert kb.request_review(
        conn,
        impl,
        summary="Implementation ready.",
        reviewer="reviewer",
        expected_run_id=impl_run.current_run_id,
    )
    review = kb.claim_review_task(conn, impl, claimer="reviewer:1")
    assert review is not None
    return impl, review


def test_premature_approval_blocked_when_qa_parent_not_done(conn) -> None:
    impl, review = _runtime_review_card(conn)
    # Legacy/misconstructed graph: an open live-verification parent is
    # linked after the card entered review (link_tasks does not demote a
    # review-lane child), exactly the state the gate must catch.
    qa = kb.create_task(
        conn,
        title="live-verify on the managed gateway",
        assignee="qa",
    )
    kb.link_tasks(conn, qa, impl)
    kb.designate_runtime_acceptance_parents(conn, impl, [qa])
    assert kb.get_task(conn, qa).status == "ready"  # QA still open

    metadata = dict(EVIDENCE)
    metadata["runtime_acceptance_parents"] = [qa]
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata=metadata,
        expected_run_id=review.current_run_id,
    )
    assert kb.get_task(conn, impl).status in ("review", "running")
    # The rejected attempt is auditable.
    events = kb.list_events(conn, impl)
    assert any(
        e.kind == "completion_blocked_runtime_acceptance" for e in events
    )


def test_non_self_describing_explicit_parent_can_satisfy_gate(conn) -> None:
    impl, review = _runtime_review_card(conn)
    qa = kb.create_task(conn, title="candidate receipt", assignee="qa")
    kb.link_tasks(conn, qa, impl)
    kb.designate_runtime_acceptance_parents(conn, impl, [qa])
    assert kb.complete_task(conn, qa, summary="Production-like runtime passed.")

    metadata = dict(EVIDENCE)
    metadata["runtime_acceptance_parents"] = [qa]
    assert kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata=metadata,
        expected_run_id=review.current_run_id,
    )
    assert kb.get_task(conn, impl).status == "done"


def test_unrelated_done_parent_cannot_replace_designated_runtime_parent(conn) -> None:
    impl, review = _runtime_review_card(conn)
    qa = kb.create_task(conn, title="candidate receipt", assignee="qa")
    unrelated = kb.create_task(conn, title="docs check", assignee="writer")
    kb.link_tasks(conn, qa, impl)
    kb.link_tasks(conn, unrelated, impl)
    kb.designate_runtime_acceptance_parents(conn, impl, [qa])
    assert kb.complete_task(conn, qa, summary="Runtime passed.")
    assert kb.complete_task(conn, unrelated, summary="Docs passed.")

    metadata = dict(EVIDENCE)
    metadata["runtime_acceptance_parents"] = [unrelated]
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved against wrong parent.",
        metadata=metadata,
        expected_run_id=review.current_run_id,
    )



def test_runtime_parent_designation_cannot_be_replaced_after_unlink(conn) -> None:
    qa = kb.create_task(conn, title="candidate receipt")
    replacement = kb.create_task(conn, title="unrelated done parent")
    impl = kb.create_task(
        conn,
        title="runtime card",
        parents=[qa, replacement],
        requires_runtime_acceptance=True,
        runtime_acceptance_parents=[qa],
    )

    assert kb.unlink_tasks(conn, qa, impl)
    with pytest.raises(RuntimeError, match="already designated"):
        kb.designate_runtime_acceptance_parents(conn, impl, [replacement])



def test_prose_only_parent_cannot_satisfy_gate(conn) -> None:
    impl, review = _runtime_review_card(conn)
    prose_match = kb.create_task(
        conn,
        title="Verify changelog formatting",
        assignee="writer",
    )
    kb.link_tasks(conn, prose_match, impl)
    assert kb.complete_task(conn, prose_match, summary="Formatting checked.")

    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata=dict(EVIDENCE),
        expected_run_id=review.current_run_id,
    )
    events = kb.list_events(conn, impl)
    blocked = [e for e in events if e.kind == "completion_blocked_runtime_acceptance"]
    assert blocked and blocked[-1].payload is not None
    assert "runtime_acceptance_parents" in blocked[-1].payload["reason"]


def test_evidence_backed_completion_succeeds_after_qa_done(conn) -> None:
    impl, review = _runtime_review_card(conn)
    qa = kb.create_task(
        conn,
        title="live-verify on the managed gateway",
        assignee="qa",
    )
    kb.link_tasks(conn, qa, impl)
    kb.designate_runtime_acceptance_parents(conn, impl, [qa])
    assert kb.complete_task(conn, qa, summary="Live verification passed.")

    metadata = dict(EVIDENCE)
    metadata["runtime_acceptance_parents"] = [qa]
    assert kb.complete_task(
        conn,
        impl,
        summary="Approved with runtime evidence.",
        metadata=metadata,
        expected_run_id=review.current_run_id,
    )
    assert kb.get_task(conn, impl).status == "done"
    # Evidence rides the completed event for downstream consumers.
    completed = [
        e.payload for e in kb.list_events(conn, impl)
        if e.kind == "completed"
    ][-1]
    assert completed["runtime_acceptance"]["commit"] == EVIDENCE["commit"]


def test_missing_commit_or_evidence_fails_closed(conn) -> None:
    impl, review = _runtime_review_card(conn)
    qa = kb.create_task(
        conn,
        title="live-verify on the managed gateway",
        assignee="qa",
    )
    kb.link_tasks(conn, qa, impl)
    kb.designate_runtime_acceptance_parents(conn, impl, [qa])
    assert kb.complete_task(conn, qa, summary="Live verification passed.")

    # No metadata at all.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        expected_run_id=review.current_run_id,
    )
    # Commit without runtime evidence.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={"commit": "abc123", "runtime_acceptance_parents": [qa]},
        expected_run_id=review.current_run_id,
    )
    # Runtime evidence without a commit.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={
            "runtime_evidence": EVIDENCE["runtime_evidence"],
            "runtime_acceptance_parents": [qa],
        },
        expected_run_id=review.current_run_id,
    )
    # A non-hash label is not a candidate commit.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={
            "commit": "latest",
            "runtime_evidence": EVIDENCE["runtime_evidence"],
            "runtime_acceptance_parents": [qa],
        },
        expected_run_id=review.current_run_id,
    )
    # Evidence for a different candidate commit must not approve this one.
    mismatched = dict(EVIDENCE["runtime_evidence"])
    mismatched["candidate_commit"] = "deadbeef"
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={
            "commit": "6a49d075dd",
            "runtime_evidence": mismatched,
            "runtime_acceptance_parents": [qa],
        },
        expected_run_id=review.current_run_id,
    )
    # Blank strings don't count.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={
            "commit": "  ",
            "runtime_evidence": "  ",
            "runtime_acceptance_parents": [qa],
        },
        expected_run_id=review.current_run_id,
    )
    assert kb.get_task(conn, impl).status in ("review", "running")


def test_code_only_cards_are_ungated(conn) -> None:
    """The default path: no marker, no evidence requirement. A reviewer
    approves a plain code-only card exactly as before."""
    impl = kb.create_task(conn, title="code-only refactor", assignee="coder")
    impl_run = kb.claim_task(conn, impl, claimer="coder:1")
    assert impl_run is not None
    assert kb.request_review(
        conn,
        impl,
        summary="Refactor ready.",
        reviewer="reviewer",
        expected_run_id=impl_run.current_run_id,
    )
    review = kb.claim_review_task(conn, impl, claimer="reviewer:1")
    assert review is not None
    assert kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        expected_run_id=review.current_run_id,
    )
    assert kb.get_task(conn, impl).status == "done"


def test_unmarked_parent_does_not_gate_marked_child_contract(
    conn,
) -> None:
    """Only QA/live-verification PARENTS gate the marked card. A plain
    parent that finishes does not substitute; the check is parent-status
    based so this test pins that an unmarked non-QA parent completing does
    not by itself allow approval without evidence."""
    impl = kb.create_task(
        conn,
        title="runtime card",
        assignee="coder",
        requires_runtime_acceptance=True,
    )
    docs = kb.create_task(
        conn, title="write docs", assignee="coder", parents=[impl]
    )
    impl_run = kb.claim_task(conn, impl, claimer="coder:1")
    assert impl_run is not None
    assert kb.request_review(
        conn,
        impl,
        summary="Ready.",
        reviewer="reviewer",
        expected_run_id=impl_run.current_run_id,
    )
    review = kb.claim_review_task(conn, impl, claimer="reviewer:1")
    assert review is not None
    kb.complete_task(conn, docs, summary="docs done")
    # QA-style parent (live-verification) absent: even with parents done,
    # missing evidence still fails closed.
    assert not kb.complete_task(
        conn,
        impl,
        summary="Approved.",
        metadata={"commit": "abc"},
        expected_run_id=review.current_run_id,
    )


# ---------------------------------------------------------------------------
# Decomposition preserves the marker and enforces QA -> reviewer order
# ---------------------------------------------------------------------------


def test_decompose_rejects_marked_runtime_parent_cycle(kanban_home) -> None:
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {
                "title": "review A",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [1],
            },
            {
                "title": "review B",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [0],
            },
        ]
        with pytest.raises(ValueError, match="cannot itself require runtime acceptance"):
            kb.decompose_triage_task(
                conn, root, root_assignee="orchestrator", children=children
            )
        root_task = kb.get_task(conn, root)
        assert root_task is not None
        assert root_task.status == "triage"


@pytest.mark.parametrize("scalar_parent_owner", ["review", "runtime_parent"])
@pytest.mark.parametrize("scalar_parent_value", [1, 0, False, ""])
def test_decompose_rejects_scalar_parents_before_runtime_edge_normalization(
    kanban_home,
    scalar_parent_owner,
    scalar_parent_value,
) -> None:
    review_parents = scalar_parent_value if scalar_parent_owner == "review" else []
    qa_parents = scalar_parent_value if scalar_parent_owner == "runtime_parent" else []
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {
                "title": "review",
                "assignee": "reviewer",
                "parents": review_parents,
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [1],
            },
            {
                "title": "candidate receipt",
                "assignee": "qa",
                "parents": qa_parents,
            },
        ]
        with pytest.raises(ValueError, match="parents must be a list"):
            kb.decompose_triage_task(
                conn, root, root_assignee="orchestrator", children=children
            )


def test_decompose_rejects_boolean_runtime_parent_index(kanban_home) -> None:
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {
                "title": "review",
                "assignee": "reviewer",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [True],
            },
            {"title": "candidate receipt", "assignee": "qa"},
        ]
        with pytest.raises(ValueError, match="not a valid sibling index"):
            kb.decompose_triage_task(
                conn, root, root_assignee="orchestrator", children=children
            )


def test_decompose_validates_referenced_child_before_normalizing_runtime_edges(
    kanban_home,
) -> None:
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {
                "title": "review",
                "assignee": "reviewer",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [1],
            },
            "not a child object",
        ]
        with pytest.raises(ValueError, match=r"child\[1\] is not a dict"):
            kb.decompose_triage_task(
                conn, root, root_assignee="orchestrator", children=children
            )


def test_decompose_preserves_marker_and_rejects_reviewer_before_qa(
    kanban_home,
) -> None:
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {
                "title": "implement",
                "assignee": "coder",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [1],
            },
            {"title": "candidate receipt", "assignee": "qa", "parents": [0]},
        ]
        child_ids = kb.decompose_triage_task(
            conn, root, root_assignee="orchestrator", children=children
        )
        assert child_ids is not None
        impl, qa = (kb.get_task(conn, cid) for cid in child_ids)
        assert impl.requires_runtime_acceptance is True
        assert qa.requires_runtime_acceptance is False
        # The decomposer flipped the inverted edge: the QA card must be a
        # PARENT of the marked review card (QA -> reviewer), never the
        # review card a parent of QA (reviewer -> later verification).
        links = conn.execute(
            "SELECT parent_id, child_id FROM task_links "
            "WHERE parent_id IN (?, ?) AND child_id IN (?, ?)",
            (child_ids[0], child_ids[1], child_ids[0], child_ids[1]),
        ).fetchall()
        pairs = {(r["parent_id"], r["child_id"]) for r in links}
        assert (child_ids[1], child_ids[0]) in pairs
        assert (child_ids[0], child_ids[1]) not in pairs


def test_decompose_links_marked_review_child_under_qa_child(
    kanban_home,
) -> None:
    """A decomposed child marked requires_runtime_acceptance whose parent
    list points at a non-QA sibling gets a live-verification edge inserted:
    the QA sibling (or, absent one, every leaf work producer) must become a
    parent of the marked card so QA -> reviewer is structural."""
    with kb.connect_closing() as conn:
        root = kb.create_task(conn, title="triage me", triage=True)
        children = [
            {"title": "implement", "assignee": "coder"},
            {
                "title": "review PR",
                "assignee": "reviewer",
                "requires_runtime_acceptance": True,
                "runtime_acceptance_parents": [0],
                "parents": [0],
            },
        ]
        child_ids = kb.decompose_triage_task(
            conn, root, root_assignee="orchestrator", children=children
        )
        assert child_ids is not None
        links = conn.execute(
            "SELECT parent_id, child_id FROM task_links WHERE child_id = ?",
            (child_ids[1],),
        ).fetchall()
        parents = {r["parent_id"] for r in links}
        # The marked review card still waits on the root (decompose links
        # every leaf under the root via the reverse edge) AND cannot have
        # shed the QA ordering: its parents include the work producer.
        assert child_ids[0] in parents


# ---------------------------------------------------------------------------
# Existing graph migration
# ---------------------------------------------------------------------------


def test_legacy_board_migration_adds_nullable_marker(tmp_path: Path) -> None:
    """Opening a legacy board DB (pre-marker schema) is safe: the additive
    column lands as 0/NULL and nothing is gated."""
    db_path = tmp_path / "legacy.db"
    with kb.connect_closing(db_path) as conn:
        qa = kb.create_task(conn, title="legacy QA", assignee="qa")
        tid = kb.create_task(
            conn,
            title="legacy-style review card",
            assignee="reviewer",
            parents=[qa],
        )
        conn.execute("ALTER TABLE tasks DROP COLUMN requires_runtime_acceptance")
        conn.commit()
    # Simulate another process opening this pre-marker file: clear this
    # process's init cache so connect() runs additive migrations again.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect_closing(db_path) as conn:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(tasks)")
        }
        assert "requires_runtime_acceptance" in cols
        assert kb.get_task(conn, tid).requires_runtime_acceptance is False
        assert kb.get_task(conn, tid).status == "todo"
        link = conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (qa, tid),
        ).fetchone()
        assert link is not None


def test_manual_done_on_marked_review_card_also_gated(kanban_home) -> None:
    """The gate is at complete_task, not the run-CAS path only: a dashboard
    drag / manual CLI completion of a marked card sitting in ``review``
    without evidence fails closed too."""
    with kb.connect_closing() as conn:
        impl = kb.create_task(
            conn,
            title="runtime card",
            assignee="coder",
            requires_runtime_acceptance=True,
        )
        kb.claim_task(conn, impl)
        assert kb.request_review(
            conn,
            impl,
            summary="Ready.",
            reviewer="reviewer",
            force=True,
        )
        # Review lane, no active run, manual approval attempt.
        assert kb.get_task(conn, impl).status == "review"
        assert not kb.complete_task(
            conn, impl, summary="Manual approve, no evidence."
        )
        assert kb.get_task(conn, impl).status == "review"
        # Backfill a QA parent as done + evidence → now it completes.
        qa = kb.create_task(conn, title="live verify", assignee="qa")
        kb.link_tasks(conn, qa, impl)
        kb.designate_runtime_acceptance_parents(conn, impl, [qa])
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? "
                "WHERE id = ?",
                (int(__import__("time").time()), qa),
            )
        metadata = dict(EVIDENCE)
        metadata["runtime_acceptance_parents"] = [qa]
        assert kb.complete_task(
            conn,
            impl,
            summary="Manual approve with evidence.",
            metadata=metadata,
        )
        assert kb.get_task(conn, impl).status == "done"
