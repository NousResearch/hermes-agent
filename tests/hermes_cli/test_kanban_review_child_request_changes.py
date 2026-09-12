"""#106583: separate review children must be able to request changes.

Impulse-style graphs keep implementation, parallel review children, and a
finalize join as distinct cards. Completing implementation promotes each
review from todo to ready; the dispatcher then claims a review from ready.
``request_changes`` used to reject that run because it was not claimed from
first-class ``review`` and had no ``review_requested`` event.

These tests pin the replacement contract: an *explicit* workflow role
(implementation / review / finalize; default ordinary) is durable provenance.
Graph topology may validate that role but never establishes it. Ordinary
A->B + [A,B]->J diamonds stay ordinary.
"""

from __future__ import annotations

import argparse
import json as jsonlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as db:
        yield db


def _explicit_review_graph(conn):
    """I (implementation) -> review children; finalize joins I + reviews."""
    impl_id = kb.create_task(
        conn, title="Implement export", assignee="builder",
        workflow_role="implementation",
    )
    review_ids = [
        kb.create_task(
            conn, title=title, assignee=assignee, parents=[impl_id],
            workflow_role="review",
        )
        for title, assignee in (
            ("QA lane", "qa-reviewer"),
            ("Tech audit", "tech-reviewer"),
            ("Scope check", "scope-reviewer"),
        )
    ]
    finalize_id = kb.create_task(
        conn,
        title="Ship it",
        assignee="releaser",
        parents=[impl_id, *review_ids],
        workflow_role="finalize",
    )
    ordinary_id = kb.create_task(
        conn,
        title="Review-shaped sequential child",
        assignee="docs",
        parents=[impl_id],
    )
    return impl_id, review_ids, finalize_id, ordinary_id


def _ordinary_diamond(conn):
    """Ordinary A->B and [A,B]->J. Same shape as a review diamond; no roles."""
    a_id = kb.create_task(conn, title="A", assignee="alice")
    b_id = kb.create_task(conn, title="B", assignee="bob", parents=[a_id])
    j_id = kb.create_task(
        conn, title="Join", assignee="joiner", parents=[a_id, b_id],
    )
    return a_id, b_id, j_id


def test_review_child_request_changes_reopens_implementation_without_duplicates(conn):
    """A review child claimed from ready must route request_changes onto I.

    Production break this catches: request_changes refuses with
    'active run was not claimed from review', the reviewer falls back to
    block, and later re-completion of I would spawn duplicate review cards
    if the graph were rebuilt instead of re-gated.
    """
    impl_id, review_ids, finalize_id, ordinary_id = _explicit_review_graph(conn)
    rejecting_id, sibling_a, sibling_b = review_ids

    assert kb.complete_task(conn, impl_id)
    kb.recompute_ready(conn)
    for rid in review_ids:
        assert kb.get_task(conn, rid).status == "ready"
    assert kb.get_task(conn, finalize_id).status == "todo"
    assert kb.get_task(conn, ordinary_id).status == "ready"

    ordinary = kb.claim_task(conn, ordinary_id, claimer="docs:1")
    assert ordinary is not None
    ok, detail = kb.request_changes(
        conn,
        ordinary_id,
        reason="ordinary children are not review children",
        expected_run_id=ordinary.current_run_id,
    )
    assert ok is False
    assert "claimed from review" in (detail or "")
    still_ordinary = kb.get_task(conn, ordinary_id)
    assert still_ordinary is not None
    assert still_ordinary.status == "running"
    kb.complete_task(conn, ordinary_id, expected_run_id=ordinary.current_run_id)

    claimed = kb.claim_task(conn, rejecting_id, claimer="qa-reviewer:1")
    assert claimed is not None
    assert claimed.status == "running"

    ok, implementer = kb.request_changes(
        conn,
        rejecting_id,
        reason="missing acceptance criterion",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is True
    assert implementer == "builder"

    impl = kb.get_task(conn, impl_id)
    assert impl is not None
    assert impl.status == "ready"
    assert impl.assignee == "builder"
    assert impl.completed_at is None

    rejecting = kb.get_task(conn, rejecting_id)
    assert rejecting is not None
    assert rejecting.status == "todo"
    assert rejecting.block_kind is None
    assert (rejecting.block_recurrences or 0) == 0
    assert rejecting.current_run_id is None

    for tid in (sibling_a, sibling_b, finalize_id):
        child = kb.get_task(conn, tid)
        assert child is not None
        assert child.status == "todo"

    change_events = [
        event for event in kb.list_events(conn, rejecting_id)
        if event.kind == "changes_requested"
    ]
    assert len(change_events) == 1
    payload = change_events[0].payload or {}
    assert payload["reason"] == "missing acceptance criterion"
    assert payload["reviewer"] == "qa-reviewer"
    assert payload["implementer"] == "builder"

    impl_events = [
        event for event in kb.list_events(conn, impl_id)
        if event.kind == "changes_requested"
    ]
    assert impl_events
    impl_payload = impl_events[-1].payload or {}
    assert impl_payload["reason"] == "missing acceptance criterion"
    assert impl_payload["reviewer"] == "qa-reviewer"

    before_ids = {
        row["id"]
        for row in conn.execute("SELECT id FROM tasks").fetchall()
    }
    assert kb.complete_task(conn, impl_id)
    kb.recompute_ready(conn)
    after_ids = {
        row["id"]
        for row in conn.execute("SELECT id FROM tasks").fetchall()
    }
    assert after_ids == before_ids
    for rid in review_ids:
        assert kb.get_task(conn, rid).status == "ready"
        assert kb.get_task(conn, rid).id == rid
    assert kb.get_task(conn, finalize_id).status == "todo"


def test_ordinary_diamond_request_changes_rejected_and_decompose_eligible(conn):
    """A->B and [A,B]->J is ordinary work, not a review/finalize workflow.

    Production break this catches: topology-only diamond inference treats B
    as a review child and J as finalize, so request_changes on B reopens A
    and auto-decompose skips both cards.
    """
    a_id, b_id, j_id = _ordinary_diamond(conn)
    assert kb.complete_task(conn, a_id)
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, b_id, claimer="bob:1")
    assert claimed is not None

    ok, detail = kb.request_changes(
        conn,
        b_id,
        reason="ordinary sequential child is not a review",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is False
    assert "claimed from review" in (detail or "")
    still_b = kb.get_task(conn, b_id)
    assert still_b is not None
    assert still_b.status == "running"
    assert kb.get_task(conn, a_id).status == "done"

    kb.complete_task(conn, b_id, expected_run_id=claimed.current_run_id)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id IN (?, ?)",
            (b_id, j_id),
        )
    b_task = kb.get_task(conn, b_id)
    j_task = kb.get_task(conn, j_id)
    assert decomp.decompose_eligible(conn, b_task) is True
    assert decomp.decompose_eligible(conn, j_task) is True


def test_late_verdict_does_not_resurrect_archived_implementation(conn):
    """A late review verdict must not un-archive a finished implementation.

    Production break this catches: request_changes reopens archived as well
    as done, racing a GC/archive pass that already closed the work.
    """
    impl_id, review_ids, finalize_id, _ordinary_id = _explicit_review_graph(conn)
    review_id = review_ids[0]
    assert kb.complete_task(conn, impl_id)
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, review_id, claimer="qa-reviewer:1")
    assert claimed is not None
    assert kb.archive_task(conn, impl_id)
    assert kb.get_task(conn, impl_id).status == "archived"

    ok, implementer = kb.request_changes(
        conn,
        review_id,
        reason="late findings after archive",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is True
    assert implementer == "builder"
    impl = kb.get_task(conn, impl_id)
    assert impl is not None
    assert impl.status == "archived"
    assert impl.completed_at is not None
    rejecting = kb.get_task(conn, review_id)
    assert rejecting is not None
    assert rejecting.status == "todo"
    assert kb.get_task(conn, finalize_id).status == "todo"


def test_list_triage_ids_includes_ordinary_excludes_explicit_review_finalize(conn):
    """list_triage_ids is the auto-decompose intake: role, not topology."""
    rough_id = kb.create_task(conn, title="rough idea", triage=True)
    a_id, b_id, j_id = _ordinary_diamond(conn)
    impl_id, review_ids, finalize_id, _ordinary_id = _explicit_review_graph(conn)
    review_id = review_ids[0]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id IN (?, ?, ?, ?)",
            (b_id, j_id, review_id, finalize_id),
        )

    triage_ids = set(decomp.list_triage_ids())
    assert rough_id in triage_ids
    assert b_id in triage_ids
    assert j_id in triage_ids
    assert review_id not in triage_ids
    assert finalize_id not in triage_ids


def test_ambiguous_implementation_parents_fail_closed_without_mutation(conn):
    """Two implementation parents cannot be disambiguated by link order.

    Production break this catches: request_changes falls back to
    impl_parents[0] (sorted link order) and reopens the wrong implementation.
    Explicit workflow_role plus graph membership still leave two candidates,
    so the only safe rule is exactly one implementation parent.
    """
    impl_a = kb.create_task(
        conn, title="Implement path A", assignee="builder-a",
        workflow_role="implementation",
    )
    impl_b = kb.create_task(
        conn, title="Implement path B", assignee="builder-b",
        workflow_role="implementation",
    )
    review_id = kb.create_task(
        conn, title="QA both paths", assignee="qa-reviewer",
        parents=[impl_a, impl_b], workflow_role="review",
    )
    descendant_id = kb.create_task(
        conn, title="Ship both paths", assignee="releaser",
        parents=[impl_a, impl_b, review_id], workflow_role="finalize",
    )
    sibling_id = kb.create_task(
        conn, title="Docs for path A", assignee="docs", parents=[impl_a],
    )

    assert kb.complete_task(conn, impl_a)
    assert kb.complete_task(conn, impl_b)
    kb.recompute_ready(conn)
    assert kb.get_task(conn, review_id).status == "ready"
    assert kb.get_task(conn, sibling_id).status == "ready"
    assert kb.get_task(conn, descendant_id).status == "todo"

    claimed = kb.claim_task(conn, review_id, claimer="qa-reviewer:1")
    assert claimed is not None
    assert claimed.status == "running"
    review_run_id = claimed.current_run_id

    before = {
        tid: kb.get_task(conn, tid)
        for tid in (impl_a, impl_b, review_id, descendant_id, sibling_id)
    }
    before_events = {
        tid: [(event.kind, event.payload) for event in kb.list_events(conn, tid)]
        for tid in (impl_a, impl_b, review_id, descendant_id, sibling_id)
    }

    ok, detail = kb.request_changes(
        conn,
        review_id,
        reason="which implementation should reopen?",
        expected_run_id=review_run_id,
    )
    assert ok is False
    assert "unambiguous" in (detail or "")
    assert "implementation parent" in (detail or "")

    after = {
        tid: kb.get_task(conn, tid)
        for tid in (impl_a, impl_b, review_id, descendant_id, sibling_id)
    }
    for tid in (impl_a, impl_b):
        assert after[tid].status == "done"
        assert after[tid].completed_at == before[tid].completed_at
        assert after[tid].assignee == before[tid].assignee
    review = after[review_id]
    assert review.status == "running"
    assert review.current_run_id == review_run_id
    assert review.block_kind is None
    assert after[descendant_id].status == "todo"
    assert after[sibling_id].status == "ready"

    after_events = {
        tid: [(event.kind, event.payload) for event in kb.list_events(conn, tid)]
        for tid in (impl_a, impl_b, review_id, descendant_id, sibling_id)
    }
    assert after_events == before_events
    for tid in (impl_a, impl_b, review_id, descendant_id, sibling_id):
        assert not any(kind == "changes_requested" for kind, _payload in after_events[tid])
        assert not any(kind == "descendant_invalidated" for kind, _payload in after_events[tid])


def test_current_step_key_does_not_establish_review_role(conn):
    """v2 step keys are not workflow-role provenance."""
    impl_id = kb.create_task(
        conn, title="I", assignee="builder", workflow_role="implementation",
    )
    with kb.write_txn(conn):
        child_id = kb.create_task(
            conn, title="looks like review", assignee="qa", parents=[impl_id],
        )
        conn.execute(
            "UPDATE tasks SET current_step_key = 'review' WHERE id = ?",
            (child_id,),
        )
    kb.create_task(
        conn, title="join", assignee="rel", parents=[impl_id, child_id],
    )
    assert kb.complete_task(conn, impl_id)
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, child_id, claimer="qa:1")
    assert claimed is not None
    ok, detail = kb.request_changes(
        conn, child_id, reason="step key is not a role",
        expected_run_id=claimed.current_run_id,
    )
    assert ok is False
    assert "claimed from review" in (detail or "")


def test_create_task_defaults_ordinary_and_rejects_unknown_role(conn):
    task_id = kb.create_task(conn, title="plain", assignee="alice")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.workflow_role == "ordinary"
    with pytest.raises(ValueError, match="workflow_role"):
        kb.create_task(conn, title="bad", assignee="alice", workflow_role="auditor")


def test_legacy_db_gains_workflow_role_column(tmp_path, monkeypatch):
    """Additive migration fills ordinary for boards that predate the column."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(db_path))
    raw.row_factory = sqlite3.Row
    raw.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    raw.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    raw.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    raw.commit()
    before = {r[1] for r in raw.execute("PRAGMA table_info(tasks)")}
    assert "workflow_role" not in before
    kbc._migrate_add_optional_columns(raw)
    after = {r[1] for r in raw.execute("PRAGMA table_info(tasks)")}
    assert "workflow_role" in after
    row = raw.execute("SELECT workflow_role FROM tasks WHERE id = 'legacy'").fetchone()
    assert row["workflow_role"] == "ordinary"
    kbc._migrate_add_optional_columns(raw)
    raw.close()


def test_cli_create_parser_exposes_role():
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        ["kanban", "create", "QA the export", "--assignee", "qa", "--role", "review"],
    )
    assert args.workflow_role == "review"


def test_cli_request_changes_on_explicit_review_child_claimed_from_ready(
    conn, monkeypatch: pytest.MonkeyPatch,
):
    """Public CLI: --role on create, then request-changes from a ready claim."""
    impl_raw = kc.run_slash(
        'create "Implement export" --assignee builder --role implementation --json'
    )
    impl = jsonlib.loads(impl_raw)
    assert impl["workflow_role"] == "implementation"
    review_raw = kc.run_slash(
        f'create "QA lane" --assignee qa-reviewer --parent {impl["id"]} '
        "--role review --json"
    )
    review = jsonlib.loads(review_raw)
    assert review["workflow_role"] == "review"
    finalize_raw = kc.run_slash(
        f'create "Ship it" --assignee releaser --parent {impl["id"]} '
        f'--parent {review["id"]} --role finalize --json'
    )
    finalize = jsonlib.loads(finalize_raw)
    assert finalize["workflow_role"] == "finalize"

    assert kb.complete_task(conn, impl["id"])
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, review["id"], claimer="qa-reviewer:1")
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", review["id"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "qa-reviewer")

    output = kc.run_slash(
        f"request-changes {review['id']} 'missing acceptance criterion'"
    )
    assert "Requested changes" in output
    assert "builder" in output

    impl_task = kb.get_task(conn, impl["id"])
    assert impl_task is not None
    assert impl_task.status == "ready"
    assert impl_task.assignee == "builder"
    review_task = kb.get_task(conn, review["id"])
    assert review_task is not None
    assert review_task.status == "todo"
    assert kb.get_task(conn, finalize["id"]).status == "todo"


def test_kanban_request_changes_tool_parity_on_explicit_review_child(
    conn, monkeypatch: pytest.MonkeyPatch,
):
    """kanban_create role + kanban_request_changes must match the CLI graph."""
    from tools import kanban_tools as tools

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    impl_out = jsonlib.loads(tools._handle_create({
        "title": "Implement export",
        "assignee": "builder",
        "role": "implementation",
    }))
    assert impl_out["ok"] is True
    assert impl_out["workflow_role"] == "implementation"
    review_out = jsonlib.loads(tools._handle_create({
        "title": "QA lane",
        "assignee": "qa-reviewer",
        "parents": [impl_out["task_id"]],
        "role": "review",
    }))
    assert review_out["ok"] is True
    assert review_out["workflow_role"] == "review"
    finalize_out = jsonlib.loads(tools._handle_create({
        "title": "Ship it",
        "assignee": "releaser",
        "parents": [impl_out["task_id"], review_out["task_id"]],
        "role": "finalize",
    }))
    assert finalize_out["ok"] is True
    assert finalize_out["workflow_role"] == "finalize"

    assert kb.complete_task(conn, impl_out["task_id"])
    kb.recompute_ready(conn)
    claimed = kb.claim_task(conn, review_out["task_id"], claimer="qa-reviewer:1")
    assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", review_out["task_id"])
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    monkeypatch.setenv("HERMES_PROFILE", "qa-reviewer")

    changed = jsonlib.loads(tools._handle_request_changes({
        "reason": "missing acceptance criterion",
    }))
    assert changed["ok"] is True
    assert changed["implementer"] == "builder"
    assert changed["status"] == "todo"

    impl = kb.get_task(conn, impl_out["task_id"])
    assert impl is not None
    assert impl.status == "ready"
    review = kb.get_task(conn, review_out["task_id"])
    assert review is not None
    assert review.status == "todo"
    assert kb.get_task(conn, finalize_out["task_id"]).status == "todo"


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _patch_profiles(names: list[str]):
    from types import SimpleNamespace

    fake = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0]),
    ]


def test_review_and_finalize_cards_are_not_eligible_for_auto_decomposition(
    conn, monkeypatch: pytest.MonkeyPatch,
):
    """Review/finalize workflow cards that land in triage must not fan out.

    Production break this catches: a blocked review child is auto-decomposed
    into unrelated children because eligibility was prompt-only.
    """
    impl_id, review_ids, finalize_id, _ordinary_id = _explicit_review_graph(conn)
    review_id = review_ids[0]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id IN (?, ?)",
            (review_id, finalize_id),
        )
    rough_id = kb.create_task(conn, title="rough idea", triage=True)

    fanout = jsonlib.dumps({
        "fanout": True,
        "rationale": "should never apply to review/finalize cards",
        "tasks": [
            {"title": "nonsense A", "body": "a", "assignee": "builder", "parents": []},
            {"title": "nonsense B", "body": "b", "assignee": "builder", "parents": []},
        ],
    })
    patches = _patch_profiles(["orchestrator", "builder"])
    for p in patches:
        p.start()
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **_kwargs: _fake_aux_response(fanout),
    )
    try:
        review_outcome = decomp.decompose_task(review_id, author="auto-decomposer")
        finalize_outcome = decomp.decompose_task(finalize_id, author="auto-decomposer")
        rough_outcome = decomp.decompose_task(rough_id, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert review_outcome.ok is False
    assert finalize_outcome.ok is False
    assert kb.child_ids(conn, review_id) == [finalize_id]
    assert kb.child_ids(conn, finalize_id) == []
    assert not any(
        event.kind == "decomposed" for event in kb.list_events(conn, review_id)
    )
    assert not any(
        event.kind == "decomposed" for event in kb.list_events(conn, finalize_id)
    )
    assert rough_outcome.ok is True
    assert rough_outcome.fanout is True
    assert rough_outcome.child_ids
