"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"
    with kb.connect() as conn:
        for child_id in child_ids:
            assessments = [
                event
                for event in kb.list_events(conn, child_id)
                if event.kind == "granularity_assessed"
            ]
            assert len(assessments) == 1
            assert assessments[0].payload["source"] == "decomposer"


def test_decompose_db_persists_review_scheduler_gate(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="freeze then schedule review")
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=[
                {
                    "title": "Freeze postimage",
                    "body": "Run closure and compute exact hashes.",
                    "kind": "closure",
                    "parents": [],
                },
                {
                    "title": "Schedule exact review",
                    "body": "Create the reviewer after hashes exist.",
                    "kind": "review_scheduler",
                    "parents": [0],
                },
            ],
            author="decomposer",
        )

    assert child_ids is not None
    with kb.connect() as conn:
        scheduler = kb.get_task(conn, child_ids[1])
    assert scheduler is not None
    assert (scheduler.body or "").startswith("REVIEW_SCHEDULER_GATE:")
    assert scheduler.idempotency_key is None


def _exact_review_children(review_key):
    return [
        {"title": "Freeze exact postimage", "kind": "closure", "parents": []},
        {
            "title": "Review exact postimage",
            "kind": "review",
            "review_key": review_key,
            "parents": [0],
        },
    ]


def test_decompose_persists_durable_exact_review_identity(kanban_home):
    postimage = "a" * 64
    claims = "b" * 64
    review_key = f"{postimage}:{claims}"

    with kb.connect() as conn:
        root_id = _create_triage(conn, title="create durable exact review")
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=_exact_review_children(review_key),
            author="decomposer",
        )
        assert child_ids is not None
        row = conn.execute(
            """SELECT idempotency_key, review_contract,
                      review_postimage_sha256, review_claims_sha256
                 FROM tasks WHERE id = ?""",
            (child_ids[1],),
        ).fetchone()

    assert row["idempotency_key"] == f"review:v1:{review_key}"
    assert row["review_contract"] == "exact_review_v1"
    assert row["review_postimage_sha256"] == postimage
    assert row["review_claims_sha256"] == claims


def test_decompose_rejects_legacy_exact_review_incumbent_atomically(kanban_home):
    postimage = "c" * 64
    claims = "d" * 64
    review_key = f"{postimage}:{claims}"
    reserved = f"review:v1:{review_key}"

    with kb.connect() as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, title, status, created_at, idempotency_key)
               VALUES ('poisoned', 'legacy incumbent', 'ready', 1, ?)""",
            (reserved,),
        )
        root_id = _create_triage(conn, title="reject legacy exact review")
        conn.commit()

        with pytest.raises(ValueError, match="ambiguous legacy exact-review"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=_exact_review_children(review_key),
                author="decomposer",
            )
        root = kb.get_task(conn, root_id)
        linked = kb.child_ids(conn, root_id)

    assert root is not None and root.status == "triage"
    assert linked == []


def test_decompose_rejects_duplicate_exact_review_incumbents_atomically(kanban_home):
    postimage = "e" * 64
    claims = "f" * 64
    review_key = f"{postimage}:{claims}"
    reserved = f"review:v1:{review_key}"

    with kb.connect() as conn:
        conn.executemany(
            """INSERT INTO tasks
               (id, title, status, created_at, idempotency_key,
                review_contract, review_postimage_sha256, review_claims_sha256)
               VALUES (?, 'duplicate exact review', 'ready', 1, ?,
                       'exact_review_v1', ?, ?)""",
            (
                ("duplicate-a", reserved, postimage, claims),
                ("duplicate-b", reserved, postimage, claims),
            ),
        )
        root_id = _create_triage(conn, title="reject duplicate exact reviews")
        conn.commit()

        with pytest.raises(ValueError, match="ambiguous legacy exact-review"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=_exact_review_children(review_key),
                author="decomposer",
            )
        root = kb.get_task(conn, root_id)
        linked = kb.child_ids(conn, root_id)

    assert root is not None and root.status == "triage"
    assert linked == []


@pytest.mark.parametrize(
    "review_body",
    [
        f"REVIEW_KEY: {'c' * 64}:{'d' * 64}\n\nReview the wrong postimage.",
        (
            f"REVIEW_KEY: {'a' * 64}:{'b' * 64}\n"
            f"REVIEW_KEY: {'a' * 64}:{'b' * 64}\n\nReview the postimage."
        ),
    ],
)
def test_decompose_rejects_conflicting_or_multiple_review_body_identity_atomically(
    kanban_home,
    review_body,
):
    review_key = f"{'a' * 64}:{'b' * 64}"
    children = _exact_review_children(review_key)
    children[1]["body"] = review_body

    with kb.connect() as conn:
        root_id = _create_triage(conn, title="reject contradictory review identity")
        with pytest.raises(ValueError, match="REVIEW_KEY"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=children,
                author="decomposer",
            )
        root = kb.get_task(conn, root_id)
        linked = kb.child_ids(conn, root_id)

    assert root is not None and root.status == "triage"
    assert linked == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("idempotency_key", f"review:v1:{'c' * 64}:{'d' * 64}"),
        ("review_contract", "legacy_review"),
        ("review_postimage_sha256", "c" * 64),
        ("review_claims_sha256", "d" * 64),
    ],
)
def test_decompose_rejects_conflicting_durable_review_discriminator_atomically(
    kanban_home,
    field,
    value,
):
    review_key = f"{'a' * 64}:{'b' * 64}"
    children = _exact_review_children(review_key)
    children[1][field] = value

    with kb.connect() as conn:
        root_id = _create_triage(conn, title="reject conflicting review discriminator")
        with pytest.raises(ValueError, match="exact review"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=children,
                author="decomposer",
            )
        root = kb.get_task(conn, root_id)
        linked = kb.child_ids(conn, root_id)

    assert root is not None and root.status == "triage"
    assert linked == []


def test_decompose_rejects_reused_review_with_conflicting_body_identity_atomically(
    kanban_home,
):
    review_key = f"{'a' * 64}:{'b' * 64}"
    conflicting_key = f"{'c' * 64}:{'d' * 64}"

    with kb.connect() as conn:
        review_id = kb.create_task(
            conn,
            title="Existing exact review",
            body=f"REVIEW_KEY: {review_key}",
            review_key=review_key,
        )
        conn.execute(
            "UPDATE tasks SET body = ? WHERE id = ?",
            (f"REVIEW_KEY: {conflicting_key}", review_id),
        )
        root_id = _create_triage(conn, title="reject conflicting reused review")
        conn.commit()

        with pytest.raises(ValueError, match="ambiguous legacy exact-review"):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=_exact_review_children(review_key),
                author="decomposer",
            )
        root = kb.get_task(conn, root_id)
        linked = kb.child_ids(conn, root_id)

    assert root is not None and root.status == "triage"
    assert linked == []


@pytest.mark.parametrize("review_status", ["running", "done"])
def test_reused_exact_review_keeps_input_parents_immutable(
    kanban_home,
    review_status,
):
    review_key = f"{'a' * 64}:{'b' * 64}"
    with kb.connect() as conn:
        review_id = kb.create_task(
            conn,
            title="Existing exact review",
            review_key=review_key,
        )
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (review_status, review_id),
        )
        root_id = _create_triage(conn, title="reuse exact review")
        conn.commit()
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[
                {
                    "title": "Freeze exact postimage",
                    "kind": "closure",
                    "parents": [],
                },
                {
                    "title": "Review exact postimage",
                    "kind": "review",
                    "review_key": review_key,
                    "parents": [0],
                },
                {
                    "title": "Controller",
                    "kind": "controller",
                    "parents": [1],
                },
            ],
            author="decomposer",
        )

        assert child_ids is not None
        closure_id, reused_id, controller_id = child_ids
        review = kb.get_task(conn, review_id)
        review_parents = kb.parent_ids(conn, review_id)
        controller_parents = kb.parent_ids(conn, controller_id)

    assert reused_id == review_id
    assert review is not None and review.status == review_status
    assert closure_id not in review_parents
    assert review_id in controller_parents
    assert closure_id in controller_parents


def test_reused_ready_review_is_demoted_behind_new_closure(kanban_home):
    review_key = f"{'c' * 64}:{'d' * 64}"
    with kb.connect() as conn:
        review_id = kb.create_task(
            conn,
            title="Ready exact review",
            review_key=review_key,
        )
        root_id = _create_triage(conn, title="gate ready review")
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[
                {"title": "Freeze", "kind": "closure", "parents": []},
                {
                    "title": "Review",
                    "kind": "review",
                    "review_key": review_key,
                    "parents": [0],
                },
            ],
            author="decomposer",
        )

        assert child_ids is not None
        closure_id, reused_id = child_ids
        review = kb.get_task(conn, review_id)
        review_parents = kb.parent_ids(conn, review_id)

    assert reused_id == review_id
    assert review is not None and review.status == "todo"
    assert closure_id in review_parents


def test_decompose_returns_none_when_task_missing(kanban_home):
    with kb.connect() as conn:
        result = kb.decompose_triage_task(
            conn,
            "nonexistent",
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_returns_none_when_task_not_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already a real task")  # not triage
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_empty_children_returns_none(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[],
            author="me",
        )
    assert result is None


def test_decompose_db_rejects_single_child_fanout_without_mutation(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="at least 2"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "one child", "body": "one atomic task"}],
                author="me",
            )
        task = kb.get_task(conn, tid)

    assert task is not None and task.status == "triage"


def test_decompose_db_rejects_split_child_even_under_warn_policy(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="remains too broad"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {
                        "title": "Implement, verify, review, and control",
                        "body": (
                            "Implement patch.py, run the full suite, freeze receipts, "
                            "conduct an independent review, and issue a controller verdict."
                        ),
                    },
                    {"title": "Prepare notes", "body": "Summarize atomic notes."},
                ],
                author="me",
            )
        task = kb.get_task(conn, tid)

    assert task is not None and task.status == "triage"


def test_decompose_db_rejects_controller_semantics_mislabeled_as_work(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="controller.*kind"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {
                        "title": "Issue controller verdict",
                        "body": "Decide GO or NO-GO immediately.",
                        "kind": "work",
                    },
                    {"title": "Prepare notes", "body": "Summarize atomic notes."},
                ],
                author="me",
            )
        task = kb.get_task(conn, tid)

    assert task is not None and task.status == "triage"


def test_decompose_db_rejects_activation_semantics_mislabeled_as_work(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="activation.*kind"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {
                        "title": "Release version 12 to production",
                        "body": "Ship the approved build now.",
                        "kind": "work",
                    },
                    {"title": "Prepare notes", "body": "Summarize atomic notes."},
                ],
                author="me",
            )
        task = kb.get_task(conn, tid)
        children = kb.child_ids(conn, tid)

    assert task is not None and task.status == "triage"
    assert children == []


def test_decompose_db_rejects_localized_activation_mislabeled_as_work(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="activation.*kind"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {
                        "title": "Publica la aplicación en producción ahora",
                        "body": "Hazla disponible para todos los usuarios externos.",
                        "kind": "work",
                    },
                    {"title": "Documenta el resultado", "body": "Resume el resultado."},
                ],
                author="me",
            )
        task = kb.get_task(conn, tid)
        children = kb.child_ids(conn, tid)

    assert task is not None and task.status == "triage"
    assert children == []


def test_decompose_db_explicit_activation_kind_is_always_blocked(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {
                    "title": "Release version 12 to production",
                    "body": "Ship the approved build now.",
                    "kind": "activation",
                },
                {"title": "Prepare notes", "body": "Summarize atomic notes."},
            ],
            author="me",
        )
        assert child_ids is not None
        activation = kb.get_task(conn, child_ids[0])

    assert activation is not None and activation.status == "blocked"
    assert activation.block_kind == "needs_input"
    assert "PREPARE_ONLY" in (activation.body or "")


def test_auto_decompose_attempt_is_claimed_once_per_triage_card(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)

        assert kb.claim_auto_decompose_attempt(conn, tid) is True
        assert kb.claim_auto_decompose_attempt(conn, tid) is False
        events = kb.list_events(conn, tid)

    attempts = [event for event in events if event.kind == "auto_decompose_attempted"]
    assert len(attempts) == 1


def test_auto_decompose_claims_are_cause_scoped_and_transient_retry_is_bounded(
    kanban_home,
):
    with kb.connect() as conn:
        tid = _create_triage(conn)

        assert kb.claim_auto_decompose_attempt(
            conn,
            tid,
            cause="granularity",
            generation="assessment:a",
        ) is True
        assert kb.finish_auto_decompose_attempt(
            conn,
            tid,
            cause="granularity",
            generation="assessment:a",
            success=False,
            transient=True,
        ) == "released"
        assert kb.claim_auto_decompose_attempt(
            conn,
            tid,
            cause="granularity",
            generation="assessment:a",
        ) is True
        assert kb.finish_auto_decompose_attempt(
            conn,
            tid,
            cause="granularity",
            generation="assessment:a",
            success=False,
            transient=True,
        ) == "abandoned"
        assert kb.claim_auto_decompose_attempt(
            conn,
            tid,
            cause="granularity",
            generation="assessment:a",
        ) is False

        # A later budget-triage generation is not consumed by the earlier
        # granularity claim on the same card.
        assert kb.claim_auto_decompose_attempt(
            conn,
            tid,
            cause="budget_triage_once",
            generation="run:42",
        ) is True
        events = kb.list_events(conn, tid)

    attempts = [event for event in events if event.kind == "auto_decompose_attempted"]
    assert all(isinstance(event.payload, dict) for event in attempts)
    scopes = [
        (event.payload.get("cause"), event.payload.get("generation"))
        for event in attempts
        if isinstance(event.payload, dict)
    ]
    assert scopes == [
        ("granularity", "assessment:a"),
        ("granularity", "assessment:a"),
        ("budget_triage_once", "run:42"),
    ]


def test_auto_decompose_scope_advances_to_latest_budget_triage_run(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        first_cause, first_generation = kb.auto_decompose_scope(conn, tid)
        kb._append_event(
            conn,
            tid,
            "budget_triaged",
            {"run_id": 42},
            run_id=42,
        )
        second_cause, second_generation = kb.auto_decompose_scope(conn, tid)

    assert first_cause == "granularity"
    assert first_generation.startswith("assessment:")
    assert second_cause == "budget_triage_once"
    assert second_generation == "run:42"


def test_decompose_rejects_self_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cannot list itself"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [0]}],
                author="me",
            )


def test_decompose_rejects_out_of_range_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="not a valid index"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [5]}],
                author="me",
            )


def test_decompose_rejects_cyclic_parents(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cyclic dependency"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {"title": "A", "parents": [1]},
                    {"title": "B", "parents": [0]},
                ],
                author="me",
            )


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[
                {"title": "task A", "assignee": "researcher"},
                {"title": "task B", "assignee": "researcher"},
            ],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_children_inherit_dir_workspace(kanban_home):
    """Fan-out children inherit the root's dir workspace, not scratch."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="codegen root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "part A"}, {"title": "part B", "parents": [0]}],
            author="decomposer",
        )
    assert child_ids and len(child_ids) == 2
    with kb.connect() as conn:
        for cid in child_ids:
            t = kb.get_task(conn, cid)
            assert t.workspace_kind == "dir"
            assert t.workspace_path == proj


def test_decompose_children_stay_scratch_when_root_scratch(kanban_home):
    """No regression: a scratch root still fans out into scratch children."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="scratch root", assignee="worker",
            workspace_kind="scratch", triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "s1"}, {"title": "s2"}], author="decomposer",
        )
        assert child_ids is not None
    with kb.connect() as conn:
        tasks = [kb.get_task(conn, child_id) for child_id in child_ids]
    assert all(task is not None for task in tasks)
    assert all(task.workspace_kind == "scratch" for task in tasks if task is not None)
    assert all(task.workspace_path is None for task in tasks if task is not None)


def test_decompose_per_child_workspace_override(kanban_home):
    """An explicit per-child workspace beats inheritance."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[
                {"title": "override", "workspace_kind": "dir",
                 "workspace_path": "/other/repo"},
                {"title": "inherit"},
            ],
            author="decomposer",
        )
    with kb.connect() as conn:
        over = kb.get_task(conn, child_ids[0])
        inh = kb.get_task(conn, child_ids[1])
    assert over.workspace_path == "/other/repo"
    assert inh.workspace_path == proj
