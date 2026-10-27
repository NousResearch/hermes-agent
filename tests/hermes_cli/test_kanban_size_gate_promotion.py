"""Promotion-path and decomposition coverage for the kanban size gate.

The creation-time gate is covered by ``test_kanban_size_gate.py``. This file
pins the three contract gaps that pass closed:

1. ``decompose_triage_task`` cannot emit a child without recording the child's
   budget estimate, and the child's body leads with the budget line.
2. Every path that parks a card on ``ready`` runs the same decision
   (``gate_ready_entry``), so an oversized card parked in ``todo``/``blocked``
   behind an open parent cannot be promoted into the dispatch lane ungated.
3. ``triage=True`` creation is measured too, and a legacy (pre-gate) card is
   measured but grandfathered - forward-only, never retro-diverted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_size_gate import (
    DEFAULT_SIZE_GATE_THRESHOLD,
    estimate_card_size,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _oversized_body() -> str:
    """A brief that scores well past the default threshold (54 with 12 AC + 8 files)."""
    return """
## Acceptance Criteria
1. First
2. Second
3. Third
4. Fourth
5. Fifth
6. Sixth
7. Seventh
8. Eighth
9. Ninth
10. Tenth
11. Eleventh
12. Twelfth

## Files
- src/a.py
- src/b.py
- src/c.py
- src/d.py
- src/e.py
- src/f.py
- src/g.py
- src/h.py
"""


def _estimates(conn, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'size_estimate' "
        "ORDER BY id", (task_id,),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows]


def _latest_estimate(conn, task_id: str) -> dict:
    estimates = _estimates(conn, task_id)
    assert estimates, f"{task_id} has no size_estimate event"
    return estimates[-1]


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    ]


def test_decomposed_children_carry_recorded_size_estimate(kanban_home):
    """Every emitted child records its own estimate and leads with the budget line."""
    from hermes_cli.kanban_db_graph import decompose_triage_task

    child_bodies = [
        "## Files\n- src/a.py\n- src/b.py\n",
        "## Files\n- src/c.py\n",
    ]
    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn, title="Root task", body="Some work", assignee="orchestrator",
            created_by="test", triage=True,
        )
        children = [
            {"title": "Child one", "body": child_bodies[0], "assignee": "junior-dev"},
            {"title": "Child two", "body": child_bodies[1], "assignee": "junior-dev"},
        ]
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator", children=children,
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2
        observed = []
        for child_id, original_body in zip(child_ids, child_bodies):
            task = kb.get_task(conn, child_id)
            observed.append({
                "estimates": _estimates(conn, child_id),
                "body": task.body,
                "title": task.title,
                "original": original_body,
            })

    for row in observed:
        # Exactly one recorded estimate, decided at decomposition time.
        assert len(row["estimates"]) == 1, row["estimates"]
        payload = row["estimates"][0]
        assert payload["decision"] == "allowed"
        assert payload["path"] == "decompose"
        assert payload["threshold"] == DEFAULT_SIZE_GATE_THRESHOLD
        # Measured on the UNSTAMPED body.
        assert payload["score"] == estimate_card_size(row["title"], row["original"]).score

        expected_line = (
            f"Budget estimate (size gate): score {payload['score']}/{payload['threshold']} - "
            "single-worker unit, measured at decomposition."
        )
        first_line, remainder = row["body"].split("\n", 1)
        assert first_line == expected_line
        assert remainder == "\n" + row["original"]  # blank line, then the original body
        # The stamp is estimator-neutral: re-measuring the stamped body agrees.
        assert estimate_card_size(row["title"], row["body"]).score == payload["score"]


def test_parked_oversized_card_is_gated_on_promotion(kanban_home):
    """An oversized card parked behind an open parent is diverted at promotion."""
    with kbc.connect() as conn:
        parent_id = kb.create_task(
            conn, title="Open parent", body="Small work", assignee="worker",
            created_by="test",
        )
        child_id = kb.create_task(
            conn, title="Oversized parked card", body=_oversized_body(),
            assignee="junior-dev", created_by="test", parents=[parent_id],
        )
        parked = kb.get_task(conn, child_id)
        parked_estimate = _latest_estimate(conn, child_id)

        assert parked.status == "todo"
        assert parked_estimate["decision"] == "advisory"
        assert parked_estimate["path"] == "creation_parked"
        assert parked_estimate["reason"] == "oversized"

        assert kb.complete_task(conn, parent_id, summary="parent finished")
        promoted = kb.recompute_ready(conn)

        gated = kb.get_task(conn, child_id)
        gated_estimate = _latest_estimate(conn, child_id)
        kinds = _event_kinds(conn, child_id)

    assert gated.status == "triage"
    assert gated_estimate["decision"] == "triage"
    assert gated_estimate["path"] == "promotion"
    assert gated_estimate["reason"] == "oversized"
    assert gated_estimate["score"] > gated_estimate["threshold"]
    # A diverted card is not a promotion: no event, not counted in the total.
    assert promoted == 0
    assert "promoted" not in kinds


def test_triage_entry_oversized_card_is_measured(kanban_home):
    """``triage=True`` creation records the estimate even though the status stays."""
    with kbc.connect() as conn:
        oversized_id = kb.create_task(
            conn, title="Oversized triage entry", body=_oversized_body(),
            assignee="junior-dev", created_by="test", triage=True,
        )
        small_id = kb.create_task(
            conn, title="Small triage entry", body="Just fix the bug",
            assignee="junior-dev", created_by="test", triage=True,
        )
        oversized_task = kb.get_task(conn, oversized_id)
        oversized_estimate = _latest_estimate(conn, oversized_id)
        small_task = kb.get_task(conn, small_id)
        small_estimate = _latest_estimate(conn, small_id)

    assert oversized_task.status == "triage"
    assert oversized_estimate["decision"] == "triage"
    assert oversized_estimate["path"] == "creation_triage"
    assert oversized_estimate["reason"] == "oversized"
    assert oversized_estimate["score"] > oversized_estimate["threshold"]

    assert small_task.status == "triage"
    assert small_estimate["decision"] == "triage"
    assert small_estimate["path"] == "creation_triage"
    assert small_estimate["reason"] == "fits"


def test_legacy_card_is_measured_but_not_diverted(kanban_home):
    """A pre-gate card (no estimate event) is measured and grandfathered to ready."""
    legacy_id = "t_legacytest01"
    with kbc.connect() as conn:
        parent_id = kb.create_task(
            conn, title="Open parent", body="Small work", assignee="worker",
            created_by="test",
        )
        # Pre-gate shape: row + link, no events at all.
        conn.execute(
            "INSERT INTO tasks (id, title, body, status, created_at, created_by) "
            "VALUES (?, ?, ?, 'todo', ?, 'legacy')",
            (legacy_id, "Legacy oversized card", _oversized_body(), 1_700_000_000),
        )
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, legacy_id),
        )
        assert _estimates(conn, legacy_id) == []

        # Unblock the dependency without going through recompute_ready first.
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent_id,))

        ok, error = kb.promote_task(conn, legacy_id, actor="tester")
        task = kb.get_task(conn, legacy_id)
        estimate = _latest_estimate(conn, legacy_id)

    assert ok and error is None
    assert task.status == "ready"
    assert estimate["decision"] == "allowed"
    assert estimate["reason"] == "legacy"
    assert estimate["score"] > estimate["threshold"]


def test_decomposed_epic_root_is_not_re_triaged(kanban_home):
    """An epic root that already decomposed wakes on ``ready`` instead of looping."""
    from hermes_cli.kanban_db_graph import decompose_triage_task

    with kbc.connect() as conn:
        parent_id = kb.create_task(
            conn, title="Gating parent", body="Small work", assignee="worker",
            created_by="test",
        )
        root_id = kb.create_task(
            conn, title="Epic root", body=_oversized_body(), assignee="orchestrator",
            created_by="test", parents=[parent_id],
        )
        # Parked oversized: advisory, not yet decided.
        assert kb.get_task(conn, root_id).status == "todo"
        assert _latest_estimate(conn, root_id)["decision"] == "advisory"

        # Operator sends it to triage, where it is decomposed into one child.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator",
            children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
            author="decomposer",
        )
        assert child_ids
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        # Close the gating parent without going through complete_task's own
        # recompute pass, so the explicit sweep below is what promotes the root.
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent_id,))
        promoted = kb.recompute_ready(conn)

        root = kb.get_task(conn, root_id)
        root_estimate = _latest_estimate(conn, root_id)

    assert root.status == "ready", (
        f"epic root must wake on ready, got {root.status!r}; a re-triage would "
        "strand it (a root decomposes once)"
    )
    assert root_estimate["decision"] == "allowed"
    assert root_estimate["path"] == "promotion"
    assert root_estimate["reason"] == "epic_root"
    assert promoted == 1


def test_binary_body_is_measured_not_skipped():
    """A BLOB body is decoded and measured instead of raising past the gate."""
    # Deliberately non-str input: one live card stores typeof(body)='blob'.
    estimate = estimate_card_size(
        "t", b"## Acceptance Criteria\n1. a\n2. b\n3. c\n",  # type: ignore[arg-type]
    )
    assert estimate.acceptance_criteria_count == 3
    assert estimate.deliverable_count == 1
    assert estimate.score == 10
