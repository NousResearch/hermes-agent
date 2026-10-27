"""Promotion-path and decomposition coverage for the kanban size gate.

The creation-time gate is covered by ``test_kanban_size_gate.py``. This file
pins the contract gaps that pass closed:

1. ``decompose_triage_task`` cannot emit a child without recording the child's
   budget estimate, and the child's body leads with the budget line.
2. Every path that parks a card on ``ready`` runs the same decision
   (``gate_ready_entry``), so an oversized card parked in ``todo``/``blocked``
   behind an open parent cannot be promoted into the dispatch lane ungated.
3. ``triage=True`` creation is measured too, and a legacy (pre-gate) card is
   measured but grandfathered - forward-only, never retro-diverted.
4. A decomposed root stuck back in ``triage`` by the size gate is repaired by
   ``decompose_task`` (the verb) without calling the LLM, and the repair
   helper is a no-op on a non-triage root.
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


def _status(conn, task_id: str) -> str:
    task = kb.get_task(conn, task_id)
    assert task is not None, f"{task_id} is missing"
    return task.status


def _decompose_parked_root(conn) -> tuple[str, str, list[str]]:
    """Create a PARKED oversized root behind an open parent, decompose it.

    A root with no open parent lands on ``ready``, where the creation gate
    diverts an oversized card straight to ``triage`` (``decision=triage``) and
    the exemption under test never decides anything. Parking it behind an open
    parent keeps it in ``todo`` with ``decision=advisory`` — the one state in
    which ``gate_ready_entry`` can still divert an oversized card — and an
    operator then sends it to triage, where it decomposes into one child.

    The returned parent is the root's *ordinary* parent, linked exactly like the
    decomposed child (``parent -> root``), so it doubles as the proof that
    ``task_links`` cannot answer lineage.

    Returns ``(parent_id, root_id, child_ids)``; the root lands in ``todo``.
    """
    from hermes_cli.kanban_db_graph import decompose_triage_task

    parent_id = kb.create_task(
        conn, title="Gating parent", body="Small work", assignee="worker",
        created_by="test",
    )
    root_id = kb.create_task(
        conn, title="Epic root", body=_oversized_body(), assignee="orchestrator",
        created_by="test", parents=[parent_id],
    )
    assert _status(conn, root_id) == "todo"
    assert _latest_estimate(conn, root_id)["decision"] == "advisory"
    conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
    child_ids = decompose_triage_task(
        conn, root_id, root_assignee="orchestrator",
        children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
        author="decomposer",
    )
    assert child_ids
    return parent_id, root_id, child_ids


def test_divert_promotion_to_triage_returns_false_for_decomposed_root(kanban_home):
    """A decomposed root is never re-diverted to triage, even though it is oversized.

    Drives the diverting path (oversized ``decision=advisory`` -> triage ->
    decompose) so the ``_already_decomposed`` exemption in ``gate_ready_entry``
    is genuinely what decides: deleting it re-diverts the root here and re-parks
    it in the terminal ``triage`` state, which fails this test.
    """
    from hermes_cli.kanban_size_gate import divert_promotion_to_triage

    with kbc.connect() as conn:
        _, root_id, _ = _decompose_parked_root(conn)
        # Back in `todo`, still undecided: nothing but the exemption can keep
        # the gate from diverting an oversized card in this state.
        assert _status(conn, root_id) == "todo"
        assert _latest_estimate(conn, root_id)["decision"] == "advisory"

        diverted = divert_promotion_to_triage(conn, root_id)
        root_after = _status(conn, root_id)
        estimate_after = _latest_estimate(conn, root_id)

    assert diverted is False, "decomposed root must not be diverted to triage"
    assert root_after == "todo", "a re-divert would re-park the root in triage"
    assert estimate_after["decision"] == "allowed"
    assert estimate_after["reason"] == "epic_root"


def test_promote_gate_verdict_returns_none_for_decomposed_root(kanban_home):
    """A decomposed root can be manually promoted instead of being refused for size.

    Same advisory path as the divert test: the manual-promotion gate must reach
    the ``_already_decomposed`` exemption and allow the card. Deleting the
    exemption makes ``promote_gate_verdict`` return the oversized refusal and
    re-park the root, so this test dies with the guard.
    """
    from hermes_cli.kanban_size_gate import promote_gate_verdict

    with kbc.connect() as conn:
        _, root_id, _ = _decompose_parked_root(conn)
        assert _latest_estimate(conn, root_id)["decision"] == "advisory"

        verdict = promote_gate_verdict(conn, root_id)
        root_after = _status(conn, root_id)
        estimate_after = _latest_estimate(conn, root_id)

    assert verdict is None, f"decomposed root must be promotable, got {verdict!r}"
    assert root_after == "todo", "a refusal would re-park the root in triage"
    assert estimate_after["decision"] == "allowed"
    assert estimate_after["reason"] == "epic_root"


def test_repair_stuck_root_moves_to_ready_when_children_done(kanban_home):
    """`repair_stuck_decomposed_root` on a triage card with a decomposed event repairs it."""
    from hermes_cli.kanban_db_graph import decompose_triage_task, repair_stuck_decomposed_root

    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn, title="Epic root", body=_oversized_body(), assignee="orchestrator",
            created_by="test", triage=True,
        )
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator",
            children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
            author="decomposer",
        )
        assert child_ids
        # Complete the child.
        assert kb.complete_task(conn, child_ids[0], summary="done")
        # Force root back to triage to simulate the bug.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
        assert kb.get_task(conn, root_id).status == "triage"
        # Now call the repair function.
        new_status = repair_stuck_decomposed_root(conn, root_id)
        root_after = kb.get_task(conn, root_id)
        events = _event_kinds(conn, root_id)

    assert new_status == "ready", "all children done -> root should be ready"
    assert root_after.status == "ready"
    assert "repaired" in events
    # No duplicate decomposed event.
    assert events.count("decomposed") == 1


def test_repair_stuck_root_moves_to_todo_with_open_children(kanban_home):
    """Repair moves to `todo` when children are still open."""
    from hermes_cli.kanban_db_graph import decompose_triage_task, repair_stuck_decomposed_root

    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn, title="Epic root", body=_oversized_body(), assignee="orchestrator",
            created_by="test", triage=True,
        )
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator",
            children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
            author="decomposer",
        )
        assert child_ids
        # Child is in ready (auto-promoted by decompose_triage_task).
        child_status = kb.get_task(conn, child_ids[0]).status
        assert child_status in ("todo", "ready"), f"child should be open, got {child_status}"
        # Force root back to triage.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
        # Repair should move to todo (waiting on children).
        new_status = repair_stuck_decomposed_root(conn, root_id)
        root_after = kb.get_task(conn, root_id)

    assert new_status == "todo", "open children -> root should be todo"
    assert root_after.status == "todo"


def _parent_ids(conn, task_id: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ?", (task_id,),
    )]


def test_decompose_task_repairs_stuck_root_before_the_llm(kanban_home, monkeypatch):
    """`decompose_task` on a stuck decomposed root repairs it without calling the LLM."""
    from hermes_cli import kanban_decompose as decomp
    from hermes_cli.kanban_db_graph import decompose_triage_task

    def _boom(*args, **kwargs):
        raise AssertionError("repair path must not call the aux LLM")

    monkeypatch.setattr(decomp, "_call_aux", _boom)

    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn, title="Epic root", body=_oversized_body(),
            assignee="orchestrator", created_by="test", triage=True,
        )
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator",
            children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
            author="decomposer",
        )
        assert child_ids
        assert kb.complete_task(conn, child_ids[0], summary="done")
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
        parents_before = _parent_ids(conn, root_id)

    outcome = decomp.decompose_task(root_id, author="test")

    with kbc.connect() as conn:
        root_after = kb.get_task(conn, root_id)
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (root_id,))]
        parents_after = _parent_ids(conn, root_id)

    assert outcome.ok is True, outcome.reason
    assert outcome.mode == "repair"
    assert outcome.new_status == "ready"
    assert root_after.status == "ready"
    assert kinds.count("decomposed") == 1
    assert kinds.count("repaired") == 1
    assert parents_after == parents_before   # no new children


def test_repair_stuck_root_is_noop_on_non_triage_card(kanban_home):
    """`repair_stuck_decomposed_root` returns None and leaves status unchanged on a non-triage root."""
    from hermes_cli.kanban_db_graph import decompose_triage_task, repair_stuck_decomposed_root

    with kbc.connect() as conn:
        root_id = kb.create_task(
            conn, title="Epic root", body=_oversized_body(), assignee="orchestrator",
            created_by="test", triage=True,
        )
        child_ids = decompose_triage_task(
            conn, root_id, root_assignee="orchestrator",
            children=[{"title": "Slice", "body": "Small slice", "assignee": "junior-dev"}],
            author="decomposer",
        )
        assert child_ids
        # Root is now in todo (auto-promoted by decompose_triage_task).
        root_before = kb.get_task(conn, root_id)
        assert root_before.status == "todo"
        events_before = _event_kinds(conn, root_id)
        # Call repair on a non-triage card.
        result = repair_stuck_decomposed_root(conn, root_id)
        root_after = kb.get_task(conn, root_id)
        events_after = _event_kinds(conn, root_id)

    assert result is None, "repair should return None for non-triage card"
    assert root_after.status == "todo", "status should be unchanged"
    assert events_after == events_before, "no repaired event should be appended"


def _park_by_block_loop_brake(conn, root_id: str) -> None:
    """Drive the real block-loop brake until it routes ``root_id`` to ``triage``.

    ``_route_block`` counts unblock-loop recurrences: block -> blocked,
    unblock -> ready, block again reaches ``BLOCK_RECURRENCE_LIMIT`` and parks
    the card in ``triage`` with a ``block_loop_detected`` event. No raw status
    writes, so the event stream is exactly what production produces.
    """
    assert kb.block_task(conn, root_id, reason="needs a human", kind="needs_input")
    assert _status(conn, root_id) == "blocked"
    assert kb.unblock_task(conn, root_id)
    assert _status(conn, root_id) == "ready"
    assert kb.block_task(conn, root_id, reason="needs a human again", kind="needs_input")


def test_repair_leaves_a_loop_braked_root_in_triage(kanban_home):
    """A decomposed root parked by the block-loop brake is NOT repaired.

    The brake parks a card in ``triage`` for a human. Writing ``ready`` there
    bypasses ``recompute_ready``'s sticky-block check and failure breaker, so the
    card would block -> triage -> repaired -> ready -> block ... one worker spawn
    per dispatcher tick, forever, with no human in the loop.
    """
    from hermes_cli.kanban_db_graph import (
        repair_stuck_decomposed_root, triage_park_cause,
    )

    with kbc.connect() as conn:
        parent_id, root_id, child_ids = _decompose_parked_root(conn)
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        # Closing the ordinary parent is what lets recompute_ready release the
        # root; the brake only fires from ready.
        assert kb.complete_task(conn, parent_id, summary="parent done")
        assert _status(conn, root_id) == "ready"

        _park_by_block_loop_brake(conn, root_id)
        assert _status(conn, root_id) == "triage", (
            "the second consecutive block must route the card to triage"
        )
        assert triage_park_cause(conn, root_id) == "block_loop_detected"

        new_status = repair_stuck_decomposed_root(conn, root_id)
        status_after = _status(conn, root_id)
        kinds_after = _event_kinds(conn, root_id)

    assert new_status is None, "the brake parked this card for a human, not for repair"
    assert status_after == "triage", "the card must stay parked for a human"
    assert kinds_after.count("repaired") == 0, "no repaired event for a braked card"
    assert kinds_after.count("decomposed") == 1


def test_decompose_task_declines_a_loop_braked_root_without_the_llm(kanban_home, monkeypatch):
    """The auto-decompose sweep must not burn an LLM call on a braked root.

    A decomposed root can never fan out a second time, and the sweep runs on
    every tick, so the declined repair has to surface as a typed failure instead
    of falling through to the aux model.
    """
    from hermes_cli import kanban_decompose as decomp

    def _boom(*args, **kwargs):
        raise AssertionError("a braked decomposed root must not reach the aux LLM")

    monkeypatch.setattr(decomp, "_call_aux", _boom)

    with kbc.connect() as conn:
        parent_id, root_id, child_ids = _decompose_parked_root(conn)
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        assert kb.complete_task(conn, parent_id, summary="parent done")
        _park_by_block_loop_brake(conn, root_id)

    outcome = decomp.decompose_task(root_id, author="test")

    with kbc.connect() as conn:
        kinds_after = _event_kinds(conn, root_id)
        status_after = _status(conn, root_id)

    assert outcome.ok is False
    assert "already decomposed" in outcome.reason
    assert "block_loop_detected" in outcome.reason
    assert status_after == "triage"
    assert kinds_after.count("repaired") == 0


def test_repair_is_a_compare_and_set_against_a_concurrent_move(kanban_home, monkeypatch):
    """A move that lands between the repair's read and its write is never overwritten.

    ``archive_task`` accepts ``triage``, so the row can leave triage while the
    repair is deciding. A read-then-blind-write resurrects the card; the repair
    must lose the race, return None, and leave the winner's status alone.
    """
    import contextlib

    from hermes_cli import kanban_db as kbmod
    from hermes_cli.kanban_db_graph import repair_stuck_decomposed_root

    with kbc.connect() as conn:
        parent_id, root_id, child_ids = _decompose_parked_root(conn)
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        assert kb.complete_task(conn, parent_id, summary="parent done")
        # Parked in triage, so the repair has a row to win or lose.
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))

    real_write_txn = kbmod.write_txn
    raced = {"fired": False}

    @contextlib.contextmanager
    def racing_write_txn(conn):
        if not raced["fired"]:
            raced["fired"] = True
            # The concurrent archive commits after the repair read 'triage'.
            with kbc.connect() as other:
                other.execute(
                    "UPDATE tasks SET status = 'archived' WHERE id = ?", (root_id,),
                )
        with real_write_txn(conn):
            yield

    monkeypatch.setattr(kbmod, "write_txn", racing_write_txn)

    with kbc.connect() as conn:
        result = repair_stuck_decomposed_root(conn, root_id)
        status_after = _status(conn, root_id)
        kinds_after = _event_kinds(conn, root_id)

    assert raced["fired"] is True, "the race hook never ran"
    assert result is None, "a lost race must not report a repaired status"
    assert status_after == "archived", "the concurrent move must win"
    assert kinds_after.count("repaired") == 0


def test_repair_reads_lineage_from_the_decomposed_payload(kanban_home):
    """Only the decomposed payload's children are lineage, not every link.

    ``decompose_triage_task`` links a child as ``child -> root`` and an ordinary
    parent is stored the same way, so deriving the child set from ``task_links``
    counts an unrelated open parent as a child and lands the root on ``todo``
    even though its only decomposed child is done.
    """
    from hermes_cli.kanban_db_graph import repair_stuck_decomposed_root

    with kbc.connect() as conn:
        parent_id, root_id, child_ids = _decompose_parked_root(conn)
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        # The real parent is still open and is linked exactly like the child.
        assert _status(conn, parent_id) != "done"
        assert sorted(_parent_ids(conn, root_id)) == sorted([parent_id, child_ids[0]])

        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
        new_status = repair_stuck_decomposed_root(conn, root_id)
        status_after = _status(conn, root_id)
        repaired = [
            json.loads(row["payload"]) for row in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'repaired'",
                (root_id,),
            )
        ]

    assert new_status == "ready", (
        "the payload names one child and it is done; the open parent is not lineage"
    )
    assert status_after == "ready"
    assert len(repaired) == 1
    assert repaired[0]["children_count"] == 1
    assert repaired[0]["all_children_done"] is True


def test_repair_survives_a_malformed_decomposed_payload(kanban_home):
    """An unreadable ``child_ids`` payload must not crash or strand the card.

    Lineage cannot be read, so the repair cannot know the children are done: it
    lands on ``todo`` — conservative, never ``ready`` — instead of raising or
    leaving the card in the terminal ``triage`` state.
    """
    from hermes_cli.kanban_db_graph import repair_stuck_decomposed_root

    corruptions = (
        {"root_assignee": "orchestrator"},          # child_ids missing
        {"child_ids": "t_abc"},                     # wrong type
        {"child_ids": [None, 42]},                  # unusable entries
        "not-json-at-all",                          # unparseable payload
    )

    with kbc.connect() as conn:
        _, root_id, child_ids = _decompose_parked_root(conn)
        assert kb.complete_task(conn, child_ids[0], summary="slice done")
        seen = []
        for payload in corruptions:
            stored = payload if isinstance(payload, str) else json.dumps(payload)
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE task_id = ? AND kind = 'decomposed'",
                (stored, root_id),
            )
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root_id,))
            seen.append((repair_stuck_decomposed_root(conn, root_id), _status(conn, root_id)))

    for payload, (new_status, status_after) in zip(corruptions, seen):
        assert new_status == "todo", f"{payload!r} must not release the root"
        assert status_after == "todo", f"{payload!r} must leave triage"
