"""The specify/decompose sweep must not resurrect cards the unblock-loop
breaker escalated into ``triage``.

Observed loop: a worker called ``kanban_block(kind='capability')`` because the
task needed a live human action. On the second same-cause block ``block_task``
hit ``BLOCK_RECURRENCE_LIMIT`` and routed the card to ``triage`` — deliberately,
to force a human decision. But ``triage`` is also the auto-decomposer's input
queue, so the gateway's auto-decompose tick re-specified the card and promoted
it back to ``todo`` about a minute later, the dispatcher spawned a worker, and
the worker blocked again for the same reason. Every cycle burned an aux-LLM
call, a worker run and a near-duplicate kanban notification.

The fix filters escalated cards out of the *sweep* listings only; specifying or
decomposing one by id is still allowed, because that is the human-in-the-loop
decision the breaker was asking for. ``promote_task`` accepts ``triage`` so a
human has a cheap door out that does not cost an aux-LLM rewrite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _escalate_via_block_loop(conn, *, kind: str = "capability") -> str:
    """Drive a task through block -> unblock -> same-cause block, i.e. the exact
    path that trips the loop breaker, and return its id."""
    tid = kb.create_task(conn, title="needs a live human click", assignee="alice")
    for _ in range(kb.BLOCK_RECURRENCE_LIMIT):
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="needs a human to click the consent screen",
            kind=kind,
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        if kb.get_task(conn, tid).status == "blocked":
            kb.unblock_task(conn, tid)
    return tid


def test_block_loop_escalation_lands_in_triage(kanban_home: Path) -> None:
    """Precondition for everything below: the breaker really does park the
    card in ``triage`` with its block state intact."""
    with kb.connect() as conn:
        tid = _escalate_via_block_loop(conn)
        task = kb.get_task(conn, tid)
    assert task.status == "triage"
    assert task.block_kind == "capability"
    assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
    assert kb.awaiting_human_after_block_loop(task)


@pytest.mark.parametrize("module", [spec, decomp], ids=["specify", "decompose"])
def test_sweep_skips_block_loop_escalations(kanban_home: Path, module) -> None:
    """The ``--all`` / gateway sweep must not pick up an escalated card."""
    with kb.connect() as conn:
        escalated = _escalate_via_block_loop(conn)
        fresh = kb.create_task(conn, title="a rough idea", triage=True)

    ids = module.list_triage_ids()
    assert fresh in ids, "a hand-dropped triage idea must still be swept"
    assert escalated not in ids, (
        "card escalated by the unblock-loop breaker must wait for a human, "
        "not be re-specified back into the worker pool"
    )


def test_dependency_blocks_are_not_treated_as_human_escalations(
    kanban_home: Path,
) -> None:
    """``dependency`` blocks never reach triage via the breaker; a card that
    merely carries that block_kind must stay sweepable."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="waits on a sibling", triage=True)
        conn.execute(
            "UPDATE tasks SET block_kind = 'dependency', block_recurrences = ? "
            "WHERE id = ?",
            (kb.BLOCK_RECURRENCE_LIMIT + 1, tid),
        )
        conn.commit()
        assert not kb.awaiting_human_after_block_loop(kb.get_task(conn, tid))

    assert tid in spec.list_triage_ids()
    assert tid in decomp.list_triage_ids()


def test_escalated_card_can_still_be_specified_by_id(kanban_home: Path) -> None:
    """The guard is sweep-only: a human acting on the card by id still gets the
    normal triage -> todo promotion."""
    with kb.connect() as conn:
        tid = _escalate_via_block_loop(conn)
        assert kb.specify_triage_task(
            conn, tid, body="human decided: park it, I will click myself", author="me",
        )
        # todo, or ready once the inline recompute_ready pass runs (no parents).
        assert kb.get_task(conn, tid).status in {"todo", "ready"}


def test_promote_is_the_humans_door_out_of_an_escalation(kanban_home: Path) -> None:
    """Closing the sweep would leave the escalated card with no cheap way back:
    the sweep skips it, ``unblock`` only applies to blocked/scheduled, and
    specify costs an aux-LLM rewrite of text that is already correct.
    ``promote`` is that door, and it records who opened it."""
    with kb.connect() as conn:
        tid = _escalate_via_block_loop(conn)
        assert not kb.unblock_task(conn, tid), "unblock must not touch triage"

        ok, err = kb.promote_task(conn, tid, actor="me", reason="clicked it myself")
        assert (ok, err) == (True, None)
        assert kb.get_task(conn, tid).status == "ready"

        ev = [e for e in kb.list_events(conn, tid) if e.kind == "promoted_manual"][-1]
        assert ev.payload["from_status"] == "triage"
        assert ev.payload["released_escalation"] is True


def test_promote_does_not_reset_the_recurrence_counter(kanban_home: Path) -> None:
    """A human saying "go" must not hand the card a fresh loop budget: if the
    worker re-blocks for the same human reason it goes straight back to triage
    instead of burning another unblock/re-block cycle."""
    with kb.connect() as conn:
        tid = _escalate_via_block_loop(conn)
        assert kb.promote_task(conn, tid, actor="me")[0]
        after = kb.get_task(conn, tid)
        assert after.block_kind == "capability"
        assert after.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT

        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid, reason="still needs the same live click", kind="capability",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "triage"


def test_promote_still_refuses_terminal_and_unknown_states(kanban_home: Path) -> None:
    """Widening promote to ``triage`` must not make it a universal status
    setter."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already finished", assignee="alice")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="done")
        ok, err = kb.promote_task(conn, tid, actor="me")
        assert not ok and "promote only applies" in (err or "")

        ok, err = kb.promote_task(conn, "t_nope", actor="me")
        assert not ok and "not found" in (err or "")
