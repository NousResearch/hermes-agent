"""Manual lane order for kanban cards (``list --sort manual`` + ``kanban reorder``).

#88640: users could only influence card position with the coarse, write-once
``priority`` integer, which is shared across every lane. These tests pin the two
invariants of the per-lane position: ``order_by="manual"`` reads
``tasks.sort_order`` (and nothing else changes), and ``reorder_task`` moves one
card relative to a neighbour in the same lane without renumbering on every move.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _titles(conn, **kw) -> list[str]:
    return [t.title for t in kb.list_tasks(conn, **kw)]


def _stamp_creation_order(conn, ids) -> None:
    """Give ``ids`` distinct ``created_at`` seconds, in the order passed.

    ``create_task`` stamps ``created_at`` at second resolution, so cards created
    inside the same second tie there and fall back to a random id; stamping the
    intended creation order keeps the order assertions deterministic instead of
    occasionally red.
    """
    for created_at, task_id in enumerate(ids, start=1_700_000_000):
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (created_at, task_id))


def test_manual_order_is_per_lane_and_leaves_the_default_view_alone(kanban_home):
    """``--sort manual`` follows ``sort_order`` within a lane; the default stays creation order."""
    with kbc.connect_closing() as conn:
        created = {name: kb.create_task(conn, title=name) for name in ("alpha", "bravo", "charlie")}
        parked = kb.create_task(conn, title="parked", triage=True)  # a second lane
        _stamp_creation_order(conn, [created["alpha"], created["bravo"], created["charlie"], parked])

        # Reverse the ready lane: charlie first, bravo last.
        assert kb.reorder_task(conn, created["charlie"], top=True)
        assert kb.reorder_task(conn, created["bravo"], bottom=True)

        manual = _titles(conn, status="ready", order_by="manual")
        default = _titles(conn)
        manual_triage = _titles(conn, status="triage", order_by="manual")
        parked_order = kb.get_task(conn, parked).sort_order

    assert manual == ["charlie", "alpha", "bravo"]
    # Untouched views: default sort is still priority DESC, created_at ASC, and
    # a lane nobody reordered keeps its reading order.
    assert default == ["alpha", "bravo", "charlie", "parked"]
    assert manual_triage == ["parked"]
    assert parked_order == 0


def test_reorder_task_places_one_card_and_only_renumbers_when_a_gap_is_gone(kanban_home):
    """A move targets a same-lane neighbour; the lane is renumbered only when no midpoint is left."""
    with kbc.connect_closing() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c")
        triaged = kb.create_task(conn, title="parked", triage=True)
        _stamp_creation_order(conn, [a, b, c, triaged])

        # Every row starts at sort_order 0, so the first between-neighbours move
        # has no integer midpoint: the lane is renumbered (0, 2, 4 …) around the
        # new position, which is what leaves room for the next move.
        assert kb.reorder_task(conn, c, before_id=b)
        assert _titles(conn, status="ready", order_by="manual") == ["a", "c", "b"]

        # The spacing left by that renumber means this second move needs no
        # rewrite at all — 'b' takes the free midpoint between a (0) and c (2).
        assert kb.reorder_task(conn, b, before_id=c)
        assert _titles(conn, status="ready", order_by="manual") == ["a", "b", "c"]

        with pytest.raises(ValueError):
            kb.reorder_task(conn, a, after_id=triaged)  # manual order is per lane
        with pytest.raises(ValueError):
            kb.reorder_task(conn, a, before_id=a)
        with pytest.raises(ValueError):
            kb.reorder_task(conn, a, top=True, bottom=True)

        assert _titles(conn, status="ready", order_by="manual") == ["a", "b", "c"]
        assert kb.reorder_task(conn, "t_missing", top=True) is False


def test_kanban_cli_reorder_round_trip(kanban_home):
    """``hermes kanban reorder`` writes the position ``kanban list --sort manual`` reads."""
    with kbc.connect_closing() as conn:
        first = kb.create_task(conn, title="first")
        second = kb.create_task(conn, title="second")
        _stamp_creation_order(conn, [first, second])

    out = kc.run_slash(f"reorder {second} --top")
    assert "Moved" in out

    listed = json.loads(kc.run_slash("list --sort manual --json"))
    assert [t["title"] for t in listed] == ["second", "first"]
    # The default listing (what the board shows unless asked otherwise) is untouched.
    assert [t["title"] for t in json.loads(kc.run_slash("list --json"))] == ["first", "second"]
