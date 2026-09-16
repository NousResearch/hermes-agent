"""Cross-board ``t_<hex>`` citations must not read as fabricated references.

The post-completion prose scan looked the cited ids up in the completing board's
own ``tasks`` table, so a worker that correctly cited a real task living on
ANOTHER board — the sibling writer that caused the condition it was reporting —
was recorded as ``suspected_hallucinated_references`` (board ``hermes-growth``,
``task_events`` row 1066). Both cited ids were real. A citation is a mention, not
a claim of authorship: it resolves on any LIVE board.

The companion arms are the boundaries this must not cross: an id no board
resolves is still flagged, and ``created_cards`` — which IS a claim of
authorship — stays board-local and keeps blocking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized default board."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        )
    ]


@pytest.mark.parametrize("pinned", [False, True], ids=["clean-env", "worker-pinned"])
def test_cross_board_citation_is_not_a_phantom_reference(kanban_home, monkeypatch, pinned):
    """A real id from another board resolves; no accusation event is emitted.

    ``worker-pinned`` is the production shape: the dispatcher pins
    ``HERMES_KANBAN_DB`` to the completing worker's own board, and the ordinary
    board resolver then maps EVERY slug to that one file — so the walk has to
    open each board by its own path rather than through the pin.
    """
    default_db = str(kb.kanban_db_path())

    kb.create_board("other")
    with kbc.connect(board="other") as conn:
        other_id = kb.create_task(conn, title="sibling writer", assignee="alice")

    if pinned:
        monkeypatch.setenv("HERMES_KANBAN_DB", default_db)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reporter", assignee="alice")
        assert kb.complete_task(
            conn, tid,
            summary=(
                f"Handoff seen: {other_id} completed at 14:00:07 and this writer "
                "claimed one second later, so the condition is unchanged."
            ),
        )

    with kbc.connect() as conn:
        kinds = _event_kinds(conn, tid)

    assert "completed" in kinds
    assert "suspected_hallucinated_references" not in kinds


def test_unresolvable_reference_is_still_flagged(kanban_home):
    """Widening the lookup must not turn the scan into a no-op."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reporter", assignee="alice")
        assert kb.complete_task(conn, tid, summary="Blocked on t_deadbeef1234.")

    with kbc.connect() as conn:
        kinds = _event_kinds(conn, tid)

    assert "suspected_hallucinated_references" in kinds


def test_created_cards_gate_stays_board_local(kanban_home):
    """``created_cards`` claims authorship, so another board's id does not verify."""
    kb.create_board("other")
    with kbc.connect(board="other") as conn:
        other_id = kb.create_task(conn, title="not created by this worker", assignee="alice")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reporter", assignee="alice")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(conn, tid, summary="done", created_cards=[other_id])

        assert _event_kinds(conn, tid) == ["created", "completion_blocked_hallucination"]