"""Cross-board resolution of prose ``t_<hex>`` references (#127641).

``complete_task`` flags summary/result ids that don't resolve in the completing
board's database as ``suspected_hallucinated_references``. On a multi-board
install that mislabels valid citations of another board's cards (91.9% of one
install's flags), so the phantom scan resolves same-board misses read-only
against the other boards first. Branch-name tokens (``fix/t_<hex>``) are not
card references at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def boards_home(tmp_path, monkeypatch):
    """Two boards beside ``default``; tests open per-board connections."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.create_board("alpha")
    kb.create_board("beta")
    return home


def _flagged(conn, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY created_at",
        (task_id, "suspected_hallucinated_references"),
    ).fetchall()
    return [json.loads(row[0]) for row in rows]


def test_same_board_reference_still_not_flagged(boards_home):
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker", assignee="ops")
        peer = kb.create_task(alpha, title="peer", assignee="ops")
        kb._flag_phantom_prose_refs(alpha, worker, None, f"finished after {peer}", None, [])
        assert _flagged(alpha, worker) == []


def test_cross_board_reference_is_not_phantom(boards_home):
    with kbc.connect(board="beta") as beta:
        real_id = kb.create_task(beta, title="real card on beta", assignee="ops")
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker on alpha", assignee="ops")
        kb._flag_phantom_prose_refs(
            alpha, worker, None, f"unblocked by parent {real_id} on the beta board", None, [],
        )
        assert _flagged(alpha, worker) == []


def test_unresolvable_on_any_board_still_flags(boards_home):
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker", assignee="ops")
        kb._flag_phantom_prose_refs(alpha, worker, None, "depends on t_00000000dead", None, [])
        events = _flagged(alpha, worker)
        assert len(events) == 1
        assert events[0]["phantom_refs"] == ["t_00000000dead"]
        assert "cross_board_refs" not in events[0]


def test_event_records_cross_board_resolutions(boards_home):
    with kbc.connect(board="beta") as beta:
        real_id = kb.create_task(beta, title="beta card", assignee="ops")
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker", assignee="ops")
        kb._flag_phantom_prose_refs(
            alpha, worker, None, f"unblocked by {real_id}; saw t_00000000cafe", None, [],
        )
        events = _flagged(alpha, worker)
        assert len(events) == 1
        assert events[0]["phantom_refs"] == ["t_00000000cafe"]
        assert events[0]["cross_board_refs"] == {real_id: "beta"}


def test_unreadable_other_board_database_does_not_crash(boards_home):
    # A corrupt sibling database must degrade to same-board semantics, not
    # break the completion path that the scan rides on.
    kb.kanban_db_path("beta").write_bytes(b"not a sqlite database" * 64)
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker", assignee="ops")
        kb._flag_phantom_prose_refs(alpha, worker, None, "saw t_00000000beef", None, [])
        events = _flagged(alpha, worker)
        assert [event["phantom_refs"] for event in events] == [["t_00000000beef"]]


def test_branch_name_tokens_are_not_card_references(boards_home):
    with kbc.connect(board="alpha") as alpha:
        worker = kb.create_task(alpha, title="worker", assignee="ops")
        kb._flag_phantom_prose_refs(
            alpha, worker, None,
            "landed via fix/t_66a9d5c0 and wt/t_ffd2508c; next up t_0123456789ab",
            None, [],
        )
        events = _flagged(alpha, worker)
        # The slash-prefixed branch/worktree tokens resolve to nothing yet must
        # not be flagged; only the bare prose reference is.
        assert [event["phantom_refs"] for event in events] == [["t_0123456789ab"]]
