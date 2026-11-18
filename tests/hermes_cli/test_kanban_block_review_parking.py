"""Parking a ``review`` card: block accepts it, unblock restores it.

Measured failure this pins (2026-09-20, registry-pin cards t_3fd73eba /
t_86d437a7): a card whose implementation is landed and approved but whose
``kanban_complete`` is structurally refused could not be parked.

* ``block_task`` only matched ``running``/``ready``, so a ``review`` card
  returned False (``cannot block <id>``) and the review dispatcher kept
  spawning reviewers — four rate-limited spawns in 22 minutes.
* Even after a manual DB edit to force ``blocked``, ``unblock_task`` sent the
  card to ``ready`` (a fresh *implementation* spawn) instead of back to
  ``review``, because ``block_task`` never recorded ``source_status=review``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _review_task(conn, title: str = "t") -> str:
    """Create a task and drive it through a real run into ``review``."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    ok = kb.request_review(
        conn, tid, summary="done", reviewer="reviewer",
        expected_run_id=claimed.current_run_id,
    )
    assert ok, f"request_review refused: {ok!r}"
    assert kb.get_task(conn, tid).status == "review"
    return tid


# ---------------------------------------------------------------------------
# (a) block accepts a review card
# ---------------------------------------------------------------------------


def test_block_accepts_a_review_card(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _review_task(conn)
        assert kb.block_task(
            conn, tid, reason="cannot complete: survivor guard", kind="capability",
        ) is True
        assert kb.get_task(conn, tid).status == "blocked"


def test_blocked_review_card_is_not_auto_recovered(kanban_home: Path) -> None:
    """Parking must actually stop dispatch, not just relabel the row."""
    with kb.connect_closing() as conn:
        tid = _review_task(conn)
        assert kb.block_task(conn, tid, reason="parked", kind="capability") is True
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"


def test_block_from_review_records_source_status(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _review_task(conn)
        kb.block_task(conn, tid, reason="parked", kind="capability")
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        assert blocked, "expected a blocked event"
        assert (blocked[-1].payload or {}).get("source_status") == "review"


# ---------------------------------------------------------------------------
# (b) unblock restores the PRE-block phase
# ---------------------------------------------------------------------------


def test_unblock_restores_review_not_ready(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _review_task(conn)
        assert kb.block_task(conn, tid, reason="parked", kind="capability") is True
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "review"


def test_unblock_from_a_running_block_still_lands_ready(kanban_home: Path) -> None:
    """Control: the implementation lane must keep its historical behaviour."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.block_task(conn, tid, reason="need input", kind="needs_input")
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
