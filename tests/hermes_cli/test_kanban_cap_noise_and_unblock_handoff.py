"""Dispatcher cap noise and the unblock handoff.

Cap noise: cards held back only by ``kanban.max_in_progress_per_profile`` are "busy,
retry later", not "stuck" — ``ready_nonempty`` must not count them, and the
suppression line must name them.

Unblock handoff: a human ``unblocked`` newer than a PR-quoting comment is a handoff (the
``active_pr`` respawn guard must not hold the card), and reassigning an unblocked
card to a different profile restarts the block-loop count instead of tripping it.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PR_COMMENT = "Opened https://github.com/example/repo/pull/77 for review."


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, all_assignees_spawnable) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _run(conn, assignee: str) -> str:
    """A task for ``assignee`` driven to ``running``."""
    tid = kb.create_task(conn, title=f"run {assignee}", assignee=assignee)
    assert kb.claim_task(conn, tid, claimer=assignee) is not None
    return tid


def _dispatcher(monkeypatch, cap):
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, cap)
    monkeypatch.setattr(kwd, "_board_slugs", lambda _kb: ["default"])
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: False)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


# --- cap noise -------------------------------------------------------------------


def test_cap_held_ready_card_is_not_spawnable_but_other_profile_is(kanban_home):
    with kbc.connect() as conn:
        _run(conn, "alpha")
        kb.create_task(conn, title="held", assignee="alpha")
        assert kbd.has_spawnable_ready(conn) is True  # historic, cap-blind answer
        assert kbd.has_spawnable_ready(conn, per_profile_cap=1) is False
        assert kbd.has_spawnable_ready(conn, per_profile_cap=2) is True  # below the cap
        kb.create_task(conn, title="free", assignee="beta")
        assert kbd.has_spawnable_ready(conn, per_profile_cap=1) is True


def test_ready_nonempty_ignores_cap_held_cards(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        _run(conn, "alpha")
        kb.create_task(conn, title="held", assignee="alpha")
    assert _dispatcher(monkeypatch, cap=1).ready_nonempty() is False
    assert _dispatcher(monkeypatch, cap=None).ready_nonempty() is True
    with kbc.connect() as conn:
        kb.create_task(conn, title="free", assignee="beta")
    assert _dispatcher(monkeypatch, cap=1).ready_nonempty() is True


def test_describe_suppression_names_per_profile_capped():
    capped = kbd.DispatchResult()
    capped.skipped_per_profile_capped = [("t_1", "alpha", 1), ("t_2", "alpha", 1)]
    capped.respawn_guarded = [("t_3", "active_pr")]
    assert kbd.describe_suppression([capped]) == "active_pr=1, per_profile_capped=2"
    assert kbd.describe_suppression([None, kbd.DispatchResult()]) == ""


# --- unblock handoff -------------------------------------------------------------------


def _stamp(conn, table: str, row_filter: str, args: tuple, ts: int) -> None:
    with kb.write_txn(conn):
        conn.execute(f"UPDATE {table} SET created_at = ? WHERE {row_filter}", (ts, *args))


def _blocked_with_pr_comment(conn) -> str:
    tid = _run(conn, "worker")
    assert kb.block_task(conn, tid, reason="needs yes", kind="needs_input")
    kb.add_comment(conn, tid, author="default", body=PR_COMMENT)
    return tid


def test_unblock_after_pr_comment_is_a_handoff(kanban_home):
    with kbc.connect() as conn:
        tid = _blocked_with_pr_comment(conn)
        assert kb.unblock_task(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_unblock_before_pr_comment_stays_guarded(kanban_home):
    with kbc.connect() as conn:
        tid = _blocked_with_pr_comment(conn)
        assert kb.unblock_task(conn, tid)
        now = int(time.time())
        _stamp(conn, "task_events", "task_id = ? AND kind = 'unblocked'", (tid,), now - 100)
        _stamp(conn, "task_comments", "task_id = ?", (tid,), now - 50)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_pr_comment_without_unblock_stays_guarded(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_comment(conn, tid, author="worker", body=PR_COMMENT)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def _block_unblock_reassign_block(conn, new_owner: str):
    tid = _run(conn, "alpha")
    assert kb.block_task(conn, tid, reason="needs yes", kind="needs_input")
    assert kb.get_task(conn, tid).block_recurrences == 1  # type: ignore[union-attr]
    assert kb.unblock_task(conn, tid)
    assert kb.assign_task(conn, tid, new_owner)
    assert kb.claim_task(conn, tid, claimer=new_owner) is not None
    assert kb.block_task(conn, tid, reason="still needs yes", kind="needs_input")
    task = kb.get_task(conn, tid)
    assert task is not None
    return task


def test_reassign_after_unblock_counts_as_one_block(kanban_home):
    with kbc.connect() as conn:
        task = _block_unblock_reassign_block(conn, "beta")
        assert (task.status, task.block_recurrences) == ("blocked", 1)
        assert "block_loop_detected" not in [e.kind for e in kb.list_events(conn, task.id)]


def test_same_owner_reblock_after_unblock_still_trips_the_loop_breaker(kanban_home):
    with kbc.connect() as conn:
        task = _block_unblock_reassign_block(conn, "alpha")  # no-op reassign
        assert (task.status, task.block_recurrences) == ("triage", 2)
