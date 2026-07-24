"""Tests for kanban_board_render.py (pure render layer, no DB)."""

from hermes_cli import kanban_board_render as r
from hermes_cli.kanban_db import Task


def _make_task(task_id: str, title: str, status: str, assignee: str = None) -> Task:
    """Build a Task with all required (no-default) fields, matching the
    live dataclass field set. Everything else uses dataclass defaults."""
    return Task(
        id=task_id,
        title=title,
        body="",
        assignee=assignee,
        status=status,
        priority=0,
        created_by="test",
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )


def test_render_returns_rich_table():
    result = r.render_board([_make_task("t_001", "test task", "todo")], board_slug="default")
    assert "Table" in type(result).__name__


def test_render_column_count():
    tasks = [
        _make_task("t_001", "todo task", "todo"),
        _make_task("t_002", "ready task", "ready"),
        _make_task("t_003", "running task", "running"),
    ]
    out = r._render_to_string(r.render_board(tasks, board_slug="default"))
    assert "todo" in out.lower()
    assert "ready" in out.lower()
    assert "running" in out.lower()
    # done/archived collapsed by default
    assert "archived" not in out.lower()


def test_render_includes_triage_and_review_columns():
    """triage and review are valid statuses and must never be dropped."""
    out = r._render_to_string(r.render_board([], board_slug="default", other_board_count=0))
    # empty board short-circuits to a placeholder, so check a populated board instead
    tasks = [
        _make_task("t_001", "needs triage", "triage"),
        _make_task("t_002", "in review", "review"),
    ]
    out = r._render_to_string(r.render_board(tasks, board_slug="default"))
    assert "triage" in out.lower()
    assert "review" in out.lower()
    assert "needs triage" in out.lower()
    assert "in review" in out.lower()


def test_render_groups_tasks_by_status():
    tasks = [
        _make_task("t_001", "first", "todo"),
        _make_task("t_002", "second", "todo"),
        _make_task("t_003", "runner", "running"),
    ]
    out = r._render_to_string(r.render_board(tasks, board_slug="default"))
    assert "first" in out
    assert "second" in out
    assert "runner" in out


def test_render_collapses_excess():
    tasks = [_make_task(f"t_{i:03d}", f"task {i}", "todo") for i in range(10)]
    out = r._render_to_string(r.render_board(tasks, board_slug="default", limit=5))
    assert "(+5 hidden)" in out.lower()


def test_render_all_statuses():
    tasks = [
        _make_task("t_001", "d", "done"),
        _make_task("t_002", "a", "archived"),
    ]
    out = r._render_to_string(r.render_board(tasks, board_slug="default", show_all=True))
    assert "done" in out.lower()
    assert "archived" in out.lower()


def test_render_empty_board():
    out = r._render_to_string(r.render_board([], board_slug="default"))
    assert "(no tasks)" in out.lower()


def test_render_shows_multi_board_hint():
    out = r._render_to_string(r.render_board([], board_slug="default", other_board_count=2))
    assert "other board" in out.lower()


def test_render_no_hint_for_single_board():
    out = r._render_to_string(r.render_board([], board_slug="default", other_board_count=0))
    assert "other board" not in out.lower()


# ── Stacked (narrow-terminal) layout ──────────────────────────────────


def test_should_stack_narrow_is_true():
    assert r.should_stack(50, show_all=False) is True


def test_should_stack_wide_is_false():
    assert r.should_stack(200, show_all=False) is False


def test_should_stack_show_all_needs_more_width():
    # 9 columns need more room than 7; a width that fits 7 may not fit 9.
    assert r.should_stack(120, show_all=True) is True
    assert r.should_stack(120, show_all=False) is False


def test_stacked_returns_group():
    result = r.render_board_stacked([_make_task("t_001", "x", "todo")], board_slug="default")
    assert "Group" in type(result).__name__


def test_stacked_lists_tasks_under_status_headers():
    tasks = [
        _make_task("t_001", "first todo", "todo"),
        _make_task("t_002", "the runner", "running"),
    ]
    out = r._render_to_string(r.render_board_stacked(tasks, board_slug="default"))
    assert "todo" in out.lower()
    assert "running" in out.lower()
    assert "first todo" in out.lower()
    assert "the runner" in out.lower()


def test_stacked_omits_empty_sections():
    out = r._render_to_string(
        r.render_board_stacked([_make_task("t_001", "only todo", "todo")], board_slug="default")
    )
    assert "todo" in out.lower()
    # ready/running/blocked sections have no tasks → not printed
    assert "ready" not in out.lower()
    assert "blocked" not in out.lower()


def test_stacked_collapses_excess():
    tasks = [_make_task(f"t_{i:03d}", f"task {i}", "todo") for i in range(10)]
    out = r._render_to_string(r.render_board_stacked(tasks, board_slug="default", limit=5))
    assert "(+5 hidden)" in out.lower()


def test_stacked_show_all_includes_done():
    tasks = [_make_task("t_001", "finished", "done")]
    out = r._render_to_string(
        r.render_board_stacked(tasks, board_slug="default", show_all=True)
    )
    assert "done" in out.lower()
    assert "finished" in out.lower()


def test_stacked_done_hidden_by_default():
    tasks = [_make_task("t_001", "finished", "done")]
    out = r._render_to_string(r.render_board_stacked(tasks, board_slug="default"))
    # done is collapsed; with no other tasks the board reads as empty
    assert "finished" not in out.lower()
    assert "(no tasks)" in out.lower()


def test_stacked_empty_board():
    out = r._render_to_string(r.render_board_stacked([], board_slug="default"))
    assert "(no tasks)" in out.lower()


def test_stacked_multi_board_hint():
    out = r._render_to_string(
        r.render_board_stacked([], board_slug="default", other_board_count=2)
    )
    assert "other board" in out.lower()
