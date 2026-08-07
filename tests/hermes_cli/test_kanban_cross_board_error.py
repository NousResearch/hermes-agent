"""Tests for the cross-board failure hint in kanban CLI verbs (#65101).

When a kanban verb fails because the task id lives on a *different* board, the
bare ``unknown task {id}`` (or ``cannot <verb> {id}``) line is misleading: the
id is real, it's just on the wrong board. The dispatch layer and the per-verb
``False`` branches now attach an actionable hint naming the owning board and
the exact ``boards switch`` command.

These tests pin that hint for the real failure paths (the ``unknown task``
exception raised when a verb touches a task before its state check, and the
``cannot <verb>`` line when a verb's task op returns ``False``), and confirm no
hint is shown when the failure is a genuine wrong-state case.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home():
    """Init the kanban DB under the hermetic HERMES_HOME that conftest sets."""
    kb.init_db()


def _task_on_board(board: str, *, title: str = "remote task") -> str:
    """Create a task on ``board`` and return its id."""
    with kb.connect(board=board) as conn:
        return kb.create_task(conn, title=title, initial_status="running")


# ---------------------------------------------------------------------------
# Real failure path: "unknown task" exception (verb touches the task before
# its state check — e.g. block with a reason calls add_comment first).
# ---------------------------------------------------------------------------


def test_block_unknown_task_on_other_board_suggests_switch(kanban_home, monkeypatch):
    """Blocking (with a reason) a task on another board raises 'unknown task'
    from add_comment; the dispatch layer attaches the owning-board hint."""
    kb.create_board("default")
    kb.create_board("cqgambit")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "cqgambit")

    tid = _task_on_board("default")

    out = kc.run_slash(f"block {tid} stuck --kind needs_input")

    assert "unknown task" in out
    assert f"'{tid}' is on board 'default'" in out
    assert "hermes kanban boards switch default" in out


def test_archive_unknown_task_on_other_board_suggests_switch(kanban_home, monkeypatch):
    """Archiving a task on another board attaches the owning-board hint."""
    kb.create_board("default")
    kb.create_board("other")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "other")

    tid = _task_on_board("default")

    out = kc.run_slash(f"archive {tid}")

    assert f"'{tid}' is on board 'default'" in out
    assert "hermes kanban boards switch default" in out


def test_complete_unknown_task_on_other_board_suggests_switch(kanban_home, monkeypatch):
    """Completing a task on another board attaches the owning-board hint."""
    kb.create_board("default")
    kb.create_board("work")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "work")

    tid = _task_on_board("default")

    out = kc.run_slash(f"complete {tid}")

    assert f"'{tid}' is on board 'default'" in out
    assert "hermes kanban boards switch default" in out


def test_unknown_task_on_current_board_has_no_switch_hint(kanban_home, monkeypatch):
    """When the task genuinely doesn't exist (not on any board) we must NOT
    suggest switching boards — that would chase the operator to boards where
    the task still isn't."""
    kb.create_board("default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    out = kc.run_slash("block t_does_not_exist why --kind needs_input")

    assert "unknown task" in out
    assert "boards switch" not in out
    assert "is on board" not in out


# ---------------------------------------------------------------------------
# Per-verb "cannot <verb>" path: the task op returns False (no reason, so
# add_comment isn't called and the verb reaches its own failure print).
# ---------------------------------------------------------------------------


def test_block_returns_false_on_other_board_suggests_switch(kanban_home, monkeypatch):
    """block without a reason reaches block_task→False and prints
    'cannot block' plus the cross-board hint."""
    kb.create_board("default")
    kb.create_board("cqgambit")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "cqgambit")

    tid = _task_on_board("default")

    out = kc.run_slash(f"block {tid} --kind needs_input")

    assert "cannot block" in out
    assert "hermes kanban boards switch default" in out


def test_wrong_state_failure_has_no_board_hint(kanban_home, monkeypatch):
    """A task present on the *current* board but in the wrong state must not
    suggest switching boards — the failure is a state problem, not a
    wrong-board problem."""
    kb.create_board("default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    # Transition a task to 'done', then block it: 'done' isn't in the
    # running/ready set block accepts → wrong-state failure, no board hint.
    with kb.connect(board="default") as conn:
        tid = kb.create_task(conn, title="finished", initial_status="running")
        kb.complete_task(conn, tid)

    out = kc.run_slash(f"block {tid} --kind needs_input")

    assert "cannot block" in out
    assert "is on board" not in out
    assert "boards switch" not in out


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_cross_board_hint_returns_none_when_task_on_current_board(
    kanban_home, monkeypatch
):
    kb.create_board("default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    with kb.connect(board="default") as conn:
        tid = kb.create_task(conn, title="here", initial_status="running")
        assert kc._cross_board_hint(conn, [tid]) is None


def test_cross_board_hint_names_owning_board(kanban_home, monkeypatch):
    kb.create_board("default")
    kb.create_board("alpha")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    tid = _task_on_board("alpha")
    with kb.connect(board="default") as conn:
        hint = kc._cross_board_hint(conn, [tid])

    assert hint is not None
    assert "board 'alpha'" in hint
    assert "hermes kanban boards switch alpha" in hint


def test_cross_board_hint_with_db_pinned_to_active(kanban_home, monkeypatch):
    """The hint must still find the owning board when the dispatcher has
    pinned ``HERMES_KANBAN_DB`` to the active board (#65128).

    Dispatcher-spawned workers always carry this pin (see
    ``kanban_db.py`` worker env handoff). A naive scan that reconnects via
    ``connect(board=other)`` honours the pin, reopens the active board's
    DB, and never finds the task — so the scan must bypass the env
    override with an explicit ``db_path``.
    """
    kb.create_board("default")
    kb.create_board("alpha")
    # Create the task on "alpha" BEFORE the DB pin is set, otherwise
    # ``connect(board="alpha")`` inside _task_on_board would itself be
    # hijacked by the pin and write the task to the default board.
    tid = _task_on_board("alpha")

    # Pin the active ("default") board's DB exactly as the dispatcher does.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="default")))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    with kb.connect(board="default") as conn:
        hint = kc._cross_board_hint(conn, [tid])

    assert hint is not None
    assert "board 'alpha'" in hint
    assert "hermes kanban boards switch alpha" in hint


def test_task_ids_parsed_from_unknown_task_message():
    assert kc._task_ids_from_unknown_task_message("unknown task t_abc123") == [
        "t_abc123"
    ]
    assert kc._task_ids_from_unknown_task_message("unknown task(s): t_abc, t_def") == [
        "t_abc",
        "t_def",
    ]
    # dedupes
    msg = "unknown task(s): t_abc, t_abc, t_def"
    assert kc._task_ids_from_unknown_task_message(msg) == ["t_abc", "t_def"]
    # non-task tokens ignored
    assert kc._task_ids_from_unknown_task_message("unknown task not-an-id") == []
