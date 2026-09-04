"""`kanban block` requires a reason, like `kanban_block` always has.

A blocked task is a request for a human decision, and a request with no sentence
is one nobody can act on. The worker tool (tools/kanban_tools.py `_handle_block`)
has always refused an empty reason with "reason is required — explain what input
you need"; the CLI accepted one, so the requirement depended on which surface you
used.

Measured on one install: 126 of 141 `blocked` events carried a reason, and 13 of
the 15 that did not came from outside any run — this path.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli import kanban as kanban_cmd
from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kb, "_HERMES_HOME_OVERRIDE", str(tmp_path), raising=False)
    yield


def _task():
    conn = kb.connect()
    try:
        return kb.create_task(conn, title="Wire the thing")
    finally:
        conn.close()


def _block(task_id, reason_words):
    args = argparse.Namespace(task_id=task_id, reason=reason_words, ids=[],
                              kind=None)
    return kanban_cmd._cmd_block(args)


def _status(task_id):
    conn = kb.connect()
    try:
        return kb.get_task(conn, task_id).status
    finally:
        conn.close()


def test_block_without_a_reason_is_refused_and_changes_nothing():
    tid = _task()
    before = _status(tid)

    assert _block(tid, []) == 2, "exit 2: a usage error, not a silent success"
    # The refusal has to be total. A task left blocked with no reason is exactly
    # the state this check exists to prevent.
    assert _status(tid) == before


@pytest.mark.parametrize("words", [[], [""], ["   "], ["", "  "]])
def test_whitespace_is_not_a_reason(words):
    tid = _task()
    assert _block(tid, words) == 2
    assert _status(tid) != "blocked"


def test_a_real_reason_blocks_and_is_recorded_where_a_human_will_read_it():
    tid = _task()
    assert _block(tid, ["waiting", "on", "the", "vendor", "key"]) == 0
    assert _status(tid) == "blocked"

    conn = kb.connect()
    try:
        # Two places, on purpose: the event payload is what a report reads, the
        # comment is what a person reads on the card.
        eventos = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'",
            (tid,),
        ).fetchall()
        assert eventos and "waiting on the vendor key" in (eventos[-1][0] or "")
        comentarios = conn.execute(
            "SELECT body FROM task_comments WHERE task_id=?", (tid,)
        ).fetchall()
        assert any("BLOCKED: waiting on the vendor key" in (c[0] or "")
                   for c in comentarios)
    finally:
        conn.close()


def test_the_cli_and_the_worker_tool_now_agree():
    """The point of the change: one rule, whichever surface you use."""
    import inspect

    from tools import kanban_tools

    fonte = inspect.getsource(kanban_tools._handle_block)
    assert "reason is required" in fonte, (
        "the worker tool's requirement is the precedent this mirrors; if it "
        "moved, this test should be the one that notices"
    )
