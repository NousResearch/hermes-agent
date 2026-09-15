"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
    task_run_is_live,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_disabled_inside_delegated_child(clear_kanban_env):
    from agent.delegation_context import delegated_child_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with delegated_child_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_disabled_inside_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent")

    assert kanban_stop_nudge_enabled() is True
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.






# ── Review transitions are terminal for the run (regression) ─────────
# A worker that ends its run with ``kanban_request_review`` /
# ``kanban_request_changes`` has left ``running``: the card sits in ``review``
# (or back in ``ready``) with ``current_run_id`` cleared. The old guard only
# knew ``kanban_complete`` / ``kanban_block``, so it kept injecting a
# "still running — protocol violation" nag that pressured the worker into
# marking unmerged, review-rejected work DONE.


def _terminal_tool_messages(tool_name: str) -> list[dict]:
    return [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]


@pytest.mark.parametrize("tool_name", ["kanban_request_review", "kanban_request_changes"])
def test_review_transitions_count_as_terminal(clear_kanban_env, tool_name):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_a3f07679")
    messages = _terminal_tool_messages(tool_name)
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None


# ── Liveness is re-read from the board, not cached dispatch state ────


@pytest.fixture
def kanban_board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized kanban DB."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    kb.init_db()
    return kb


def test_task_run_is_live_true_while_running(kanban_board, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc

    kb = kanban_board
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    assert task_run_is_live(tid) is True
    # No terminal tool in the transcript and a genuinely live run → nag fires.
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None


def test_no_nag_after_request_review_lands_on_board(kanban_board, monkeypatch):
    """Acceptance: the card sits in ``review`` and the worker must exit clean.

    Even with a transcript that shows no terminal tool call (e.g. the review
    request happened in an earlier, compressed turn), the durable board read
    must suppress the protocol-violation nag.
    """
    from hermes_cli import kanban_db_connect as kbc

    kb = kanban_board
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary="implemented + verified", expected_run_id=run_id
        ) is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.current_run_id is None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    assert task_run_is_live(tid) is False
    assert build_kanban_stop_nudge(messages=[], attempts=0) is None


def test_no_nag_after_request_changes_requeues_task(kanban_board, monkeypatch):
    from hermes_cli import kanban_db_connect as kbc

    kb = kanban_board
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="implemented",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        ) is True
        # Reviewer claims the review lane and rejects the work.
        assert kb.claim_review_task(conn, tid) is not None
        review_run = kb.get_task(conn, tid).current_run_id
        ok, _ = kb.request_changes(
            conn, tid, reason="fix the contract defect", expected_run_id=review_run
        )
        assert ok is True
        assert kb.get_task(conn, tid).current_run_id is None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review_run))
    assert task_run_is_live(tid) is False
    assert build_kanban_stop_nudge(messages=[], attempts=0) is None


def test_liveness_read_failure_fails_open(kanban_board, monkeypatch):
    """A DB problem must degrade to the old behaviour, never silence the guard.

    Uses a card whose run is genuinely over (``review``), so a broken read is
    distinguishable from a correct terminal-state read: the guard must still
    fire rather than silently trust the failure.
    """
    from hermes_cli import kanban_db as kb_mod
    from hermes_cli import kanban_db_connect as kbc

    kb = kanban_board
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="implemented",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        ) is True

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert task_run_is_live(tid) is False  # healthy read sees the terminal state

    def _boom(*_a, **_kw):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(kb_mod, "get_task", _boom)
    assert task_run_is_live(tid) is True
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None
