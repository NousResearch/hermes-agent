"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import sqlite3

import pytest

from agent import kanban_stop as agent_mod
from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    reset_run_outcome_cache,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_RUN_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _msgs_with_tool_call(name: str) -> list[dict]:
    """Minimal history in which the assistant invoked terminal tool ``name``."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": name, "tool_call_id": "1", "content": "ok"},
    ]


def _seed_board_run(
    db_path,
    monkeypatch,
    *,
    outcome=None,
) -> int:
    """Create a real board DB with one task and one run row.

    ``outcome=None`` seeds an open (live) run; any string seeds a terminal
    run the way the dispatcher closes it. Returns the run id. The DB is
    pinned via ``HERMES_KANBAN_DB`` so ``kanban_db.connect()`` inside the
    guard resolves to this file.
    """
    from hermes_cli import kanban_db

    conn = kanban_db.connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO tasks (id, title, body, status, created_at) "
            "VALUES ('t_run', 'seed', NULL, 'running', strftime('%s','now'))"
        )
        assert cur.rowcount == 1
        cur = conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, "
            "ended_at, outcome) VALUES ('t_run', 'forge', 'running', "
            "strftime('%s','now'), ?, ?)",
            (
                None if outcome is None else int(__import__("time").time()),
                outcome,
            ),
        )
        run_id = int(cur.lastrowid or 0)
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    reset_run_outcome_cache()
    return run_id






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


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


# ── Review-lane terminals count as board-terminal (#98107) ────────────


@pytest.mark.parametrize(
    "tool_name",
    [
        "kanban_request_review",
        "kanban_request_changes",
        "kanban_complete",
        "kanban_block",
    ],
)
def test_review_lane_terminals_suppress_nudge(clear_kanban_env, tool_name):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _msgs_with_tool_call(tool_name)
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_non_terminal_tools_still_nudge(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = _msgs_with_tool_call("kanban_heartbeat")
    assert session_called_kanban_terminal(messages) is False
    assert build_kanban_stop_nudge(messages=messages) is not None


# ── Terminal run outcome suppresses the nudge (#98750) ─────────────────


def _nudge_with_run_env(monkeypatch, run_id, task_id="t_run"):
    """Invoke the guard with the env a dispatcher-spawned worker would have.

    ``build_kanban_stop_nudge`` reads os.environ directly, so the env is
    set inside the caller's active monkeypatch context.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return build_kanban_stop_nudge(messages=[])


def test_no_nudge_when_run_outcome_set(tmp_path, monkeypatch):
    run_id = _seed_board_run(tmp_path / "board.db", monkeypatch,
                             outcome="changes_requested")
    nudge = _nudge_with_run_env(monkeypatch, run_id)
    assert nudge is None


@pytest.mark.parametrize("outcome", ["completed", "blocked", "reclaimed"])
def test_no_nudge_any_terminal_outcome(tmp_path, monkeypatch, outcome):
    run_id = _seed_board_run(tmp_path / f"board-{outcome}.db", monkeypatch,
                             outcome=outcome)
    assert _nudge_with_run_env(monkeypatch, run_id) is None


def test_nudge_still_fires_while_run_open(tmp_path, monkeypatch):
    run_id = _seed_board_run(tmp_path / "board.db", monkeypatch, outcome=None)
    nudge = _nudge_with_run_env(monkeypatch, run_id)
    assert nudge is not None
    assert "t_run" in nudge


def test_run_outcome_read_is_memoized(tmp_path, monkeypatch):
    db_path = tmp_path / "board.db"
    run_id = _seed_board_run(db_path, monkeypatch, outcome="completed")
    assert agent_mod.build_kanban_stop_nudge(
        messages=[], attempts=0, max_attempts=5,
    ) is None
    # Empty env → guard disabled; re-enable and confirm the cached
    # terminal outcome still suppresses even if the row were edited.
    assert _nudge_with_run_env(monkeypatch, run_id) is None
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE task_runs SET outcome=NULL WHERE id=?", (run_id,))
        conn.commit()
    finally:
        conn.close()
    # Cached read → still suppressed; cache reset → nudge fires again.
    assert _nudge_with_run_env(monkeypatch, run_id) is None
    reset_run_outcome_cache()
    assert _nudge_with_run_env(monkeypatch, run_id) is not None


def test_unknown_run_id_fails_open(clear_kanban_env, tmp_path, monkeypatch):
    _seed_board_run(tmp_path / "board.db", monkeypatch, outcome=None)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_run")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "999999")
    assert build_kanban_stop_nudge(messages=[]) is not None


def test_missing_run_id_keeps_history_only_behavior(clear_kanban_env):
    # Older dispatchers never set HERMES_KANBAN_RUN_ID; the guard must
    # stay purely history-driven there (fail open).
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert build_kanban_stop_nudge(messages=[]) is not None


# ── Template asserts nothing it did not read (DoD c) ───────────────────


def test_template_makes_no_unconditional_card_status_assertion(
    tmp_path, monkeypatch,
):
    run_id = _seed_board_run(tmp_path / "board.db", monkeypatch, outcome=None)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_run")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    nudge = build_kanban_stop_nudge(messages=[])
    assert nudge is not None
    lowered = nudge.lower()
    # The old template hardcoded "task `X` is still `running`" — the card
    # may have moved on while this session lagged. Nothing in the nudge
    # may assert a card/run state.
    for forbidden in (
        "is still `running`",
        "is still running",
        "still `running`",
        "status is",
        "card is",
        "run is",
        "claim is",
    ):
        assert forbidden not in lowered, forbidden
    # It still names the task and the required tools.
    assert "t_run" in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




