"""Cost-cap enforcement tests (tasks.max_cost).

A per-card dollar cap (the spend analogue of the existing
``max_runtime_seconds``) must trip a runaway worker off its budget, BLOCK the
card with kind ``cost_cap``, record both numbers (spend and cap) in a comment,
and NEVER retry it — a ``cost_cap`` block is a dead letter that routes to the
jobsy triage lane. The 64-run dispatch storm and the 7-round review card were
dollar runaways runtime caps alone did not stop, so this mirrors the runtime
cap exactly, reading spend from the state.db session ledger
(``session_model_usage.estimated_cost_usd``).
"""

from __future__ import annotations

import argparse
import signal
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _host_claimer() -> str:
    """A host-local claim lock that enforce_max_cost will treat as ours."""
    return f"{kb._claimer_id().split(':', 1)[0]}:worker"


def _claim_running(conn, task_id):
    claimed = kb.claim_task(conn, task_id, claimer=_host_claimer())
    assert claimed is not None, "task should be claimable (ready)"
    kb._set_worker_pid(conn, task_id, 77777)
    return claimed


def _make_state_db(path: Path, rows) -> Path:
    """Create a minimal state.db ledger: one session per (cwd, cost) row."""
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY,
          cwd TEXT,
          source TEXT
        );
        CREATE TABLE session_model_usage (
          session_id TEXT NOT NULL,
          model TEXT NOT NULL DEFAULT '',
          billing_provider TEXT NOT NULL DEFAULT '',
          billing_base_url TEXT NOT NULL DEFAULT '',
          billing_mode       TEXT NOT NULL DEFAULT '',
          task               TEXT NOT NULL DEFAULT '',
          estimated_cost_usd REAL NOT NULL DEFAULT 0
        );
        """
    )
    for i, (cwd, cost) in enumerate(rows, start=1):
        con.execute(
            "INSERT INTO sessions (id, cwd, source) VALUES (?, ?, ?)",
            (f"s{i}", cwd, "kanban"),
        )
        con.execute(
            "INSERT INTO session_model_usage (session_id, estimated_cost_usd) "
            "VALUES (?, ?)",
            (f"s{i}", cost),
        )
    con.commit()
    con.close()
    return path


def _noop_signal(pid, sig):
    return None


def _workspace(kanban_home, suffix):
    return str(kanban_home / "workspaces" / f"t_{suffix}")


# ---------------------------------------------------------------------------
# Tiny cap trips + blocks WITHOUT retry
# ---------------------------------------------------------------------------


def test_cost_cap_trip_blocks_without_retry(kanban_home):
    conn = kb.connect()
    W = _workspace(kanban_home, "cap")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_cost=0.01, workspace_path=W
    )
    _claim_running(conn, tid)
    state_db = _make_state_db(kanban_home / "state.db", [(W, 1.25)])

    sigs: list[tuple] = []
    capped = kb.enforce_max_cost(
        conn, signal_fn=lambda p, s: sigs.append((p, s)), state_db_path=state_db
    )

    assert capped == [tid]
    # SIGTERM was sent (the shared teardown; SIGKILL is the follow-up only if
    # the worker is still alive after the grace window, which a fake pid is).
    assert sigs and sigs[0][1] == signal.SIGTERM

    t = kb.get_task(conn, tid)
    assert t.status == "blocked"
    assert t.block_kind == "cost_cap"
    # NOT requeued: blocked cards are never re-promoted by the dispatcher.
    assert t.status != "ready" and t.status != "running"
    # The run is terminal (recorded as blocked), not left dangling.
    runs = kb.list_runs(conn, tid)
    assert runs and runs[-1].outcome == "blocked"
    # No re-fire: a second tick sees no 'running' max_cost card any more.
    assert (
        kb.enforce_max_cost(
            conn, signal_fn=_noop_signal, state_db_path=state_db
        )
        == []
    )
    conn.close()


# ---------------------------------------------------------------------------
# Block comment records spend AND cap
# ---------------------------------------------------------------------------


def test_cost_cap_block_comment_records_both_numbers(kanban_home):
    conn = kb.connect()
    W = _workspace(kanban_home, "capc")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_cost=0.01, workspace_path=W
    )
    _claim_running(conn, tid)
    state_db = _make_state_db(kanban_home / "state.db", [(W, 1.25)])

    kb.enforce_max_cost(conn, signal_fn=_noop_signal, state_db_path=state_db)

    comments = kb.list_comments(conn, tid)
    joined = "\n".join(c.body for c in comments)
    assert "1.25" in joined  # ~spend
    assert "0.01" in joined  # ~cap
    assert "cost" in joined
    conn.close()


# ---------------------------------------------------------------------------
# No cap => unaffected (backward compat)
# ---------------------------------------------------------------------------


def test_cost_cap_unset_does_not_block(kanban_home):
    conn = kb.connect()
    W = _workspace(kanban_home, "nocap")
    tid = kb.create_task(conn, title="x", assignee="bob", workspace_path=W)
    _claim_running(conn, tid)
    state_db = _make_state_db(kanban_home / "state.db", [(W, 999.0)])

    capped = kb.enforce_max_cost(
        conn, signal_fn=_noop_signal, state_db_path=state_db
    )
    assert capped == []
    t = kb.get_task(conn, tid)
    assert t.status == "running"
    conn.close()


# ---------------------------------------------------------------------------
# Cumulative across retries / multiple run sessions under one workspace
# ---------------------------------------------------------------------------


def test_cost_cap_cumulative_across_run_sessions(kanban_home):
    conn = kb.connect()
    W = _workspace(kanban_home, "cumulative")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_cost=0.50, workspace_path=W
    )
    _claim_running(conn, tid)
    # Three retries already racked up 0.30 + 0.11 + 0.55 = 0.96 > 0.50 cap,
    # across DISTINCT sessions under the SAME workspace path.
    state_db = _make_state_db(
        kanban_home / "state.db",
        [(W, 0.30), (W, 0.11), (W, 0.55)],
    )
    capped = kb.enforce_max_cost(
        conn, signal_fn=_noop_signal, state_db_path=state_db
    )
    assert capped == [tid]
    assert kb.get_task(conn, tid).status == "blocked"
    conn.close()


def test_cost_cap_below_cap_cumulative_not_blocked(kanban_home):
    conn = kb.connect()
    W = _workspace(kanban_home, "below")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_cost=2.50, workspace_path=W
    )
    _claim_running(conn, tid)
    state_db = _make_state_db(
        kanban_home / "state.db", [(W, 1.00), (W, 1.20)]
    )
    capped = kb.enforce_max_cost(
        conn, signal_fn=_noop_signal, state_db_path=state_db
    )
    assert capped == []
    assert kb.get_task(conn, tid).status == "running"
    conn.close()


# ---------------------------------------------------------------------------
# CLI arg parsed and persisted
# ---------------------------------------------------------------------------


def test_cli_create_parses_max_cost(kanban_home):
    from hermes_cli import kanban as kcli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    kcli.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "create", "--max-cost", "1.5", "--assignee", "bob", "hello"]
    )
    assert getattr(args, "max_cost", None) == 1.5


def test_create_task_persists_max_cost(kanban_home):
    conn = kb.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_cost=3.25)
    t = kb.get_task(conn, tid)
    assert t.max_cost == 3.25
    # Round-trips through the DB (REAL column storage).
    row = conn.execute(
        "SELECT max_cost FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert abs(float(row["max_cost"]) - 3.25) < 1e-6
    conn.close()


def test_create_task_rejects_negative_max_cost(kanban_home):
    conn = kb.connect()
    with pytest.raises(ValueError):
        kb.create_task(conn, title="x", assignee="bob", max_cost=-1.0)
    conn.close()


def test_create_task_without_max_cost_stays_uncapped(kanban_home):
    conn = kb.connect()
    tid = kb.create_task(conn, title="x", assignee="bob")
    assert kb.get_task(conn, tid).max_cost is None
    conn.close()