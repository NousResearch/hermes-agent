"""Quota-cap enforcement tests (tasks.max_quota_tokens).

2026-09-08 (option C): the dispatcher heartbeat now reads the cumulative
``prompt+completion+reasoning`` tokens across a card's worker sessions and
blocks with kind ``quota_cap`` when the total exceeds the cap. Mirrors the
runtime-cap / cost-cap machinery exactly — same dead-letter routing, same
shared teardown — but the unit is tokens, not dollars.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _host_claimer() -> str:
    return f"{kb._claimer_id().split(':', 1)[0]}:worker"


def _claim_running(conn, task_id):
    claimed = kb.claim_task(conn, task_id, claimer=_host_claimer())
    assert claimed is not None, "task should be claimable (ready)"
    kb._set_worker_pid(conn, task_id, 77777)
    return claimed


def _workspace(kanban_home, suffix):
    return str(kanban_home / "workspaces" / f"t_{suffix}")


def _make_state_db_with_tokens(
    path: Path, rows, task_id: str | None = None,
) -> Path:
    """Create a state.db ledger where each session has prompt/completion/reasoning tokens."""
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY,
          cwd TEXT,
          source TEXT,
          title TEXT,
          estimated_cost_usd REAL
        );
        CREATE TABLE IF NOT EXISTS session_model_usage (
          session_id TEXT NOT NULL,
          model TEXT NOT NULL DEFAULT '',
          billing_provider TEXT NOT NULL DEFAULT '',
          billing_base_url TEXT NOT NULL DEFAULT '',
          billing_mode       TEXT NOT NULL DEFAULT '',
          task               TEXT NOT NULL DEFAULT '',
          estimated_cost_usd REAL NOT NULL DEFAULT 0,
          input_tokens       INTEGER NOT NULL DEFAULT 0,
          output_tokens      INTEGER NOT NULL DEFAULT 0,
          reasoning_tokens   INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    for i, (cwd, prompt, completion, reasoning) in enumerate(rows, start=1):
        title = f"Work kanban task {task_id} #{i}" if task_id else None
        con.execute(
            "INSERT INTO sessions (id, cwd, source, title, estimated_cost_usd) "
            "VALUES (?, ?, 'kanban', ?, 0.0)",
            (f"s{i}", cwd, title),
        )
        con.execute(
            "INSERT INTO session_model_usage "
            "(session_id, input_tokens, output_tokens, reasoning_tokens) "
            "VALUES (?, ?, ?, ?)",
            (f"s{i}", prompt, completion, reasoning),
        )
    con.commit()
    con.close()
    return path


def _noop_signal(pid, sig):
    return None


# ---------------------------------------------------------------------------
# Token sums hit the cap; legacy dollar axis is untouched
# ---------------------------------------------------------------------------


def test_quota_cap_trips_when_tokens_exceed_cap(kanban_home):
    """600_000 tokens (>500k cap) on the assignee's ledger must trip the
    quota enforcement and block with kind=quota_cap, never retry."""
    conn = kb.connect()
    W = _workspace(kanban_home, "quota_trip")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_quota_tokens=500_000, workspace_path=W,
    )
    _claim_running(conn, tid)
    state = _make_state_db_with_tokens(
        kanban_home / "state.db",
        [(W, 400_000, 100_000, 100_000)],  # 600_000 tokens, > 500_000 cap
        task_id=tid,
    )

    blocked = kb.enforce_max_quota(conn, signal_fn=_noop_signal, state_db_path=state)
    assert blocked == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    assert task.block_kind == "quota_cap"
    conn.close()


def test_quota_cap_under_threshold_does_not_block(kanban_home):
    """400_000 tokens (<500k cap) on the assignee's ledger must NOT trip the quota."""
    conn = kb.connect()
    W = _workspace(kanban_home, "under")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_quota_tokens=500_000, workspace_path=W,
    )
    _claim_running(conn, tid)
    state = _make_state_db_with_tokens(
        kanban_home / "state.db",
        [(W, 300_000, 80_000, 20_000)],  # 400_000 tokens, < cap
        task_id=tid,
    )

    blocked = kb.enforce_max_quota(conn, signal_fn=_noop_signal, state_db_path=state)
    assert blocked == []
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    conn.close()


def test_quota_cap_reviewers_ledger_does_not_count(kanban_home):
    """Reviewer spend on a DIFFERENT profile's ledger must not breach the
    assignee's quota (09-06 C1 rule applied to the quota axis)."""
    conn = kb.connect()
    W = _workspace(kanban_home, "reviewer")
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_quota_tokens=500_000, workspace_path=W,
    )
    _claim_running(conn, tid)
    state_path = kanban_home / "state.db"
    bob_db = _make_state_db_with_tokens(
        state_path,
        [(W, 200_000, 50_000, 0)],  # 250_000 tokens
        task_id=tid,
    )
    (kanban_home / "profiles" / "rodge").mkdir(parents=True)
    rodge_db = _make_state_db_with_tokens(
        kanban_home / "profiles" / "rodge" / "state.db",
        [(W, 400_000, 100_000, 0)],  # 500_000 reviewer tokens, on a different profile
        task_id=tid,
    )
    # Assignee under cap -> not blocked.
    assert kb.enforce_max_quota(conn, signal_fn=_noop_signal, state_db_path=bob_db) == []
    # Add a second session to bob's ledger that pushes over the cap.
    con = sqlite3.connect(str(state_path))
    title = f"Work kanban task {tid} #2"
    con.execute(
        "INSERT INTO sessions (id, cwd, source, title, estimated_cost_usd) "
        "VALUES ('s-extra', ?, 'kanban', ?, 0.0)",
        (W, title),
    )
    con.execute(
        "INSERT INTO session_model_usage (session_id, input_tokens, output_tokens, reasoning_tokens) "
        "VALUES ('s-extra', 250_000, 60_000, 0)",
    )
    con.commit()
    con.close()
    assert kb.enforce_max_quota(conn, signal_fn=_noop_signal, state_db_path=state_path) == [tid]
    assert kb.get_task(conn, tid).block_kind == "quota_cap"
    conn.close()


# ---------------------------------------------------------------------------
# Overwatch extension: once per card, never past hard ceiling
# ---------------------------------------------------------------------------


def test_set_quota_once_and_never_past_hard_ceiling(kanban_home):
    """Mirror of set_task_max_cost's one-extension rule on the quota axis."""
    conn = kb.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_quota_tokens=500_000)
    new_cap = kb.resolve_max_quota_hard_ceiling()
    assert kb.set_task_max_quota(conn, tid, new_cap, by="default", reason="legitimate") == new_cap
    assert kb.get_task(conn, tid).max_quota_tokens == new_cap
    bodies = [c.body for c in kb.list_comments(conn, tid)]
    assert any(b.startswith("quota-extension:") for b in bodies)
    # Second extension refused — second breach is Richie's.
    with pytest.raises(ValueError, match="already been extended"):
        kb.set_task_max_quota(conn, tid, new_cap, by="default")
    conn.close()


def test_set_quota_refuses_above_hard_ceiling_and_non_increase(kanban_home):
    hard = kb.resolve_max_quota_hard_ceiling()
    conn = kb.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_quota_tokens=500_000)
    with pytest.raises(ValueError, match="hard ceiling"):
        kb.set_task_max_quota(conn, tid, hard + 1, by="default")
    with pytest.raises(ValueError, match="not above"):
        kb.set_task_max_quota(conn, tid, 400_000, by="default")
    assert kb.get_task(conn, tid).max_quota_tokens == 500_000
    conn.close()


# ---------------------------------------------------------------------------
# Mint-time conversion: max_cost alias -> max_quota_tokens
# ---------------------------------------------------------------------------


def test_mint_with_only_max_cost_converts_to_quota(kanban_home):
    """The legacy alias path: a card minted with max_cost=1.0 ends up with a
    quota cap converted at the blended proxy rate (5_000_000 tokens per USD),
    then clamped to the quota ceiling (500_000 by default)."""
    conn = kb.connect()
    tid = kb.create_task(conn, title="x", assignee="bob", max_cost=1.00)
    task = kb.get_task(conn, tid)
    # Legacy column kept for audit.
    assert task.max_cost == pytest.approx(1.00)
    # Quota column populated and clamped to the ceiling (500_000 default).
    ceiling = kb.resolve_max_quota_ceiling()
    assert task.max_quota_tokens == ceiling
    conn.close()


def test_mint_with_only_max_quota_tokens_does_not_touch_max_cost(kanban_home):
    """Explicit quota cap stays; max_cost stays NULL (no proxy conversion)."""
    conn = kb.connect()
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_quota_tokens=250_000,
    )
    task = kb.get_task(conn, tid)
    assert task.max_quota_tokens == 250_000
    assert task.max_cost is None
    conn.close()


def test_mint_with_explicit_quota_clamps_to_ceiling(kanban_home):
    """Explicit max_quota_tokens above the ceiling is clamped (500_000 default)."""
    conn = kb.connect()
    tid = kb.create_task(
        conn, title="x", assignee="bob", max_quota_tokens=10_000_000,
    )
    task = kb.get_task(conn, tid)
    ceiling = kb.resolve_max_quota_ceiling()
    assert task.max_quota_tokens == ceiling
    conn.close()