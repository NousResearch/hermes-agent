"""Regression: block_task/schedule_task on a RUNNING task must terminate the
live host-local worker after the transition commits.

Every other running→X reclaim path (reclaim_task, archive_task, reopen
invalidation) snapshots pid+claim inside the txn and signals the worker
post-commit (#76196: clearing ``worker_pid`` alone left the OS process running
and pushing work against an untracked card). ``block_task`` and
``schedule_task`` cleared the pid without signalling — same bug class.
"""

from __future__ import annotations

import json
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as c:
        yield c


def _make_running(conn, *, run_row=True):
    tid = kb.create_task(conn, title="live worker", assignee="p1")
    lock = kb._host_prefix() + ":claim-test"
    future = int(time.time()) + 3600
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=? WHERE id=?",
        (lock, future, 12345, tid),
    )
    run_id = None
    if run_row:
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, worker_pid, started_at)"
            " VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, future, 12345, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()
    return tid, run_id


def _signals(fn_calls):
    return [sig for _pid, sig in fn_calls]


def test_block_task_terminates_running_worker(conn):
    tid, _ = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.block_task(conn, tid, reason="stop it", signal_fn=lambda pid, sig: calls.append((pid, sig)))
    assert 12345 in [pid for pid, _ in calls]
    evs = [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='block_worker_termination'", (tid,)
        )
    ]
    assert evs and evs[0]["prev_pid"] == 12345


def test_block_task_worker_self_call_does_not_signal(conn):
    tid, run_id = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.block_task(
        conn, tid, reason="dependency handoff", kind="needs_input",
        expected_run_id=run_id, signal_fn=lambda pid, sig: calls.append((pid, sig)),
    )
    assert calls == []


def test_schedule_task_terminates_running_worker(conn):
    tid, _ = _make_running(conn)
    calls: list[tuple[int, int]] = []
    assert kb.schedule_task(conn, tid, reason="park", signal_fn=lambda pid, sig: calls.append((pid, sig)))
    assert 12345 in [pid for pid, _ in calls]
    evs = [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='schedule_worker_termination'", (tid,)
        )
    ]
    assert evs and evs[0]["prev_pid"] == 12345


# --- Review findings on #124775: admission fence (P1) + post-commit observer (P2) ---
#
# P1: a resumable card must not become claimable while its predecessor worker
# is still live. Termination runs BEFORE the transition (same sanction as
# ``release_stale_claims``), and a survived/unrefused-live worker leaves the
# card ``running`` with its claim and run intact — unblock/claim cannot admit
# a successor beside it.
#
# P2: the ordinary ``kanban_task_blocked`` observer runs AFTER the txn commits
# (``_fire_kanban_lifecycle_hook`` contract): it reads committed state and can
# take the write lock itself.

from hermes_cli.plugins import get_plugin_manager  # noqa: E402

_LIVE_FINGERPRINT = "987654|1234567"


def _running_card_fingerprinted(conn, *, worker_started_at=_LIVE_FINGERPRINT):
    """A running card whose worker row carries a real spawn fingerprint (and
    whose ``worker_pid`` is preserved through the helpers under test)."""
    tid = kb.create_task(conn, title="live predecessor", assignee="p1")
    lock = kb._host_prefix() + ":claim-fence"
    future = int(time.time()) + 3600
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, worker_pid=?,"
        " worker_started_at=? WHERE id=?",
        (lock, future, 51000, worker_started_at, tid),
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, worker_pid,"
        " worker_started_at, started_at) VALUES (?, 'running', ?, ?, ?, ?, ?)",
        (tid, lock, future, 51000, worker_started_at, int(time.time())),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()
    return tid, run_id, lock


def _live_worker(monkeypatch, *, pid=51000):
    """A live host-local worker of ours: pid alive, fingerprint agrees."""
    monkeypatch.setattr(kb, "_pid_alive", lambda p: p == pid)
    import hermes_cli.kanban_db_dispatch as _dispatch

    monkeypatch.setattr(_dispatch, "_pid_recycled", lambda p, started_at: False)


def _noop_signal(pid, sig):
    pass


def _terminated_events(conn, tid, kind):
    return [
        json.loads(r["payload"])
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind=?", (tid, kind)
        )
    ]


@pytest.mark.parametrize("op", ("block", "schedule"))
def test_transition_refused_while_worker_survives(conn, monkeypatch, op):
    """P1: signal refused (worker never dies) -> the card stays running with
    its claim, run and pid evidence intact; a successor cannot be admitted."""
    tid, run_id, lock = _running_card_fingerprinted(conn)
    _live_worker(monkeypatch)
    calls: list[tuple[int, int]] = []
    if op == "block":
        rv = kb.block_task(conn, tid, reason="stop", signal_fn=lambda p, s: calls.append((p, s)))
        kind = "block_worker_termination"
    else:
        rv = kb.schedule_task(conn, tid, reason="park", signal_fn=lambda p, s: calls.append((p, s)))
        kind = "schedule_worker_termination"
    assert calls, "refusal must still attempt (and audit) the signal"
    assert rv is False, "survived worker must fence the resumable transition"

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running", "card must stay running: not claimable by anyone"
    assert task.claim_lock == lock
    assert kb._current_run_id(conn, tid) == run_id
    row = conn.execute(
        "SELECT worker_pid, worker_started_at FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == 51000 and row["worker_started_at"] == _LIVE_FINGERPRINT

    evs = _terminated_events(conn, tid, kind)
    assert len(evs) == 1 and evs[0]["terminated"] is False and evs[0]["prev_pid"] == 51000


@pytest.mark.parametrize("op", ("block", "schedule"))
def test_transition_refused_for_unverified_live_worker(conn, monkeypatch, op):
    """P1: an ``unverified`` fingerprint live worker is never signalled and the
    card stays running (no successor spawn beside an unidentified process)."""
    tid, run_id, _ = _running_card_fingerprinted(conn, worker_started_at="unverified")
    monkeypatch.setattr(kb, "_pid_alive", lambda p: True)
    calls: list[tuple[int, int]] = []
    if op == "block":
        rv = kb.block_task(conn, tid, reason="stop", signal_fn=lambda p, s: calls.append((p, s)))
        kind = "block_worker_termination"
    else:
        rv = kb.schedule_task(conn, tid, reason="park", signal_fn=lambda p, s: calls.append((p, s)))
        kind = "schedule_worker_termination"
    assert calls == [], "an unverified live process must never be signalled"
    assert rv is False
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "running"
    assert kb._current_run_id(conn, tid) == run_id
    row = conn.execute(
        "SELECT worker_pid, worker_started_at FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == 51000 and row["worker_started_at"] == "unverified"
    evs = _terminated_events(conn, tid, kind)
    assert evs and evs[0]["signal_refused"] is True and evs[0]["terminated"] is False


@pytest.mark.parametrize("op", ("block", "schedule"))
def test_transition_proceeds_once_worker_dies(conn, monkeypatch, op):
    """P1 control: a worker that dies on signal settles the transition —
    block/schedule commits, termination lands as its own event."""
    tid, _, _ = _running_card_fingerprinted(conn)
    # Worker alive until signalled, then gone (fingerprint row is preserved by
    # _pid_recycled's pid-check ordering).
    state = {"alive": True}

    def fake_alive(pid):
        return bool(state["alive"]) and pid == 51000

    monkeypatch.setattr(kb, "_pid_alive", fake_alive)
    import hermes_cli.kanban_db_dispatch as _dispatch

    monkeypatch.setattr(_dispatch, "_pid_recycled", lambda p, started_at: False)

    def killer(pid, sig):
        state["alive"] = False

    if op == "block":
        assert kb.block_task(conn, tid, reason="stop", signal_fn=killer) is True
        kind = "block_worker_termination"
    else:
        assert kb.schedule_task(conn, tid, reason="park", signal_fn=killer) is True
        kind = "schedule_worker_termination"
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == ("blocked" if op == "block" else "scheduled")
    evs = _terminated_events(conn, tid, kind)
    assert evs and evs[0]["terminated"] is True


def test_block_task_ordinary_observer_post_commit(conn, monkeypatch):
    """P2: the ordinary ``kanban_task_blocked`` observer runs after the txn
    commits — it reads ``blocked`` (not the stale ``ready``) and can take the
    write lock on a second connection without a ``database is locked`` error."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    seen_status: list[str] = []
    wrote: list[bool] = []

    def observer(**kw):
        import sqlite3

        c2 = sqlite3.connect(kb.kanban_db_path(), timeout=0.05)
        try:
            c2.execute("PRAGMA busy_timeout=50")
            seen_status.append(
                c2.execute("SELECT status FROM tasks WHERE id=?", (kw["task_id"],)).fetchone()[0]
            )
            c2.execute("UPDATE tasks SET priority = priority + 1 WHERE id=?", (kw["task_id"],))
            c2.commit()
            wrote.append(True)
        finally:
            c2.close()

    mgr._hooks.setdefault("kanban_task_blocked", []).append(observer)
    try:
        tid = kb.create_task(conn, title="observed", assignee="p1")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
        assert kb.block_task(conn, tid, reason="observe me")
    finally:
        mgr._hooks = saved

    assert seen_status == ["blocked"], "observer must see committed state"
    assert wrote == [True], "observer must be able to write (no write lock held)"
