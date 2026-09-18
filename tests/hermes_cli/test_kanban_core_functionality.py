"""Core-functionality tests for the kanban kernel + CLI additions.

Complements tests/hermes_cli/test_kanban_db.py (schema + CAS atomicity)
and tests/hermes_cli/test_kanban_cli.py (end-to-end run_slash).  The
tests here exercise the pieces added as part of the kanban hardening
pass: circuit breaker, crash detection, daemon loop, idempotency,
retention/gc, stats, notify subscriptions, worker log accessor, run_slash
parity across every registered verb.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli.kanban import run_slash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Existing crash-detection tests pre-date the grace window; pin to 0
    # so they keep their immediate-reclaim semantics.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Disable the detect_crashed_workers grace period for legacy tests in
    # this file that claim a task and immediately expect
    # ``detect_crashed_workers`` to act on it. The grace period (30s by
    # default, see ``DEFAULT_CRASH_GRACE_SECONDS``) prevents the
    # multi-dispatcher reap race in production; setting it to 0 here
    # restores the pre-fix instant-reclaim semantics these tests were
    # written against. The grace-period itself is covered by dedicated
    # tests in tests/hermes_cli/test_kanban_db.py.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Spawn-failure circuit breaker
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Worker aliveness / crash detection
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Stats + age
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Notify subscriptions
# ---------------------------------------------------------------------------

def test_notify_sub_crud(kanban_home):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123", user_id="u1",
            notifier_profile="default",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "42",
            },
        )
        subs = kbn.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["platform"] == "telegram"
        assert subs[0]["notifier_profile"] == "default"
        assert subs[0]["delivery_metadata"] == {
            "chat_type": "dm",
            "telegram_reply_to_message_id": "42",
        }
        # Duplicate add is a no-op.
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "43",
            },
        )
        assert len(kbn.list_notify_subs(conn, tid)) == 1
        assert kbn.list_notify_subs(conn, tid)[0]["delivery_metadata"][
            "telegram_reply_to_message_id"
        ] == "43"
        # Distinct thread is a new row.
        kbn.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            thread_id="5",
        )
        assert len(kbn.list_notify_subs(conn, tid)) == 2
        # Remove one.
        ok = kbn.remove_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert ok is True
        assert len(kbn.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()


def test_notify_claim_is_single_owner_and_rewindable(kanban_home):
    conn1 = kbc.connect()
    conn2 = kbc.connect()
    try:
        tid = kb.create_task(conn1, title="x", assignee="w")
        kbn.add_notify_sub(conn1, task_id=tid, platform="telegram", chat_id="123")
        # New subs start caught up at the task's current MAX(task_events.id)
        # (the `created` event) — issue #29905.
        initial_cursor = int(kbn.list_notify_subs(conn1, tid)[0]["last_event_id"])
        kb.complete_task(conn1, tid, result="ok")

        old_cursor, claimed_cursor, events = kbn.claim_unseen_events_for_sub(
            conn1,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert old_cursor == initial_cursor
        assert claimed_cursor > old_cursor
        assert [ev.kind for ev in events] == ["completed"]

        # A concurrent notifier instance sees the advanced cursor and cannot
        # claim/send the same event range.
        _, _, duplicate_events = kbn.claim_unseen_events_for_sub(
            conn2,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert duplicate_events == []

        assert kbn.rewind_notify_cursor(
            conn1,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            claimed_cursor=claimed_cursor,
            old_cursor=old_cursor,
        ) is True
        _, retried_events = kbn.unseen_events_for_sub(
            conn2,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert [ev.kind for ev in retried_events] == ["completed"]
    finally:
        conn1.close()
        conn2.close()


# ---------------------------------------------------------------------------
# GC + retention
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Log rotation + accessor
# ---------------------------------------------------------------------------





def test_read_worker_log_tail(kanban_home):
    log_dir = kanban_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "t_beef.log"
    # 10 lines
    p.write_text("\n".join(f"line {i}" for i in range(10)))
    full = kb.read_worker_log("t_beef")
    assert full is not None and "line 0" in full
    tail = kb.read_worker_log("t_beef", tail_bytes=30)
    assert tail is not None
    # Tail should not include line 0.
    assert "line 0" not in tail
    # Missing log returns None.
    assert kb.read_worker_log("t_missing") is None


# ---------------------------------------------------------------------------
# CLI bulk verbs
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CLI stats / watch / log / notify / daemon parity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_slash parity — every verb returns a sensible, non-crashy string
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Max-runtime enforcement (item 1 from the Multica audit)
# ---------------------------------------------------------------------------

def test_max_runtime_terminates_overrun_worker(kanban_home):
    """A running task whose elapsed time exceeds max_runtime_seconds gets
    SIGTERM'd, emits a ``timed_out`` event, and goes back to ready."""
    killed = []
    def _signal_fn(pid, sig):
        killed.append((pid, sig))

    # We bypass _pid_alive by stubbing it so the grace-poll exits fast.
    import hermes_cli.kanban_db as _kb
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False  # pretend SIGTERM worked immediately

    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(
                conn, title="long job", assignee="worker",
                max_runtime_seconds=1,  # one second cap
            )
            # Spawn by hand: claim + set pid + set active run start to the past.
            kb.claim_task(conn, tid)
            kbd._set_worker_pid(conn, tid, os.getpid())   # any live pid works
            # Backdate both the task-level first-start timestamp and the active
            # run timestamp so elapsed > limit under the per-run runtime model.
            old_started = int(time.time()) - 30
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET started_at = ? WHERE id = ?",
                    (old_started, tid),
                )
                conn.execute(
                    "UPDATE task_runs SET started_at = ? "
                    "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                    (old_started, tid),
                )

            timed_out = kbd.enforce_max_runtime(conn, signal_fn=_signal_fn)
            assert tid in timed_out
            assert killed and killed[0][0] == os.getpid()

            task = kb.get_task(conn, tid)
            assert task.status == "ready",                 f"timed-out task should reset to ready, got {task.status}"
            assert task.worker_pid is None
            assert task.last_heartbeat_at is None

            events = kb.list_events(conn, tid)
            assert any(e.kind == "timed_out" for e in events)
            to_event = next(e for e in events if e.kind == "timed_out")
            assert to_event.payload["limit_seconds"] == 1
            assert to_event.payload["elapsed_seconds"] >= 30
        finally:
            conn.close()
    finally:
        _kb._pid_alive = original_alive








# ---------------------------------------------------------------------------
# Heartbeat (item 2 from the Multica audit)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Event vocab rename + spawned event (item 3 from Multica)
# ---------------------------------------------------------------------------







def test_migration_renames_legacy_event_kinds(tmp_path, monkeypatch):
    """A DB created with the old vocab must have its event rows renamed
    in place on init_db()."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Init fresh.
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="x")
        # Inject legacy event kinds directly.
        now = int(time.time())
        with kb.write_txn(conn):
            for old in ("ready", "priority", "spawn_auto_blocked"):
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) "
                    "VALUES (?, ?, NULL, ?)",
                    (tid, old, now),
                )
        # Re-run init_db — the migration pass should rename them.
        kb.init_db()
        rows = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        kinds = [r["kind"] for r in rows]
        assert "ready" not in kinds
        assert "priority" not in kinds
        assert "spawn_auto_blocked" not in kinds
        assert "promoted" in kinds
        assert "reprioritized" in kinds
        assert "gave_up" in kinds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Assignees (item 4 from Multica)
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# CLI --max-runtime flag + duration parser
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Runs as first-class (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------








def test_stale_run_cannot_block_or_heartbeat_new_attempt(kanban_home, monkeypatch):
    """Stale retry attempts cannot mutate the active run lifecycle."""
    import hermes_cli.kanban_db as _kb

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="retry heartbeat guarded", assignee="worker")

        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        kbd._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kbd.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kbd.heartbeat_worker(conn, tid, note="late", expected_run_id=run1.id)
        assert not kb.block_task(conn, tid, reason="late block", expected_run_id=run1.id)
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id == run2.id
        assert task.last_heartbeat_at is None

        assert kbd.heartbeat_worker(conn, tid, note="current", expected_run_id=run2.id)
        assert kb.block_task(conn, tid, reason="current block", expected_run_id=run2.id)
        assert kb.get_task(conn, tid).status == "blocked"
    finally:
        conn.close()








def test_relative_age_renders_coarse_buckets():
    """Freshness helper turns epoch seconds into coarse human ages, and
    degrades safely on missing / future timestamps."""
    now = 1_000_000
    assert kb._relative_age(now, now) == "just now"
    assert kb._relative_age(now - 30, now) == "just now"
    assert kb._relative_age(now - 5 * 60, now) == "5m ago"
    assert kb._relative_age(now - 18 * 3600, now) == "18h ago"
    assert kb._relative_age(now - 2 * 86400, now) == "2d ago"
    # Clock skew across machines/profiles must not claim "in the future".
    assert kb._relative_age(now + 500, now) == "just now"
    # Missing / unparseable timestamps render empty so callers can append
    # unconditionally.
    assert kb._relative_age(None, now) == ""
    # Defensive: an unparseable value (e.g. a stray string) renders empty
    # rather than raising.
    assert kb._relative_age("garbage", now) == ""  # type: ignore[arg-type]


def test_migration_backfills_inflight_run_for_legacy_db(kanban_home):
    """An existing 'running' task from before task_runs existed should
    get a synthesized run row so subsequent operations (complete,
    heartbeat) have something to write to."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="pre-migration", assignee="worker")
        # Simulate legacy: set running + claim_lock directly, leave
        # current_run_id NULL and delete the run row the claim created.
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id = ?", (tid,))
            conn.execute(
                "UPDATE tasks SET current_run_id = NULL WHERE id = ?",
                (tid,),
            )

        # Sanity: no runs, no pointer.
        assert kb.list_runs(conn, tid) == []
        assert kb.get_task(conn, tid).current_run_id is None

        # Re-run init_db — migration backfill should kick in.
        kb.init_db()
        conn2 = kbc.connect()
        try:
            runs = kb.list_runs(conn2, tid)
            assert len(runs) == 1
            assert runs[0].status == "running"
            assert runs[0].profile == "worker"
            task = kb.get_task(conn2, tid)
            assert task.current_run_id == runs[0].id

            # Subsequent complete closes the backfilled run cleanly.
            kb.complete_task(conn2, tid, result="done", summary="ok")
            r = kb.latest_run(conn2, tid)
            assert r.outcome == "completed"
            assert r.summary == "ok"
        finally:
            conn2.close()
    finally:
        conn.close()






# -------------------------------------------------------------------------
# Integration hardening (Apr 2026 audit fixes)
# -------------------------------------------------------------------------









# -------------------------------------------------------------------------
# Deep-scan fixes (Apr 2026 second audit)
# -------------------------------------------------------------------------







def test_claim_task_recovers_from_invariant_leak(kanban_home):
    """Belt-and-suspenders: if a prior run somehow leaked (stranded
    current_run_id on a ready task), claim_task should recover rather
    than strand it further."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="invariant test", assignee="worker")
        # Manually engineer the invariant violation: create a run, then
        # flip status back to 'ready' without closing the run.
        kb.claim_task(conn, tid)
        leaked_run_id = kb.latest_run(conn, tid).id
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL "
            "WHERE id = ?", (tid,),
        )
        conn.commit()
        # The leaked run is still open.
        assert kb.get_run(conn, leaked_run_id).ended_at is None

        # Now re-claim — the defensive recovery must close the leak.
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        leaked = kb.get_run(conn, leaked_run_id)
        assert leaked.ended_at is not None
        assert leaked.outcome == "reclaimed"
        # New run opened and pointed to.
        new_run = kb.latest_run(conn, tid)
        assert new_run.id != leaked_run_id
        assert new_run.ended_at is None
    finally:
        conn.close()


# -------------------------------------------------------------------------
# Live-test findings (Apr 2026 third pass: auto-init, show --json carries runs)
# -------------------------------------------------------------------------




# -------------------------------------------------------------------------
# Pre-merge audit by @erosika (issue #16102 comment 4331125835) — fixes
# -------------------------------------------------------------------------

def test_unblock_invariant_recovery(kanban_home):
    """unblock_task must leave current_run_id NULL even if some other
    code path left it dangling. Engineer the leak, verify recovery."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="unblock invariant", assignee="worker")
        # Start on running, then open a run, then force to 'blocked' but
        # leave current_run_id pointing at the open run — simulate the
        # invariant violation erosika flagged.
        kb.claim_task(conn, tid)
        leaked_run_id = kb.latest_run(conn, tid).id
        # Force the bad state.
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,),
        )
        conn.commit()
        # current_run_id is still set; run is still open.
        assert kb.get_task(conn, tid).current_run_id == leaked_run_id
        assert kb.get_run(conn, leaked_run_id).ended_at is None

        # Unblock — the defensive recovery must close the leaked run.
        assert kb.unblock_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        leaked = kb.get_run(conn, leaked_run_id)
        assert leaked.outcome == "reclaimed"
        assert leaked.ended_at is not None
    finally:
        conn.close()


def test_migration_backfill_idempotent_under_re_run(tmp_path, monkeypatch):
    """init_db must be safe to re-run repeatedly. Each call should leave
    at most one run row per in-flight task, even if called while a
    dispatcher is simultaneously claiming."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Fresh DB, one task left in 'running' with a claim but no run row.
    # Simulates a pre-runs-era DB.
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="legacy inflight", assignee="worker")
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='old', "
            "claim_expires=?, started_at=?, current_run_id=NULL WHERE id=?",
            (now + 900, now, tid),
        )
        # Drop any synthetic run the normal claim path would have made.
        conn.execute("DELETE FROM task_runs WHERE task_id=?", (tid,))
        conn.commit()

        # Re-run init_db 3x — each should detect the orphan-inflight and
        # install exactly ONE run row, not three.
        for _ in range(3):
            kb.init_db()

        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1, f"expected exactly 1 backfilled run, got {len(runs)}"
        # Pointer should be installed.
        assert kb.get_task(conn, tid).current_run_id == runs[0].id
    finally:
        conn.close()




# -------------------------------------------------------------------------
# Battle-test findings (May 2026: stress/ suite exposed zombie + id collision)
# -------------------------------------------------------------------------

@pytest.mark.skipif("linux" not in __import__("sys").platform,
                    reason="zombie detection is Linux-specific")
def test_pid_alive_detects_zombie(kanban_home):
    """_pid_alive must return False for a zombie process.

    Without the /proc check, kill(pid, 0) succeeds against zombies
    (process table entry exists until parent reaps), so the dispatcher
    would treat a dead-but-unreaped worker as alive. This catches a
    worker that exited normally but whose parent hasn't called wait().
    """
    import subprocess as _sp
    proc = _sp.Popen(
        ["sleep", "3600"],
        stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )
    pid = proc.pid
    try:
        assert kb._pid_alive(pid) is True  # live non-zombie
        os.kill(pid, 9)
        time.sleep(0.3)
        # Verify /proc reports zombie state so the test is actually
        # exercising the zombie path and not some other liveness failure
        with open(f"/proc/{pid}/status") as f:
            state_line = next(
                (l for l in f if l.startswith("State:")), ""
            )
        assert "Z" in state_line, f"expected zombie, got {state_line!r}"
        # And _pid_alive must see through it.
        assert kb._pid_alive(pid) is False
    finally:
        try:
            proc.wait(timeout=1)
        except Exception:
            pass










def test_default_spawn_does_not_auto_load_any_skill(kanban_home, monkeypatch):
    """The dispatcher no longer auto-loads a bundled kanban skill.

    The kanban lifecycle (formerly the kanban-worker/kanban-orchestrator
    skills) is now injected into every worker's system prompt via
    KANBAN_GUIDANCE, so _default_spawn must NOT append a `--skills` flag
    when the task carries no per-task skills.

    We intercept Popen to capture the argv without actually spawning a
    hermes subprocess (which would hang trying to call an LLM).
    """
    captured = {}

    class FakeProc:
        def __init__(self):
            self.pid = 99999

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="skill-loading test",
                             assignee="some-profile")
        task = kb.get_task(conn, tid)
        workspace = kbw.resolve_workspace(task)
        pid = kbd._default_spawn(task, str(workspace))
        assert pid == 99999
    finally:
        conn.close()

    cmd = captured["cmd"]
    assert "--skills" not in cmd, (
        f"spawn argv should not auto-load any skill: {cmd}"
    )
    assert "--accept-hooks" in cmd, f"spawn argv missing --accept-hooks: {cmd}"
    assert cmd.index("--accept-hooks") < cmd.index("chat"), (
        f"--accept-hooks must come before 'chat' in argv: {cmd}"
    )
    # Assignee + task env are still present
    assert "some-profile" in cmd
    env = captured["env"]
    assert env.get("HERMES_KANBAN_TASK") == tid
    assert env.get("HERMES_PROFILE") == "some-profile"


# ---------------------------------------------------------------------------
# Per-task force-loaded skills
# ---------------------------------------------------------------------------







def test_legacy_db_without_skills_column_migrates(tmp_path):
    """_migrate_add_optional_columns is idempotent and adds skills
    when absent. Run it twice on a pared-down schema to confirm."""
    import sqlite3
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Build a pared-down legacy tasks table that lacks all the
    # optional columns _migrate_add_optional_columns knows how to
    # add. We deliberately omit `skills` so we can observe its
    # introduction.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is also touched by the migrator for run_id backfill.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    conn.commit()

    before = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "skills" not in before

    # Run the migrator directly — the same function connect() calls.
    kbc._migrate_add_optional_columns(conn)
    after = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "skills" in after, f"migration did not add skills column: {after}"

    # Idempotent: running again must not raise.
    kbc._migrate_add_optional_columns(conn)

    # Legacy row has skills=NULL -> Task.skills=None.
    row = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    # from_row needs additional columns; build a Task manually via the
    # path from_row takes for a skills NULL/missing.
    keys = set(row.keys())
    assert "skills" in keys
    assert row["skills"] is None
    conn.close()


def test_legacy_spawn_failure_columns_are_copied_not_renamed(tmp_path):
    """Legacy failure counters survive migration without fragile column renames."""
    import sqlite3
    db_path = tmp_path / "legacy-failures.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            spawn_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_spawn_error TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is required: _migrate_add_optional_columns also runs a
    # PRAGMA on it to back-fill the run_id column and raises
    # OperationalError if the table is absent.
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, priority, created_by, created_at, "
        "started_at, completed_at, workspace_kind, workspace_path, claim_lock, "
        "claim_expires, tenant, result, idempotency_key, spawn_failures, "
        "worker_pid, last_spawn_error) "
        "VALUES ('legacy', 'old task', NULL, 'default', 'ready', 0, NULL, 1, "
        "NULL, NULL, 'scratch', NULL, NULL, NULL, NULL, NULL, NULL, 4, NULL, "
        "'missing profile')"
    )
    conn.commit()

    kbc._migrate_add_optional_columns(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "spawn_failures" in cols
    assert "consecutive_failures" in cols
    assert "last_spawn_error" in cols
    assert "last_failure_error" in cols

    row = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    assert row["consecutive_failures"] == 4
    assert row["last_failure_error"] == "missing profile"
    task = kb.Task.from_row(row)
    assert task.consecutive_failures == 4
    assert task.last_failure_error == "missing profile"

    kbc._migrate_add_optional_columns(conn)
    row_again = conn.execute("SELECT * FROM tasks WHERE id = 'legacy'").fetchone()
    assert row_again["consecutive_failures"] == 4
    assert row_again["last_failure_error"] == "missing profile"
    conn.close()


def test_legacy_migration_no_legacy_columns_at_all(tmp_path):
    """Scenario A: DB has neither spawn_failures nor consecutive_failures.

    This is the exact crash scenario from issue #20842 — a very old DB that
    predates the spawn_failures column entirely.  The old RENAME COLUMN path
    raised ``sqlite3.OperationalError: no such column: spawn_failures``.
    The ADD-first approach adds consecutive_failures with default 0.
    """
    import sqlite3

    db_path = tmp_path / "ancient.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    # task_events is required: _migrate_add_optional_columns also runs a
    # PRAGMA on it to back-fill the run_id column and raises
    # OperationalError if the table is absent.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('t1', 'ancient task', 'ready', 1)"
    )
    conn.commit()

    # Must not raise (this was the crash before this fix).
    kbc._migrate_add_optional_columns(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "consecutive_failures" in cols, "migration must add consecutive_failures"
    assert "last_failure_error" in cols, "migration must add last_failure_error"
    assert "spawn_failures" not in cols, "no legacy column should be synthesised"

    row = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row["consecutive_failures"] == 0
    assert row["last_failure_error"] is None

    # Idempotent second run must not raise either.
    kbc._migrate_add_optional_columns(conn)
    row_again = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row_again["consecutive_failures"] == 0
    assert row_again["last_failure_error"] is None
    conn.close()


# ---------------------------------------------------------------------------
# Gateway-embedded dispatcher: config, CLI warnings, daemon deprecation stub
# ---------------------------------------------------------------------------

def test_config_default_dispatch_in_gateway_is_true():
    """Default config must enable gateway-embedded dispatch out of the box.
    Flipping this default to false is a user-visible behaviour change and
    should require a conscious migration."""
    from hermes_cli.config import DEFAULT_CONFIG
    kanban = DEFAULT_CONFIG.get("kanban", {})
    assert kanban.get("dispatch_in_gateway") is True, (
        "kanban.dispatch_in_gateway default should be True; got "
        f"{kanban.get('dispatch_in_gateway')!r}"
    )
    interval = kanban.get("dispatch_interval_seconds")
    assert isinstance(interval, (int, float)) and interval >= 1, (
        f"dispatch_interval_seconds must be a positive number, got {interval!r}"
    )






def _make_create_ns(**overrides):
    """Build a Namespace suitable for kb_cli._cmd_create()."""
    ns = argparse.Namespace(
        title="x", body=None, assignee="worker",
        created_by="user", workspace="scratch", tenant=None,
        priority=0, parent=None, triage=False,
        idempotency_key=None, max_runtime=None, skills=None,
        json=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_cli_daemon_help_marks_deprecated():
    """The argparse help string on `daemon` mentions deprecation so users
    scanning `--help` see the migration before running the stub."""
    import argparse as _ap
    from hermes_cli import kanban as kb_cli
    root = _ap.ArgumentParser()
    subs = root.add_subparsers()
    kb_cli.build_parser(subs)
    # Walk the subparser tree to find the daemon action.
    daemon_help = None
    for action in root._actions:
        if isinstance(action, _ap._SubParsersAction):
            for name, parser in action.choices.items():
                if name == "kanban":
                    for sub_action in parser._actions:
                        if isinstance(sub_action, _ap._SubParsersAction):
                            for sname, _ in sub_action.choices.items():
                                if sname == "daemon":
                                    daemon_help = sub_action._choices_actions
                                    break
    # _choices_actions is a list of _ChoicesPseudoAction-like objects with .help
    found_deprecation = False
    if daemon_help:
        for act in daemon_help:
            if getattr(act, "dest", "") == "daemon":
                if "DEPRECATED" in (act.help or ""):
                    found_deprecation = True
                    break
    assert found_deprecation, (
        "daemon subparser help should be marked DEPRECATED so users see "
        "the migration guidance in `hermes kanban --help` output"
    )


# ---------------------------------------------------------------------------
# Gateway embedded dispatcher watcher
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("corrupt_exc", ["sqlite", "guard"])
def test_gateway_dispatcher_disables_corrupt_board_without_traceback(
    monkeypatch, tmp_path, caplog, corrupt_exc
):
    """Corrupt board DBs log one actionable error and stop retrying per tick."""
    import asyncio
    import logging
    import sqlite3

    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_connect as _kbc
    from hermes_cli import kanban_db_dispatch as _kbd

    runner = object.__new__(GatewayRunner)
    runner._running = True
    corrupt_db = tmp_path / "kanban.db"
    corrupt_db.write_text("not sqlite", encoding="utf-8")

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
            }
        },
    )
    monkeypatch.setattr(
        _kb,
        "list_boards",
        lambda include_archived=False: [{"slug": _kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(
        _kb,
        "read_board_metadata",
        lambda slug: {"slug": slug},
    )
    monkeypatch.setattr(_kb, "kanban_db_path", lambda board=None: corrupt_db)

    calls = {"connect": 0, "to_thread": 0}

    def _connect(*args, **kwargs):
        calls["connect"] += 1
        if corrupt_exc == "guard":
            raise _kbc.KanbanDbCorruptError(
                corrupt_db,
                corrupt_db.with_suffix(".db.corrupt.test.bak"),
                "sqlite refused to open file: database disk image is malformed",
            )
        raise sqlite3.DatabaseError("file is not a database")

    async def _to_thread(fn, *args, **kwargs):
        # PR salvage (#32857 commit 7): the dispatcher now reaps zombies at
        # the top of each tick via ``asyncio.to_thread(_kbd.reap_worker_zombies)``
        # BEFORE the per-board tick work. Each tick now issues 3 ``to_thread``
        # calls (reaper + ``_tick_once`` + ``_ready_nonempty``) instead of 2,
        # so this counter must reach 6 to allow the same 2 dispatch ticks the
        # pre-reaper test expected at 4. Connect counts in the assertion below
        # are unchanged.
        calls["to_thread"] += 1
        result = fn(*args, **kwargs)
        if calls["to_thread"] >= 6:
            runner._running = False
        return result

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(_kbc, "connect", _connect)
    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(
                runner._kanban_dispatcher_watcher(),
                timeout=3.0,
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    assert sum("not a valid SQLite database" in msg for msg in messages) == 1
    assert not any("tick failed on board" in msg for msg in messages)
    assert not any(record.exc_info for record in caplog.records)
    # First tick connect (dispatch) + two probes per `_has_ready_work` call
    # (ready then review, both via _kbc.connect). The second dispatch tick
    # skips the dispatch connect because the corrupt board fingerprint is
    # disabled, but the ready/review probes still each connect. PR f55d94a1e
    # added the review-column probe alongside the existing ready-column
    # probe, bumping this from 3 → 5.
    assert calls["connect"] == 5


# ---------------------------------------------------------------------------
# Hallucination gate (created_cards verify + prose scan)
# ---------------------------------------------------------------------------




def test_complete_can_retry_after_phantom_rejection(kanban_home):
    """A worker that hits the hallucinated-card gate must be able to
    retry kanban_complete on the same task — both with a corrected
    created_cards list and with an empty list (the documented escape
    hatch). Regression test for #22923, where workers were believed to
    be unrecoverable after the first rejection.
    """
    conn = kbc.connect()
    try:
        # Two parallel completing tasks so we can exercise both retry
        # shapes without status interference.
        parent_a = kb.create_task(conn, title="retry-empty", assignee="alice")
        kb.claim_task(conn, parent_a)
        parent_b = kb.create_task(conn, title="retry-corrected", assignee="alice")
        kb.claim_task(conn, parent_b)
        real = kb.create_task(
            conn, title="real-child", assignee="x", created_by="alice",
        )

        # First attempt: phantom in the list rejects, task stays running.
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent_a,
                summary="oops",
                created_cards=["t_phantomdeadbeef"],
            )
        assert kb.get_task(conn, parent_a).status == "running"

        # Retry with [] (escape hatch): gate is skipped, completion lands.
        ok = kb.complete_task(
            conn, parent_a,
            summary="retry without claims",
            created_cards=[],
        )
        assert ok is True
        assert kb.get_task(conn, parent_a).status == "done"

        # Same flow on parent_b, but recover via a corrected list rather
        # than the empty escape hatch.
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent_b,
                summary="oops",
                created_cards=[real, "t_anotherphantom"],
            )
        assert kb.get_task(conn, parent_b).status == "running"

        ok = kb.complete_task(
            conn, parent_b,
            summary="retry with corrected list",
            created_cards=[real],
        )
        assert ok is True
        assert kb.get_task(conn, parent_b).status == "done"

        # Both audit events landed; the eventual completion event is
        # also present on each task.
        for parent in (parent_a, parent_b):
            kinds = [
                r["kind"] for r in conn.execute(
                    "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
                    (parent,),
                )
            ]
            assert kinds.count("completion_blocked_hallucination") == 1
            assert kinds.count("completed") == 1
    finally:
        conn.close()




# ---------------------------------------------------------------------------
# Recovery helpers (reclaim + reassign)
# ---------------------------------------------------------------------------

def test_reclaim_task_resets_running_to_ready(kanban_home, monkeypatch):
    """Manual reclaim releases the claim, resets status, and emits a
    ``reclaimed`` event even when claim_expires has not passed."""
    import signal
    import time
    import secrets
    import hermes_cli.kanban_db as _kb
    conn = kbc.connect()
    try:
        t = kb.create_task(conn, title="stuck", assignee="broken")
        # Simulate a live claim (not expired).
        lock = f"{_kb._claimer_id().split(':', 1)[0]}:{secrets.token_hex(8)}"
        future = int(time.time()) + 3600
        killed: list[int] = []
        state = {"alive": True}

        def _signal(pid, sig):
            killed.append(sig)
            if sig == signal.SIGTERM:
                state["alive"] = False

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: state["alive"])
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 12345, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 12345, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()

        # release_stale_claims should NOT reclaim (not expired).
        assert kb.release_stale_claims(conn) == 0

        # reclaim_task should work immediately.
        assert kb.reclaim_task(conn, t, reason="test reason", signal_fn=_signal) is True

        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None

        import json as _json
        reclaim_evs = [
            _json.loads(r["payload"])
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='reclaimed'",
                (t,),
            )
        ]
        assert len(reclaim_evs) == 1
        assert reclaim_evs[0].get("manual") is True
        assert reclaim_evs[0].get("reason") == "test reason"
        assert reclaim_evs[0].get("termination_attempted") is True
        assert reclaim_evs[0].get("terminated") is True
        assert killed == [signal.SIGTERM]
    finally:
        conn.close()






# ---------------------------------------------------------------------------
# Unified failure counter — timeout + crash paths increment the same counter
# as spawn failures, and the circuit breaker trips after N consecutive
# failures regardless of which outcome caused them.
# ---------------------------------------------------------------------------




def _drive_worker_exit(conn, tid, fake_pid, raw_status):
    """Claim ``tid``, record ``raw_status`` for its dead worker pid, and run
    one reaper pass.

    Deliberately resolves ``hermes_cli.kanban_db`` fresh and uses that single
    module object for the exit registry, the liveness patch, AND the reaper:
    earlier tests in a full-suite run can reload the module, and recording
    the exit into one module object while reaping through another (stale)
    one makes ``_classify_worker_exit`` return ``unknown`` — silently turning
    a clean-exit protocol violation into a plain crash.
    """
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd
    host_prefix = _kb._claimer_id().split(":", 1)[0]
    claimed = _kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable for the next attempt"
    _kbd._set_worker_pid(conn, tid, fake_pid)
    _kbd._record_worker_exit(fake_pid, raw_status)
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        return _kbd.detect_crashed_workers(conn)
    finally:
        _kb._pid_alive = original_alive


def _drive_protocol_violation(conn, tid, fake_pid):
    """One clean-exit protocol violation reaper pass for ``tid``.

    os.W_EXITCODE(status=0, signal=0) == 0 on POSIX.
    """
    return _drive_worker_exit(conn, tid, fake_pid, 0)


def _drive_nonzero_crash(conn, tid, fake_pid):
    """One plain non-zero-exit crash reaper pass for ``tid``.

    W_EXITCODE(1, 0) == 256 — WIFEXITED True, WEXITSTATUS == 1.
    """
    return _drive_worker_exit(conn, tid, fake_pid, 256)


def test_protocol_violation_budget_not_consumed_by_other_failures(kanban_home):
    """Mixed failure kinds must not consume the violation retry budget.

    Regression for the #61233 review finding: expressed as a plain
    ``failure_limit`` over the unified ``consecutive_failures`` counter, the
    violation budget was consumed by earlier timeouts / nonzero exits. As a
    violation-only streak, a prior real crash must not eat violation
    retries, and below-budget violations must leave the unified counter
    untouched (so the two budgets stay independent).
    """
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="mixed", assignee="worker")

        # One real crash: unified counter ticks to 1 (below
        # DEFAULT_FAILURE_LIMIT=2 — task stays ready).
        _drive_nonzero_crash(conn, tid, 991000)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        # Two violations after it: streak 1 and 2 — both retry, unified
        # counter untouched. (Pre-fix: the crash consumed the budget and the
        # violations blocked well before three of them happened.)
        for i, pid in enumerate((991001, 991002)):
            _drive_protocol_violation(conn, tid, pid)
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"violation {i + 1} after a crash must still retry, "
                f"got {task.status}"
            )
            assert task.consecutive_failures == 1, (
                "below-budget violations must not tick the unified counter"
            )

        # Third consecutive violation: streak hits the bound — blocked.
        _drive_protocol_violation(conn, tid, 991003)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("protocol_violations") == \
            _kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT
    finally:
        conn.close()












def test_notify_sub_starts_caught_up_on_active_task(kanban_home):
    """A new subscription must NOT replay historical terminal events.

    Regression for issue #29905: `kanban_notify_subs.last_event_id` defaulted
    to 0, so subscribing to a task that already had terminal events in
    `task_events` replayed the entire backlog on the next notifier tick — 27
    stale subs produced a 100+ message burst at gateway boot. The cursor now
    snaps to the task's MAX(task_events.id) at creation: only events that
    occur AFTER subscribing are delivered.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="old task", assignee="w")
        # Historical terminal activity BEFORE anyone subscribes.
        kb.complete_task(conn, tid, result="done long ago")

        kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="123")
        sub = kbn.list_notify_subs(conn, tid)[0]
        assert int(sub["last_event_id"]) > 0, (
            "cursor must snap to MAX(task_events.id) at subscription time"
        )
        _, events = kbn.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        assert events == [], "historical events must not replay to a new sub"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Respawn guard — prev_worker_alive (defence in depth against a still-running
# worker being misclassified as dead and respawned beside itself)
# ---------------------------------------------------------------------------

def _spawn_stand_in_worker():
    """A genuine long-lived, portable child process to stand in for a worker
    pid -- ``sys.executable`` rather than the Unix ``sleep`` binary so this
    helper (used from unmarked, cross-platform tests) works on Windows too."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _end_run_as_reclaimed(conn, tid, *, outcome="crashed"):
    """Simulate a (possibly WRONG) crash/timeout/reclaim classification: reset
    the task row to ``ready`` and close its run — WITHOUT touching the actual
    worker process, exactly like a misclassified live worker. Returns the
    closed run's id."""
    run_id = kb.get_task(conn, tid).current_run_id
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        kb._end_run(conn, tid, outcome=outcome, status=outcome, error="test-induced close")
    return run_id


def _tamper_recorded_start_ticks(conn, tid, run_id):
    """Corrupt the ``spawned`` event's recorded ``start_ticks`` for ``run_id``
    so the non-reusable-identity check (signal 0 in
    ``_prev_worker_identity_plausible``) sees a mismatch against the pid's
    CURRENT live ``/proc`` value, forcing a fall-through to the weaker
    cmdline/timing signals.

    Models genuine pid reuse: the recorded value belonged to the ORIGINAL
    worker that has since exited; the pid was recycled to an unrelated
    process (the fixture's still-alive stand-in) afterward. Registering a
    still-alive process's pid via ``_set_worker_pid`` and then probing that
    same still-alive process necessarily reproduces its OWN identity exactly
    -- so, without this tamper, the fixture models "the same process,
    correctly identified" rather than a recycled pid (see PR 109491 review
    finding F-1)."""
    row = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'spawned' ORDER BY id DESC LIMIT 1",
        (tid, run_id),
    ).fetchone()
    assert row is not None, "test setup: no spawned event to tamper"
    payload = kb._json_dict(row["payload"])
    assert "start_ticks" in payload, (
        "test setup: this host did not stamp start_ticks (non-Linux or /proc "
        "unreadable) -- the tamper is meaningless without it"
    )
    payload["start_ticks"] = payload["start_ticks"] + 999_999
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = ? WHERE id = ?",
            (json.dumps(payload), row["id"]),
        )


def test_respawn_guard_blocks_spawn_when_prev_worker_pid_alive_on_this_host(
    kanban_home, all_assignees_spawnable,
):
    """A reclaimed card whose previous worker pid IS alive on this host must
    not respawn: ``check_respawn_guard`` returns ``prev_worker_alive``,
    ``dispatch_once`` does not spawn, a ``respawn_guarded`` event carries the
    pid/host/run id, and the card stays ``ready``."""
    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="prev-alive", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            run_id = claimed.current_run_id
            kbd._set_worker_pid(conn, tid, proc.pid)

            _end_run_as_reclaimed(conn, tid)

            assert kbd.check_respawn_guard(conn, tid) == "prev_worker_alive"

            spawned: list[int] = []
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
            )
            assert tid not in [s[0] for s in res.spawned]
            guarded = dict(res.respawn_guarded)
            assert guarded.get(tid) == "prev_worker_alive"
            assert not spawned, "guarded task must not be spawned"
            assert kb.get_task(conn, tid).status == "ready"

            events = [e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"]
            assert len(events) == 1
            payload = events[0].payload
            assert payload.get("reason") == "prev_worker_alive"
            assert payload.get("pid") == proc.pid
            assert payload.get("run_id") == run_id
            assert payload.get("host")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.linux_only
def test_respawn_guard_survives_deleted_spawned_claimed_events_via_run_metadata(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A garbage-collection pass on done/archived tasks can delete the
    spawned/claimed event rows for an older run, and a pid-source that only
    looked at those events would go blind the moment that happened (returned
    None, letting a second worker spawn beside the still-live first one).
    The run row's own ``metadata`` JSON is a second, durable pid source
    stamped at close time — reclaim via the real production path
    (``detect_crashed_workers``), delete the spawned/claimed events for that
    run, and the guard must still block on the metadata stamp alone."""
    import hermes_cli.kanban_db as _kb

    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="metadata-survives-event-gc", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            run_id = claimed.current_run_id
            kbd._set_worker_pid(conn, tid, proc.pid)

            # Force the reclaim path to classify this run as crashed even
            # though the decoy process is genuinely alive.
            monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
            monkeypatch.setattr(kbd, "_pid_alive", lambda pid: False)
            crashed = kbd.detect_crashed_workers(conn)
            assert tid in crashed

            # Sanity: the run row's metadata really was stamped at close time.
            run = kb.list_runs(conn, tid)[-1]
            assert run.id == run_id
            assert run.metadata.get("prev_worker_pid") == proc.pid
            assert run.metadata.get("prev_worker_host")

            # Delete the immutable spawned/claimed events for that run —
            # exactly what an event-retention gc pass does for a done/archived task.
            deleted = conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND run_id = ? "
                "AND kind IN ('spawned', 'claimed')",
                (tid, run_id),
            )
            conn.commit()
            assert deleted.rowcount > 0, "test setup: expected spawned/claimed rows to delete"

            # The guard must still block via the run-row metadata stamp.
            assert kbd.check_respawn_guard(conn, tid) == "prev_worker_alive"
            info = kbd._prev_worker_alive_guard_info(conn, tid)
            assert info == {
                "pid": proc.pid, "host": _kb._host_prefix().rstrip(":"), "run_id": run_id,
                "reason": "prev_worker_alive",
            }

            spawned: list[int] = []
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
            )
            assert tid not in [s[0] for s in res.spawned]
            assert not spawned, "metadata-backed guard must still block the respawn"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def test_respawn_guard_prev_worker_alive_escalates_after_five_ticks(
    kanban_home, all_assignees_spawnable,
):
    """A genuinely alive previous worker must not park the card forever
    invisibly. The first four ticks each append one ``respawn_guarded``
    event; the fifth consecutive tick stops appending per-tick rows, emits
    exactly one ``respawn_guard_escalated`` event naming the pid/run, and
    blocks the card ``needs_input`` so a human sees it."""
    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="prev-alive-escalate", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            kbd._set_worker_pid(conn, tid, proc.pid)
            _end_run_as_reclaimed(conn, tid)
            assert kbd.check_respawn_guard(conn, tid) == "prev_worker_alive"

            for tick in range(1, kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER):
                res = kbd.dispatch_once(
                    conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("must not spawn while guarded"),
                    ),
                )
                guarded = dict(res.respawn_guarded)
                assert guarded.get(tid) == "prev_worker_alive", f"tick {tick}"
                assert kb.get_task(conn, tid).status == "ready", f"tick {tick}"
                guarded_events = [
                    e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"
                ]
                assert len(guarded_events) == tick, f"tick {tick}"
                escalated_events = [
                    e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
                ]
                assert escalated_events == [], f"tick {tick}"

            # Fifth consecutive tick: escalate instead of another guarded row.
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("must not spawn on escalation"),
                ),
            )
            assert tid not in [s[0] for s in res.spawned]

            guarded_events = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"
            ]
            assert len(guarded_events) == kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER - 1, (
                "escalation must stop appending per-tick respawn_guarded rows"
            )
            escalated_events = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated_events) == 1
            assert escalated_events[0].payload.get("pid") == proc.pid
            assert escalated_events[0].payload.get("consecutive") == (
                kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER
            )

            task = kb.get_task(conn, tid)
            assert task.status == "blocked"
            assert task.block_kind == "needs_input"

            # A sixth tick must not fire a second escalation event or touch
            # the (now blocked) card again.
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("must not spawn after escalation"),
                ),
            )
            escalated_events = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated_events) == 1, "must not re-escalate the same pid"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.linux_only
def test_respawn_guard_recycled_pid_does_not_block_genuine_pid_does(
    kanban_home, all_assignees_spawnable,
):
    """A pid that merely OCCUPIES the recorded slot but is NOT our worker --
    its /proc start time predates the run by far more than the identity
    tolerance, and its cmdline does not reference the run's workspace --
    must NOT block a legitimate respawn (closes M5: an identity check that
    always returns True would wrongly block here). A genuine worker pid in
    an otherwise identical fixture -- its cmdline carries the workspace path
    verbatim, the strong identity signal -- DOES block, even though the same
    far-backdated run start time means the timing signal alone could not
    have decided it."""
    # Case (a): a decoy/recycled pid.
    decoy = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="recycled-pid", assignee="alice")
            workspace = f"/tmp/kanban-test-workspace-{tid}"
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_path = ? WHERE id = ?", (workspace, tid),
                )
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, decoy.pid)
            run_id = _end_run_as_reclaimed(conn, tid)
            # Tamper the recorded non-reusable identity (start_ticks) so the
            # decoy -- alive and genuinely occupying this pid slot -- no
            # longer matches its OWN recorded identity: this models the pid
            # having been reused by an unrelated process since the original
            # worker recorded here exited (signal 0 must fall through to the
            # weaker cmdline/timing signals below, not short-circuit on a
            # self-match). See F-1, PR 109491 review.
            _tamper_recorded_start_ticks(conn, tid, run_id)
            # Backdate the closed run's started_at far outside the identity
            # tolerance so ONLY the cmdline signal could plausibly accept
            # it -- and the decoy's cmdline carries no workspace reference.
            old = int(time.time()) - 100_000
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id),
                )
            with open(f"/proc/{decoy.pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode()
            assert workspace not in cmdline, "test setup: decoy must not carry the workspace"

            assert kbd.check_respawn_guard(conn, tid) is None, (
                "a recycled/unrelated pid must not block a legitimate respawn"
            )
            spawned: list[int] = []
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
            )
            assert tid in [s[0] for s in res.spawned]
            assert spawned, "a recycled pid must not guard a legitimate respawn"
    finally:
        decoy.terminate()
        try:
            decoy.wait(timeout=5)
        except Exception:
            decoy.kill()
            decoy.wait(timeout=5)

    # Case (b): a genuine worker -- same fixture shape (run backdated
    # identically far outside the time tolerance), but the process's cmdline
    # carries the task's workspace path verbatim.
    with kbc.connect() as conn:
        tid2 = kb.create_task(conn, title="genuine-worker", assignee="alice")
        workspace2 = f"/tmp/kanban-test-workspace-{tid2}"
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path = ? WHERE id = ?", (workspace2, tid2),
            )
        claimed2 = kb.claim_task(conn, tid2)
        assert claimed2 is not None
    genuine = subprocess.Popen(
        ["/bin/sh", "-c", f"sleep 300  # worker for {workspace2}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.2)
        with kbc.connect() as conn:
            kbd._set_worker_pid(conn, tid2, genuine.pid)
            run_id2 = _end_run_as_reclaimed(conn, tid2)
            old2 = int(time.time()) - 100_000
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?", (old2, run_id2),
                )
            with open(f"/proc/{genuine.pid}/cmdline", "rb") as f:
                cmdline2 = f.read().replace(b"\x00", b" ").decode()
            assert workspace2 in cmdline2, "test setup: genuine worker must carry the workspace"

            assert kbd.check_respawn_guard(conn, tid2) == "prev_worker_alive", (
                "a genuine worker's pid, identified via the workspace-path "
                "cmdline signal, must block the respawn"
            )
            res2 = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("must not spawn while a genuine worker is alive"),
                ),
            )
            assert tid2 not in [s[0] for s in res2.spawned]
    finally:
        genuine.terminate()
        try:
            genuine.wait(timeout=5)
        except Exception:
            genuine.kill()
            genuine.wait(timeout=5)


def _end_run_as_reclaimed_into_review(conn, tid, *, outcome="crashed"):
    """Simulate a misclassified crash landing the card back in the REVIEW
    lane -- the shape a reviewer's worker leaves behind: the run is closed
    exactly like :func:`_end_run_as_reclaimed`, but the card is parked in
    ``review`` instead of ``ready``. ``block_task`` cannot move a ``review``
    row, so the guard's own escalation bound is the only thing that stops
    it from re-parking the card forever."""
    run_id = kb.get_task(conn, tid).current_run_id
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='review', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        kb._end_run(conn, tid, outcome=outcome, status=outcome, error="test-induced close")
    return run_id


def test_respawn_guard_prev_worker_alive_escalation_bound_holds_in_review_lane(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The prev_worker_alive escalation bound's already-escalated early
    return is a no-op in the ready lane once ``block_task`` moves the card
    out of ``ready`` -- the REVIEW lane is the only lane where it is
    load-bearing, because ``block_task`` cannot touch a ``review`` row and
    nothing else halts the tick loop (closes M4). Hold a card in review past
    the bound with a still-alive previous worker, then keep ticking well
    past it (10 further ticks): exactly ONE escalated marker must ever
    appear, and no further per-tick ``respawn_guarded`` rows may be
    appended."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="review-lane-escalate", assignee="reviewer")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, proc.pid)
            _end_run_as_reclaimed_into_review(conn, tid)
            assert kb.get_task(conn, tid).status == "review"
            assert kbd.check_respawn_guard(conn, tid, lane="review") == "prev_worker_alive"

            for tick in range(1, kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER):
                res = kbd.dispatch_once(
                    conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("must not spawn while guarded"),
                    ),
                )
                guarded = dict(res.respawn_guarded)
                assert guarded.get(tid) == "prev_worker_alive", f"tick {tick}"
                assert kb.get_task(conn, tid).status == "review", f"tick {tick}"

            # Escalation tick.
            kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("must not spawn on escalation"),
                ),
            )
            escalated = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated) == 1
            guarded_at_escalation = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"
            ]
            assert len(guarded_at_escalation) == kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER - 1

            # block_task cannot move a 'review' row: the card MUST still be
            # 'review', and the stall must be visible via last_failure_error
            # instead of a block event.
            task = kb.get_task(conn, tid)
            assert task.status == "review"
            assert task.last_failure_error and str(proc.pid) in task.last_failure_error

            # At least 10 further ticks past the bound: exactly one
            # escalated marker must EVER exist, and no further per-tick
            # respawn_guarded rows may be appended.
            for tick in range(10):
                res = kbd.dispatch_once(
                    conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("must not spawn after escalation"),
                    ),
                )
                assert tid not in [s[0] for s in res.spawned], f"post-escalation tick {tick}"

            escalated_after = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated_after) == 1, (
                "the review lane must never re-escalate the same still-alive pid"
            )
            guarded_after = [
                e for e in kb.list_events(conn, tid) if e.kind == "respawn_guarded"
            ]
            assert len(guarded_after) == kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER - 1, (
                "escalation must stop appending per-tick respawn_guarded "
                "rows in the review lane too"
            )
            assert kb.get_task(conn, tid).status == "review"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def test_respawn_guard_consults_latest_ended_run_not_an_older_one(
    kanban_home, all_assignees_spawnable,
):
    """With several ended runs on one card, the guard must consult only the
    LATEST ended run's pid -- an older run's still-alive pid must never
    block a card whose most recent attempt already ended with a dead pid,
    and conversely a stale dead pid on an older run must never mask a
    genuinely alive pid on the latest run (closes M6). Both directions are
    asserted."""

    def _dead_pid() -> int:
        p = subprocess.Popen(
            [sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        p.wait(timeout=5)
        return p.pid

    # Direction 1: older run's pid is ALIVE, latest run's pid is DEAD ->
    # must NOT block.
    alive_proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="latest-run-wins-dead", assignee="alice")
            kb.claim_task(conn, tid)
            kbd._set_worker_pid(conn, tid, alive_proc.pid)
            _end_run_as_reclaimed(conn, tid)  # run 1 (older): alive pid

            dead_pid = _dead_pid()
            kb.claim_task(conn, tid)
            kbd._set_worker_pid(conn, tid, dead_pid)
            _end_run_as_reclaimed(conn, tid)  # run 2 (latest): dead pid

            runs = kb.list_runs(conn, tid, include_active=False)
            assert len(runs) == 2, "test setup: expected exactly two closed runs"

            assert kbd.check_respawn_guard(conn, tid) is None, (
                "an older run's still-alive pid must not block a card whose "
                "latest attempt ended with a dead pid"
            )
            spawned: list[int] = []
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
            )
            assert tid in [s[0] for s in res.spawned]
            assert spawned
    finally:
        alive_proc.terminate()
        try:
            alive_proc.wait(timeout=5)
        except Exception:
            alive_proc.kill()
            alive_proc.wait(timeout=5)

    # Direction 2: older run's pid is DEAD, latest run's pid is ALIVE ->
    # must block.
    dead_pid_2 = _dead_pid()
    alive_proc_2 = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid2 = kb.create_task(conn, title="latest-run-wins-alive", assignee="alice")
            kb.claim_task(conn, tid2)
            kbd._set_worker_pid(conn, tid2, dead_pid_2)
            _end_run_as_reclaimed(conn, tid2)  # run 1 (older): dead pid

            kb.claim_task(conn, tid2)
            kbd._set_worker_pid(conn, tid2, alive_proc_2.pid)
            _end_run_as_reclaimed(conn, tid2)  # run 2 (latest): alive pid

            runs2 = kb.list_runs(conn, tid2, include_active=False)
            assert len(runs2) == 2, "test setup: expected exactly two closed runs"

            assert kbd.check_respawn_guard(conn, tid2) == "prev_worker_alive", (
                "the latest run's alive pid must block even though an "
                "older run's pid is dead"
            )
            res2 = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                    AssertionError("must not spawn while the latest run's worker is alive"),
                ),
            )
            assert tid2 not in [s[0] for s in res2.spawned]
    finally:
        alive_proc_2.terminate()
        try:
            alive_proc_2.wait(timeout=5)
        except Exception:
            alive_proc_2.kill()
            alive_proc_2.wait(timeout=5)


# ---------------------------------------------------------------------------
# Respawn guard — review findings on PR 109491 (F-1 fixture repair above;
# F-2 the three missing test groups below; F-3 the narrowed cross-host hold)
# ---------------------------------------------------------------------------

@pytest.mark.windows_only
def test_respawn_guard_windows_probe_does_not_signal_live_child():
    """(Review ask 1) On native Windows the liveness probe must route
    through ``gateway.status._pid_exists`` (psutil, else the
    OpenProcess/WaitForSingleObject ctypes path), never a bare
    ``os.kill(pid, 0)`` -- ``sig=0`` on Windows is ``CTRL_C_EVENT``
    broadcast to the whole console process group (bpo-14484), which could
    signal or kill an unrelated process sharing the console, not merely
    probe existence. A genuinely live child process must be reported alive,
    and must remain alive and unharmed by the probe call itself -- this is
    the real child-survival proof the review asked for, run on native
    Windows rather than a platform-patched Linux stand-in (this repo's
    testing rule forbids faking ``sys.platform`` for host-dependent
    behaviour; see ``tests/conftest.py``)."""
    proc = _spawn_stand_in_worker()
    try:
        assert kbd._prev_worker_alive_probe(proc.pid) is True
        assert proc.poll() is None, (
            "the liveness probe must not have signalled or terminated the child"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def test_respawn_guard_cross_host_unclean_close_holds_card(
    kanban_home, all_assignees_spawnable,
):
    """(Review ask 2 / F-3) A run closed as a crash/timeout/reclaim/stale
    misclassification on a DIFFERENT host must hold the card fail-closed:
    this host has no ``/proc`` route to verify the remote pid, and a live
    remote worker sharing that run outcome is still plausible."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cross-host-crash", assignee="alice")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        _end_run_as_reclaimed(conn, tid, outcome="crashed")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps({"prev_worker_pid": 424242, "prev_worker_host": "otherbox"}), run_id),
            )
        info = kbd._prev_worker_alive_guard_info(conn, tid)
        assert info == {
            "pid": 424242, "host": "otherbox", "run_id": run_id,
            "reason": "prev_worker_cross_host_unknown",
        }, info
        assert kbd.check_respawn_guard(conn, tid) == "prev_worker_cross_host_unknown"

        spawned: list[int] = []
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
        )
        assert not spawned, "a crash-closed cross-host run must hold the card"
        assert tid not in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "prev_worker_cross_host_unknown"


@pytest.mark.parametrize("clean_outcome", ["completed", "review_requested"])
def test_respawn_guard_cross_host_clean_close_fails_open(
    kanban_home, all_assignees_spawnable, clean_outcome,
):
    """(Review ask 2 / F-3) A run closed CLEANLY (``completed`` -- an
    ordinary ``kanban_complete``, or ``review_requested`` -- the standard
    implementer-to-reviewer handoff) on a DIFFERENT host must NOT hold the
    card: the remote worker is known to have stopped on purpose, not merely
    unreachable, so this is the pre-existing fail-open behaviour, unlike the
    genuinely-still-running case above. Holding this case was the F-3
    regression: it would have blocked every multi-host board's ordinary
    handoff after 5 ticks."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=f"cross-host-clean-{clean_outcome}", assignee="alice")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        _end_run_as_reclaimed(conn, tid, outcome=clean_outcome)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps({"prev_worker_pid": 424242, "prev_worker_host": "otherbox"}), run_id),
            )
            if clean_outcome == "completed":
                # ``check_respawn_guard``'s UNRELATED ``recent_success`` guard
                # (a separate, correct duplicate-work protection, not part of
                # this fix) would otherwise hold a just-completed task within
                # its own success window regardless of host. Append a
                # ``status`` event, exactly the "explicit re-queue after
                # success" exception that guard documents, so this test
                # isolates the cross-host behaviour under review.
                kb._append_event(conn, tid, "status", {"to": "ready"})
        assert kbd._prev_worker_alive_guard_info(conn, tid) is None, clean_outcome
        assert kbd.check_respawn_guard(conn, tid) is None, clean_outcome

        spawned: list[int] = []
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
        )
        assert spawned, (
            f"a cleanly-closed ({clean_outcome}) cross-host run must not hold the card"
        )
        assert tid in [s[0] for s in res.spawned]


def test_respawn_guard_cross_host_claim_lock_fallback_unclean_close_holds_card(
    kanban_home, all_assignees_spawnable,
):
    """(QA pr109491-v2 finding N-2) The SAME F-3 narrowing also guards the
    claim-lock fallback path in ``_prev_worker_alive_guard_info`` -- the
    ``if lock and cross_host_holds`` branch used for rows with no
    ``prev_worker_pid`` run-metadata stamp (older rows, or any row where the
    durable metadata source was never written). A run closed as a
    crash/timeout/reclaim/stale misclassification, whose only host evidence
    is a ``claimed`` event's lock naming a DIFFERENT host, must still hold
    the card fail-closed via this fallback -- exactly like the
    metadata-sourced case above, just reached through the older source."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cross-host-crash-claimlock", assignee="alice")
        claimed = kb.claim_task(conn, tid, claimer="otherbox:424243")
        assert claimed is not None
        kbd._set_worker_pid(conn, tid, 424243)
        _end_run_as_reclaimed(conn, tid, outcome="crashed")
        info = kbd._prev_worker_alive_guard_info(conn, tid)
        assert info == {
            "pid": 424243, "host": "otherbox", "run_id": claimed.current_run_id,
            "reason": "prev_worker_cross_host_unknown",
        }, info
        assert kbd.check_respawn_guard(conn, tid) == "prev_worker_cross_host_unknown"

        spawned: list[int] = []
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
        )
        assert not spawned, (
            "a crash-closed cross-host run must hold the card via the claim-lock fallback"
        )
        assert tid not in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "prev_worker_cross_host_unknown"


@pytest.mark.parametrize("clean_outcome", ["completed", "review_requested"])
def test_respawn_guard_cross_host_claim_lock_fallback_clean_close_fails_open(
    kanban_home, all_assignees_spawnable, clean_outcome,
):
    """(QA pr109491-v2 finding N-2) The claim-lock fallback counterpart of
    ``test_respawn_guard_cross_host_clean_close_fails_open``: a run closed
    CLEANLY on a different host, with no ``prev_worker_pid`` run-metadata
    stamp so the guard falls back to the ``claimed`` event's lock, must NOT
    hold the card -- the remote worker is known to have stopped on
    purpose."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title=f"cross-host-clean-claimlock-{clean_outcome}", assignee="alice",
        )
        claimed = kb.claim_task(conn, tid, claimer="otherbox:424243")
        assert claimed is not None
        kbd._set_worker_pid(conn, tid, 424243)
        _end_run_as_reclaimed(conn, tid, outcome=clean_outcome)
        if clean_outcome == "completed":
            # Same isolation as the metadata-sourced clean-close test: keep
            # the unrelated 'recent_success' guard from parking this task
            # inside its own success window regardless of host.
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "status", {"to": "ready"})
        assert kbd._prev_worker_alive_guard_info(conn, tid) is None, clean_outcome
        assert kbd.check_respawn_guard(conn, tid) is None, clean_outcome

        spawned: list[int] = []
        res = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
        )
        assert spawned, (
            f"a cleanly-closed ({clean_outcome}) cross-host run (claim-lock fallback) "
            "must not hold the card"
        )
        assert tid in [s[0] for s in res.spawned]


@pytest.mark.linux_only
def test_respawn_guard_identity_mismatch_does_not_block_genuine_match_does(
    kanban_home, all_assignees_spawnable,
):
    """(Review ask 3) Non-reusable identity (``start_ticks`` + ``boot_id``)
    is signal 0, decisive over the weaker cmdline/timing signals: a live pid
    whose RECORDED identity does not match its CURRENT identity (the
    pid-reuse shape) must not block, even though the run started only
    moments ago (the pre-existing ~600s time-window signal alone would have
    wrongly accepted it as the same worker). The identical fixture with an
    UNTAMPERED recorded identity DOES block, purely on the identity signal,
    even with the run backdated far outside that time window and a cmdline
    that does not reference the task's workspace (so neither of the two
    weaker signals could have decided it either way)."""
    # Case (a): recorded identity mismatch (genuine pid reuse) -- must NOT block.
    decoy = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="ask3-mismatch", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, decoy.pid)
            run_id = _end_run_as_reclaimed(conn, tid)
            _tamper_recorded_start_ticks(conn, tid, run_id)
            assert kbd._prev_worker_alive_probe(decoy.pid) is True
            assert kbd._prev_worker_alive_guard_info(conn, tid) is None, (
                "a live pid whose recorded identity mismatches must not block"
            )
            assert kbd.check_respawn_guard(conn, tid) is None
    finally:
        decoy.terminate()
        try:
            decoy.wait(timeout=5)
        except Exception:
            decoy.kill()
            decoy.wait(timeout=5)

    # Case (b): untampered identity -- must block on the identity signal alone.
    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid2 = kb.create_task(conn, title="ask3-match", assignee="alice")
            workspace = f"/tmp/kanban-test-workspace-{tid2}"
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET workspace_path = ? WHERE id = ?", (workspace, tid2),
                )
            claimed2 = kb.claim_task(conn, tid2)
            assert claimed2 is not None
            kbd._set_worker_pid(conn, tid2, proc.pid)
            run_id2 = _end_run_as_reclaimed(conn, tid2)
            old = int(time.time()) - 100_000
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id2),
                )
            with open(f"/proc/{proc.pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode()
            assert workspace not in cmdline, "test setup: must not carry the workspace"
            info = kbd._prev_worker_alive_guard_info(conn, tid2)
            assert info is not None and info["reason"] == "prev_worker_alive", (
                "an untampered identity match must block purely on signal 0"
            )
            assert kbd.check_respawn_guard(conn, tid2) == "prev_worker_alive"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def test_respawn_guard_cross_host_takeover_lifecycle_unblocks_and_respawns(
    kanban_home, all_assignees_spawnable,
):
    """(PR 109491 review, remaining blocker) Full lifecycle in order:

    1. A run closes unclean (``crashed``) on a DIFFERENT host, in the
       REVIEW lane -- the guard holds fail-closed
       (``prev_worker_cross_host_unknown``): this host cannot verify the
       remote pid's liveness.
    2. The hold survives ``_PREV_WORKER_ALIVE_ESCALATE_AFTER`` consecutive
       dispatch ticks and escalates (``last_failure_error`` stamped; the
       review lane can't be moved to ``blocked`` by ``block_task``, so the
       card stays ``review`` -- see
       ``test_respawn_guard_prev_worker_alive_escalation_bound_holds_in_review_lane``).
    3. A bare ``reopen-review`` (no takeover) is the CONTROL proving the
       reviewer's exact complaint: it restores the card to ``ready`` but
       does NOT retire the guard's evidence, so the very next tick would
       hold and eventually re-escalate again.
    4. A human instead runs ``hermes kanban reopen-review --takeover`` (via
       ``run_slash``, the real CLI path). This durably retires the
       cross-host guard's evidence for that specific closed run.
    5. The very next dispatch tick spawns the card -- it does NOT see
       ``prev_worker_cross_host_unknown`` again and does NOT re-escalate.

    (The READY lane's own escalation incidentally synthesizes a fresh
    ``blocked`` run with no prev-worker metadata, which happens to become
    the new "latest ended run" the guard reads next -- masking the original
    evidence as a side effect of ``block_task``, not a deliberate retirement.
    The REVIEW lane cannot do that (``block_task`` never touches a
    ``review`` row), so it is the faithful reproduction of the reviewer's
    complaint and the case ``--takeover`` exists for.)
    """
    import hermes_cli.config as cfgmod

    def _dispatch_once_review(conn):
        return kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not spawn while cross-host guarded"),
            ),
        )

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cross-host-review-takeover", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        _end_run_as_reclaimed_into_review(conn, tid, outcome="crashed")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps({"prev_worker_pid": 424242, "prev_worker_host": "otherbox"}), run_id),
            )
        assert kb.get_task(conn, tid).status == "review"

        # Step 1: the cross-host hold is in effect in the review lane.
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "prev_worker_cross_host_unknown"

        # Step 2: drive it to escalation, same shape as the existing
        # review-lane escalation-bound test.
        for tick in range(1, kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER):
            res = _dispatch_once_review(conn)
            guarded = dict(res.respawn_guarded)
            assert guarded.get(tid) == "prev_worker_cross_host_unknown", f"tick {tick}"
            assert kb.get_task(conn, tid).status == "review", f"tick {tick}"

        res = _dispatch_once_review(conn)
        assert tid not in [s[0] for s in res.spawned]
        task = kb.get_task(conn, tid)
        assert task.status == "review", "block_task cannot move a review row"
        assert task.last_failure_error and "prev_worker_cross_host_unknown" in task.last_failure_error
        escalated_events = [
            e for e in kb.list_events(conn, tid) if e.kind == "respawn_guard_escalated"
        ]
        assert len(escalated_events) == 1
        assert escalated_events[0].payload.get("reason") == "prev_worker_cross_host_unknown"

        # Step 3 (control): a bare reopen-review restores 'ready' but does
        # NOT retire the guard's evidence -- the reviewer's exact complaint.
        assert kb.reopen_review_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "prev_worker_cross_host_unknown", (
            "control: a bare reopen-review must not have retired the guard evidence"
        )

    # Re-create the same held-and-escalated review-lane state for the
    # takeover path (the control run above already consumed/reopened tid).
    prior_load_config = cfgmod.load_config
    cfgmod.load_config = lambda *a, **k: {"kanban": {"review_dispatch": True}}
    try:
        with kbc.connect() as conn:
            tid2 = kb.create_task(conn, title="cross-host-review-takeover-2", assignee="reviewer")
            claimed2 = kb.claim_task(conn, tid2)
            assert claimed2 is not None
            run_id2 = claimed2.current_run_id
            _end_run_as_reclaimed_into_review(conn, tid2, outcome="crashed")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps({"prev_worker_pid": 424242, "prev_worker_host": "otherbox"}), run_id2),
                )
            for _tick in range(1, kbd._PREV_WORKER_ALIVE_ESCALATE_AFTER):
                _dispatch_once_review(conn)
            _dispatch_once_review(conn)
            assert kb.get_task(conn, tid2).status == "review"
            escalated_events2 = [
                e for e in kb.list_events(conn, tid2) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated_events2) == 1

        # Step 4: human takeover via the real CLI path (``run_slash`` drives
        # the actual argparse + _cmd_reopen_review code).
        out = run_slash(f"reopen-review {tid2} --takeover")
        assert "Reopened" in out and tid2 in out
        assert "takeover" in out.lower()

        with kbc.connect() as conn:
            task2 = kb.get_task(conn, tid2)
            assert task2.status == "ready", "takeover must restore the card to ready"

            # The guard's evidence for that closed run is durably retired.
            assert kbd._prev_worker_alive_guard_info(conn, tid2) is None
            assert kbd.check_respawn_guard(conn, tid2) is None

            ack_events = [e for e in kb.list_events(conn, tid2) if e.kind == "prev_worker_ack"]
            assert len(ack_events) == 1
            assert ack_events[0].payload.get("run_id") == run_id2
            assert ack_events[0].payload.get("reason") == "prev_worker_cross_host_unknown"

            run = kb.list_runs(conn, tid2)[-1]
            assert run.id == run_id2
            assert run.metadata.get("prev_worker_ack") is True
            # The original evidence is preserved for audit, not deleted/rewritten.
            assert run.metadata.get("prev_worker_pid") == 424242
            assert run.metadata.get("prev_worker_host") == "otherbox"

            # Step 5: the very next dispatch tick actually spawns the card
            # instead of re-guarding/re-escalating.
            spawned: list[int] = []
            res = kbd.dispatch_once(
                conn, spawn_fn=lambda *a, **k: (spawned.append(1), 999)[1],
            )
            assert tid2 in [s[0] for s in res.spawned], (
                "the card must spawn on the tick after takeover, not re-escalate"
            )
            assert spawned == [1]
            assert dict(res.respawn_guarded).get(tid2) is None
            # No second escalation was recorded.
            escalated_events3 = [
                e for e in kb.list_events(conn, tid2) if e.kind == "respawn_guard_escalated"
            ]
            assert len(escalated_events3) == 1, "the takeover tick must not re-escalate"
    finally:
        cfgmod.load_config = prior_load_config


def test_takeover_ack_does_not_commit_when_the_flip_it_precedes_fails(
    kanban_home, all_assignees_spawnable,
):
    """(F-1, PR 109491 QA) A failed ``reopen-review --takeover`` -- the id
    isn't actually in ``review`` -- must leave the prev-worker guard's
    evidence completely intact: no ``prev_worker_ack`` event, the run row's
    ``metadata`` unstamped, and ``check_respawn_guard`` still returning the
    original hold. Before the fix, ``acknowledge_prev_worker_guard`` ran in
    its own transaction ahead of the flip and ``_bulk_apply`` had no
    rollback, so the ack committed unconditionally even though
    ``reopen_review_task`` never touched the row (QA's live reproduction:
    ``cannot reopen t_f587abf9 (not in review?)`` while the ack stamp was
    already ``True`` and the guard had gone silent)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="takeover-ack-atomicity", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        # Cross-host unclean close -- the guard holds fail-closed, and the
        # card lands back in READY (not review), so a subsequent
        # ``reopen-review`` genuinely cannot apply.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL WHERE id=?",
                (tid,),
            )
            kb._end_run(conn, tid, outcome="crashed", status="crashed", error="test-induced close")
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps({"prev_worker_pid": 424242, "prev_worker_host": "otherbox"}), run_id),
            )
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "prev_worker_cross_host_unknown"

        # RED (pre-fix): this used to report failure while silently
        # retiring the guard anyway.
        out = run_slash(f"reopen-review {tid} --takeover")
        assert "cannot reopen" in out and tid in out, out
        assert "Reopened" not in out

        task = kb.get_task(conn, tid)
        assert task.status == "ready", "a failed reopen must not move the card"

        ack_events = [e for e in kb.list_events(conn, tid) if e.kind == "prev_worker_ack"]
        assert ack_events == [], (
            "a failed flip must leave the guard's ack evidence untouched -- "
            "the ack must not commit when the flip it precedes fails"
        )
        assert kbd.check_respawn_guard(conn, tid) == "prev_worker_cross_host_unknown", (
            "guard evidence must survive a failed takeover unchanged"
        )
        run = kb.list_runs(conn, tid)[-1]
        assert run.id == run_id
        assert not run.metadata.get("prev_worker_ack"), (
            "the run row must not be stamped by a takeover whose flip failed"
        )


@pytest.mark.linux_only
def test_takeover_refuses_a_same_host_alive_prev_worker(
    kanban_home, all_assignees_spawnable,
):
    """(F-2, PR 109491 QA) ``--takeover`` must refuse to retire a SAME-HOST
    ``prev_worker_alive`` hold when the recorded pid is genuinely still
    running -- verified with a real child process, not a mock. Before the
    fix, the short-circuit in ``_prev_worker_alive_guard_info`` returned
    ``None`` for ANY ack reason before ever reading a pid, so ``--takeover``
    retired a hold the host could directly prove was still true -- reopening
    exactly the duplicate-writer case this PR exists to prevent (QA's live
    reproduction: guard info named a real, running pid; takeover reported
    success; the guard cleared; the child was still alive)."""
    proc = _spawn_stand_in_worker()
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="takeover-refuses-alive", assignee="alice")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            kbd._set_worker_pid(conn, tid, proc.pid)
            _end_run_as_reclaimed(conn, tid)
            assert kbd.check_respawn_guard(conn, tid) == "prev_worker_alive"

            # Move the card to 'blocked' (no reason -> no run synthesized,
            # so the guard-held closed run stays the latest one) so
            # ``unblock --takeover`` has a flip that WOULD otherwise
            # succeed -- isolating the F-2 refusal from the F-1 rollback.
            assert kb.block_task(conn, tid) is True
            assert kb.get_task(conn, tid).status == "blocked"

        assert proc.poll() is None, "test setup: the worker must genuinely be alive"

        out = run_slash(f"unblock {tid} --takeover")
        assert "Unblocked" not in out
        assert str(proc.pid) in out and "still alive" in out.lower(), out

        with kbc.connect() as conn:
            task = kb.get_task(conn, tid)
            assert task.status == "blocked", "a refused takeover must not move the card"
            ack_events = [e for e in kb.list_events(conn, tid) if e.kind == "prev_worker_ack"]
            assert ack_events == [], "a refused takeover must not stamp the ack"
            assert kbd.check_respawn_guard(conn, tid) == "prev_worker_alive", (
                "the guard must still hold on the genuinely-alive pid"
            )
        assert proc.poll() is None, "the refused takeover must not have touched the live worker"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


_WORKER_LOG_TAIL = (
    "Query: work kanban task\n"
    "╭─ ☤ Hermes ───────────────────╮\n"
    "│ the board protocol requires reassigning this card to orchestrator, but the │\n"
    "│ native kanban_* tools available here have no reassignment operation.       │\n"
    "╰──────────────────────────────╯\n"
    "\nResume this session with:\n  hermes --resume 20260915_000000_abc\n\n"
    "Session:        20260915_000000_abc\nMessages:       3 (1 user, 2 tool calls)\n"
)


@pytest.mark.parametrize("drive", [_drive_protocol_violation, _drive_nonzero_crash])
def test_dead_worker_reap_surfaces_the_workers_own_last_output(kanban_home, drive):
    """Regression for #88603 / #46593: a worker that explained why it could not comply
    (or printed a provider error) and then exited must have that text on the board and
    on the reap event — with the CLI exit summary trimmed — instead of only the canned label."""
    import hermes_cli.kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="handoff", assignee="worker")
        log_path = kb.worker_log_path(tid)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(_WORKER_LOG_TAIL)

        drive(conn, tid, 991100)

        task = kb.get_task(conn, tid)
        assert "no reassignment operation" in (task.last_failure_error or "")
        assert "Resume this session" not in (task.last_failure_error or "")
        assert "Query:" not in (task.last_failure_error or "")
        assert "│" not in (task.last_failure_error or "")
        events = [e for e in kb.list_events(conn, tid) if e.kind in ("protocol_violation", "crashed")]
        assert len(events) == 1
        assert "no reassignment operation" in (events[0].payload or {}).get("worker_output", "")
    finally:
        conn.close()


def test_dead_worker_reap_reads_the_log_of_the_dispatching_board(kanban_home):
    """The reap must read the worker log under the board the tick runs for, not the
    ambient "current" board — otherwise every non-default board silently gets the canned
    message (the #88603 review finding)."""
    import hermes_cli.kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    assert kb.get_current_board() == "default"
    board = "other-board"
    conn = kbc.connect(board=board)
    try:
        tid = kb.create_task(conn, title="handoff", assignee="worker")
        log_path = kb.worker_log_path(tid, board=board)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(_WORKER_LOG_TAIL)
        host_prefix = kb._claimer_id().split(":", 1)[0]
        assert kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock") is not None
        kbd._set_worker_pid(conn, tid, 991101)
        kbd._record_worker_exit(991101, 0)
        original_alive = kb._pid_alive
        kb._pid_alive = lambda p: False
        try:
            kbd.detect_crashed_workers(conn, board=board)
        finally:
            kb._pid_alive = original_alive
        task = kb.get_task(conn, tid)
        assert "no reassignment operation" in (task.last_failure_error or "")
    finally:
        conn.close()
