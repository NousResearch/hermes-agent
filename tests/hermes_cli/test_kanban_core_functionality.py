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
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
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
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123", user_id="u1",
            notifier_profile="default",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "42",
            },
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["platform"] == "telegram"
        assert subs[0]["notifier_profile"] == "default"
        assert subs[0]["delivery_metadata"] == {
            "chat_type": "dm",
            "telegram_reply_to_message_id": "42",
        }
        # Duplicate add is a no-op.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            delivery_metadata={
                "chat_type": "dm",
                "telegram_reply_to_message_id": "43",
            },
        )
        assert len(kb.list_notify_subs(conn, tid)) == 1
        assert kb.list_notify_subs(conn, tid)[0]["delivery_metadata"][
            "telegram_reply_to_message_id"
        ] == "43"
        # Distinct thread is a new row.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            thread_id="5",
        )
        assert len(kb.list_notify_subs(conn, tid)) == 2
        # Remove one.
        ok = kb.remove_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
        )
        assert ok is True
        assert len(kb.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()


def test_notify_claim_is_single_owner_and_rewindable(kanban_home):
    conn1 = kb.connect()
    conn2 = kb.connect()
    try:
        tid = kb.create_task(conn1, title="x", assignee="w")
        kb.add_notify_sub(conn1, task_id=tid, platform="telegram", chat_id="123")
        # New subs start caught up at the task's current MAX(task_events.id)
        # (the `created` event) — issue #29905.
        initial_cursor = int(kb.list_notify_subs(conn1, tid)[0]["last_event_id"])
        kb.complete_task(conn1, tid, result="ok")

        old_cursor, claimed_cursor, events = kb.claim_unseen_events_for_sub(
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
        _, _, duplicate_events = kb.claim_unseen_events_for_sub(
            conn2,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            kinds=["completed", "blocked"],
        )
        assert duplicate_events == []

        assert kb.rewind_notify_cursor(
            conn1,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            claimed_cursor=claimed_cursor,
            old_cursor=old_cursor,
        ) is True
        _, retried_events = kb.unseen_events_for_sub(
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
        conn = kb.connect()
        try:
            tid = kb.create_task(
                conn, title="long job", assignee="worker",
                max_runtime_seconds=1,  # one second cap
            )
            # Spawn by hand: claim + set pid + set active run start to the past.
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, os.getpid())   # any live pid works
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

            timed_out = kb.enforce_max_runtime(conn, signal_fn=_signal_fn)
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
    conn = kb.connect()
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

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="retry heartbeat guarded", assignee="worker")

        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        kb._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kb.heartbeat_worker(conn, tid, note="late", expected_run_id=run1.id)
        assert not kb.block_task(conn, tid, reason="late block", expected_run_id=run1.id)
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id == run2.id
        assert task.last_heartbeat_at is None

        assert kb.heartbeat_worker(conn, tid, note="current", expected_run_id=run2.id)
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
    conn = kb.connect()
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
        conn2 = kb.connect()
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
    conn = kb.connect()
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
    conn = kb.connect()
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
    conn = kb.connect()
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

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="skill-loading test",
                             assignee="some-profile")
        task = kb.get_task(conn, tid)
        workspace = kb.resolve_workspace(task)
        pid = kb._default_spawn(task, str(workspace))
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


def test_skill_preflight_matches_worker_resolver(kanban_home, tmp_path):
    """The dispatcher probe honors local, external, alias, and collision rules."""
    skills_root = kanban_home / "skills"
    local_skill = skills_root / "kanban-test-category" / "local-skill-for-kanban-test"
    local_skill.mkdir(parents=True)
    (local_skill / "SKILL.md").write_text(
        "---\nname: local-skill-for-kanban-test\n---\nlocal\n",
        encoding="utf-8",
    )

    external_root = tmp_path / "external-skills"
    aliased = external_root / "filesystem-name"
    aliased.mkdir(parents=True)
    (aliased / "SKILL.md").write_text(
        "---\nname: alias-skill-for-kanban-test\n---\nalias\n",
        encoding="utf-8",
    )
    collision = external_root / "external-collision"
    collision.mkdir(parents=True)
    (collision / "SKILL.md").write_text(
        "---\nname: collision-skill-for-kanban-test\n---\nexternal\n",
        encoding="utf-8",
    )
    local_collision = skills_root / "local-collision"
    local_collision.mkdir()
    (local_collision / "SKILL.md").write_text(
        "---\nname: collision-skill-for-kanban-test\n---\nlocal\n",
        encoding="utf-8",
    )
    (kanban_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_root}\n",
        encoding="utf-8",
    )

    assert kb._skill_available_for_home(
        "local-skill-for-kanban-test", str(kanban_home)
    ) is True
    assert kb._skill_available_for_home(
        "alias-skill-for-kanban-test", str(kanban_home)
    ) is True
    assert kb._skill_available_for_home(
        "collision-skill-for-kanban-test", str(kanban_home)
    ) is False
    assert kb._skill_available_for_home(
        "missing-skill-for-kanban-test", str(kanban_home)
    ) is False


def test_default_spawn_drops_unresolvable_task_skills(kanban_home, monkeypatch, capfd):
    """Only skills resolved by the real filesystem resolver reach child argv."""
    skill = kanban_home / "skills" / "good-skill-for-kanban-test"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: good-skill-for-kanban-test\n---\nusable\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeProc:
        pid = 7

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="real skill preflight",
            assignee="some-profile",
            skills=["good-skill-for-kanban-test", "missing-skill-for-kanban-test"],
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        workspace = kb.resolve_workspace(task)
        assert kb._default_spawn(task, str(workspace)) == 7
    finally:
        conn.close()

    cmd = captured["cmd"]
    skill_names = [
        cmd[index + 1]
        for index, token in enumerate(cmd)
        if token == "--skills" and index + 1 < len(cmd)
    ]
    assert skill_names == ["good-skill-for-kanban-test"]
    err = capfd.readouterr().err
    assert "missing-skill-for-kanban-test" in err
    assert tid in err


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
    kb._migrate_add_optional_columns(conn)
    after = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "skills" in after, f"migration did not add skills column: {after}"

    # Idempotent: running again must not raise.
    kb._migrate_add_optional_columns(conn)

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

    kb._migrate_add_optional_columns(conn)
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

    kb._migrate_add_optional_columns(conn)
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
    kb._migrate_add_optional_columns(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "consecutive_failures" in cols, "migration must add consecutive_failures"
    assert "last_failure_error" in cols, "migration must add last_failure_error"
    assert "spawn_failures" not in cols, "no legacy column should be synthesised"

    row = conn.execute("SELECT * FROM tasks WHERE id = 't1'").fetchone()
    assert row["consecutive_failures"] == 0
    assert row["last_failure_error"] is None

    # Idempotent second run must not raise either.
    kb._migrate_add_optional_columns(conn)
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
            raise _kb.KanbanDbCorruptError(
                corrupt_db,
                corrupt_db.with_suffix(".db.corrupt.test.bak"),
                "sqlite refused to open file: database disk image is malformed",
            )
        raise sqlite3.DatabaseError("file is not a database")

    async def _to_thread(fn, *args, **kwargs):
        # PR salvage (#32857 commit 7): the dispatcher now reaps zombies at
        # the top of each tick via ``asyncio.to_thread(_kb.reap_worker_zombies)``
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

    monkeypatch.setattr(_kb, "connect", _connect)
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
    # (ready then review, both via _kb.connect). The second dispatch tick
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
    conn = kb.connect()
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
    conn = kb.connect()
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
    host_prefix = _kb._claimer_id().split(":", 1)[0]
    claimed = _kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable for the next attempt"
    _kb._set_worker_pid(conn, tid, fake_pid)
    _kb._record_worker_exit(fake_pid, raw_status)
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        return _kb.detect_crashed_workers(conn)
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
    conn = kb.connect()
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
            _kb._PROTOCOL_VIOLATION_FAILURE_LIMIT
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
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="old task", assignee="w")
        # Historical terminal activity BEFORE anyone subscribes.
        kb.complete_task(conn, tid, result="done long ago")

        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="123")
        sub = kb.list_notify_subs(conn, tid)[0]
        assert int(sub["last_event_id"]) > 0, (
            "cursor must snap to MAX(task_events.id) at subscription time"
        )
        _, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="123",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        assert events == [], "historical events must not replay to a new sub"
    finally:
        conn.close()


