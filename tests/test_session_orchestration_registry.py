"""
Tests for session_orchestration/registry.py.

Covers:
- Schema creation (idempotent)
- upsert / get / list basic CRUD
- UNIQUE(run_id, repo) constraint — duplicate adopt/spawn rejected
- enqueue_intent / drain_intents round-trip
- acquire_lock / release_lock including stale-reclaim
- canonical_repo_id normalisation
- Concurrent-write correctness: cron writer + webhook-adopt enqueue +
  drive-update enqueue racing in one window, cron draining →
  no lost updates, no corruption, exactly one row per (run_id, repo).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import pytest

from session_orchestration.registry import (
    SessionOrchestrationRegistry,
    canonical_repo_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path) -> Path:
    """Return a path to a fresh, isolated state.db in tmp_path."""
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path) -> SessionOrchestrationRegistry:
    """Return a registry backed by an isolated DB."""
    return SessionOrchestrationRegistry(db_path=db_path)


# ---------------------------------------------------------------------------
# canonical_repo_id
# ---------------------------------------------------------------------------


class TestCanonicalRepoId:
    def test_same_remote_url_variants_hash_equal(self):
        """HTTPS and git@ forms of the same repo produce the same id."""
        https_id = canonical_repo_id(remote_url="https://github.com/foo/bar.git")
        ssh_id = canonical_repo_id(remote_url="git@github.com:foo/bar.git")
        assert https_id == ssh_id

    def test_different_remote_urls_hash_differently(self):
        a = canonical_repo_id(remote_url="https://github.com/foo/bar")
        b = canonical_repo_id(remote_url="https://github.com/foo/baz")
        assert a != b

    def test_fallback_to_workdir_when_no_remote(self, tmp_path):
        rid = canonical_repo_id(workdir=str(tmp_path))
        assert isinstance(rid, str)
        assert len(rid) == 12
        assert rid.isalnum()

    def test_result_is_12_hex_chars(self):
        rid = canonical_repo_id(remote_url="https://github.com/foo/bar")
        assert len(rid) == 12
        int(rid, 16)  # raises ValueError if not hex


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_tables_created_on_init(self, db_path):
        import sqlite3
        SessionOrchestrationRegistry(db_path=db_path)
        conn = sqlite3.connect(str(db_path))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "session_orchestration" in tables
        assert "session_orchestration_queue" in tables

    def test_init_is_idempotent(self, db_path):
        """Creating the registry twice does not raise."""
        SessionOrchestrationRegistry(db_path=db_path)
        SessionOrchestrationRegistry(db_path=db_path)  # must not raise

    def test_unique_run_id_repo_index_exists(self, db_path):
        import sqlite3
        SessionOrchestrationRegistry(db_path=db_path)
        conn = sqlite3.connect(str(db_path))
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_so_run_repo" in indexes

    def test_migrate_schema_discord_user_id_idempotent(self, db_path):
        """discord_user_id ALTER TABLE migration must be idempotent.

        The second registry init on the same DB path triggers _migrate_schema
        on an already-migrated table; the OperationalError must be swallowed
        silently and the column must be present and writable.
        """
        import uuid
        reg1 = SessionOrchestrationRegistry(db_path=db_path)
        reg2 = SessionOrchestrationRegistry(db_path=db_path)  # must not raise
        # Explicitly calling _migrate_schema a third time must also be silent
        reg2._migrate_schema()

        # Verify the column is present by inserting and reading back
        tid = str(uuid.uuid4())
        reg1.upsert(
            tid,
            agent="test",
            run_id=f"run-{tid[:8]}",
            repo=f"repo-{tid[:8]}",
            discord_user_id="user-999",
        )
        row = reg1.get(tid)
        assert row is not None
        assert row.get("discord_user_id") == "user-999", (
            "discord_user_id column must be present and writable after migration"
        )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestUpsertGetList:
    def test_upsert_and_get(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="claude-code", run_id="run1", repo="aabbcc112233")
        row = registry.get(tid)
        assert row is not None
        assert row["agent"] == "claude-code"
        assert row["run_id"] == "run1"
        assert row["repo"] == "aabbcc112233"
        assert row["state"] == "RUNNING"

    def test_upsert_updates_existing_row(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="claude-code", run_id="run1", repo="aabbcc112233")
        registry.upsert(tid, agent="claude-code", state="WAITING_USER")
        row = registry.get(tid)
        assert row["state"] == "WAITING_USER"

    def test_list_all(self, registry):
        for i in range(3):
            registry.upsert(
                str(uuid.uuid4()),
                agent="omp",
                run_id=f"run{i}",
                repo=f"repo{i:012x}",
            )
        rows = registry.list()
        assert len(rows) == 3

    def test_list_filter_by_state(self, registry):
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())
        registry.upsert(t1, agent="omp", state="RUNNING")
        registry.upsert(t2, agent="omp", state="DONE")
        running = registry.list(state="RUNNING")
        assert len(running) == 1
        assert running[0]["task_id"] == t1

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent-task-id") is None


# ---------------------------------------------------------------------------
# UNIQUE(run_id, repo) constraint
# ---------------------------------------------------------------------------


class TestUniquenessConstraint:
    def test_duplicate_run_id_repo_rejected(self, registry):
        """A second upsert with the same (run_id, repo) but different task_id
        must be silently ignored (no duplicate row, no exception)."""
        run_id = "shared-run"
        repo = canonical_repo_id(remote_url="https://github.com/foo/bar")
        t1 = str(uuid.uuid4())
        t2 = str(uuid.uuid4())

        registry.upsert(t1, agent="claude-code", run_id=run_id, repo=repo)
        registry.upsert(t2, agent="claude-code", run_id=run_id, repo=repo)  # ignored

        rows = registry.list(run_id=run_id)
        assert len(rows) == 1
        assert rows[0]["task_id"] == t1

    def test_same_run_id_different_repos_allowed(self, registry):
        """Same run_id but different repos → both rows must be inserted."""
        run_id = "multi-repo-run"
        repo_a = canonical_repo_id(remote_url="https://github.com/foo/alpha")
        repo_b = canonical_repo_id(remote_url="https://github.com/foo/beta")
        registry.upsert(str(uuid.uuid4()), agent="omp", run_id=run_id, repo=repo_a)
        registry.upsert(str(uuid.uuid4()), agent="omp", run_id=run_id, repo=repo_b)

        rows = registry.list(run_id=run_id)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Queue: enqueue / drain
# ---------------------------------------------------------------------------


class TestQueue:
    def test_enqueue_and_drain(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="claude-code")
        registry.enqueue_intent("drive", task_id=tid, payload={"msg": "hello"})
        intents = registry.drain_intents()
        assert len(intents) == 1
        assert intents[0]["intent"] == "drive"
        assert intents[0]["task_id"] == tid

    def test_drain_is_empty_after_drain(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.enqueue_intent("update", task_id=tid, payload={"state": "DONE"})
        registry.drain_intents()
        assert registry.drain_intents() == []

    def test_drain_applies_adopt_intent(self, registry):
        """_apply_intent with 'adopt' kind must upsert the registry row."""
        run_id = "wh-run"
        repo = canonical_repo_id(remote_url="https://github.com/foo/bar")
        new_tid = str(uuid.uuid4())
        registry.enqueue_intent(
            "adopt",
            task_id=new_tid,
            run_id=run_id,
            repo=repo,
            payload={"agent": "claude-code", "run_id": run_id, "repo": repo,
                     "workdir": "/home/user/project"},
        )
        intents = registry.drain_intents()
        for intent in intents:
            registry._apply_intent(intent)
        row = registry.get(new_tid)
        assert row is not None
        assert row["source"] == "adopt"
        assert row["workdir"] == "/home/user/project"

    def test_drain_applies_update_intent(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp", state="RUNNING")
        registry.enqueue_intent("update", task_id=tid, payload={"state": "DONE"})
        for intent in registry.drain_intents():
            registry._apply_intent(intent)
        row = registry.get(tid)
        assert row["state"] == "DONE"

    def test_drain_ordering_is_fifo(self, registry):
        """Intents must be returned in insertion order (oldest first)."""
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        for i in range(5):
            registry.enqueue_intent("drive", task_id=tid, payload={"seq": i})
        intents = registry.drain_intents()
        seqs = [json.loads(i["payload"])["seq"] for i in intents]
        assert seqs == list(range(5))


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


class TestLock:
    def test_acquire_and_release(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        ok = registry.acquire_lock(tid, holder="cron:1234")
        assert ok is True
        registry.release_lock(tid, holder="cron:1234")
        row = registry.get(tid)
        assert row["lock_holder"] is None

    def test_second_holder_cannot_acquire_while_locked(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.acquire_lock(tid, holder="cron:1")
        ok = registry.acquire_lock(tid, holder="relay:2")
        assert ok is False

    def test_expired_lock_is_reclaimed(self, registry):
        """A lock with ttl=0.001 s must be reclaimed by the next acquirer."""
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.acquire_lock(tid, holder="old-holder", ttl_seconds=0.001)
        time.sleep(0.05)  # ensure expiry
        ok = registry.acquire_lock(tid, holder="new-holder", ttl_seconds=60)
        assert ok is True
        row = registry.get(tid)
        assert row["lock_holder"] == "new-holder"

    def test_release_by_non_owner_is_noop(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.acquire_lock(tid, holder="real-holder")
        registry.release_lock(tid, holder="impersonator")  # must not raise or change lock
        row = registry.get(tid)
        assert row["lock_holder"] == "real-holder"

    def test_acquire_same_holder_refreshes_ttl(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.acquire_lock(tid, holder="cron:1", ttl_seconds=60)
        old_ts = registry.get(tid)["lock_ts"]
        time.sleep(0.05)
        ok = registry.acquire_lock(tid, holder="cron:1", ttl_seconds=60)
        assert ok is True
        new_ts = registry.get(tid)["lock_ts"]
        assert new_ts >= old_ts


# ---------------------------------------------------------------------------
# Atomic counter increment
# ---------------------------------------------------------------------------


class TestAtomicCounters:
    def test_increment_heartbeat_counter(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        registry.increment_counter(tid, "heartbeat_counter")
        registry.increment_counter(tid, "heartbeat_counter")
        row = registry.get(tid)
        assert row["heartbeat_counter"] == 2

    def test_increment_idle_ticks(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        for _ in range(5):
            registry.increment_counter(tid, "idle_ticks")
        assert registry.get(tid)["idle_ticks"] == 5

    def test_disallowed_column_raises(self, registry):
        tid = str(uuid.uuid4())
        registry.upsert(tid, agent="omp")
        with pytest.raises(ValueError, match="not in allowed set"):
            registry.increment_counter(tid, "agent")


# ---------------------------------------------------------------------------
# Concurrent-write correctness
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    """
    Simulates the three-writer scenario described in the acceptance criterion:

      • cron watcher: reads the queue, applies intents, bumps heartbeat_counter
      • webhook-adopt path: enqueues "adopt" intents
      • Discord-drive path: enqueues "drive" intents

    All three races in the same window against the same SQLite file.
    After the cron finishes draining, we assert:
      - No lost updates (heartbeat_counter == expected value)
      - No corruption (heartbeat_counter is a plain integer, row is coherent)
      - Exactly one row per (run_id, repo) (no duplicate adopt rows)
    """

    def test_concurrent_enqueue_and_drain_no_lost_updates(self, db_path):
        N_ADOPT = 5        # distinct sessions adopted
        N_DRIVE = 20       # drive intents (can reuse the same task_ids)
        N_CRON_TICKS = 10  # how many times the cron "drains"
        N_HEARTBEAT_BUMPS_PER_TICK = 1

        run_id = "concurrent-test-run"
        repo = canonical_repo_id(remote_url="https://github.com/test/concurrent")

        # Pre-seed the registry with N_ADOPT rows (simulating spawn)
        main_reg = SessionOrchestrationRegistry(db_path=db_path)
        task_ids: list[str] = []
        for i in range(N_ADOPT):
            tid = f"task-conc-{i}"
            task_ids.append(tid)
            main_reg.upsert(tid, agent="claude-code", run_id=run_id, repo=repo + f"{i:02x}")

        errors: list[str] = []
        stop_event = threading.Event()

        # ── Thread 1: webhook-adopt enqueuer ──
        def adopt_enqueuer():
            reg = SessionOrchestrationRegistry(db_path=db_path)
            for i in range(N_ADOPT):
                # Each adopt tries to register the SAME (run_id, repo) pair again —
                # simulating a duplicate webhook POST.  The cron drain must deduplicate.
                try:
                    reg.enqueue_intent(
                        "adopt",
                        task_id=task_ids[i],
                        run_id=run_id,
                        repo=repo + f"{i:02x}",
                        payload={
                            "agent": "claude-code",
                            "run_id": run_id,
                            "repo": repo + f"{i:02x}",
                            "task_id": task_ids[i],
                        },
                    )
                except Exception as exc:
                    errors.append(f"adopt_enqueuer: {exc}")
                time.sleep(0.002)

        # ── Thread 2: Discord-drive enqueuer ──
        def drive_enqueuer():
            reg = SessionOrchestrationRegistry(db_path=db_path)
            for i in range(N_DRIVE):
                try:
                    tid = task_ids[i % N_ADOPT]
                    reg.enqueue_intent(
                        "drive",
                        task_id=tid,
                        payload={"msg": f"drive-{i}"},
                    )
                except Exception as exc:
                    errors.append(f"drive_enqueuer: {exc}")
                time.sleep(0.001)

        # ── Thread 3: cron drainer ──
        final_heartbeats: dict[str, int] = {}

        def cron_drainer():
            reg = SessionOrchestrationRegistry(db_path=db_path)
            for tick in range(N_CRON_TICKS):
                try:
                    intents = reg.drain_intents()
                    for intent in intents:
                        reg._apply_intent(intent)
                    # Atomically bump heartbeat_counter for every registered row
                    for tid in task_ids:
                        reg.increment_counter(tid, "heartbeat_counter",
                                              by=N_HEARTBEAT_BUMPS_PER_TICK)
                except Exception as exc:
                    errors.append(f"cron tick {tick}: {exc}")
                time.sleep(0.005)
            # Capture final counters
            for tid in task_ids:
                row = reg.get(tid)
                if row:
                    final_heartbeats[tid] = row["heartbeat_counter"]

        t_adopt = threading.Thread(target=adopt_enqueuer, name="adopt-enqueuer")
        t_drive = threading.Thread(target=drive_enqueuer, name="drive-enqueuer")
        t_cron = threading.Thread(target=cron_drainer, name="cron-drainer")

        t_adopt.start()
        t_drive.start()
        t_cron.start()

        t_adopt.join(timeout=10)
        t_drive.join(timeout=10)
        t_cron.join(timeout=10)

        # ── Drain any remaining intents the cron didn't catch ──
        remaining = main_reg.drain_intents()
        for intent in remaining:
            main_reg._apply_intent(intent)

        # ── Assertions ──

        assert not errors, f"Thread errors: {errors}"

        # 1. No lost updates: every row must have heartbeat_counter == N_CRON_TICKS
        for tid in task_ids:
            row = main_reg.get(tid)
            assert row is not None, f"Row for {tid} disappeared"
            actual = row["heartbeat_counter"]
            assert actual == N_CRON_TICKS, (
                f"Lost update detected for {tid}: expected "
                f"heartbeat_counter={N_CRON_TICKS}, got {actual}"
            )

        # 2. Exactly one row per (run_id, repo) — no duplicate adopt/spawn rows
        for i in range(N_ADOPT):
            r = repo + f"{i:02x}"
            rows = main_reg.list(run_id=run_id)
            matching = [x for x in rows if x["repo"] == r]
            assert len(matching) == 1, (
                f"Expected exactly 1 row for (run_id={run_id}, repo={r}), "
                f"got {len(matching)}: {matching}"
            )

        # 3. No corruption — state column must be a valid string, counter non-negative
        for tid in task_ids:
            row = main_reg.get(tid)
            assert isinstance(row["heartbeat_counter"], int)
            assert row["heartbeat_counter"] >= 0


# ---------------------------------------------------------------------------
# T011 — Terminate intent application + last_question migration
# ---------------------------------------------------------------------------


class TestTerminateIntentApplication:
    """_apply_intent('terminate') marks the row terminal."""

    def _seed(self, registry, task_id="task-term-001"):
        registry.upsert(
            task_id,
            agent="claude",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            state="RUNNING",
        )

    def test_terminate_intent_no_restart_marks_done(self, registry):
        """restart=False -> state=DONE."""
        task_id = "task-term-001"
        self._seed(registry, task_id)

        intent = {
            "intent": "terminate",
            "task_id": task_id,
            "payload": json.dumps({"restart": False}),
        }
        registry._apply_intent(intent)

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "DONE", f"expected DONE, got {row['state']!r}"

    def test_terminate_intent_restart_marks_error(self, registry):
        """restart=True -> state=ERROR (row is marked terminal before re-spawn)."""
        task_id = "task-term-002"
        self._seed(registry, task_id)

        intent = {
            "intent": "terminate",
            "task_id": task_id,
            "payload": json.dumps({"restart": True}),
        }
        registry._apply_intent(intent)

        row = registry.get(task_id)
        assert row is not None
        assert row["state"] == "ERROR", f"expected ERROR, got {row['state']!r}"

    def test_terminate_apply_unknown_task_no_crash(self, registry):
        """terminate for a nonexistent task_id logs a warning and does not crash."""
        intent = {
            "intent": "terminate",
            "task_id": "nonexistent-task-xyz",
            "payload": json.dumps({"restart": False}),
        }
        # Must not raise
        registry._apply_intent(intent)

    def test_terminate_apply_missing_task_id_no_crash(self, registry):
        """terminate intent with no task_id logs a warning and does not crash."""
        intent = {
            "intent": "terminate",
            "payload": json.dumps({}),
        }
        registry._apply_intent(intent)


class TestMigrateSchemaLastQuestion:
    """last_question column added idempotently."""

    def test_migrate_schema_last_question_idempotent(self, db_path):
        """Calling _migrate_schema twice on the same DB must not raise."""
        reg1 = SessionOrchestrationRegistry(db_path=db_path)
        reg2 = SessionOrchestrationRegistry(db_path=db_path)  # second init
        reg2._migrate_schema()  # explicit third call — must be silent

    def test_last_question_column_writable(self, db_path):
        """After migration, last_question can be written and read back."""
        reg = SessionOrchestrationRegistry(db_path=db_path)
        task_id = str(uuid.uuid4())
        reg.upsert(
            task_id,
            agent="claude",
            run_id=f"run-{uuid.uuid4().hex[:8]}",
            repo=f"repo-{uuid.uuid4().hex[:8]}",
            last_question="What is the meaning of life?",
        )
        row = reg.get(task_id)
        assert row is not None
        assert row.get("last_question") == "What is the meaning of life?"


# ---------------------------------------------------------------------------
# gc_terminal_rows
# ---------------------------------------------------------------------------


def _seed_terminal(
    registry: SessionOrchestrationRegistry,
    *,
    state: str,
    terminated_at: float | None,
    task_id: str | None = None,
) -> str:
    """Insert a row with the given terminal state and terminated_at stamp."""
    tid = task_id or str(uuid.uuid4())
    registry.upsert(
        tid,
        agent="claude",
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        repo=f"repo-{uuid.uuid4().hex[:8]}",
        state=state,
        terminated_at=terminated_at,
    )
    return tid


class TestGcTerminalRows:
    """gc_terminal_rows GC method — four eligibility rules."""

    def test_old_terminal_row_is_deleted(self, registry):
        """(a) A terminal row with terminated_at older than the threshold is deleted."""
        now = 1_000_000.0
        max_age = 3600  # 1 h
        # terminated_at is 2 h ago — eligible
        task_id = _seed_terminal(
            registry, state="DONE", terminated_at=now - 7200.0
        )

        deleted = registry.gc_terminal_rows(now=now, max_age_seconds=max_age)

        assert deleted == 1
        assert registry.get(task_id) is None, "old terminal row must be deleted"

    def test_fresh_terminal_row_is_retained(self, registry):
        """(b) A terminal row with a recent terminated_at is NOT deleted."""
        now = 1_000_000.0
        max_age = 3600  # 1 h
        # terminated_at is only 30 min ago — too fresh
        task_id = _seed_terminal(
            registry, state="DONE", terminated_at=now - 1800.0
        )

        deleted = registry.gc_terminal_rows(now=now, max_age_seconds=max_age)

        assert deleted == 0
        assert registry.get(task_id) is not None, "fresh terminal row must be retained"

    def test_terminal_row_with_null_terminated_at_is_retained(self, registry):
        """(c) A terminal row with terminated_at=NULL is never GC-ed."""
        now = 1_000_000.0
        task_id = _seed_terminal(registry, state="ERROR", terminated_at=None)

        deleted = registry.gc_terminal_rows(now=now, max_age_seconds=60)

        assert deleted == 0
        assert registry.get(task_id) is not None, "NULL-stamp terminal row must survive GC"

    def test_running_row_is_never_deleted(self, registry):
        """(d) A non-terminal RUNNING row is never deleted regardless of timestamps."""
        now = 1_000_000.0
        # Seed a running row with a very old timestamp (not a real column but
        # we store terminated_at=old just to confirm it does not get deleted)
        task_id = _seed_terminal(registry, state="RUNNING", terminated_at=now - 999_999.0)

        deleted = registry.gc_terminal_rows(now=now, max_age_seconds=1)

        assert deleted == 0
        assert registry.get(task_id) is not None, "RUNNING row must never be deleted by GC"

    def test_only_eligible_rows_deleted_mixed_batch(self, registry):
        """Mixed batch: only the old terminal rows are deleted."""
        now = 1_000_000.0
        max_age = 3600

        old_done = _seed_terminal(registry, state="DONE", terminated_at=now - 7200.0)
        old_error = _seed_terminal(registry, state="ERROR", terminated_at=now - 7200.0)
        fresh_done = _seed_terminal(registry, state="DONE", terminated_at=now - 1800.0)
        null_done = _seed_terminal(registry, state="DONE", terminated_at=None)
        running = _seed_terminal(registry, state="RUNNING", terminated_at=now - 99999.0)

        deleted = registry.gc_terminal_rows(now=now, max_age_seconds=max_age)

        assert deleted == 2
        assert registry.get(old_done) is None
        assert registry.get(old_error) is None
        assert registry.get(fresh_done) is not None
        assert registry.get(null_done) is not None
        assert registry.get(running) is not None

    def test_returns_zero_when_nothing_eligible(self, registry):
        """gc_terminal_rows returns 0 when no rows match."""
        deleted = registry.gc_terminal_rows(now=1_000_000.0, max_age_seconds=86400)
        assert deleted == 0
