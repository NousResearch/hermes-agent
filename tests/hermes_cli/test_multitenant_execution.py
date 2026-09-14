"""Multi-tenant execution: isolation, fair scheduling, and recovery.

These are the tests for the hardening pass that turned ``tasks.tenant`` from a
label into a boundary. Each one is written against a behaviour that was broken
or absent before it, and several assert the *negative* case that used to
succeed — cross-tenant read, cross-tenant archive, idempotency-key collision.

Concurrency here is real: threads with their own connections and a barrier to
force overlap, and in one case genuinely separate OS processes. A test that
serialises its "concurrent" workers proves nothing about a queue.

Restart recovery is simulated the way it actually happens: the worker process
is killed without warning, its claim is left behind in the database, and a
fresh dispatcher has to notice and recover it.
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_tenant as kt


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated board with the reclaim grace windows collapsed.

    The 30 s crash grace exists to stop two live dispatchers reaping each
    other's workers; these tests have one dispatcher and need the reclaim to
    happen in-test, so it is pinned to 0 exactly as the existing kanban suites
    do.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    # Every profile name is spawnable: these tests are about the scheduler and
    # the tenant boundary, not about profile resolution.
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    return home


@pytest.fixture(autouse=True)
def _unbound_tenant():
    """Each test starts unscoped; tests bind explicitly with ``kt.use``."""
    token = kt._ACTIVE.set(None)
    try:
        yield
    finally:
        kt._ACTIVE.reset(token)


def _recording_spawn(spawned):
    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 424242  # a PID that is not this process and is not alive
    return spawn


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_reads_do_not_cross_tenants(board):
    conn = kbc.connect()
    a = kb.create_task(conn, title="A payroll", tenant="tenant-a", assignee="ops")
    b = kb.create_task(conn, title="B payroll", tenant="tenant-b", assignee="ops")

    with kt.use("tenant-b"):
        # Absent, not forbidden: answering "forbidden" for a row that exists and
        # "not found" for one that does not would confirm other tenants' ids.
        assert kb.get_task(conn, a) is None
        assert kb.get_task(conn, b) is not None
        assert [t.title for t in kb.list_tasks(conn)] == ["B payroll"]

    with kt.use("tenant-a"):
        assert kb.get_task(conn, b) is None
        assert [t.title for t in kb.list_tasks(conn)] == ["A payroll"]


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda conn, tid: kb.assign_task(conn, tid, "attacker"), id="assign"),
        pytest.param(lambda conn, tid: kb.archive_task(conn, tid), id="archive"),
        pytest.param(lambda conn, tid: kb.delete_task(conn, tid), id="delete"),
        pytest.param(lambda conn, tid: kb.claim_task(conn, tid), id="claim"),
        pytest.param(lambda conn, tid: kb.complete_task(conn, tid, result="x"), id="complete"),
        pytest.param(lambda conn, tid: kb.block_task(conn, tid, reason="x"), id="block"),
    ],
)
def test_mutations_do_not_cross_tenants(board, operation):
    """Every one of these succeeded against another tenant's task before."""
    conn = kbc.connect()
    victim = kb.create_task(conn, title="A secret", tenant="tenant-a", assignee="ops")
    before = kb.get_task(conn, victim)

    with kt.use("tenant-b"):
        assert not operation(conn, victim)

    after = kb.get_task(conn, victim)
    assert after is not None, "the task was destroyed across a tenant boundary"
    assert (after.status, after.assignee, after.title) == (
        before.status, before.assignee, before.title
    )


def test_tenant_may_still_operate_on_its_own_tasks(board):
    """The boundary must not be a wall around everything."""
    conn = kbc.connect()
    mine = kb.create_task(conn, title="B work", tenant="tenant-b", assignee="ops")
    with kt.use("tenant-b"):
        assert kb.assign_task(conn, mine, "ops2")
        assert kb.claim_task(conn, mine) is not None
        assert kb.complete_task(conn, mine, result="done")
    assert kb.get_task(conn, mine).status == "done"


def test_child_rows_carry_their_tenant(board):
    """Events, runs, comments and attachments must carry tenant_id themselves.

    Recovering it by joining back to ``tasks`` is not equivalent: nothing was
    doing that join, and it breaks the moment a task is hard-deleted.
    """
    conn = kbc.connect()
    t = kb.create_task(conn, title="A job", tenant="tenant-a", assignee="ops")
    kb.add_comment(conn, t, "alice", "a note")
    kb.claim_task(conn, t, claimer="w1")
    kb.complete_task(conn, t, result="ok")
    conn.commit()

    for table in ("task_events", "task_runs", "task_comments"):
        rows = conn.execute(
            f"SELECT DISTINCT tenant FROM {table} WHERE task_id = ?", (t,)
        ).fetchall()
        assert [r["tenant"] for r in rows] == ["tenant-a"], table


def test_unscoped_callers_still_see_everything(board):
    """The CLI, the TUI and the dispatcher bind no tenant and must be unaffected."""
    conn = kbc.connect()
    kb.create_task(conn, title="A", tenant="tenant-a", assignee="ops")
    kb.create_task(conn, title="B", tenant="tenant-b", assignee="ops")
    kb.create_task(conn, title="untenanted", assignee="ops")
    assert kt.current() is None
    assert len(kb.list_tasks(conn)) == 3


def test_worker_subprocess_inherits_its_boundary(board):
    """A spawned worker is scoped by HERMES_TENANT, which the dispatcher exports.

    Run as a real subprocess: the binding happens on first kanban connect, and
    an in-process test could not show that a *fresh interpreter* picks it up.
    """
    conn = kbc.connect()
    a = kb.create_task(conn, title="A secret", tenant="tenant-a", assignee="ops")
    b = kb.create_task(conn, title="B work", tenant="tenant-b", assignee="ops")
    conn.commit()

    script = (
        "from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, "
        "kanban_tenant as kt\n"
        "conn = kbc.connect()\n"
        f"print('bound', kt.current())\n"
        f"print('foreign', kb.get_task(conn, {a!r}))\n"
        f"print('own', kb.get_task(conn, {b!r}).title)\n"
        "print('visible', sorted(t.title for t in kb.list_tasks(conn)))\n"
    )
    env = {
        **os.environ,
        "HERMES_HOME": str(board),
        "HERMES_TENANT": "tenant-b",
        "PYTHONPATH": str(REPO_ROOT),
    }
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120,
    ).stdout

    assert "bound tenant-b" in out
    assert "foreign None" in out
    assert "own B work" in out
    assert "visible ['B work']" in out


# ---------------------------------------------------------------------------
# Idempotency / duplicate execution
# ---------------------------------------------------------------------------

def test_same_key_in_two_tenants_is_two_tasks(board):
    """The collision that silently dropped a tenant's job.

    Two tenants naming a nightly job the same obvious thing is ordinary. Before,
    the second submission returned the FIRST tenant's task id: tenant B's work
    never ran, and B held a readable handle to A's card.
    """
    conn = kbc.connect()
    a = kb.create_task(conn, title="A export", tenant="tenant-a", idempotency_key="nightly")
    b = kb.create_task(conn, title="B export", tenant="tenant-b", idempotency_key="nightly")

    assert a != b
    assert kb.get_task(conn, b).tenant == "tenant-b"
    assert kb.get_task(conn, b).title == "B export"


def test_same_key_same_tenant_still_dedupes(board):
    conn = kbc.connect()
    first = kb.create_task(conn, title="A export", tenant="tenant-a", idempotency_key="nightly")
    again = kb.create_task(conn, title="A export", tenant="tenant-a", idempotency_key="nightly")
    assert first == again


def test_untenanted_board_keeps_global_dedupe(board):
    """Single-tenant installs must not silently stop deduplicating.

    A bare UNIQUE(tenant, key) would not do this: SQLite treats NULLs as
    distinct, so every untenanted row would be unique. Hence COALESCE.
    """
    conn = kbc.connect()
    first = kb.create_task(conn, title="job", idempotency_key="shared")
    again = kb.create_task(conn, title="job", idempotency_key="shared")
    assert first == again


def test_concurrent_submits_create_exactly_one_task(board):
    """Sixteen threads, one barrier, one idempotency key."""
    workers = 16
    barrier = threading.Barrier(workers)
    ids: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def submit(i):
        conn = kbc.connect()
        try:
            barrier.wait(timeout=30)
            tid = kb.create_task(
                conn, title=f"submit {i}", tenant="tenant-a", idempotency_key="once"
            )
            with lock:
                ids.append(tid)
        except Exception as exc:  # noqa: BLE001 — the assertion reports it
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            conn.close()

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert len(set(ids)) == 1, f"{len(set(ids))} distinct ids returned: {set(ids)}"

    conn = kbc.connect()
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = 'once'"
    ).fetchone()
    assert rows["n"] == 1


def test_unique_index_backstops_the_idempotency_key(board):
    """The constraint is in the database, not only in the lookup.

    The in-transaction lookup is what wins the race under SQLite's single
    writer; this asserts the index exists so the guarantee does not quietly
    depend on that being true of every future engine or connection mode.
    """
    conn = kbc.connect()
    kb.create_task(conn, title="job", tenant="tenant-a", idempotency_key="k")
    conn.commit()
    with pytest.raises(Exception) as caught:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, tenant, idempotency_key) "
            "VALUES ('t_forced', 'dupe', 'ready', 1, 'tenant-a', 'k')"
        )
    assert "unique" in str(caught.value).lower()


# ---------------------------------------------------------------------------
# Concurrent execution across tenants
# ---------------------------------------------------------------------------

def test_two_tenants_execute_concurrently(board):
    """A and B run at the same time, and each claim goes to exactly one worker."""
    conn = kbc.connect()
    for i in range(4):
        kb.create_task(conn, title=f"A{i}", tenant="tenant-a", assignee="ops-a")
        kb.create_task(conn, title=f"B{i}", tenant="tenant-b", assignee="ops-b")
    conn.commit()

    claimed: list[tuple[str, str]] = []
    lock = threading.Lock()
    ready = [t.id for t in kb.list_tasks(conn, status="ready")]
    barrier = threading.Barrier(len(ready))

    def worker(task_id):
        c = kbc.connect()
        try:
            barrier.wait(timeout=30)
            got = kb.claim_task(c, task_id, claimer=f"w-{task_id}")
            if got is not None:
                with lock:
                    claimed.append((got.tenant, got.id))
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(t,)) for t in ready]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    by_tenant = collections.Counter(tenant for tenant, _ in claimed)
    assert by_tenant["tenant-a"] == 4
    assert by_tenant["tenant-b"] == 4
    # No task claimed twice.
    assert len({tid for _, tid in claimed}) == len(claimed)


def test_a_claim_is_won_by_exactly_one_worker(board):
    """Many workers, one task: the CAS must admit a single winner."""
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="contested", tenant="tenant-a", assignee="ops")
    conn.commit()

    winners: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def contend(i):
        c = kbc.connect()
        try:
            barrier.wait(timeout=30)
            if kb.claim_task(c, task_id, claimer=f"w{i}") is not None:
                with lock:
                    winners.append(f"w{i}")
        finally:
            c.close()

    threads = [threading.Thread(target=contend, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(winners) == 1, f"{len(winners)} workers claimed the same task: {winners}"


# ---------------------------------------------------------------------------
# Fair scheduling / noisy neighbour
# ---------------------------------------------------------------------------

def test_flooding_tenant_does_not_starve_a_quiet_one(board):
    """The noisy-neighbour case, with the flood holding every advantage.

    The loud tenant enqueues first AND at maximum priority, which under the
    original global ``ORDER BY priority DESC, created_at ASC`` put it at the
    head of the queue for every tick.
    """
    conn = kbc.connect()
    for i in range(50):
        kb.create_task(conn, title=f"loud {i}", tenant="loud", assignee="ops", priority=9)
    for i in range(2):
        kb.create_task(conn, title=f"quiet {i}", tenant="quiet", assignee="ops", priority=0)
    conn.commit()

    spawned: list[str] = []
    kbd.dispatch_once(
        conn, spawn_fn=_recording_spawn(spawned), max_spawn=6,
        max_in_progress=100, max_in_progress_per_tenant=3,
    )

    tenants = collections.Counter(kb.get_task(conn, t).tenant for t in spawned)
    assert tenants["quiet"] > 0, "the quiet tenant was starved by the flood"
    assert tenants["loud"] <= 3, "the loud tenant exceeded its concurrency cap"


def test_fair_ordering_alone_interleaves_tenants(board):
    """Ordering is a separate mechanism from the cap; prove it works unaided."""
    conn = kbc.connect()
    for i in range(20):
        kb.create_task(conn, title=f"loud {i}", tenant="loud", assignee="ops", priority=9)
    for i in range(3):
        kb.create_task(conn, title=f"quiet {i}", tenant="quiet", assignee="ops", priority=0)
    conn.commit()

    spawned: list[str] = []
    kbd.dispatch_once(
        conn, spawn_fn=_recording_spawn(spawned), max_spawn=6,
        max_in_progress=100,  # no per-tenant cap at all
    )
    tenants = collections.Counter(kb.get_task(conn, t).tenant for t in spawned)
    assert tenants["quiet"] > 0
    assert tenants["loud"] > 0


def test_per_tenant_cap_is_not_exceeded_within_one_tick(board):
    """The within-tick counter matters: every row would otherwise read a stale
    count and the whole budget would land on one tenant."""
    conn = kbc.connect()
    for i in range(10):
        kb.create_task(conn, title=f"a{i}", tenant="tenant-a", assignee="ops")
    conn.commit()

    spawned: list[str] = []
    result = kbd.dispatch_once(
        conn, spawn_fn=_recording_spawn(spawned), max_spawn=10,
        max_in_progress=100, max_in_progress_per_tenant=2,
    )
    assert len(spawned) == 2
    assert len(result.skipped_per_tenant_capped) == 8


def test_single_tenant_board_keeps_its_priority_order(board):
    """Fair ordering must be a no-op when there is nothing to be fair between."""
    conn = kbc.connect()
    low = kb.create_task(conn, title="low", tenant="solo", assignee="ops", priority=0)
    high = kb.create_task(conn, title="high", tenant="solo", assignee="ops", priority=9)
    conn.commit()

    spawned: list[str] = []
    kbd.dispatch_once(
        conn, spawn_fn=_recording_spawn(spawned), max_spawn=1, max_in_progress=100,
    )
    assert spawned == [high], "priority order was disturbed on a single-tenant board"
    assert low not in spawned


def test_least_loaded_tenant_is_served_first(board):
    """A tenant already holding workers yields to one holding none."""
    conn = kbc.connect()
    busy = kb.create_task(conn, title="busy running", tenant="busy", assignee="ops")
    kb.claim_task(conn, busy, claimer="w-existing")   # busy now has 1 in flight
    queued_busy = kb.create_task(conn, title="busy queued", tenant="busy",
                                 assignee="ops", priority=9)
    idle = kb.create_task(conn, title="idle queued", tenant="idle",
                          assignee="ops", priority=0)
    conn.commit()

    spawned: list[str] = []
    # max_spawn is a LIVE concurrency cap (running + spawned this tick), not a
    # per-tick budget — so 2 here means "one free slot" given busy's worker.
    kbd.dispatch_once(
        conn, spawn_fn=_recording_spawn(spawned), max_spawn=2, max_in_progress=100,
    )
    assert spawned == [idle], (
        "the tenant with a worker already running took the slot ahead of an idle tenant"
    )
    assert queued_busy not in spawned


# ---------------------------------------------------------------------------
# Restart / crash recovery
# ---------------------------------------------------------------------------

def test_claim_survives_process_restart_and_is_recovered(board, monkeypatch):
    """A worker dies without releasing its claim; a fresh dispatcher recovers it.

    This is the container-restart case: the task row and its claim are durable,
    so recovery is a matter of noticing the PID is gone — not of remembering
    anything that was only ever in memory.
    """
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="long job", tenant="tenant-a", assignee="ops")
    conn.commit()

    # Claim it, then make its worker disappear the way a killed container does.
    kb.claim_task(conn, task_id)  # host-prefixed lock, as a real worker gets
    kbd._set_worker_pid(conn, task_id, 98765)
    conn.commit()
    assert kb.get_task(conn, task_id).status == "running"

    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    # A fresh connection: the restarted process sees only what was made durable.
    fresh = kbc.connect()
    assert kbd.detect_crashed_workers(fresh) == [task_id]

    recovered = kb.get_task(fresh, task_id)
    assert recovered.status in {"ready", "todo"}, (
        f"a task orphaned by a dead worker was left in {recovered.status!r}"
    )
    assert recovered.tenant == "tenant-a", "tenant was lost across recovery"


def test_recovery_preserves_run_history(board, monkeypatch):
    """The failed attempt stays on the record; recovery is not amnesia."""
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="job", tenant="tenant-a", assignee="ops")
    kb.claim_task(conn, task_id)  # host-prefixed lock, as a real worker gets
    kbd._set_worker_pid(conn, task_id, 98765)
    conn.commit()

    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    kbd.detect_crashed_workers(kbc.connect())

    runs = conn.execute(
        "SELECT status, outcome, tenant FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchall()
    assert runs, "the crashed attempt left no run record"
    assert all(r["tenant"] == "tenant-a" for r in runs)


def test_one_tenants_crash_does_not_disturb_another(board, monkeypatch):
    conn = kbc.connect()
    crashing = kb.create_task(conn, title="A crashes", tenant="tenant-a", assignee="ops")
    healthy = kb.create_task(conn, title="B healthy", tenant="tenant-b", assignee="ops")
    kb.claim_task(conn, crashing)
    kb.claim_task(conn, healthy)
    kbd._set_worker_pid(conn, crashing, 98765)     # A's worker is gone
    kbd._set_worker_pid(conn, healthy, os.getpid())  # B's is this very process
    conn.commit()

    # Only A's pid is dead; B's is the running interpreter.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == os.getpid())
    kbd.detect_crashed_workers(kbc.connect())

    assert kb.get_task(conn, crashing).status in {"ready", "todo"}
    assert kb.get_task(conn, healthy).status == "running", (
        "a healthy tenant's running task was reclaimed by another tenant's crash"
    )


# ---------------------------------------------------------------------------
# Retry backoff
# ---------------------------------------------------------------------------

def test_retry_backoff_grows_and_is_capped(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RETRY_BACKOFF_SECONDS", "10")
    monkeypatch.setenv("HERMES_KANBAN_RETRY_BACKOFF_MAX_SECONDS", "100")
    assert kbd.retry_backoff_seconds(0) == 0
    assert kbd.retry_backoff_seconds(1) == 10
    assert kbd.retry_backoff_seconds(2) == 20
    assert kbd.retry_backoff_seconds(3) == 40
    assert kbd.retry_backoff_seconds(9) == 100      # capped


def test_retry_backoff_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RETRY_BACKOFF_SECONDS", "0")
    assert kbd.retry_backoff_seconds(5) == 0


def test_a_failing_task_is_held_off_instead_of_hot_looping(board, monkeypatch):
    """Without backoff a crashed card re-spawned every tick until the breaker
    tripped, spending a shared spawn slot each time."""
    monkeypatch.setenv("HERMES_KANBAN_RETRY_BACKOFF_SECONDS", "300")
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="crasher", tenant="tenant-a", assignee="ops")
    conn.commit()

    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET consecutive_failures = 2, status = 'ready' WHERE id = ?", (task_id,)
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at, tenant) "
        "VALUES (?, 'crashed', 'crashed', ?, ?, 'tenant-a')",
        (task_id, now - 5, now - 5),
    )
    conn.commit()

    assert kbd.check_respawn_guard(conn, task_id) == "failure_backoff"

    # Once the wait has elapsed the card is spawnable again.
    conn.execute(
        "UPDATE task_runs SET ended_at = ? WHERE task_id = ?", (now - 10_000, task_id)
    )
    conn.commit()
    assert kbd.check_respawn_guard(conn, task_id) != "failure_backoff"


def test_backoff_does_not_hold_a_task_that_never_failed(board):
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="fine", tenant="tenant-a", assignee="ops")
    conn.commit()
    assert kbd.check_respawn_guard(conn, task_id) is None




# ---------------------------------------------------------------------------
# Long-running jobs, heartbeats, graceful shutdown
# ---------------------------------------------------------------------------

def test_a_long_running_job_holds_its_claim_via_heartbeats(board):
    """A job that outlives its claim TTL keeps the claim by heartbeating.

    The TTL exists so an abandoned claim is eventually reclaimed; a job that is
    merely *slow* must not be mistaken for an abandoned one.
    """
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="12 hour export", tenant="tenant-a", assignee="ops")
    claimed = kb.claim_task(conn, task_id, ttl_seconds=1)
    assert claimed is not None
    lock = kb.get_task(conn, task_id).claim_lock

    first_expiry = kb.get_task(conn, task_id).claim_expires
    assert kb.heartbeat_claim(conn, task_id, ttl_seconds=3600, claimer=lock)
    assert kb.get_task(conn, task_id).claim_expires > first_expiry

    # Still running, still owned, still the right tenant.
    still = kb.get_task(conn, task_id)
    assert still.status == "running"
    assert still.tenant == "tenant-a"


def test_a_heartbeat_from_another_claimer_is_refused(board):
    """Heartbeating is proof of ownership, not a way to take ownership."""
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="job", tenant="tenant-a", assignee="ops")
    kb.claim_task(conn, task_id)
    assert not kb.heartbeat_claim(conn, task_id, claimer="somebody-else")


def test_stale_claim_is_reclaimed_when_heartbeats_stop(board):
    """The other half of the contract: stop heartbeating and you lose the claim."""
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="job", tenant="tenant-a", assignee="ops")
    kb.claim_task(conn, task_id, ttl_seconds=1)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 60, task_id)
    )
    conn.commit()

    assert kb.release_stale_claims(conn) >= 1
    reclaimed = kb.get_task(conn, task_id)
    assert reclaimed.status in {"ready", "todo"}
    assert reclaimed.tenant == "tenant-a", "tenant survived the reclaim"


def test_dispatch_loop_stops_on_request(board):
    """Graceful shutdown: the loop exits on the stop event, mid-flight."""
    conn = kbc.connect()
    kb.create_task(conn, title="job", tenant="tenant-a", assignee="ops")
    conn.commit()

    stop = threading.Event()
    ticks = []

    def on_tick(_result):
        ticks.append(1)
        stop.set()          # ask it to stop from inside a tick

    done = threading.Event()

    def run():
        kbd.run_daemon(interval=0.05, stop_event=stop, on_tick=on_tick)
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(timeout=30), "dispatch_loop did not exit after stop_event was set"
    assert ticks, "the loop never ran a tick"


# ---------------------------------------------------------------------------
# Multi-process: the real deployment shape
# ---------------------------------------------------------------------------

_SUBMIT_SCRIPT = """
import os, sys
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
conn = kbc.connect()
print(kb.create_task(conn, title="p%s" % os.getpid(),
                     tenant=sys.argv[1], idempotency_key=sys.argv[2]))
"""


def test_separate_processes_cannot_duplicate_an_idempotent_job(board):
    """Eight OS processes, one key. Threads share a GIL and a cache; processes
    share only the database, which is where the guarantee has to live."""
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _SUBMIT_SCRIPT, "tenant-a", "cross-process"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "HERMES_HOME": str(board), "PYTHONPATH": str(REPO_ROOT)},
        )
        for _ in range(8)
    ]
    outs = []
    for proc in procs:
        out, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err[-2000:]
        outs.append(out.strip().splitlines()[-1])

    assert len(set(outs)) == 1, f"processes created {len(set(outs))} distinct tasks: {set(outs)}"
    conn = kbc.connect()
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = 'cross-process'"
    ).fetchone()["n"]
    assert n == 1


def test_separate_processes_in_different_tenants_each_get_their_own_job(board):
    """The same key in two tenants must produce two tasks, across processes."""
    outs = {}
    for tenant in ("tenant-a", "tenant-b"):
        proc = subprocess.run(
            [sys.executable, "-c", _SUBMIT_SCRIPT, tenant, "nightly"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "HERMES_HOME": str(board), "PYTHONPATH": str(REPO_ROOT)},
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outs[tenant] = proc.stdout.strip().splitlines()[-1]

    assert outs["tenant-a"] != outs["tenant-b"]
    conn = kbc.connect()
    assert kb.get_task(conn, outs["tenant-a"]).tenant == "tenant-a"
    assert kb.get_task(conn, outs["tenant-b"]).tenant == "tenant-b"


# ---------------------------------------------------------------------------
# Child rows: hiding the task is not enough
# ---------------------------------------------------------------------------

def _task_with_history(conn):
    task_id = kb.create_task(conn, title="A confidential", tenant="tenant-a", assignee="ops")
    kb.add_comment(conn, task_id, "alice", "the merger closes Friday")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, result="done")
    conn.commit()
    return task_id


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(lambda conn, tid: kb.list_comments(conn, tid), id="comments"),
        pytest.param(lambda conn, tid: kb.list_comments_after(conn, tid), id="comments_after"),
        pytest.param(lambda conn, tid: kb.list_events(conn, tid), id="events"),
        pytest.param(lambda conn, tid: kb.list_runs(conn, tid), id="runs"),
        pytest.param(lambda conn, tid: kb.list_attachments(conn, tid), id="attachments"),
    ],
)
def test_child_rows_do_not_leak_across_tenants(board, reader):
    """A task id is guessable. Before this, ``get_task`` correctly returned None
    while the same id still yielded that task's comment bodies verbatim, its run
    history and its attachment paths."""
    conn = kbc.connect()
    task_id = _task_with_history(conn)

    with kt.use("tenant-b"):
        assert reader(conn, task_id) == []

    with kt.use("tenant-a"):
        assert reader(conn, task_id) is not None  # own data unaffected


def test_comment_bodies_do_not_leak(board):
    """Named separately because this is the one that leaks business content."""
    conn = kbc.connect()
    task_id = _task_with_history(conn)
    with kt.use("tenant-b"):
        bodies = [c.body for c in kb.list_comments(conn, task_id)]
    assert bodies == []
    with kt.use("tenant-a"):
        assert [c.body for c in kb.list_comments(conn, task_id)] == ["the merger closes Friday"]


def test_runs_addressed_by_their_own_id_do_not_leak(board):
    """``get_run`` takes a run id, so the task-level guard never sees it."""
    conn = kbc.connect()
    task_id = _task_with_history(conn)
    run_id = kb.list_runs(conn, task_id)[0].id

    with kt.use("tenant-b"):
        assert kb.get_run(conn, run_id) is None
        assert kb.latest_run(conn, task_id) is None
    with kt.use("tenant-a"):
        assert kb.get_run(conn, run_id) is not None


def test_attachment_paths_do_not_leak(board, tmp_path):
    """``stored_path`` is an absolute filesystem path; ``get_attachment`` is
    addressed by attachment id, not task id."""
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="A", tenant="tenant-a", assignee="ops")
    blob = tmp_path / "secret.pdf"
    blob.write_text("x")
    att_id = kb.add_attachment(
        conn, task_id, filename="secret.pdf", stored_path=str(blob),
        content_type="application/pdf", size=1, uploaded_by="alice",
    )
    conn.commit()

    with kt.use("tenant-b"):
        assert kb.get_attachment(conn, att_id) is None
    with kt.use("tenant-a"):
        assert kb.get_attachment(conn, att_id) is not None


def test_unscoped_callers_still_read_child_rows(board):
    """The CLI and dispatcher bind no tenant and must keep working."""
    conn = kbc.connect()
    task_id = _task_with_history(conn)
    assert kt.current() is None
    assert len(kb.list_comments(conn, task_id)) == 1
    assert len(kb.list_runs(conn, task_id)) == 1
