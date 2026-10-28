"""Cross-process session turn lease, proven with real processes (#84234).

``tests/state/test_session_turn_lease.py`` covers the lease logic inside one
interpreter: two ``SessionDB`` handles, threads, and a stubbed ``psutil`` for
the dead-owner path. That is the right place for branch coverage, but a single
process cannot observe the property the lease exists for, which is what happens
when two Hermes processes with different PIDs open the same ``state.db``. Every
test in this file spawns real OS children through ``subprocess`` and asserts on
what they did to one shared database.

Two rules keep these tests from passing vacuously.

First, contention tests park every worker on a file barrier until all of them
report ready. Process startup on a loaded machine staggers spawns by more than
a short worker loop takes, so without a barrier the first worker finishes
before the last one starts, the workers never overlap, and "exactly one winner"
becomes a statement about scheduling rather than about the lease. The barrier
costs one file per worker and turns the spread between worker starts from
seconds into milliseconds. Each contention test also asserts that the workers'
active windows really did overlap, so a regression in the barrier itself shows
up as a failure instead of a silently weaker test.

Second, nothing here is monkeypatched. The dead-owner reclaim below runs
against the real ``psutil.pid_exists`` and a PID that really is gone, and the
write fence is crossed by a holder string minted in another process. A stub of
either would test the test.

Windows has no ``fork``, so children are launched as ``sys.executable -c``
bootstraps that import this module and call one of the worker functions below.
That is the same code path on POSIX, so the file behaves identically on every
host and needs no OS marker.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

import hermes_state
from hermes_state import SessionDB, SessionTurnLeaseLostError

# ----------------------------------------------------------------------
# Tunables
#
# Kept small enough that the whole file is a normal unit-test cost rather
# than a nightly job. The barrier is what lets small N stay meaningful: six
# processes that provably overlap prove more than twelve that do not.
# ----------------------------------------------------------------------

WORKERS = 6
STRESS_CYCLES = 10

RACE_TTL_S = 30.0
RACE_WAIT_S = 2.5
RACE_HOLD_S = 3.5

SHORT_TTL_S = 1.0
LONG_TTL_S = 60.0
DEAD_OWNER_TTL_S = 300.0

#: Wall-clock slack added after a short TTL before a rendezvous is released,
#: so "the lease has expired" is true even on a slow or oversubscribed host.
EXPIRY_PAD_S = 0.75

#: A reclaim of a dead owner's lease must not wait for the TTL. The gap
#: between this bound and ``DEAD_OWNER_TTL_S`` is the whole assertion.
FAST_RECLAIM_BOUND_S = 10.0

READY_TIMEOUT_S = 60.0
CHILD_TIMEOUT_S = 120.0
BARRIER_TIMEOUT_S = 60.0


# ----------------------------------------------------------------------
# Child bootstrap
# ----------------------------------------------------------------------

_HERE = str(Path(__file__).resolve().parent)
_REPO_ROOT = str(Path(hermes_state.__file__).resolve().parent)
_MODULE_NAME = Path(__file__).stem


def _bootstrap() -> str:
    """Source for ``python -c`` that re-imports this module in the child.

    Worker bodies stay real module-level functions this way (readable,
    lintable, importable) instead of a source string, and the child gets the
    repo root on ``sys.path`` explicitly rather than inheriting whatever
    pytest happened to prepend in the parent.
    """
    return (
        "import sys;"
        f"sys.path[:0] = [{_REPO_ROOT!r}, {_HERE!r}];"
        f"import {_MODULE_NAME} as worker;"
        "worker._child_main()"
    )


def _holder(name: str) -> str:
    """Holder in the structured format the reclaim path parses."""
    return f"pid={os.getpid()}:turn={name}:platform=test"


def _message(text: str) -> dict:
    return {"role": "user", "content": text}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _observe_lease(db: SessionDB, session_id: str) -> dict | None:
    """Read the raw lease row for ``session_id``.

    Compression locks have ``get_compression_lock_holder`` for exactly this;
    the turn lease ships no public accessor, so a test that wants to say
    *which* fencing condition fired has to read the table.
    """
    conversation_id = db._session_turn_lease_key(session_id)
    row = db._conn.execute(
        "SELECT holder, acquired_at, expires_at FROM session_turn_leases "
        "WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    if isinstance(row, sqlite3.Row):
        holder, acquired_at, expires_at = (
            row["holder"], row["acquired_at"], row["expires_at"]
        )
    else:
        holder, acquired_at, expires_at = row
    return {
        "holder": holder,
        "acquired_at": float(acquired_at),
        "expires_at": float(expires_at),
    }


def _lease_row_count(db: SessionDB) -> int:
    return int(db._conn.execute(
        "SELECT COUNT(*) FROM session_turn_leases"
    ).fetchone()[0])


# ----------------------------------------------------------------------
# Worker bodies (run in the child, never in the pytest process)
# ----------------------------------------------------------------------


def _signal_ready(payload: dict, extra: dict | None = None) -> None:
    record = {"pid": os.getpid(), "name": payload["name"]}
    record.update(extra or {})
    _write_json(Path(payload["ready_path"]), record)


def _wait_for_go(payload: dict) -> float:
    """Block until the parent releases the barrier; return the release time."""
    go_path = Path(payload["go_path"])
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while not go_path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"barrier {go_path} never opened")
        time.sleep(0.002)
    return time.time()


def _contender_worker(payload: dict) -> dict:
    """Race for one conversation's turn, hold it if won, then release."""
    db = SessionDB(Path(payload["db_path"]))
    session_id = payload["session_id"]
    holder = _holder(payload["name"])
    # Warm the connection (schema check, pragmas) before the barrier so
    # first-open work is not mistaken for lease contention.
    db.get_messages(session_id, limit=1)
    _signal_ready(payload)
    t_go = _wait_for_go(payload)

    started = time.monotonic()
    won = db.acquire_session_turn_lease(
        session_id,
        holder,
        ttl_seconds=payload["ttl_seconds"],
        wait_seconds=payload["wait_seconds"],
        poll_interval_seconds=payload["poll_interval_seconds"],
    )
    acquire_seconds = time.monotonic() - started

    if won:
        # Hold past every loser's wait budget so no loser can win later by
        # TTL expiry or by reclaim. That is what makes "exactly one winner
        # per wave" an assertion about mutual exclusion.
        time.sleep(payload["hold_seconds"])
        db.release_session_turn_lease(session_id, holder)
    result = {
        "name": payload["name"],
        "holder": holder,
        "won": bool(won),
        "acquire_seconds": round(acquire_seconds, 4),
        "t_go": t_go,
        "t_end": time.time(),
    }
    db.close()
    return result


def _holder_worker(payload: dict) -> dict:
    """Acquire the lease, then either die holding it or wait and release."""
    db = SessionDB(Path(payload["db_path"]))
    session_id = payload["session_id"]
    holder = _holder(payload["name"])
    won = db.acquire_session_turn_lease(
        session_id,
        holder,
        ttl_seconds=payload["ttl_seconds"],
        wait_seconds=10.0,
        poll_interval_seconds=0.05,
    )
    record = {"name": payload["name"], "holder": holder, "won": bool(won),
              "pid": os.getpid()}
    _signal_ready(payload, record)

    if payload.get("die_holding"):
        # Closest in-process analogue to a Hermes that was killed mid-turn:
        # no release, no close(), no atexit. The lease row outlives the PID.
        _write_json(Path(payload["result_path"]), record)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    _wait_for_go(payload)
    db.release_session_turn_lease(session_id, holder)
    db.close()
    return record


def _stale_writer_worker(payload: dict) -> dict:
    """Write under a lease, pause at a rendezvous, then write again.

    The second write is the interesting one: by then the parent has either
    reclaimed the conversation or merely watched the lease expire, and this
    process still presents the holder string it acquired with.
    """
    db = SessionDB(Path(payload["db_path"]))
    session_id = payload["session_id"]
    holder = _holder(payload["name"])
    won = db.acquire_session_turn_lease(
        session_id,
        holder,
        ttl_seconds=payload["ttl_seconds"],
        wait_seconds=10.0,
        poll_interval_seconds=0.05,
    )
    first_rows = db.append_messages_batch(
        session_id,
        [_message("first-a"), _message("first-b")],
        turn_lease_holder=holder,
    )
    _signal_ready(payload, {"holder": holder, "won": bool(won),
                            "first_rows": first_rows})
    _wait_for_go(payload)

    refreshed = None
    if payload.get("refresh_before_second_write"):
        refreshed = db.refresh_session_turn_lease(
            session_id, holder, ttl_seconds=payload["ttl_seconds"]
        )

    observed = _observe_lease(db, session_id)
    observed_at = time.time()

    second_rows = None
    error_type = None
    error_message = None
    try:
        second_rows = db.append_messages_batch(
            session_id,
            [_message("second-a"), _message("second-b")],
            turn_lease_holder=holder,
        )
    except BaseException as exc:  # noqa: BLE001 - the class is the evidence
        error_type = f"{type(exc).__module__}.{type(exc).__name__}"
        error_message = str(exc)

    # The row as it stands after the attempt. A write admitted by reviving an
    # expired lease leaves a different row behind than one admitted because
    # the lease never lapsed, and that difference is the regression signal.
    observed_after = _observe_lease(db, session_id)
    observed_after_at = time.time()

    result = {
        "name": payload["name"],
        "holder": holder,
        "won": bool(won),
        "first_rows": first_rows,
        "refreshed": refreshed,
        "observed_lease": observed,
        "observed_at": observed_at,
        "observed_lease_after": observed_after,
        "observed_after_at": observed_after_at,
        "second_rows": second_rows,
        "error_type": error_type,
        "error_message": error_message,
        "message_count": len(db.get_messages(session_id)),
    }
    db.close()
    return result


def _stress_worker(payload: dict) -> dict:
    """Barrier-synchronized acquire, fenced append, release, N times.

    Every failure mode is counted separately so the parent can tell a write
    that was refused from a write that was lost.
    """
    db = SessionDB(Path(payload["db_path"]))
    session_id = payload["session_id"]
    counters = {
        "acquired": 0,
        "acquire_timeout": 0,
        "acquire_locked": 0,
        "rows_written": 0,
        "lease_lost": 0,
        "write_locked": 0,
        "write_failed": 0,
        "release_failed": 0,
    }
    unexpected: list[str] = []

    db.get_messages(session_id, limit=1)
    _signal_ready(payload)
    t_go = _wait_for_go(payload)

    for cycle in range(payload["cycles"]):
        holder = _holder(f"{payload['name']}-{cycle}")
        try:
            won = db.acquire_session_turn_lease(
                session_id,
                holder,
                ttl_seconds=payload["ttl_seconds"],
                wait_seconds=payload["wait_seconds"],
                poll_interval_seconds=0.02,
            )
        except sqlite3.OperationalError as exc:
            counters["acquire_locked"] += 1
            unexpected.append(f"acquire: {exc}")
            continue
        if not won:
            counters["acquire_timeout"] += 1
            continue
        counters["acquired"] += 1
        try:
            counters["rows_written"] += db.append_messages_batch(
                session_id,
                [_message(f"{payload['name']}-{cycle}")],
                turn_lease_holder=holder,
            )
        except SessionTurnLeaseLostError as exc:
            counters["lease_lost"] += 1
            unexpected.append(f"fence: {exc}")
        except sqlite3.OperationalError as exc:
            counters["write_locked"] += 1
            unexpected.append(f"write: {exc}")
        except Exception as exc:  # noqa: BLE001 - counted, then reported
            counters["write_failed"] += 1
            unexpected.append(f"write: {type(exc).__name__}: {exc}")
        try:
            db.release_session_turn_lease(session_id, holder)
        except Exception as exc:  # noqa: BLE001 - counted, then reported
            counters["release_failed"] += 1
            unexpected.append(f"release: {type(exc).__name__}: {exc}")

    result = {
        "name": payload["name"],
        "session_id": session_id,
        "t_go": t_go,
        "t_end": time.time(),
        "unexpected": unexpected[:5],
        **counters,
    }
    db.close()
    return result


_ROLES = {
    "contender": _contender_worker,
    "holder": _holder_worker,
    "stale_writer": _stale_writer_worker,
    "stress": _stress_worker,
}


def _child_main() -> None:
    """Entry point for every spawned child. Never runs in the pytest process."""
    payload = _read_json(Path(sys.argv[1]))
    result_path = Path(payload["result_path"])
    try:
        result = _ROLES[payload["role"]](payload)
    except BaseException as exc:  # noqa: BLE001 - surfaced to the parent
        _write_json(result_path, {
            "name": payload.get("name"),
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    result.setdefault("pid", os.getpid())
    _write_json(result_path, result)


# ----------------------------------------------------------------------
# Parent-side process control
# ----------------------------------------------------------------------


class _Worker:
    """A spawned child plus the file paths it talks to the parent through."""

    def __init__(self, name: str, proc: subprocess.Popen, work_dir: Path,
                 go_path: Path) -> None:
        self.name = name
        self.proc = proc
        self.work_dir = work_dir
        self.ready_path = work_dir / "ready.json"
        self.result_path = work_dir / "result.json"
        self.go_path = go_path
        self.stdout = ""
        self.stderr = ""

    def ready(self) -> dict:
        return _read_json(self.ready_path)

    def result(self) -> dict:
        return _read_json(self.result_path)

    def report(self) -> str:
        parts = [f"worker {self.name} (rc={self.proc.returncode})"]
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.strip()}")
        if self.result_path.exists():
            parts.append(f"result:\n{self.result_path.read_text('utf-8')}")
        return "\n".join(parts)


def _child_env() -> dict:
    """Child environment: inherited, minus pytest's own plugin autoload.

    The hermetic conftest sets ``HERMES_HOME`` and pytest sets
    ``PYTEST_CURRENT_TEST`` on the real ``os.environ``, so children inherit
    both. That is deliberate: ``hermes_state``'s live-DB guard is env
    activated, so it stays armed inside every child spawned here.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_PLUGINS"}
    return env


def _spawn(tmp_path: Path, role: str, name: str, *, go_path: Path | None = None,
           **payload) -> _Worker:
    work_dir = tmp_path / "workers" / name
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_go = go_path or (work_dir / "go")
    payload.update(
        role=role,
        name=name,
        ready_path=str(work_dir / "ready.json"),
        result_path=str(work_dir / "result.json"),
        go_path=str(resolved_go),
    )
    args_path = work_dir / "args.json"
    _write_json(args_path, payload)
    proc = subprocess.Popen(
        [sys.executable, "-c", _bootstrap(), str(args_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_child_env(),
    )
    return _Worker(name, proc, work_dir, resolved_go)


def _drain(worker: _Worker, timeout: float) -> None:
    try:
        worker.stdout, worker.stderr = worker.proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        worker.proc.kill()
        worker.stdout, worker.stderr = worker.proc.communicate()
        raise AssertionError(
            f"worker {worker.name} did not exit within {timeout}s\n"
            f"{worker.report()}"
        )


def _await_ready(workers: list[_Worker],
                 timeout: float = READY_TIMEOUT_S) -> None:
    """Wait for every worker to report ready, failing loudly if one dies."""
    deadline = time.monotonic() + timeout
    pending = list(workers)
    while pending:
        pending = [w for w in pending if not w.ready_path.exists()]
        if not pending:
            return
        dead = [w for w in pending if w.proc.poll() is not None]
        if dead:
            for w in dead:
                _drain(w, timeout=5.0)
            raise AssertionError(
                "worker exited before reporting ready:\n"
                + "\n\n".join(w.report() for w in dead)
            )
        if time.monotonic() > deadline:
            raise AssertionError(
                f"workers never reported ready within {timeout}s: "
                + ", ".join(w.name for w in pending)
            )
        time.sleep(0.005)


def _release_barrier(go_path: Path) -> None:
    go_path.write_text("go", encoding="utf-8")


def _collect(workers: list[_Worker],
             timeout: float = CHILD_TIMEOUT_S) -> list[dict]:
    """Wait for every worker, assert clean exits, return their results."""
    deadline = time.monotonic() + timeout
    for worker in workers:
        _drain(worker, timeout=max(1.0, deadline - time.monotonic()))
    failed = [w for w in workers if w.proc.returncode != 0]
    assert not failed, "workers failed:\n" + "\n\n".join(
        w.report() for w in failed
    )
    return [w.result() for w in workers]


def _assert_overlapped(results: list[dict]) -> None:
    """The workers really did run at the same time.

    Without this, a barrier regression turns every contention assertion in
    this file into a tautology about a queue of one.
    """
    starts = [r["t_go"] for r in results]
    ends = [r["t_end"] for r in results]
    spread_ms = (max(starts) - min(starts)) * 1000.0
    assert max(starts) < min(ends), (
        "workers did not overlap, so the contention assertions are vacuous: "
        f"last start {max(starts):.4f}, first end {min(ends):.4f}, "
        f"barrier release spread {spread_ms:.1f}ms"
    )


# ----------------------------------------------------------------------
# 1. Mutual exclusion across processes
# ----------------------------------------------------------------------


def test_only_one_of_many_processes_wins_the_same_conversation(tmp_path):
    """Six real processes race one conversation; exactly one may hold it.

    This is the property the whole PR exists for, and it is the one a
    single-process test cannot state: the losers here are separate
    interpreters with their own ``SessionDB``, their own SQLite connection,
    and their own PID in the holder string, so nothing but the database is
    shared between them.

    The barrier is load-bearing. Spawning six interpreters that each import
    ``hermes_state`` staggers their starts by hundreds of milliseconds, which
    is longer than the whole race; without parking them until all six are
    ready, the "winner" would simply be whoever the OS scheduled first and no
    contention would ever occur. Every worker therefore warms its connection,
    reports ready, and blocks until the parent opens the barrier. The winner
    then holds past every loser's wait budget, so a loser cannot win later by
    TTL expiry and turn "exactly one winner" into an accident of timing.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("shared", source="test")

    go_path = tmp_path / "go"
    workers = [
        _spawn(
            tmp_path,
            "contender",
            f"contender-{i}",
            go_path=go_path,
            db_path=str(db_path),
            session_id="shared",
            ttl_seconds=RACE_TTL_S,
            wait_seconds=RACE_WAIT_S,
            hold_seconds=RACE_HOLD_S,
            poll_interval_seconds=0.1,
        )
        for i in range(WORKERS)
    ]
    _await_ready(workers)
    _release_barrier(go_path)
    results = _collect(workers)

    _assert_overlapped(results)
    winners = [r for r in results if r["won"]]
    losers = [r for r in results if not r["won"]]
    assert len(winners) == 1, (
        f"expected exactly one winner among {WORKERS} processes, got "
        f"{[r['name'] for r in winners]}"
    )
    assert len(losers) == WORKERS - 1

    # Losers were refused for the whole wait budget rather than erroring out
    # early, which is what "waited for the current owner" means.
    for loser in losers:
        assert loser["acquire_seconds"] >= RACE_WAIT_S * 0.5, loser

    # The winner released on its way out, so the conversation is free again.
    assert _lease_row_count(db) == 0
    reclaimer = f"pid={os.getpid()}:turn=after-wave"
    assert db.try_acquire_session_turn_lease("shared", reclaimer, ttl_seconds=5)
    db.release_session_turn_lease("shared", reclaimer)


# ----------------------------------------------------------------------
# 2. Dead-owner reclaim, against the real liveness probe
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    hermes_state.psutil is None and os.name == "nt",
    reason="without psutil, Windows has no safe PID probe and stays TTL-only",
)
def test_lease_of_a_process_that_died_is_reclaimed_without_stubbing_psutil(
    tmp_path,
):
    """A crashed owner's lease is reclaimed in seconds, not in its TTL.

    The existing coverage for this path replaces ``hermes_state.psutil`` with
    a ``SimpleNamespace`` whose ``pid_exists`` returns False, which proves the
    SQL branch but assumes the answer. Here a real child really acquires a
    300-second lease, really exits without releasing it (``os._exit``, so no
    ``close()`` and no ``atexit``), and the parent asks the real
    ``psutil.pid_exists`` about a PID the kernel has genuinely reaped. The
    reclaim has to happen against that answer or not at all.

    There is no barrier because there is no race to synchronize: the ordering
    is a handoff, and the parent waits on the child's actual exit rather than
    on a sleep, so a slow host cannot turn the assertion into a coin flip.

    The live-owner arm is the control. Without it, "reclaimed quickly" would
    also be satisfied by a lease that is simply never enforced.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("crashed", source="test")
    db.create_session("alive", source="test")

    # Arm A: the owner dies holding a 300s lease.
    dead = _spawn(
        tmp_path,
        "holder",
        "dies-holding",
        db_path=str(db_path),
        session_id="crashed",
        ttl_seconds=DEAD_OWNER_TTL_S,
        die_holding=True,
    )
    _await_ready([dead])
    dead_record = dead.ready()
    assert dead_record["won"] is True
    # Reap before probing. On Linux an unreaped child is a zombie, and
    # psutil.pid_exists() reports a zombie as alive, so measuring the reclaim
    # before communicate() would time the TTL path on POSIX and the dead-PID
    # path on Windows while looking like one test.
    _drain(dead, timeout=CHILD_TIMEOUT_S)
    assert dead.proc.returncode == 0, dead.report()

    lease = _observe_lease(db, "crashed")
    assert lease is not None, "the dead owner's lease row should still exist"
    assert lease["holder"] == dead_record["holder"]
    assert lease["expires_at"] - time.time() > DEAD_OWNER_TTL_S / 2, (
        "the lease should still be far from its TTL when the owner dies"
    )
    if hermes_state.psutil is not None:
        assert not hermes_state.psutil.pid_exists(dead_record["pid"]), (
            "the child PID must really be gone before the reclaim is measured"
        )

    started = time.monotonic()
    assert db.try_acquire_session_turn_lease(
        "crashed", f"pid={os.getpid()}:turn=reclaimer", ttl_seconds=30
    ) is True
    reclaim_seconds = time.monotonic() - started
    assert reclaim_seconds < FAST_RECLAIM_BOUND_S, (
        f"reclaim took {reclaim_seconds:.3f}s against a "
        f"{DEAD_OWNER_TTL_S}s TTL"
    )

    # Arm B (control): a live owner with the same TTL is not reclaimed.
    live = _spawn(
        tmp_path,
        "holder",
        "stays-alive",
        db_path=str(db_path),
        session_id="alive",
        ttl_seconds=DEAD_OWNER_TTL_S,
    )
    _await_ready([live])
    assert live.ready()["won"] is True
    for _ in range(3):
        assert db.try_acquire_session_turn_lease(
            "alive", f"pid={os.getpid()}:turn=intruder", ttl_seconds=30
        ) is False, "a live owner's unexpired lease must not be reclaimed"
        time.sleep(0.1)
    _release_barrier(live.go_path)
    _collect([live])

    assert db.try_acquire_session_turn_lease(
        "alive", f"pid={os.getpid()}:turn=after-release", ttl_seconds=5
    ) is True


# ----------------------------------------------------------------------
# 3. Write fencing across processes
# ----------------------------------------------------------------------


def test_a_stale_holder_cannot_write_after_another_process_reclaims(tmp_path):
    """A lost holder's transcript flush is refused inside the write txn.

    Refresh-loss interrupt is cooperative, so the only thing standing between
    a stalled process and a corrupted transcript is the fence in
    ``_check_transcript_write_guards``. In one interpreter that fence can only
    be crossed by a holder string the same process minted. Here the second
    process really is a different process: the child acquires, writes, and
    stalls; the parent waits for the lease to expire, takes the conversation
    under its own PID, appends its own turn; and only then does the child try
    to flush the rest of its turn.

    The rendezvous replaces a sleep, so the test does not depend on the child
    waking up at a particular moment. The child records the lease row it saw
    immediately before its second write, which is what lets this assert
    *which* condition fenced it: the row is present, unexpired, and owned by
    the parent, so the refusal is the foreign-holder branch and not a lease
    that merely timed out.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("handover", source="test")

    child = _spawn(
        tmp_path,
        "stale_writer",
        "stalled-owner",
        db_path=str(db_path),
        session_id="handover",
        ttl_seconds=SHORT_TTL_S,
    )
    _await_ready([child])
    child_record = child.ready()
    assert child_record["won"] is True
    assert child_record["first_rows"] == 2

    # Take the conversation as soon as the child's lease lapses. Polling the
    # real acquire path is the honest way to wait: it succeeds exactly when
    # the reclaim is legitimate.
    parent_holder = f"pid={os.getpid()}:turn=successor"
    deadline = time.monotonic() + 30.0
    while not db.try_acquire_session_turn_lease(
        "handover", parent_holder, ttl_seconds=LONG_TTL_S
    ):
        assert time.monotonic() < deadline, "never reclaimed the expired lease"
        time.sleep(0.05)
    assert db.append_messages_batch(
        "handover", [_message("successor-a")], turn_lease_holder=parent_holder
    ) == 1

    _release_barrier(child.go_path)
    (result,) = _collect([child])

    assert result["error_type"] == (
        "hermes_state.SessionTurnLeaseLostError"
    ), result
    assert "turn lease lost" in result["error_message"]
    assert result["second_rows"] is None

    observed = result["observed_lease"]
    assert observed is not None, "the successor's lease row should be visible"
    assert observed["holder"] == parent_holder, (
        "the child should have been fenced by the successor's holder, not by "
        f"a missing row: {observed}"
    )
    assert observed["expires_at"] > result["observed_at"], (
        "the fencing lease was unexpired, so this is the foreign-holder "
        f"branch rather than the expiry branch: {observed}"
    )

    # The stale process's late turn never landed.
    assert [m["content"] for m in db.get_messages("handover")] == [
        "first-a", "first-b", "successor-a",
    ]
    db.release_session_turn_lease("handover", parent_holder)


# ----------------------------------------------------------------------
# 4. Expired but uncontested: revival, and the limits of it
# ----------------------------------------------------------------------


def test_an_expired_lease_revives_for_its_own_owner_but_not_once_reaped(
    tmp_path,
):
    """An owner that overran its TTL keeps its turn; a reaped row does not.

    A process whose refresher was starved (a long tool call, a suspended
    machine, a wall-clock jump) can reach its flush after its own lease has
    lapsed. Since ``46d87e34`` the write fence separates lateness from
    takeover: an expired row that is still the caller's own is revived inside
    the same ``BEGIN IMMEDIATE`` transaction as the insert and the write is
    admitted, while an absent row or a foreign holder is still fenced. This
    test pins both halves, because a revival that is even slightly too
    generous would quietly undo the fencing the rest of this file checks.

    Four arms, all waiting the same wall-clock time past the same short TTL,
    so the only variable is what happened to the lease row in the meantime.

    1. Expired, uncontested, row untouched: the write is admitted and the row
       comes back unexpired. This is the arm that used to raise
       ``SessionTurnLeaseLostError``.
    2. Control: the owner refreshes first, so the row was never expired at
       write time. Distinguishes revival from "it was fine all along".
    3. Control: the same pause under a TTL that never lapses. Shows the pause
       itself is not what any of this turns on.
    4. Reaped: a second process acquires the expired lease and releases it, so
       the row is gone when the original owner returns. Still fenced. This is
       the case an over-broad revival would break, and it is only reachable
       across processes, since the owner here never released anything.

    The parent is doing something a single process cannot do in arms 1 and 4:
    a second live process observes the row while the owner sits at the
    rendezvous, which is what makes "uncontested" and "reaped" measurements
    rather than assumptions. The rendezvous replaces a sleep on the child's
    side, so a slow host cannot turn either arm into a race.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    for session_id in ("expired", "refreshed", "long-ttl", "reaped"):
        db.create_session(session_id, source="test")

    def run_arm(name: str, session_id: str, ttl: float, *,
                refresh: bool = False, parent_action: str = "none") -> dict:
        child = _spawn(
            tmp_path,
            "stale_writer",
            name,
            db_path=str(db_path),
            session_id=session_id,
            ttl_seconds=ttl,
            refresh_before_second_write=refresh,
        )
        _await_ready([child])
        ready = child.ready()
        assert ready["won"] is True
        # Every arm waits the same wall-clock time past the short TTL, so the
        # only difference between them is the refresh, the TTL, and what the
        # parent does to the row.
        time.sleep(SHORT_TTL_S + EXPIRY_PAD_S)
        if parent_action == "observe_expired":
            watched = _observe_lease(db, session_id)
            assert watched is not None, "the owner's row should still be there"
            assert watched["holder"] == ready["holder"], (
                f"a second process saw the lease change hands: {watched}"
            )
            assert watched["expires_at"] <= time.time(), watched
        elif parent_action == "take_and_release":
            # A successor legitimately claims the lapsed lease, finishes, and
            # releases. Nothing of the original owner's row survives.
            successor = f"pid={os.getpid()}:turn=successor-{name}"
            assert db.try_acquire_session_turn_lease(
                session_id, successor, ttl_seconds=LONG_TTL_S
            ), "the expired lease should have been claimable"
            db.release_session_turn_lease(session_id, successor)
            assert _observe_lease(db, session_id) is None
        _release_barrier(child.go_path)
        (result,) = _collect([child])
        return result

    expired = run_arm("expired-owner", "expired", SHORT_TTL_S,
                      parent_action="observe_expired")
    refreshed = run_arm("refreshing-owner", "refreshed", SHORT_TTL_S,
                        refresh=True)
    long_ttl = run_arm("unexpired-owner", "long-ttl", LONG_TTL_S)
    reaped = run_arm("reaped-owner", "reaped", SHORT_TTL_S,
                     parent_action="take_and_release")

    # Arm 1: the lease really had lapsed, and really was still the owner's.
    assert expired["observed_lease"]["holder"] == expired["holder"], (
        "nobody else ever held this conversation"
    )
    assert expired["observed_lease"]["expires_at"] <= expired["observed_at"]
    # ... and the write went through anyway, on a revived row.
    assert expired["error_type"] is None, expired
    assert expired["second_rows"] == 2
    assert expired["message_count"] == 4
    revived = expired["observed_lease_after"]
    assert revived is not None, "the revived row should still be there"
    assert revived["holder"] == expired["holder"]
    assert revived["expires_at"] > expired["observed_after_at"], (
        "the write should have re-extended the owner's lease, not left it "
        f"expired: {revived}"
    )
    assert revived["acquired_at"] == expired["observed_lease"]["acquired_at"], (
        "revival should extend the owner's original row rather than replace it"
    )

    # Arm 2: refreshing first also works, and shows that refresh has never
    # applied an expiry check of its own.
    assert refreshed["refreshed"] is True
    assert refreshed["error_type"] is None, refreshed
    assert refreshed["second_rows"] == 2
    assert refreshed["message_count"] == 4

    # Arm 3: a TTL that never lapses is unaffected by any of this.
    assert long_ttl["error_type"] is None, long_ttl
    assert long_ttl["observed_lease"]["expires_at"] > long_ttl["observed_at"]
    assert long_ttl["second_rows"] == 2
    assert long_ttl["message_count"] == 4

    # Arm 4: once a successor has taken and released the row, the original
    # owner is out. Revival must not resurrect a lease nobody is holding.
    assert reaped["observed_lease"] is None, reaped
    assert reaped["error_type"] == "hermes_state.SessionTurnLeaseLostError", (
        reaped
    )
    assert "turn lease lost" in reaped["error_message"]
    assert reaped["second_rows"] is None
    assert reaped["message_count"] == 2, "only the first batch survived"
    assert _observe_lease(db, "reaped") is None, (
        "a fenced write must not resurrect the row on its way out"
    )


# ----------------------------------------------------------------------
# 5. Contention stress
# ----------------------------------------------------------------------


def _run_stress_arm(tmp_path: Path, db_path: Path, arm: str,
                    sessions: list[str]) -> list[dict]:
    go_path = tmp_path / f"go-{arm}"
    workers = [
        _spawn(
            tmp_path,
            "stress",
            f"{arm}-{i}",
            go_path=go_path,
            db_path=str(db_path),
            session_id=sessions[i],
            cycles=STRESS_CYCLES,
            ttl_seconds=15.0,
            wait_seconds=30.0,
        )
        for i in range(WORKERS)
    ]
    _await_ready(workers)
    _release_barrier(go_path)
    results = _collect(workers)
    _assert_overlapped(results)
    return results


def _assert_no_failures(results: list[dict], arm: str) -> None:
    for result in results:
        for counter in ("acquire_timeout", "acquire_locked", "lease_lost",
                        "write_locked", "write_failed", "release_failed"):
            assert result[counter] == 0, (
                f"{arm}: worker {result['name']} reported {counter}="
                f"{result[counter]}; first errors: {result['unexpected']}"
            )
        assert result["acquired"] == STRESS_CYCLES, result
        assert result["rows_written"] == STRESS_CYCLES, result


def test_concurrent_processes_lose_no_writes_and_leave_no_orphan_leases(
    tmp_path,
):
    """Sustained multi-process traffic writes every row exactly once.

    Six processes each run ten cycles of acquire, fenced append, release
    against one ``state.db``. The first arm gives every worker its own
    conversation, so the leases never collide but the SQLite writers do; the
    second arm points all six at a single conversation, so every cycle has to
    be handed through the lease. Both arms assert exact row counts, meaning
    no write was silently lost and none was written twice, and both assert
    that the lease table is empty afterwards, meaning every acquire was
    matched by a release and no orphan row is left to block the next turn.

    The barrier is what makes six workers enough. Interpreter startup plus
    the ``hermes_state`` import costs far more than a ten-cycle loop, so
    unsynchronized workers run one after another and the arm degenerates into
    a sequential smoke test that would pass with no locking at all. Parking
    every worker until all six are ready compresses the starts to
    milliseconds, and ``_assert_overlapped`` fails the test if that ever
    stops being true.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    own = [f"own-{i}" for i in range(WORKERS)]
    for session_id in own + ["shared"]:
        db.create_session(session_id, source="test")

    distinct = _run_stress_arm(tmp_path, db_path, "distinct", own)
    _assert_no_failures(distinct, "distinct-sessions")
    for session_id in own:
        assert len(db.get_messages(session_id)) == STRESS_CYCLES

    shared = _run_stress_arm(tmp_path, db_path, "shared", ["shared"] * WORKERS)
    _assert_no_failures(shared, "shared-session")
    assert len(db.get_messages("shared")) == WORKERS * STRESS_CYCLES

    total = sum(r["rows_written"] for r in distinct + shared)
    assert total == 2 * WORKERS * STRESS_CYCLES
    assert _lease_row_count(db) == 0, "every acquire should have been released"
