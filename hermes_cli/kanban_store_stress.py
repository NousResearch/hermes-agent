"""Opt-in Kanban store stress-contract harnesses.

The harnesses in this module are deliberately isolated: every run creates a
fresh temporary ``HERMES_HOME`` and explicit SQLite DB path.  They are intended
for adapter contract validation and must never operate on the user's active
Kanban board.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Iterator, Sequence

from hermes_cli.kanban_store import KanbanStore
from hermes_cli.kanban_store_factory import get_default_kanban_store

_STRESS_BOARD = "db-adapter-stress"
_ENV_KEYS = (
    "HERMES_HOME",
    "HOME",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_BACKEND",
)


@dataclass(frozen=True)
class StressHarnessConfig:
    """Configuration for a bounded stress-contract run."""

    task_count: int = 12
    claim_workers: int = 4
    base_dir: Path | str | None = None
    board: str = _STRESS_BOARD
    backend: str = "sqlite"
    claim_ttl_seconds: int = 30


@dataclass(frozen=True)
class ClaimStressResult:
    backend: str
    board: str
    hermes_home: str
    db_path: str
    task_count: int
    claimed_task_ids: list[str]
    duplicate_claims: list[str]
    status_counts: dict[str, int]
    open_run_count: int


@dataclass(frozen=True)
class DispatchSignalStressResult:
    backend: str
    board: str
    hermes_home: str
    db_path: str
    signal_names: list[str]
    spawned_task_ids: list[str]
    crashed_task_ids: list[str]
    status_counts: dict[str, int]
    open_run_count: int
    crash_event_count: int
    unexpected_failures: list[str] = field(default_factory=list)


@contextmanager
def _isolated_kanban_env(config: StressHarnessConfig) -> Iterator[tuple[KanbanStore, Path, Path]]:
    """Create and activate an isolated SQLite-backed Kanban environment."""
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    base = Path(config.base_dir) if config.base_dir is not None else Path(
        tempfile.mkdtemp(prefix="hermes_kanban_store_stress_")
    )
    base.mkdir(parents=True, exist_ok=True)
    home = base / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    db_path = base / "kanban-stress.db"
    try:
        os.environ["HERMES_HOME"] = str(home)
        os.environ["HOME"] = str(home)
        os.environ["HERMES_KANBAN_DB"] = str(db_path)
        os.environ.pop("HERMES_KANBAN_HOME", None)
        os.environ["HERMES_KANBAN_BOARD"] = config.board
        os.environ["HERMES_KANBAN_BACKEND"] = config.backend
        store = get_default_kanban_store()
        store.init_db()
        yield store, home, db_path
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _status_counts(conn) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status ORDER BY status"
    ).fetchall()
    return {str(row["status"]): int(row["n"]) for row in rows}


def _open_run_count(conn) -> int:
    return int(
        conn.execute("SELECT COUNT(*) FROM task_runs WHERE ended_at IS NULL").fetchone()[0]
    )


def _seed_tasks(store, count: int, *, assignee: str = "default") -> list[str]:
    task_ids: list[str] = []
    with store.connect() as conn:
        for idx in range(count):
            task_ids.append(
                store.create_task(
                    conn,
                    title=f"stress-contract-{idx}",
                    assignee=assignee,
                    created_by="kanban-store-stress",
                    tenant="db-adapter-stress",
                )
            )
    return task_ids


def run_claim_stress_contract(config: StressHarnessConfig) -> ClaimStressResult:
    """Concurrently claim and complete tasks against the selected store.

    This is intentionally bounded and deterministic enough for CI while still
    exercising separate SQLite connections and atomic ``claim_task`` behavior.
    """
    with _isolated_kanban_env(config) as (store, home, db_path):
        _seed_tasks(store, config.task_count)

        def worker(worker_idx: int) -> list[str]:
            claimed_ids: list[str] = []
            idle_rounds = 0
            with store.connect() as conn:
                while idle_rounds < 3:
                    row = conn.execute(
                        "SELECT id FROM tasks "
                        "WHERE status='ready' AND claim_lock IS NULL "
                        "ORDER BY id LIMIT 1"
                    ).fetchone()
                    if row is None:
                        idle_rounds += 1
                        time.sleep(0.01)
                        continue
                    idle_rounds = 0
                    task_id = row["id"]
                    claimed = store.claim_task(
                        conn,
                        task_id,
                        ttl_seconds=config.claim_ttl_seconds,
                        claimer=f"stress-worker-{worker_idx}",
                    )
                    if claimed is None:
                        continue
                    claimed_ids.append(claimed.id)
                    store.complete_task(
                        conn,
                        claimed.id,
                        result="ok",
                        summary=f"stress worker {worker_idx} completed",
                        expected_run_id=claimed.current_run_id,
                    )
            return claimed_ids

        claimed_task_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=config.claim_workers) as pool:
            futures = [pool.submit(worker, idx) for idx in range(config.claim_workers)]
            for fut in as_completed(futures):
                claimed_task_ids.extend(fut.result())

        counts = Counter(claimed_task_ids)
        duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
        with store.connect() as conn:
            statuses = _status_counts(conn)
            open_runs = _open_run_count(conn)
        return ClaimStressResult(
            backend=store.capabilities.backend,
            board=config.board,
            hermes_home=str(home),
            db_path=str(db_path),
            task_count=config.task_count,
            claimed_task_ids=claimed_task_ids,
            duplicate_claims=duplicates,
            status_counts=statuses,
            open_run_count=open_runs,
        )


def _spawn_sleeping_worker(processes: list[subprocess.Popen]):
    def spawn(_task, _workspace):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        processes.append(proc)
        return proc.pid

    return spawn


def _signal_name(sig: signal.Signals | int) -> str:
    try:
        return signal.Signals(sig).name
    except Exception:
        return f"SIG{int(sig)}"


def _terminate_spawned_processes(processes: Iterable[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def run_dispatch_signal_contract(
    config: StressHarnessConfig,
    *,
    signals: Sequence[signal.Signals | int] = (signal.SIGTERM, signal.SIGKILL),
) -> DispatchSignalStressResult:
    """Spawn workers through dispatch, signal them, and verify crash recovery."""
    processes: list[subprocess.Popen] = []
    unexpected: list[str] = []
    with _isolated_kanban_env(config) as (store, home, db_path):
        try:
            task_ids = _seed_tasks(store, len(signals))
            with store.connect() as conn:
                first = store.dispatch_once(
                    conn,
                    spawn_fn=_spawn_sleeping_worker(processes),
                    max_spawn=len(signals),
                    ttl_seconds=config.claim_ttl_seconds,
                    failure_limit=99,
                )
                spawned_ids = [task_id for task_id, _assignee, _workspace in first.spawned]
                if sorted(spawned_ids) != sorted(task_ids):
                    unexpected.append(
                        f"spawned {sorted(spawned_ids)} but expected {sorted(task_ids)}"
                    )

                for proc, sig in zip(processes, signals):
                    try:
                        os.kill(proc.pid, int(sig))
                    except ProcessLookupError:
                        unexpected.append(f"pid {proc.pid} disappeared before {_signal_name(sig)}")

                # Give children time to transition to exited/zombie.  Do not
                # call Popen.wait()/poll() here: dispatch_once should reap and
                # classify the child exits itself.
                time.sleep(0.25)
                second = store.dispatch_once(
                    conn,
                    max_spawn=0,
                    ttl_seconds=config.claim_ttl_seconds,
                    failure_limit=99,
                )
                crashed_ids = list(second.crashed)
                statuses = _status_counts(conn)
                open_runs = _open_run_count(conn)
                crash_event_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM task_events WHERE kind='crashed'"
                    ).fetchone()[0]
                )
        finally:
            _terminate_spawned_processes(processes)

        return DispatchSignalStressResult(
            backend=store.capabilities.backend,
            board=config.board,
            hermes_home=str(home),
            db_path=str(db_path),
            signal_names=[_signal_name(sig) for sig in signals],
            spawned_task_ids=spawned_ids,
            crashed_task_ids=crashed_ids,
            status_counts=statuses,
            open_run_count=open_runs,
            crash_event_count=crash_event_count,
            unexpected_failures=unexpected,
        )


__all__ = [
    "ClaimStressResult",
    "DispatchSignalStressResult",
    "StressHarnessConfig",
    "run_claim_stress_contract",
    "run_dispatch_signal_contract",
]
