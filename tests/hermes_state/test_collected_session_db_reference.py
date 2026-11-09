"""A holder-less ``SessionDB`` collected unclosed returns its registry reference without deadlocking.

The cyclic GC runs a finalizer on whatever thread allocates, including one inside the registry's own lock or a
path's lifecycle lock. The finalizer therefore only queues the reference, and the next ``SessionDB`` open or close
returns it. The cases that could hang run in a child process under a faulthandler deadline, which dumps every
thread's stack and ends the child when a lock is never released.
"""

import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hermes_state_registry as registry
from hermes_state import SessionDB
from persistence.sqlite import _registry as holders

DEADLINE_S = 5
_ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = f"""
import faulthandler, gc, os, shutil, sys, threading, time
from pathlib import Path
from hermes_state import SessionDB
from persistence.sqlite import _registry as holders

faulthandler.dump_traceback_later({DEADLINE_S * 4}, exit=True)
path = Path(sys.argv[1]) / "state.db"


class Owner:
    def __init__(self, db):
        self.db = db
        self.me = self  # a cycle: only the cyclic GC frees it


def refcount():
    generation = holders._generations.get(path.resolve())
    return None if generation is None else generation.refcount


def within_deadline(target):
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join({DEADLINE_S})
    if worker.is_alive():
        print("worker blocked: a finalizer re-entered a lock its thread holds", file=sys.stderr)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(3)
"""


class _Owner:
    """Holds a SessionDB inside a reference cycle, so only the cyclic GC frees it."""

    def __init__(self, db):
        self.db = db
        self.me = self


def _run_bounded(tmp_path, scenario: str) -> str:
    """Run *scenario* in a child under the deadline; its stdout, or a failure carrying every thread's stack."""
    child = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(scenario), str(tmp_path)],
        cwd=_ROOT, capture_output=True, text=True, timeout=DEADLINE_S * 8, check=False,
    )
    if child.returncode:
        pytest.fail(f"child exited {child.returncode}:\n{child.stderr}", pytrace=False)
    return child.stdout.strip()


def test_a_collection_inside_the_registry_lock_queues_the_reference(tmp_path):
    out = _run_bounded(tmp_path, """
        anchor = SessionDB(path)
        gc.disable()
        Owner(SessionDB(path))
        before = refcount()

        def collect_under_the_registry_lock():
            with holders._lock:
                gc.collect()

        within_deadline(collect_under_the_registry_lock)
        SessionDB(path).close()  # the next open returns the queued reference
        after = refcount()
        anchor.close()
        print(before, after, refcount())
    """)
    assert out == "2 1 None"


def test_a_collection_inside_the_replacement_open_queues_the_retired_reference(tmp_path):
    out = _run_bounded(tmp_path, """
        gc.disable()
        Owner(SessionDB(path))  # the generation's last reference
        copy = path.with_suffix(".copy")
        shutil.copy2(path, copy)
        os.replace(copy, path)  # a new inode: the next acquire retires the generation
        open_holder = holders._open_holder

        def open_with_a_collection(target):
            gc.collect()  # the cyclic GC firing while acquire holds the path's lifecycle lock
            return open_holder(target)

        holders._open_holder = open_with_a_collection
        within_deadline(lambda: holders.release(holders.acquire(path)))
        holders._open_holder = open_holder
        queued = len(holders._retired)
        SessionDB(path).close()
        print(queued, len(holders._retired))
    """)
    assert out == "1 0"


def test_handles_collected_at_default_thresholds_never_stall_registry_readers(tmp_path):
    out = _run_bounded(tmp_path, """
        anchor = SessionDB(path)
        stop = time.monotonic() + 3.0
        churned = [0]

        def churn():
            while time.monotonic() < stop:
                Owner(SessionDB(path))
                churned[0] += 1

        def read_registry():
            while time.monotonic() < stop:
                holders.other_generations_for_path(path)
                holders.stats()

        churner = threading.Thread(target=churn, daemon=True)
        churner.start()
        within_deadline(read_registry)
        within_deadline(churner.join)
        gc.collect()
        SessionDB(path).close()
        print(churned[0] > 0, refcount())
    """)
    assert out == "True 1"


def test_shutdown_closes_a_generation_whose_collected_reference_is_still_queued(tmp_path):
    path = tmp_path / "state.db"
    anchor = SessionDB(path)
    holder = anchor._holder
    _Owner(SessionDB(path))
    gc.collect()
    try:
        assert registry.close_all() == 1
        assert holder._conn is None
        SessionDB(path).close()  # a reference drained after the shutdown releases nothing it does not own
        assert holders.stats()["live_generations"] == 0
    finally:
        anchor.close()
        registry.close_all()


def test_an_explicit_close_releases_at_once_and_its_collection_releases_nothing(tmp_path):
    path = tmp_path / "state.db"
    anchor = SessionDB(path)
    try:
        db = SessionDB(path)
        db.close()
        assert holders._generations[path.resolve()].refcount == 1
        del db
        gc.collect()
        SessionDB(path).close()
        assert holders._generations[path.resolve()].refcount == 1
    finally:
        anchor.close()
        registry.close_all()
    assert holders.stats()["live_generations"] == 0
