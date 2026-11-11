"""A writable ``hermes_state.SessionDB`` stand-in collected unclosed releases its store handle without deadlocking.

The cyclic GC runs a finalizer on whatever thread allocates, including one inside a lock the release takes: the store
handle's own, the shared-backend cache's, the registry's, a path's lifecycle lock, the file's read budget, the WAL
guard. The finalizer therefore only hands the handle to the reaper thread, which releases it once that lock is free.
The cases that could hang run in a child process under a faulthandler deadline, which dumps every thread's stack and
ends the child when a lock is never released.
"""

import gc
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import hermes_state
import persistence_assembly
from persistence.sqlite import _registry as holders
from persistence_assembly import close_all_stores, open_persistence, profile_scope
from tests.waiting import wait_until

DEADLINE_S = 5
_ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = f"""
import faulthandler, gc, os, shutil, sys, threading, time, warnings
from pathlib import Path
warnings.simplefilter("ignore", DeprecationWarning)
import hermes_state
import persistence_assembly
from persistence.sqlite import _registry as holders
from persistence.sqlite._state import lockguard
from persistence.sqlite._state.readpool import _read_budget_for
from tests.waiting import wait_until

faulthandler.dump_traceback_later({DEADLINE_S * 4}, exit=True)
path = Path(sys.argv[1]) / "state.db"


class Owner:
    def __init__(self, db):
        self.db = db
        self.me = self  # a cycle: only the cyclic GC frees it


def open_here():
    return holders._generations.get(path.resolve()) is not None


def within_deadline(target):
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join({DEADLINE_S})
    if worker.is_alive():
        print("worker blocked: a finalizer re-entered a lock its thread holds", file=sys.stderr)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(3)


def released(predicate):
    wait_until(predicate, timeout={DEADLINE_S}, what="the reaper releasing the collected handle")
"""


class _Owner:
    """Holds a stand-in inside a reference cycle, so only the cyclic GC frees it."""

    def __init__(self, db):
        self.db = db
        self.me = self


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(hermes_state, "_warned", True)


def _run_bounded(tmp_path, scenario: str) -> str:
    """Run *scenario* in a child under the deadline; its stdout, or a failure carrying every thread's stack."""
    child = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(scenario), str(tmp_path)],
        cwd=_ROOT, capture_output=True, text=True, timeout=DEADLINE_S * 8, check=False,
    )
    if child.returncode:
        pytest.fail(f"child exited {child.returncode}:\n{child.stderr}", pytrace=False)
    return child.stdout.strip()


def _handles(scope):
    """How many open handles share ``scope``'s backend (0 once the backend is released)."""
    entry = persistence_assembly._shared.get(scope.hermes_home_key)
    return 0 if entry is None else entry[1]


@pytest.mark.parametrize(
    "lock",
    ["handle._lock", "persistence_assembly._cache_lock", "holders._lock", "budget._lock", "lockguard._LOCK"],
    ids=["store-handle", "backend-cache", "registry", "read-budget", "wal-guard"],
)
def test_a_collection_under_a_lock_the_release_takes_releases_the_handle_once_it_is_free(tmp_path, lock):
    out = _run_bounded(tmp_path, f"""
        gc.disable()
        owner = Owner(hermes_state.SessionDB(path))  # the store's only handle: its release closes the store
        handle = owner.db._store
        budget = _read_budget_for(path)
        del owner
        before = open_here()

        def collect_under_the_lock():
            with {lock}:
                gc.collect()

        within_deadline(collect_under_the_lock)
        released(lambda: not open_here())
        print(before, open_here())
    """)
    assert out == "True False"


def test_a_collection_inside_the_replacement_open_releases_the_retired_store(tmp_path):
    out = _run_bounded(tmp_path, """
        gc.disable()
        Owner(hermes_state.SessionDB(path))  # the store's only handle
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
        released(lambda: not holders._retired)
        print(len(holders._retired), open_here())
    """)
    assert out == "0 False"


def test_handles_collected_at_default_thresholds_never_stall_registry_readers(tmp_path):
    out = _run_bounded(tmp_path, """
        scope = persistence_assembly.profile_scope(home=path.parent)
        anchor = persistence_assembly.open_persistence(scope, eager=True)
        stop = time.monotonic() + 3.0
        churned = [0]

        def churn():
            while time.monotonic() < stop:
                Owner(hermes_state.SessionDB(path))
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
        released(lambda: persistence_assembly._shared[scope.hermes_home_key][1] == 1)
        print(churned[0] > 0, persistence_assembly._shared[scope.hermes_home_key][1])
    """)
    assert out == "True 1"


def test_a_stand_in_dropped_unclosed_releases_its_handle_without_a_later_open_or_close(tmp_path):
    scope = profile_scope(home=tmp_path)
    anchor = open_persistence(scope, eager=True)
    try:
        db = hermes_state.SessionDB(tmp_path / "state.db")
        db.create_session("s1", "cli")
        assert _handles(scope) == 2
        del db  # refcount collection: nothing opens or closes a store afterwards
        wait_until(lambda: _handles(scope) == 1, timeout=DEADLINE_S, what="the dropped stand-in's handle released")
    finally:
        anchor.close()
    assert _handles(scope) == 0
    assert holders._generations.get((tmp_path / "state.db").resolve()) is None


def test_shutdown_closes_a_store_whose_collected_handle_is_still_queued(tmp_path, monkeypatch):
    scope = profile_scope(home=tmp_path)
    anchor = open_persistence(scope, eager=True)
    reaping, resume, done = threading.Event(), threading.Event(), threading.Event()
    release_shared = persistence_assembly._release_shared

    def reaper_stalls_before_releasing(key, entry):
        if threading.current_thread().name != "sessiondb-reaper":
            return release_shared(key, entry)
        reaping.set()
        resume.wait(DEADLINE_S)
        release_shared(key, entry)
        done.set()

    monkeypatch.setattr(persistence_assembly, "_release_shared", reaper_stalls_before_releasing)
    _Owner(hermes_state.SessionDB(tmp_path / "state.db"))
    gc.collect()
    try:
        assert reaping.wait(DEADLINE_S)
        assert _handles(scope) == 2  # the collected handle is not released yet
        assert close_all_stores() == 1
        resume.set()
        assert done.wait(DEADLINE_S)  # a handle released after the shutdown releases only its own reference
        assert _handles(scope) == 1
        assert holders.stats()["live_generations"] == 0
    finally:
        resume.set()
        anchor.close()
        close_all_stores()
    assert _handles(scope) == 0


def test_an_explicit_close_releases_at_once_and_its_collection_releases_nothing(tmp_path):
    scope = profile_scope(home=tmp_path)
    anchor = open_persistence(scope, eager=True)
    try:
        db = hermes_state.SessionDB(tmp_path / "state.db")
        db.close()
        assert _handles(scope) == 1
        del db
        gc.collect()
        hermes_state.SessionDB(tmp_path / "state.db").close()
        assert _handles(scope) == 1
    finally:
        anchor.close()
    assert _handles(scope) == 0


def test_a_reaper_that_fails_to_start_leaves_no_handle_behind(tmp_path, monkeypatch):
    class Unstartable:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(hermes_state, "_reaper", None)
    monkeypatch.setattr(hermes_state, "_ReaperThread", Unstartable)
    scope = profile_scope(home=tmp_path)
    anchor = open_persistence(scope, eager=True)
    try:
        with pytest.raises(RuntimeError, match="can't start new thread"):
            hermes_state.SessionDB(tmp_path / "state.db")
        assert _handles(scope) == 1
    finally:
        anchor.close()
    assert _handles(scope) == 0


def test_a_stand_in_opened_again_after_close_and_dropped_releases_the_new_handle(tmp_path):
    scope = profile_scope(home=tmp_path)
    anchor = open_persistence(scope, eager=True)
    try:
        db = hermes_state.SessionDB(tmp_path / "state.db")
        db.close()
        db.create_session("s1", "cli")  # used after close(): opens the store again
        assert _handles(scope) == 2
        del db
        wait_until(lambda: _handles(scope) == 1, timeout=DEADLINE_S, what="the reopened handle released")
    finally:
        anchor.close()
    assert _handles(scope) == 0
