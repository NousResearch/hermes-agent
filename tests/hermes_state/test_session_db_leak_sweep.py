"""Suite-wide SessionDB leak-closing contract (OOM incident 20260816).

A raw single-process ``pytest tests/hermes_cli/`` used to accumulate every
SessionDB a test constructed and forgot to close — writer connection,
pooled read connections, and (once token accounting ran) an ``atexit``
registration pinning the instance alive — ballooning to 16-25 GB RSS.

The fix is two-sided:

* ``hermes_state_guard._register_test_instance`` adds every successfully
  constructed SessionDB to a WeakSet registry when the
  ``HERMES_TEST_ISOLATION`` marker is set (test-isolation runs only).
* the autouse ``_close_leaked_session_dbs`` fixture in ``tests/conftest.py``
  closes everything in the registry at each test's teardown.

These tests pin the *behavior contract*: instances register under pytest,
close() empties them idempotently, and a leaked instance from an earlier
test is actually closed by the suite-level sweep.
"""

from __future__ import annotations

import threading
from typing import Any

import hermes_state_guard
from hermes_state import SessionDB

# Deliberate cross-test handoff: test_leaked_instance_* leaks an instance;
# the later test (pytest runs file order deterministically without a
# randomizer plugin, and the sanctioned runner executes whole files in one
# process) asserts the autouse teardown sweep closed it.
_leaked: list[SessionDB] = []


def test_constructed_sessiondb_is_registered(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db in hermes_state_guard._test_instance_registry
    finally:
        db.close()
    # close() must fully release the writer connection…
    assert db._conn is None
    # …and be idempotent: a second close (the suite sweep will call it
    # again at teardown) must not raise.
    db.close()


def test_leaked_instance_stays_open_within_the_test(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="leak-probe", source="cli", model="m")
    # Intentionally NOT closed — the suite-level sweep owns cleanup.
    assert db._conn is not None
    _leaked.append(db)


def test_previously_leaked_instance_was_closed_by_the_sweep():
    assert _leaked, "expected the previous test to have leaked an instance"
    db = _leaked.pop()
    # The autouse _close_leaked_session_dbs teardown between the two tests
    # must have closed the leaked instance (writer conn released), which is
    # what bounds fd/RSS growth in single-process runs.
    assert db._conn is None


# Cross-test handoff for the in-use-on-another-thread contract: closing a
# connection from the sweep while its owner thread is inside sqlite3_step()
# is a use-after-free that segfaulted CI (dashboard statedb-eager-reconcile).
_threaded: dict[str, Any] = {}


def test_instance_owned_by_a_running_thread_survives_the_sweep(tmp_path):
    opened = threading.Event()
    release = threading.Event()

    def _owner():
        db = SessionDB(db_path=tmp_path / "state.db", read_only=False)
        _threaded["db"] = db
        opened.set()
        release.wait(30)
        db.close()

    thread = threading.Thread(target=_owner, name="leak-sweep-owner", daemon=True)
    thread.start()
    assert opened.wait(30)
    _threaded["thread"], _threaded["release"] = thread, release
    assert hermes_state_guard._owner_thread_is_running(_threaded["db"])


def test_sweep_left_the_running_threads_instance_open_then_owner_closed_it():
    db, thread, release = _threaded["db"], _threaded["thread"], _threaded["release"]
    # The teardown sweep between the two tests ran while the owner thread was
    # still alive: it must NOT have closed the connection out from under it.
    assert thread.is_alive()
    assert db._conn is not None
    release.set()
    thread.join(30)
    assert not thread.is_alive()
    assert db._conn is None
    # Owner gone → the next sweep may close it: the skip no longer applies.
    assert not hermes_state_guard._owner_thread_is_running(db)
