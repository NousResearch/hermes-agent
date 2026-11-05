"""Each CLI and tool reader of a profile's state.db waits for a store another process holds as long as main's reader
does: a read-only connection given main's timeout (install-state evidence and the install age 1 s, the
retained-worktree checks 2 s, spillover references, approval history and the owner receipt behind a kanban reclaim
5 s) running one statement, measured right after each reader under the same hold. The store held exclusively stands
for a first boot or a repair holding it."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from functools import partial

import pytest

from hermes_cli import approvals_suggest, install_state, kanban_owner_recovery, worktree_ops
from hermes_cli.observability import shared_metrics_snapshot
from persistence import StoreContention
from persistence_assembly import open_persistence, open_store_admin, profile_scope
from tests.fixtures.storage_seed import seed_session
from tools import tool_result_storage


@pytest.fixture
def held(tmp_path):
    """A seeded state.db with a session inside ``tmp_path / "tree"``, and a context holding it exclusively."""
    home = tmp_path / "home"
    home.mkdir()
    with open_persistence(profile_scope(home=home), eager=True) as store:
        seed_session(store, "s", "cli", cwd=str(tmp_path / "tree"))
    db = home / "state.db"
    return db, lambda: open_store_admin(db, read_only=False).exclusive_repair_guard()


def _timed(call):
    start = time.monotonic()
    value = call()
    return value, time.monotonic() - start


def _main_wait(db, timeout: float) -> float:
    """How long main's reader waits on this host: a read-only connection given ``timeout`` running one statement."""
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=timeout)) as conn:
        start = time.monotonic()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            conn.execute("SELECT 1 FROM messages LIMIT 1")
        return time.monotonic() - start


def _about(waited: float, main_wait: float) -> bool:
    """About main's wait: not much shorter (a brief hold is still waited out) and never much longer."""
    return main_wait * 0.7 <= waited < main_wait * 1.3


def test_install_state_evidence_waits_as_long_as_mains_one_second_reader(held):
    db, hold = held
    with hold():
        found, waited = _timed(partial(install_state._messages, install_state._Root(db.parent)))
        main_wait = _main_wait(db, 1.0)
    assert found is False and _about(waited, main_wait), (waited, main_wait)


def test_install_age_waits_as_long_as_mains_one_second_reader(held):
    db, hold = held
    with hold():
        bucket, waited = _timed(partial(shared_metrics_snapshot.install_age_bucket, db.parent))
        main_wait = _main_wait(db, 1.0)
    assert bucket == "unknown" and _about(waited, main_wait), (waited, main_wait)


def test_retained_worktree_checks_wait_as_long_as_mains_two_second_reader(held, tmp_path):
    db, hold = held
    tree = str(tmp_path / "tree")
    with hold():
        live, live_waited = _timed(partial(worktree_ops._retained_tree_is_live, str(db), tree))
        live_main = _main_wait(db, 2.0)
        served, served_waited = _timed(partial(worktree_ops._tree_still_served, str(db), tree))
        served_main = _main_wait(db, 2.0)
    assert (live, served) == (True, "unreadable")
    assert _about(live_waited, live_main) and _about(served_waited, served_main), (
        live_waited, live_main, served_waited, served_main)


def test_spillover_approvals_and_owner_receipt_wait_as_long_as_mains_five_second_reader(held):
    db, hold = held
    spill_dir = db.parent / "cache" / "tool_results"
    receipt = {"db": str(db), "session_id": "s", "request_id": "r"}
    with hold():
        names, spill_waited = _timed(partial(tool_result_storage.referenced_spillover_names, spill_dir))
        spill_main = _main_wait(db, 5.0)
        row, owner_waited = _timed(partial(kanban_owner_recovery._admission_row, receipt))
        owner_main = _main_wait(db, 5.0)
        start = time.monotonic()
        with pytest.raises(StoreContention):
            approvals_suggest.scan_approval_history(db)
        approvals_waited = time.monotonic() - start
        approvals_main = _main_wait(db, 5.0)
    assert (names, row) == (None, None)
    pairs = ((spill_waited, spill_main), (owner_waited, owner_main), (approvals_waited, approvals_main))
    assert all(_about(waited, main_wait) for waited, main_wait in pairs), pairs
