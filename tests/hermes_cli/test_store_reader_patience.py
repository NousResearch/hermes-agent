"""Each CLI and tool reader of a profile's state.db waits for a store another process holds as long as main's reader
did: install-state evidence and the install age 1 s, the retained-worktree checks 2 s, spillover references, approval
history and the owner receipt behind a kanban reclaim 5 s. The store held exclusively stands for a first boot or a
repair holding it."""

from __future__ import annotations

import importlib.util
import time
from functools import partial

import pytest

from hermes_cli import approvals_suggest, install_state, kanban_owner_recovery, worktree_ops
from hermes_cli.observability import shared_metrics_snapshot
from persistence import StoreContention
from persistence_assembly import open_persistence, open_store_admin, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite._legacy_store import oracle_tree
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


def _about(waited: float, main_wait: float) -> bool:
    """About main's wait: not much shorter (a brief hold is still waited out) and never much longer."""
    return main_wait * 0.8 <= waited < main_wait + 0.75


def test_install_state_evidence_waits_one_second(held):
    db, hold = held
    with hold():
        found, waited = _timed(partial(install_state._messages, install_state._Root(db.parent)))
    assert found is False and _about(waited, 1.0)


def test_install_age_waits_one_second(held):
    db, hold = held
    with hold():
        bucket, waited = _timed(partial(shared_metrics_snapshot.install_age_bucket, db.parent))
    assert bucket == "unknown" and _about(waited, 1.0)


def test_retained_worktree_checks_wait_two_seconds(held, tmp_path):
    db, hold = held
    tree = str(tmp_path / "tree")
    with hold():
        live, live_waited = _timed(partial(worktree_ops._retained_tree_is_live, str(db), tree))
        served, served_waited = _timed(partial(worktree_ops._tree_still_served, str(db), tree))
    assert (live, served) == (True, "unreadable")
    assert _about(live_waited, 2.0) and _about(served_waited, 2.0)


def test_retained_worktree_checks_wait_as_main_does(held, tmp_path):
    spec = importlib.util.spec_from_file_location("main_worktree_ops", oracle_tree() / "hermes_cli" / "worktree_ops.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    db, hold = held
    tree = str(tmp_path / "tree")
    answers = {}
    with hold():
        for name, ops in (("main", main), ("head", worktree_ops)):
            answers[name] = (_timed(partial(ops._retained_tree_is_live, str(db), tree)),
                             _timed(partial(ops._tree_still_served, str(db), tree)))
    for name, ((live, live_waited), (served, served_waited)) in answers.items():
        assert (live, served) == (True, "unreadable"), name
        assert _about(live_waited, 2.0) and _about(served_waited, 2.0), (name, live_waited, served_waited)


def test_spillover_approvals_and_owner_receipt_wait_five_seconds(held):
    db, hold = held
    spill_dir = db.parent / "cache" / "tool_results"
    receipt = {"db": str(db), "session_id": "s", "request_id": "r"}
    with hold():
        names, spill_waited = _timed(partial(tool_result_storage.referenced_spillover_names, spill_dir))
        row, owner_waited = _timed(partial(kanban_owner_recovery._admission_row, receipt))
        start = time.monotonic()
        with pytest.raises(StoreContention):
            approvals_suggest.scan_approval_history(db)
        approvals_waited = time.monotonic() - start
    assert (names, row) == (None, None)
    assert all(_about(waited, 5.0) for waited in (spill_waited, owner_waited, approvals_waited)), (
        spill_waited, owner_waited, approvals_waited)
