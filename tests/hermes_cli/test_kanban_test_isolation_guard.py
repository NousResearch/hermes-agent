"""Regression tests for the 2026-09-17 kanban fixture-leak wave.

Two defects collaborated to write dispatcher-test fixture rows (``owner`` /
``child`` / ``a0``–``a4`` / ``b0``–``b2``) into the LIVE ``~/.hermes/kanban.db``:

1. ``hermes_constants.get_default_hermes_root()`` collapsed a sandboxed
   ``HERMES_HOME`` parked under the native home (pytest ``--basetemp
   ~/.hermes/tmp/...``) back to the PRODUCTION root, so every kanban path
   resolved to the live board.
2. The conftest ``_kanban_write_guard`` only patches modules already in
   ``sys.modules`` at fixture time — the FIRST test in a file that lazily
   imports ``hermes_cli.kanban_db`` inside its body wrote before any patch
   existed.

The choke-point guard ``kanban_db_connect._ensure_test_isolation`` now arms on
env/ancestry test signals and refuses production board paths regardless of
patch timing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# get_default_hermes_root: sandbox-under-root no longer collapses
# ---------------------------------------------------------------------------


def test_default_root_keeps_sandboxed_home_under_native_home(monkeypatch, tmp_path):
    from hermes_constants import get_default_hermes_root

    # tmp_path here may itself live under the real ~/.hermes (conductor
    # delegate TMPDIR): the sandbox is NOT a profile home, so it must be
    # treated as the root itself instead of collapsing to the native home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "sandbox"))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    root = get_default_hermes_root()
    assert root == Path(tmp_path / "sandbox")


def test_default_root_profile_home_still_maps_to_parent(monkeypatch, tmp_path):
    from hermes_constants import get_default_hermes_root

    profiles = tmp_path / "hermes" / "profiles" / "worker"
    profiles.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles))
    root = get_default_hermes_root()
    assert root == tmp_path / "hermes"


def test_default_root_dot_profile_home_keeps_existing_mapping(monkeypatch, tmp_path):
    """Dotted profile dirs (<root>/profiles/.name) keep the pre-existing
    grandparent mapping (ValueError branch) — this test pins it so the
    sandbox-under-root change cannot silently alter it."""
    from hermes_constants import get_default_hermes_root

    profiles = tmp_path / "hermes" / "profiles" / ".hidden"
    profiles.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles))
    root = get_default_hermes_root()
    assert root == tmp_path / "hermes"


# ---------------------------------------------------------------------------
# kanban choke-point guard: production board refused from a test context,
# regardless of conftest patch timing
# ---------------------------------------------------------------------------


@pytest.fixture()
def _pytest_context(monkeypatch):
    """Arm the env-based test signal the way the hermetic conftest does."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_self.py::test_x")
    yield


def test_guard_refuses_production_kanban_db(_pytest_context):
    from hermes_state_guard import _real_platform_state_root
    from hermes_cli.kanban_db_connect import _ensure_test_isolation

    root = _real_platform_state_root()
    assert root is not None
    with pytest.raises(RuntimeError, match="test-isolation guard"):
        _ensure_test_isolation(root / "kanban.db")


def test_guard_refuses_production_board_dirs_and_profiles(_pytest_context):
    from hermes_state_guard import _real_platform_state_root
    from hermes_cli.kanban_db_connect import _ensure_test_isolation

    root = _real_platform_state_root()
    assert root is not None
    with pytest.raises(RuntimeError, match="test-isolation guard"):
        _ensure_test_isolation(root / "kanban" / "boards" / "second" / "kanban.db")
    with pytest.raises(RuntimeError, match="test-isolation guard"):
        _ensure_test_isolation(root / "profiles" / "worker" / "kanban.db")


def test_guard_allows_sandboxed_db_under_real_root(_pytest_context):
    """pytest tmpdirs parked under ~/.hermes are legitimate sandboxes — only
    the production BOARD paths are denied."""
    from hermes_state_guard import _real_platform_state_root
    from hermes_cli.kanban_db_connect import _ensure_test_isolation

    root = _real_platform_state_root()
    assert root is not None
    sandbox_db = root / "tmp" / "pytest-something" / ".hermes" / "kanban.db"
    _ensure_test_isolation(sandbox_db)  # must not raise


def test_guard_bypass_env(_pytest_context, monkeypatch):
    from hermes_state_guard import _real_platform_state_root
    from hermes_cli.kanban_db_connect import _ensure_test_isolation

    root = _real_platform_state_root()
    assert root is not None
    monkeypatch.setenv("HERMES_KANBAN_GUARD_BYPASS", "1")
    _ensure_test_isolation(root / "kanban.db")  # must not raise


def test_guard_inert_outside_test_context(monkeypatch):
    from hermes_state_guard import _real_platform_state_root
    from hermes_cli import kanban_db_connect as kbc

    root = _real_platform_state_root()
    assert root is not None
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    monkeypatch.delenv("HERMES_TEST_ISOLATION", raising=False)
    # Not memoised in this process yet: without psutil ancestors there is
    # nothing pytest-shaped above us inside the test runner either, so the
    # guard must stay silent and simply return.
    kbc._ensure_test_isolation(root / "kanban.db")  # must not raise


# ---------------------------------------------------------------------------
# Write-boundary choke (2026-09-18 wave 8): write_txn guards externally-opened
# conns too — a raw sqlite3.connect to the live board must not become a write
# path for dispatcher helpers (claim_task / create_task / _set_worker_pid).
# ---------------------------------------------------------------------------


def test_write_txn_refuses_raw_conn_to_live_board(_pytest_context):
    """connect() refuses the live board, but a fixture that opens the live DB
    with raw sqlite3.connect and drives kb.create_task goes through
    write_txn — that boundary must refuse as well."""
    import sqlite3

    from hermes_state_guard import _real_platform_state_root
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    root = _real_platform_state_root()
    assert root is not None
    raw = sqlite3.connect(root / "kanban.db")
    try:
        with pytest.raises(RuntimeError, match="test-isolation guard"):
            with kbc.write_txn(raw):
                kb.create_task(raw, title="wave8-probe", assignee="alpha")
    finally:
        raw.close()


def test_real_platform_state_root_ignores_redirected_home(monkeypatch, tmp_path):
    """Invariant (wave-8 forensics item 1): the deny-root follows the REAL
    home, not the HOME env a test redirected — so the guard keeps matching
    the production board even while HOME points at a sandbox."""
    from hermes_state_guard import _real_platform_state_root

    real = _real_platform_state_root()
    assert real is not None
    monkeypatch.setenv("HOME", str(tmp_path / "sandbox-home"))
    assert _real_platform_state_root() == real, (
        "deny-root must not follow a redirected HOME env"
    )
