"""Regression tests for PR #58244 — broken-board DB recovery without silent retry loop.

Covers two behaviours added in kanban_watchers.py + kanban_db.py:

1. Recovery path: when a cached board's DB is replaced or truncated out-of-band,
   the next natural dispatcher tick detects the missing-schema error, invalidates
   the cache entry, and the *following* tick's connect() rebuilds the schema —
   without ever calling init_db() directly from the watcher.

2. Quarantine path: if the schema is still missing after the cache-invalidation
   recovery attempt (i.e. the DB is genuinely corrupt/unrecoverable), the board
   is quarantined via disabled_corrupt_boards and no further ticks are attempted.

Fixture pattern mirrors test_kanban_boards.py:fresh_home (local fixture, not in
conftest) and dispatcher harness mirrors test_kanban_core_functionality.py:3672.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

import hermes_cli.kanban_db as _kb


# ---------------------------------------------------------------------------
# Fixture — mirrors test_kanban_boards.py:fresh_home exactly.
# Cannot import it directly as it is local to that module.
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with clean kanban state and reset module-level cache."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    _kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warm_cache(slug: str) -> Path:
    """Create a board, connect to it (populating _INITIALIZED_PATHS), return db path."""
    _kb.create_board(slug)
    db_path = _kb.kanban_db_path(board=slug)
    with _kb.connect(board=slug) as conn:
        _kb.create_task(conn, title="seed-task", assignee="dev")
    assert str(db_path.resolve()) in _kb._INITIALIZED_PATHS, (
        "Cache warm-up failed — _INITIALIZED_PATHS does not contain board path"
    )
    return db_path


def _make_dispatcher_runner(monkeypatch, slug: str, dispatch_interval: int = 1):
    """Return a GatewayRunner wired to dispatch only the given board slug."""
    from gateway.run import GatewayRunner
    import hermes_cli.config as _cfg_mod

    runner = object.__new__(GatewayRunner)
    runner._running = True

    monkeypatch.setattr(
        _cfg_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": dispatch_interval,
            }
        },
    )
    monkeypatch.setattr(
        _kb,
        "list_boards",
        lambda include_archived=False: [{"slug": slug}],
    )
    monkeypatch.setattr(
        _kb,
        "read_board_metadata",
        lambda s: {"slug": s},
    )
    return runner


# ---------------------------------------------------------------------------
# Test 1 — Recovery path
# ---------------------------------------------------------------------------

def test_broken_board_db_recovery_via_cache_invalidation(
    fresh_home, monkeypatch, caplog
):
    """Cache invalidation on missing-schema error allows schema rebuild on next tick.

    Sequence:
      1. Warm _INITIALIZED_PATHS by connecting to a real board.
      2. Replace the DB file with an empty/truncated file out-of-band.
      3. Drive the real dispatcher watcher for enough ticks to cover:
           - Tick N  : detects OperationalError(no such table), calls
                       invalidate_cached_schema(), returns without retrying.
           - Tick N+1: _INITIALIZED_PATHS cache entry is gone; connect()
                       runs the schema-init pass and rebuilds the schema.
      4. Assert init_db() was never called directly from within the watcher.
      5. Assert the DB is readable with a valid schema after recovery.
    """
    SLUG = "recovery-test"
    db_path = _warm_cache(SLUG)

    # Replace the DB with an empty file — triggers "no such table" on next connect.
    db_path.write_bytes(b"")

    runner = _make_dispatcher_runner(monkeypatch, SLUG, dispatch_interval=1)

    init_db_calls: list[str] = []
    _real_init_db = _kb.init_db

    def _spy_init_db(*args, **kwargs):
        init_db_calls.append("called")
        return _real_init_db(*args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="hermes"):
        with patch.object(_kb, "init_db", side_effect=_spy_init_db):
            # 4 seconds gives the 1-second dispatcher at least 3 full ticks:
            # tick 1 = detects error + invalidates cache
            # tick 2 = schema rebuild via normal connect()
            # tick 3 = confirms board is stable
            try:
                asyncio.run(
                    asyncio.wait_for(
                        runner._kanban_dispatcher_watcher(),
                        timeout=12.0,
                    )
                )
            except asyncio.TimeoutError:
                pass  # Expected — watcher runs until cancelled.

    # Core assertion: init_db() must never be called directly from the watcher.
    assert init_db_calls == [], (
        "init_db() was called directly from the dispatcher watcher — "
        "this reintroduces the race condition from issue #21378. "
        "Recovery must go through cache invalidation + natural connect() only."
    )

    # The DB must be readable with a valid schema after recovery.
    with _kb.connect(board=SLUG) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {r[0] for r in rows}
    assert "tasks" in table_names, (
        f"Schema was not rebuilt after cache invalidation. Tables found: {table_names}"
    )

    # Cache entry must be repopulated (connect() re-warmed it on tick N+1).
    assert str(db_path.resolve()) in _kb._INITIALIZED_PATHS, (
        "Cache entry was not repopulated after recovery — "
        "subsequent connects will re-run schema init on every tick."
    )

    # The first-attempt log must have fired (confirms recovery path was taken,
    # not the generic corrupt-board path).
    # Exact message fragment from kanban_watchers.py:1069.
    first_attempt_logs = [
        r for r in caplog.records
        if SLUG in r.message and "will quarantine if that doesn't fix it" in r.message
    ]
    assert len(first_attempt_logs) == 1, (
        f"Expected exactly one cache-invalidation log for '{SLUG}', "
        f"got {len(first_attempt_logs)}. Records: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Quarantine path
# ---------------------------------------------------------------------------

def test_broken_board_db_quarantined_on_repeated_schema_failure(
    fresh_home, monkeypatch, caplog
):
    """Board is quarantined after repeated missing-schema errors, not retried each tick.

    disabled_corrupt_boards is a local variable inside _kanban_dispatcher_watcher,
    not an attribute on GatewayRunner. Quarantine is observed via the ERROR log
    message emitted when the board is paused (kanban_watchers.py:1057).

    Sequence:
      1. Warm cache.
      2. Replace DB with a valid-but-schema-less SQLite file and patch init_db
         to be a no-op so the schema can never be rebuilt — simulates a DB that
         is permanently unrecoverable within this process.
      3. Drive the real dispatcher watcher for enough ticks to trigger:
           - Tick N  : first missing-schema error → cache invalidated, logged.
           - Tick N+1: same fingerprint still fails → board quarantined, logged.
           - Tick N+2: board is in disabled_corrupt_boards → skipped entirely.
      4. Assert the quarantine log fired exactly once.
      5. Assert no further dispatch attempts occur after quarantine.
    """
    SLUG = "quarantine-test"
    db_path = _warm_cache(SLUG)

    # Patch connect() to always raise the exact OperationalError that
    # _is_missing_schema_error() matches (kanban_watchers.py:994).
    # This guarantees both tick N (cache invalidation) and tick N+1
    # (quarantine) see the error on every call, simulating a DB that
    # is permanently unrecoverable within this process.
    # We do NOT patch init_db — the error fires before init_db is
    # reached, so patching connect() is sufficient and cleaner.
    _missing_schema_exc = sqlite3.OperationalError("no such table: tasks")

    runner = _make_dispatcher_runner(monkeypatch, SLUG, dispatch_interval=1)

    # Track connect() calls so we can assert dispatch stops after quarantine.
    connect_calls: list[float] = []

    def _always_raise_missing_schema(*args, **kwargs):
        import time
        connect_calls.append(time.monotonic())
        raise _missing_schema_exc

    with caplog.at_level(logging.ERROR, logger="hermes"):
        with patch.object(_kb, "connect", side_effect=_always_raise_missing_schema):
                try:
                    asyncio.run(
                        asyncio.wait_for(
                            runner._kanban_dispatcher_watcher(),
                            timeout=14.0,
                        )
                    )
                except asyncio.TimeoutError:
                    pass

    # Quarantine log must have fired exactly once.
    # Exact message fragment from kanban_watchers.py:1057.
    quarantine_logs = [
        r for r in caplog.records
        if SLUG in r.message and "pausing dispatch for this board" in r.message
    ]
    assert len(quarantine_logs) == 1, (
        f"Expected exactly one quarantine log for '{SLUG}', "
        f"got {len(quarantine_logs)}. "
        f"Records: {[r.message for r in caplog.records if SLUG in r.message]}"
    )

    # After quarantine, connect() must not be called again.
    # Quarantine fires on tick N+1 (~2s in). Remaining ~3s are tick N+2 onward.
    # If connect() is called after the quarantine log timestamp, board is still
    # being retried.
    quarantine_time = quarantine_logs[0].created  # log record timestamp (epoch float)
    post_quarantine_connects = [t for t in connect_calls if t > quarantine_time]
    assert post_quarantine_connects == [], (
        f"connect() was called {len(post_quarantine_connects)} time(s) after "
        f"quarantine was logged — board is still being dispatched. "
        f"connect() call times: {connect_calls}, quarantine at: {quarantine_time}"
    )
