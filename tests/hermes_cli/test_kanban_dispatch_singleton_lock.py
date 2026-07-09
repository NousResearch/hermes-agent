"""Tests for BUILD-263: `hermes kanban dispatch` single-dispatcher lock.

2026-07-08 incident: two orphaned shell loops ran `hermes -p orchestrator
kanban dispatch` every 60s for 6 and 19 days alongside the gateway's
internal scheduler — concurrent dispatchers over the same SQLite kanban.db
with no mutual exclusion. The CLI `dispatch` entry now takes the exact same
machine-wide singleton lock the gateway's embedded dispatcher holds
(`gateway.kanban_watchers._acquire_singleton_lock` /
`dispatcher_singleton_lock_path`), refusing loudly (nonzero exit) when
another dispatcher already holds it, unless `--force`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gateway.kanban_watchers import (
    _acquire_singleton_lock,
    _release_singleton_lock,
    dispatcher_singleton_lock_path,
)
from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


def _args(**overrides):
    base = dict(dry_run=True, max=None, failure_limit=2, json=False, force=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def _stub_dispatch_once(monkeypatch, calls):
    def fake(conn, **kwargs):
        calls.append(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake)


def test_dispatch_proceeds_when_lock_is_free(kanban_home, monkeypatch):
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    rc = kb_cli._cmd_dispatch(_args())

    assert rc == 0
    assert len(calls) == 1


def test_dispatch_releases_lock_after_running(kanban_home, monkeypatch):
    """After a normal (uncontended) run, the lock must be released — a
    subsequent acquire attempt (as the gateway would do) must succeed."""
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    rc = kb_cli._cmd_dispatch(_args())
    assert rc == 0

    lock_path = dispatcher_singleton_lock_path()
    handle, state = _acquire_singleton_lock(lock_path)
    assert state == "held", "CLI must release the lock when its run finishes"
    _release_singleton_lock(handle)


def test_dispatch_refuses_when_another_dispatcher_holds_the_lock(
    kanban_home, monkeypatch, capsys,
):
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    lock_path = dispatcher_singleton_lock_path()
    holder_handle, holder_state = _acquire_singleton_lock(lock_path)
    assert holder_state == "held"
    try:
        rc = kb_cli._cmd_dispatch(_args())
    finally:
        _release_singleton_lock(holder_handle)

    assert rc != 0
    assert calls == [], "dispatch_once must NOT run while another dispatcher holds the lock"
    err = capsys.readouterr().err
    assert "refusing" in err.lower()
    assert "--force" in err


def test_dispatch_force_bypasses_a_held_lock(kanban_home, monkeypatch):
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    lock_path = dispatcher_singleton_lock_path()
    holder_handle, holder_state = _acquire_singleton_lock(lock_path)
    assert holder_state == "held"
    try:
        rc = kb_cli._cmd_dispatch(_args(force=True))
    finally:
        _release_singleton_lock(holder_handle)

    assert rc == 0
    assert len(calls) == 1, "--force must bypass the lock and still dispatch"


def test_dispatch_reports_holder_pid_when_available(kanban_home, monkeypatch, capsys):
    """The lock file is stamped with the holder's pid (diagnostics only) —
    the refusal message should surface it when present."""
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    lock_path = dispatcher_singleton_lock_path()
    holder_handle, holder_state = _acquire_singleton_lock(lock_path)
    assert holder_state == "held"
    try:
        kb_cli._cmd_dispatch(_args())
    finally:
        _release_singleton_lock(holder_handle)

    err = capsys.readouterr().err
    assert "pid" in err.lower()


def test_stale_lock_is_reclaimed_automatically_after_holder_process_dies(
    kanban_home, monkeypatch,
):
    """Simulate a crashed holder: close its file handle WITHOUT an explicit
    flock release (mirrors what the OS does automatically when a process
    dies — every fd, and any flock tied to it, is released on process
    exit). No special "stale lock" cleanup code should be required for the
    CLI's next attempt to succeed."""
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    lock_path = dispatcher_singleton_lock_path()
    dead_handle, dead_state = _acquire_singleton_lock(lock_path)
    assert dead_state == "held"
    dead_handle.close()  # simulate process death — no _release_singleton_lock call

    rc = kb_cli._cmd_dispatch(_args())

    assert rc == 0
    assert len(calls) == 1, "a stale (crashed-holder) lock must be reclaimed automatically"


def test_dispatch_fails_open_when_lock_is_unavailable(kanban_home, monkeypatch):
    """When the locking mechanism itself can't be used (import failure,
    non-POSIX filesystem, etc.) dispatch must still proceed — mutual
    exclusion is best-effort and must never block a legitimate dispatch."""
    calls = []
    _stub_dispatch_once(monkeypatch, calls)

    import gateway.kanban_watchers as watchers_mod

    monkeypatch.setattr(
        watchers_mod, "_acquire_singleton_lock", lambda _path: (None, "unavailable"),
    )

    rc = kb_cli._cmd_dispatch(_args())

    assert rc == 0
    assert len(calls) == 1


def test_run_slash_dispatch_accepts_force_flag(kanban_home):
    """End-to-end argparse coverage: --force must parse on the real CLI
    surface (not just via a hand-built argparse.Namespace)."""
    out = kb_cli.run_slash("dispatch --dry-run --force")
    assert "Spawned:" in out
