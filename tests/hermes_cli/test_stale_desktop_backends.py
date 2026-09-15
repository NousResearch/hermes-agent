"""Stale Desktop backends: detection (locks) + the update-side recycle.

Field failure (2026-09-14T02:52:31Z gateway ImportError): ``hermes update`` pulled
``gateway.session.profile_from_session_key_namespace`` into the checkout while a Desktop
``hermes serve --isolated`` backend — REUSED across reconnects and started hours earlier —
kept the previous checkout's modules in memory. Every turn then died with
``cannot import name 'profile_from_session_key_namespace'`` until the process was restarted.

These tests pin the two halves of the fix that live in the CLI: deciding which backends predate
the update (``stale_desktop_backend_pids``) and recycling them in the restart phase.
"""

from __future__ import annotations

import json
import signal
import time

from hermes_cli.dashboard_procs import _lock_started_at_epoch, stale_desktop_backend_pids
from hermes_cli.update_cmd_fleet import _recycle_stale_desktop_backends

_OWNERSHIP_A = "a" * 32
_OWNERSHIP_B = "b" * 32


def _lock_body(ownership_id: str, pid: int, started_at: str) -> dict:
    return {
        "schemaVersion": 2,
        "protocolVersion": 1,
        "ownershipId": ownership_id,
        "spawnNonce": "0123456789abcdef",
        "pid": pid,
        "port": 41,
        "profile": "",
        "hermesPath": "/home/agent/.local/bin/hermes",
        "hermesHome": "/home/agent/.hermes",
        "logPath": f"/home/agent/.hermes/desktop-ssh/{ownership_id}/0123456789abcdef.log",
        "tokenFingerprint": "c" * 32,
        "startedAt": started_at,
    }


def _write_lock(base_dir, ownership_id: str, pid: int, started_at: str, *, body: dict | None = None) -> None:
    directory = base_dir / ownership_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "backend.lock.json").write_text(json.dumps(body or _lock_body(ownership_id, pid, started_at)))


def _ago(seconds: float) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def test_stale_pids_only_lists_live_backends_older_than_the_cutoff(tmp_path):
    _write_lock(tmp_path, _OWNERSHIP_A, 111, _ago(7200))   # 2h old: predates the update
    _write_lock(tmp_path, _OWNERSHIP_B, 222, _ago(60))     # 1m old: spawned after it
    alive = {111, 222}

    stale = stale_desktop_backend_pids(time.time() - 1800, base_dir=tmp_path, is_alive=lambda pid: pid in alive)

    assert stale == [111]


def test_dead_backends_are_not_reported(tmp_path):
    _write_lock(tmp_path, _OWNERSHIP_A, 111, _ago(7200))

    assert stale_desktop_backend_pids(time.time() - 1800, base_dir=tmp_path, is_alive=lambda pid: False) == []


def test_unparseable_started_at_is_never_a_stale_verdict(tmp_path):
    """Recycling a backend we cannot date risks killing a healthy one: fail safe, not aggressive."""
    _write_lock(tmp_path, _OWNERSHIP_A, 111, "not-a-timestamp")

    assert _lock_started_at_epoch({"startedAt": "not-a-timestamp"}) is None
    assert stale_desktop_backend_pids(1e18, base_dir=tmp_path, is_alive=lambda pid: True) == []


def test_invalid_or_foreign_locks_contribute_nothing(tmp_path):
    """A corrupt record must not hand the update a PID to kill (mirrors readLockfile's validation)."""
    _write_lock(tmp_path, _OWNERSHIP_A, 111, _ago(7200), body={"schemaVersion": 1, "pid": 111})
    (tmp_path / "not-a-lock-dir").mkdir()
    (tmp_path / "not-a-lock-dir" / "backend.lock.json").write_text("{ truncated", encoding="utf-8")
    (tmp_path / "short").mkdir()
    (tmp_path / "short" / "backend.lock.json").write_text(json.dumps(_lock_body("short", 333, _ago(7200))))

    assert stale_desktop_backend_pids(1e18, base_dir=tmp_path, is_alive=lambda pid: True) == []


def test_recycle_sigterms_stale_backends_only():
    calls: list = []
    recycled = _recycle_stale_desktop_backends(
        1234.0,
        kill=lambda pid, sig: calls.append((pid, sig)),
        stale_pids_fn=lambda cutoff: [7, 9],
    )

    assert recycled == [7, 9]
    assert calls == [(7, signal.SIGTERM), (9, signal.SIGTERM)]


def test_recycle_survives_a_failing_probe_and_dead_pids():
    def _boom(cutoff):
        raise OSError("no process table")

    assert _recycle_stale_desktop_backends(1.0, kill=lambda *a: None, stale_pids_fn=_boom) == []

    def _gone(pid, sig):
        raise ProcessLookupError(pid)

    assert _recycle_stale_desktop_backends(1.0, kill=_gone, stale_pids_fn=lambda cutoff: [3]) == []
