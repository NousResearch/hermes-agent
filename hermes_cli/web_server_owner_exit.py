"""Supersession retirement for Desktop-owned ``hermes serve --isolated`` backends reached over SSH.

Such a backend is detached on purpose (#91668) and the remote Desktop decides ownership through
``desktop-ssh/<ownershipId>/backend.lock.json``. When a reconnect cannot prove the recorded pid is
its own (dead-looking pid, foreign argv proof) it drops the lock without signalling that pid and
spawns a new nonce (#132034), so the old backend keeps running until the idle watchdog fires — never,
while a half-open tunnel still counts as a client — stacking writers on ``state.db``.

This watchdog polls the lock with two duties:

* once a VALID lock names another ``spawnNonce`` on consecutive polls, retire self through the
  retirement fence (proven idle, admission closed first) — the original behaviour;
* once the lock provably names THIS spawn, any other live backend of the same ownership slot is
  one the Desktop can no longer reach (its per-spawn token was rotated): the leftover a gateway
  restart stranded in #132133, still holding the desktop session. Retire it with the same SIGTERM
  self-retirement answers — after the settle window and two consecutive sightings, so a lock
  mid-rotation or a scan glitch kills nothing.

A missing, unreadable or skewed lock keeps the process up: those are the windows between a cleanup
and the replacement's write, or another Desktop build's file.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.web_server_skew_exit import _CONFIRMATIONS, _log, _run_retirement_watchdog

DEFAULT_OWNER_POLL_S = 15.0


def should_retire_superseded(*, lock: Optional[dict], my_nonce: str, age_s: float) -> bool:
    """Retire only when a valid lock provably names another spawn of this ownership slot. A young
    process may still see the previous spawn's lock, so it must first outlive the same settle window
    the orphan reaper gives the Desktop to write the lock."""
    from hermes_cli.dashboard_procs import _REAP_MIN_AGE_SECONDS

    return lock is not None and lock["spawnNonce"] != my_nonce and age_s >= _REAP_MIN_AGE_SECONDS


def stale_slot_sibling_pids(lock: dict, my_pid: int) -> list[int]:
    """Live backends of this ownership slot that the lock does not name. The lock names exactly one
    spawn — the Desktop's current backend — so any other live backend carrying this slot's
    token-file path on argv is a leftover: unreachable through the rotated token while it still
    holds the desktop session (#132133). Slot identity comes from the token-file path on argv,
    never from a guess on port or process age."""
    if sys.platform == "win32":  # Windows SSH teardown is the runtime's job-object tree, not os.kill
        return []
    from hermes_cli.dashboard_procs import scan_for_ssh_slot_backends

    ownership_id = str(lock.get("ownershipId") or "")
    if not ownership_id:
        return []
    try:
        named = {int(lock.get("pid") or 0), my_pid}
        return [pid for pid in scan_for_ssh_slot_backends(ownership_id) if pid not in named]
    except Exception:
        return []  # never let a scan failure widen the kill


def start_owner_watchdog(server, *, lock_path: Path, nonce: str,
                         read_lock: Optional[Callable[[Path], Optional[dict]]] = None,
                         fence=None, poll_s: float = DEFAULT_OWNER_POLL_S,
                         now: Callable[[], float] = time.monotonic,
                         max_polls: Optional[int] = None) -> threading.Thread:
    """Daemon thread that retires this backend once it is provably superseded and provably idle,
    and retires leftover same-slot siblings once the lock provably names this spawn.
    ``max_polls`` bounds the loop for tests only."""
    if read_lock is None:
        from hermes_cli.dashboard_procs import read_valid_backend_lock

        read_lock = read_valid_backend_lock
    reader: Callable[[Path], Optional[dict]] = read_lock
    started = now()
    sibling_sightings = 0
    signalled: set[int] = set()

    def _observe() -> Optional[str]:
        nonlocal sibling_sightings
        lock = reader(lock_path)
        if lock is None:
            sibling_sightings = 0  # window between cleanup and the replacement's write
            return None
        if lock.get("spawnNonce") != nonce:
            sibling_sightings = 0  # a lock naming another spawn says nothing about ours
            if should_retire_superseded(lock=lock, my_nonce=nonce, age_s=now() - started):
                return f"SSH-isolated backend {nonce} was superseded by spawn {lock['spawnNonce']}; retiring."
            return None
        from hermes_cli.dashboard_procs import _REAP_MIN_AGE_SECONDS

        if now() - started < _REAP_MIN_AGE_SECONDS:
            return None  # young: the Desktop may still be settling this slot's lock
        siblings = stale_slot_sibling_pids(lock, os.getpid())
        sibling_sightings = sibling_sightings + 1 if siblings else 0
        if sibling_sightings >= _CONFIRMATIONS:
            for pid in siblings:
                if pid in signalled:
                    continue
                _log.warning(
                    "Retiring leftover SSH backend pid %d of ownership slot %s; the lock names this spawn (%s).",
                    pid, lock.get("ownershipId"), nonce)
                signalled.add(pid)
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.kill(pid, signal.SIGTERM)
        return None

    return _run_retirement_watchdog(server, observe=_observe, fence=fence, poll_s=poll_s,
                                    max_polls=max_polls, thread_name="ssh-isolated-owner-watchdog")
