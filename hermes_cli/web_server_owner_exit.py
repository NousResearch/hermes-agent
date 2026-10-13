"""Supersession retirement for Desktop-owned ``hermes serve --isolated`` backends reached over SSH.

Such a backend is detached on purpose (#91668) and the remote Desktop decides ownership through
``desktop-ssh/<ownershipId>/backend.lock.json``. When a reconnect cannot prove the recorded pid is
its own (dead-looking pid, foreign argv proof) it drops the lock without signalling that pid and
spawns a new nonce (#132034), so the old backend keeps running until the idle watchdog fires — never,
while a half-open tunnel still counts as a client — stacking writers on ``state.db``.

This watchdog polls the lock and, once a VALID lock names another ``spawnNonce`` on consecutive
polls and the retirement fence proves the backend idle (closing admission so no turn can start
mid-teardown), exits cleanly. A missing, unreadable or skewed lock keeps the process up: those are
the windows between a cleanup and the replacement's write, or another Desktop build's file.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.web_server_skew_exit import _run_retirement_watchdog

DEFAULT_OWNER_POLL_S = 15.0


def should_retire_superseded(*, lock: Optional[dict], my_nonce: str, age_s: float) -> bool:
    """Retire only when a valid lock provably names another spawn of this ownership slot. A young
    process may still see the previous spawn's lock, so it must first outlive the same settle window
    the orphan reaper gives the Desktop to write the lock."""
    from hermes_cli.dashboard_procs import _REAP_MIN_AGE_SECONDS

    return lock is not None and lock["spawnNonce"] != my_nonce and age_s >= _REAP_MIN_AGE_SECONDS


def start_owner_watchdog(server, *, lock_path: Path, nonce: str,
                         read_lock: Optional[Callable[[Path], Optional[dict]]] = None,
                         fence=None, poll_s: float = DEFAULT_OWNER_POLL_S,
                         now: Callable[[], float] = time.monotonic,
                         max_polls: Optional[int] = None) -> threading.Thread:
    """Daemon thread that sets ``server.should_exit`` once this backend is provably superseded and
    provably idle. ``max_polls`` bounds the loop for tests only."""
    if read_lock is None:
        from hermes_cli.dashboard_procs import read_valid_backend_lock

        read_lock = read_valid_backend_lock
    reader: Callable[[Path], Optional[dict]] = read_lock
    started = now()

    def _observe() -> Optional[str]:
        lock = reader(lock_path)
        if lock is None or not should_retire_superseded(lock=lock, my_nonce=nonce, age_s=now() - started):
            return None
        return f"SSH-isolated backend {nonce} was superseded by spawn {lock['spawnNonce']}; retiring."

    return _run_retirement_watchdog(server, observe=_observe, fence=fence, poll_s=poll_s,
                                    max_polls=max_polls, thread_name="ssh-isolated-owner-watchdog")
