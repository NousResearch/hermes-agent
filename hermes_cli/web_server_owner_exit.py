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
Design from the #132034 report by @VexterZ; lock-ownership watchdog idea first drafted in #89452.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_log = logging.getLogger(__name__)

DEFAULT_OWNER_POLL_S = 15.0
# The Desktop writes the lock right after the spawn returns; a young process may still see an
# older lock for a moment, so supersession only counts once the backend has settled.
MIN_AGE_S = 90.0
_CONFIRMATIONS = 2


def should_retire_superseded(*, lock: Optional[dict], my_nonce: str, age_s: float) -> bool:
    """Retire only when a valid lock provably names another spawn of this ownership slot."""
    return lock is not None and lock["spawnNonce"] != my_nonce and age_s >= MIN_AGE_S


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
    if fence is None:
        from hermes_cli.backend_retirement import retirement

        fence = retirement
    reader: Callable[[Path], Optional[dict]] = read_lock
    started = now()

    def _loop() -> None:
        seen = polls = 0
        while not getattr(server, "should_exit", False) and (max_polls is None or polls < max_polls):
            polls += 1
            lock = reader(lock_path)
            if should_retire_superseded(lock=lock, my_nonce=nonce, age_s=now() - started):
                seen += 1
            else:
                seen = 0
            if seen >= _CONFIRMATIONS and lock is not None:
                permit = fence.prepare()
                if permit.get("ok") and fence.commit(permit.get("token")).get("ok"):
                    _log.warning("SSH-isolated backend %s was superseded by spawn %s; retiring.",
                                 nonce, lock["spawnNonce"])
                    server.should_exit = True
                    return
            time.sleep(poll_s)

    thread = threading.Thread(target=_loop, daemon=True, name="ssh-isolated-owner-watchdog")
    thread.start()
    return thread
