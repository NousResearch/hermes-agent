"""Signal selection for verified, host-local Kanban worker reclamation."""

from __future__ import annotations

import os
from typing import Callable


SignalSender = Callable[[int, int], None]


def kill_fn(signal_fn: SignalSender | None) -> SignalSender | None:
    """Injected callbacks retain their PID-only contract on every platform."""
    return signal_fn if signal_fn is not None else getattr(os, "kill", None)


def reclaimed_signal_target(pid: int, kill: SignalSender, *, allow_group: bool) -> SignalSender:
    """Group-signal only a verified worker that is its own process-group leader.

    The dispatcher checks ownership and the spawn fingerprint before calling this.
    Non-POSIX hosts and injected callbacks keep the existing single-PID behavior.
    """
    getpgid = getattr(os, "getpgid", None) if allow_group else None
    killpg = getattr(os, "killpg", None)
    if getpgid is None or killpg is None:
        return kill
    try:
        if getpgid(pid) != pid:
            return kill
    except OSError:
        return kill

    def signal_target(target_pid: int, sig: int) -> None:
        try:
            killpg(target_pid, sig)
        except ProcessLookupError:
            raise
        except OSError:
            kill(target_pid, sig)

    return signal_target
