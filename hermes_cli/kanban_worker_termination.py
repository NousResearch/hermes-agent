"""Shared worker termination for Kanban reclaim, archive, and explicit stop."""
from __future__ import annotations

import signal
from typing import Any, Optional


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    started_at=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths. ``started_at`` is the spawn-time
    fingerprint: when the live process no longer matches it, the PID was recycled and nothing is
    signalled — the worker is gone, which is what the reclaim wanted (``terminated`` = True). An
    UNVERIFIED spawn (fingerprint capture failed) that is still live is never signalled either, but
    it is reported as surviving (``signal_refused``) so the reclaim holds the claim instead of
    spawning a duplicate beside it."""
    from hermes_cli import kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as dispatch

    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info
    if not str(claim_lock).startswith(_kb._host_prefix()):
        return info
    info["host_local"] = True

    kill = dispatch._kill_fn(signal_fn)
    if kill is None:
        return info
    if started_at == dispatch.UNVERIFIED_WORKER_FINGERPRINT:
        # Never signal by bare number: a dead PID is "gone" (reclaim proceeds), a live one is held.
        info["signal_refused"] = True
        info["terminated"] = not _kb._pid_alive(pid)
        return info
    if _kb._pid_alive(pid) and dispatch._pid_recycled(pid, started_at):
        info["terminated"] = True
        info["pid_recycled"] = True
        return info

    info["termination_attempted"] = True
    if signal_fn is None:
        from tools.process_registry import ProcessRegistry

        # Snapshot descendants before the worker can exit and orphan them. The
        # shared terminal terminator also handles children in separate sessions.
        expected_start = int(started_at.rsplit("|", 1)[1]) if isinstance(started_at, str) else started_at
        escalated = ProcessRegistry._terminate_host_pid(int(pid), expected_start)
        info["termination_scope"] = "process_tree"
        info["sigkill"] = int(pid) in escalated
        info["terminated"] = dispatch._poll_worker_exit(pid, started_at)
        return info
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Already gone = successful termination. Leaving terminated=False would
        # make the reclaim guard misread a dead worker as alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    if dispatch._poll_worker_exit(pid, started_at):
        info["terminated"] = True
        return info
    if dispatch._worker_alive(pid, started_at):
        if not dispatch._sigkill(kill, pid):
            return info
        info["sigkill"] = True
    info["terminated"] = not dispatch._worker_alive(pid, started_at)
    return info
