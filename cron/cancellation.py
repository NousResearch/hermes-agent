"""On-demand cancellation of an in-flight cron run (``hermes cron cancel``).

The request has to reach the process that owns the run, and that process is usually not
the one taking the request: ``run_one_job`` hands an attempt to a detached external worker
before anything executes in-process, so an in-memory ``Event`` can never be the transport.
The request is therefore a marker file per EXECUTION — the ledger already mints one id per
attempt — polled by the same cancel seam the fire-claim heartbeat uses, so a cancel reaches
script runs (tree-kill + TERM/KILL escalation) and agent runs (interrupt) through the code
those paths already own.

Keying by execution id is what makes a leftover marker harmless: only the run it was written
for can read it, so a cancel can never bleed into the job's next fire.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

from cron.executions import live_inflight_execution

# A marker whose run was SIGKILLed before it could clean up is unreadable garbage. Canceling a
# job is rare, so the prune rides along with the write instead of needing its own sweep.
_MARKER_TTL_SECONDS = 24 * 3600

_DEFAULT_REASON = "cancelled on request (hermes cron cancel)"


def _cancels_dir() -> Path:
    """Resolved per call so a multiplexed gateway never writes into another profile's home."""
    return get_hermes_home().resolve() / "cron" / "cancels"


def _marker_path(execution_id: str) -> Path:
    execution_id = str(execution_id or "").strip()
    if not execution_id or "/" in execution_id or "\\" in execution_id or execution_id in {".", ".."}:
        raise ValueError(f"Not a usable execution id: {execution_id!r}")
    return _cancels_dir() / f"{execution_id}.cancel"


def _prune_stale_markers(now: Optional[float] = None) -> None:
    cutoff = (now if now is not None else time.time()) - _MARKER_TTL_SECONDS
    with contextlib.suppress(OSError):
        for entry in _cancels_dir().iterdir():
            with contextlib.suppress(OSError):
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()


def request_cancel(execution_id: str, *, reason: Optional[str] = None) -> Path:
    """Record a cancel for one execution. Idempotent: the marker holds the first reason."""
    path = _marker_path(execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".cancel.tmp")
    tmp.write_text((reason or _DEFAULT_REASON).strip(), encoding="utf-8")
    os.replace(tmp, path)
    _prune_stale_markers()
    return path


def cancel_requested(execution_id: str) -> bool:
    with contextlib.suppress(ValueError, OSError):
        return _marker_path(execution_id).exists()
    return False


def clear_cancel(execution_id: str) -> None:
    """Called by the run itself once it has wound down, so a finished attempt leaves no marker."""
    with contextlib.suppress(ValueError, OSError):
        _marker_path(execution_id).unlink(missing_ok=True)


class CancelRequest:
    """``_CancelEventLike`` view of one execution's cancel marker.

    Implements ``set()`` for the protocol but never latches: this source is owned by another
    process, so a local ``set()`` would claim a cancel nobody asked for.
    """

    def __init__(self, execution_id: str) -> None:
        self.execution_id = str(execution_id or "")
        self._reason: Optional[str] = None

    def is_set(self) -> bool:
        return cancel_requested(self.execution_id)

    def set(self) -> None:  # pragma: no cover — protocol shape only
        raise NotImplementedError("a cancel request is set by the requesting process, not the run")

    @property
    def reason(self) -> str:
        if self._reason is None:
            with contextlib.suppress(ValueError, OSError):
                recorded = _marker_path(self.execution_id).read_text(encoding="utf-8").strip()
                if recorded:
                    self._reason = recorded
            self._reason = self._reason or _DEFAULT_REASON
        return self._reason


def cancel_run(job_id: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
    """Request cancellation of the job's live run. A job with nothing in flight is a no-op."""
    record = live_inflight_execution(job_id)
    if record is None:
        return {
            "success": True,
            "cancelled": False,
            "execution_id": None,
            "detail": "No run of this job is in flight — nothing to cancel.",
        }
    request_cancel(record["id"], reason=reason)
    return {
        "success": True,
        "cancelled": True,
        "execution_id": record["id"],
        "status": record.get("status"),
        "pid": record.get("pid"),
        "detail": "Cancel requested; the run stops at its next cancel check.",
    }
