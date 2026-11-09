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
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from hermes_constants import get_hermes_home

from cron.executions import cancellable_execution

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
    """Record a cancel for one execution. Idempotent: the marker holds the first reason.

    A second request for the same execution keeps the reason already on disk, because the run
    may have read it into its report before the later request lands. The first ask is the one
    that describes why the operator acted.
    """
    path = _marker_path(execution_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _publish_first(path, (reason or _DEFAULT_REASON).strip().encode("utf-8"))
    _prune_stale_markers()
    return path


def _publish_first(path: Path, body: bytes) -> None:
    """Create ``path`` holding ``body`` unless it already exists (the first reason wins).

    A private temp file published with link() means overlapping requests never share a temp path
    and a reader never sees the reason half-written. A filesystem without hard links (exFAT, some
    network shares) falls back to an exclusive create: a reader racing that write sees an empty
    marker, which still reads as a cancel, with the default reason.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
        with contextlib.suppress(FileExistsError):
            os.link(tmp, path)
        linked = True
    except OSError:  # no hard links on this filesystem
        linked = False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    if not linked:
        with contextlib.suppress(FileExistsError), open(path, "xb") as handle:
            handle.write(body)


def cancel_requested(execution_id: str) -> bool:
    with contextlib.suppress(ValueError, OSError):
        return _marker_path(execution_id).exists()
    return False


def clear_cancel(execution_id: str) -> None:
    """Called by the run itself once it has wound down, so a finished attempt leaves no marker."""
    with contextlib.suppress(ValueError, OSError):
        _marker_path(execution_id).unlink(missing_ok=True)


class _CancelEventLike(Protocol):
    """Structural type for cancellation sources (``threading.Event``, ``_CombinedCancelEvent``)."""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class _CombinedCancelEvent:
    """Duck-typed ``threading.Event`` ORing several cancellation sources (fire-claim heartbeat
    ``lost_ownership`` + per-transport events). Workers only call is_set()/set(), so no pump thread.
    """

    def __init__(self, *events: Optional[_CancelEventLike]) -> None:
        self._events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def set(self) -> None:
        for event in self._events:
            event.set()

    @property
    def reason(self) -> Optional[str]:
        """Why the run is being torn down, from whichever source fired — so a script or agent
        stop never reports "fire claim lost" for an operator cancel (or the reverse)."""
        for event in self._events:
            if event.is_set():
                return getattr(event, "reason", None)
        return None


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
                recorded = _marker_path(self.execution_id).read_text(encoding="utf-8-sig").strip()
                if recorded:
                    self._reason = recorded
            self._reason = self._reason or _DEFAULT_REASON
        return self._reason


def cancel_reason(cancel_event: Any) -> str:
    """Human-facing cause for a cancelled run; the legacy wording when no source carries one."""
    return getattr(cancel_event, "reason", None) or "cron fire ownership was lost"


def finish_cancelled_run(job: dict[str, Any], execution_id: str, fire_owner: Optional[str]) -> None:
    """Book an operator-cancelled run: ledger row ``cancelled`` with the recorded reason, and the
    job's ``last_status`` — without a failure alert or a failure-streak bump, since the operator
    asked for it. A job removed mid-run has no record left to mark; its ledger row still lands."""
    from cron import scheduler
    from cron.unreachable_retry import is_retry_run

    reason = CancelRequest(execution_id).reason
    if not scheduler.self_removal_delivery_allowed(job["id"]):
        owner = {"expected_fire_owner": fire_owner} if fire_owner is not None else {}
        # A stopped ladder re-run re-ran an occurrence that already counted: no second repeat slot.
        rung = {"ladder_rung": True} if is_retry_run(job) else {}
        scheduler.mark_job_run(
            job["id"], False, f"Cancelled: {reason}", status="cancelled", **owner, **rung)
    scheduler.finish_execution(execution_id, success=False, error=reason, cancelled=True)


def finish_lost_execution(execution_id: str, error: str) -> Optional[dict[str, Any]]:
    """Close an attempt whose fire claim vanished. Removing a job mid-run cancels it AND drops the
    claim, so a run that finds its claim gone books the operator's ``cancelled`` (with the reason)
    when a cancel marker is waiting, and ``failed`` with ``error`` only for a real claim loss."""
    from cron import scheduler

    if cancel_requested(execution_id):
        return scheduler.finish_execution(
            execution_id, success=False, error=CancelRequest(execution_id).reason, cancelled=True)
    return scheduler.finish_execution(execution_id, success=False, error=error)


def run_job_unless_cancelled(run_job, job, run_kwargs, teardown, fence, execution_id, fire_owner):
    """``run_job``'s result, or None once an operator cancel stopped the run and was booked.

    Checked before the caller's claim-loss handling: removing a job mid-run cancels it AND drops
    its claim, and the run must book the cancel, not "fire claim ownership lost" (#123045).
    ``run_job`` hands the agent back even when raising, so ``teardown`` runs on every path
    (BaseException, so KeyboardInterrupt/SystemExit mid-run still tear down)."""
    try:
        result = run_job(job, **run_kwargs)
    except BaseException as exc:
        teardown()
        if not (isinstance(exc, Exception) and fence.operator_cancelled()):
            raise
        finish_cancelled_run(job, execution_id, fire_owner)
        return None
    if not fence.operator_cancelled():
        return result
    teardown()
    finish_cancelled_run(job, execution_id, fire_owner)
    return None


def cancel_run(job_id: str, *, reason: Optional[str] = None) -> dict[str, Any]:
    """Request cancellation of the job's live run. A job with nothing in flight is a no-op."""
    record = cancellable_execution(job_id)
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
