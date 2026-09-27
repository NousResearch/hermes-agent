"""Attempt-scoped cron worker drain signals for external deployers (#125513).

A terminal executions.db row, ``last_status``, or the outer run's return value
are NOT proof that the attempt's worker has finished:

- ``run_job`` can terminalize an attempt via its inactivity timeout while the
  inner ``run_conversation`` future is still live — the detached worker's
  Future owns the session finalization and agent teardown
  (``cron.scheduler_detached_worker.defer_teardown_to_running_worker``).
- ``_wait_for_external_cron_worker_body`` returns on a terminal ledger row
  before the external worker process exits (zombie reaping is a background
  thread there).

An external deployer that pauses a job before replacing code or skill content
needs a durable, attempt-scoped "the worker for THIS attempt has fully
drained" signal. This module is that signal: one JSON-lines log under
``<home>/cron/worker-drains.jsonl``, appended exactly once per attempt after
that attempt's teardown completed — inline teardown in ``run_job``'s finally,
post-delivery teardown in ``run_one_job``, or the detached worker's Future
callback — and queryable by execution id or job id.

Absence of a record is meaningful: attempts from before this signal existed,
workers that died without teardown (SIGKILL, crash), and unmatched ids all
read as ``unknown`` — a deployer keeps the deploy closed when completion is
unproven.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

_DRAIN_LOG_NAME = "worker-drains.jsonl"
# Bounded tail: append-only, atomically rewritten when it outgrows _PRUNE_BEYOND.
_PRUNE_BEYOND = 400
_PRUNE_KEEP = 200


def _drain_log_path() -> Path:
    # Resolved through cron.scheduler._get_hermes_home() at call time so
    # profile isolation and the test override both hold. Never call
    # get_hermes_home() directly here — that would freeze the shared default
    # root and break per-profile homes.
    from cron.scheduler import _get_hermes_home
    return _get_hermes_home() / "cron" / _DRAIN_LOG_NAME


def record_drain(execution_id: str, *, job_id: str = "") -> None:
    """Append this attempt's drain record. Best-effort: never raises into a run.

    The record is the LAST durable event of an attempt: callers invoke it only
    after the attempt's session finalization and agent teardown have completed,
    so a polling deployer can treat its presence as "worker fully drained".
    """
    if not execution_id:
        return
    try:
        path = _drain_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "execution_id": str(execution_id),
            "job_id": str(job_id or ""),
            "pid": os.getpid(),
            "drained_at": time.time(),
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        _prune(path)
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "cron worker drain record failed for %s", execution_id, exc_info=True)


def _prune(path: Path) -> None:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) <= _PRUNE_BEYOND:
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines[-_PRUNE_KEEP:]) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Monitoring tail only: unbounded growth is preferable to raising into
        # the teardown path that called us.
        pass


def _find_record(execution_id: str) -> Optional[Dict[str, Any]]:
    if not execution_id:
        return None
    try:
        path = _drain_log_path()
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception:
        return None
    # Tail-first: the freshest attempts drained last.
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str(row.get("execution_id") or "") == execution_id:
            return row
    return None


def drain_status(execution_id: str) -> Dict[str, Any]:
    """Drain status dict for one attempt id.

    ``status`` is ``"drained"`` when a record exists, ``"unknown"`` otherwise
    (pre-signal attempt, worker that died without teardown, or wrong id) —
    never a guess: unknown means "completion unproven", which is exactly the
    state an external deployer must treat as not-safe-to-replace.
    """
    execution_id = str(execution_id or "")
    record = _find_record(execution_id)
    if record is None:
        return {"execution_id": execution_id, "status": "unknown", "drained": False}
    return {
        "execution_id": execution_id,
        "job_id": record.get("job_id", ""),
        "status": "drained",
        "drained": True,
        "drained_at": record.get("drained_at"),
        "pid": record.get("pid"),
    }


def query_drain(target: str) -> Dict[str, Any]:
    """Resolve *target* — an execution id, or a job id (its latest attempt)."""
    target = str(target or "").strip()
    if not target:
        return {"target": target, "status": "unknown", "drained": False,
                "reason": "empty target"}

    # The drain log itself is the authoritative index for execution ids: an
    # exact record wins even without a ledger row (e.g. attempts from a worker
    # whose gateway never re-read the ledger, or pruned ledger history).
    status = drain_status(target)
    if status.get("drained"):
        status["target"] = target
        return status

    # Job id: the pause-then-drain flow names the job, not the attempt.
    from cron.executions import get_execution, list_executions
    try:
        if get_execution(target) is not None:
            status["target"] = target
            return status
    except Exception:
        pass
    try:
        latest = list_executions(job_id=target, limit=1)
    except Exception:
        latest = []
    if latest:
        execution_id = str(latest[0].get("id") or "")
        status = drain_status(execution_id)
        status["target"] = target
        return status
    return {"target": target, "status": "unknown", "drained": False,
            "reason": "no execution or job matched"}
