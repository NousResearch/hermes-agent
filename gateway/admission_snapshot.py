"""Live gateway admission snapshot for the ``admission`` control verb (#127611).

Answered live from the runner's in-memory counters plus read-only durable
state. Read-only and non-force: it never drains, interrupts, writes status,
repairs ledgers, or quarantines anything. Every unreadable source becomes
``unknown`` (count None), never zero.

The CLI (``hermes_cli/update_admission.py``) prefers this verb and falls
back fail-closed to ``unknown`` for gateways that predate it.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ADMISSION_SCHEMA_VERSION = 1


def _idle(detail: dict | None = None) -> dict[str, Any]:
    return {"state": "idle", "count": 0, "detail": detail or {"source": "gateway-live"}}


def _busy(count: int, detail: dict | None = None) -> dict[str, Any]:
    return {"state": "busy", "count": int(count), "detail": detail or {"source": "gateway-live"}}


def _unknown(reason: str, detail: dict | None = None) -> dict[str, Any]:
    merged = {"reason": reason, "source": "gateway-live"}
    merged.update(detail or {})
    return {"state": "unknown", "count": None, "detail": merged}


def _count_or_unknown(label: str, fn) -> dict[str, Any]:
    try:
        count = max(0, int(fn()))
    except Exception as exc:  # noqa: BLE001 - unreadable source => unknown, never zero
        logger.debug("admission %s unreadable: %s", label, exc)
        return _unknown(f"{label}_unreadable")
    if count > 0:
        return _busy(count, {"source": "gateway-live"})
    return _idle({"source": "gateway-live"})


def build_gateway_admission(runner: Any) -> dict[str, Any]:
    """Fresh admission work for ``runner`` (read-only; never raises)."""
    work: dict[str, Any] = {}
    # Foreground turns.
    try:
        work["foreground_turns"] = _count_or_unknown(
            "foreground_turns", runner._running_agent_count)
    except Exception:  # noqa: BLE001
        work["foreground_turns"] = _unknown("foreground_unreadable")

    # Cron jobs, including restart-safe external worker pids.
    try:
        from cron.scheduler import get_running_job_details

        details = get_running_job_details()
        if details:
            work["cron_jobs"] = _busy(len(details), {
                "source": "gateway-live",
                "jobs": [
                    {"job_id": j.get("job_id"), "worker_pid": j.get("worker_pid"),
                     "restart_safe_worker": bool(j.get("worker_pid")),
                     "elapsed_s": j.get("elapsed_s")}
                    for j in details[:20]
                ],
            })
        else:
            work["cron_jobs"] = _idle({"source": "gateway-live"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission cron unreadable: %s", exc)
        work["cron_jobs"] = _unknown("cron_jobs_unreadable")

    # API runs: handler count + executor worker count (module-level, survives
    # adapter teardown). No api_server adapter => proven zero.
    try:
        from gateway.config import Platform

        adapters = getattr(runner, "adapters", {}) or {}
        if Platform.API_SERVER not in adapters:
            work["api_runs"] = _idle({"source": "no_api_server"})
        else:
            handler = max(0, int(runner._active_api_run_count()))
            try:
                from gateway.platforms.api_server_runs import api_worker_live_count

                workers = max(0, int(api_worker_live_count()))
            except Exception as exc:  # noqa: BLE001
                logger.debug("admission api workers unreadable: %s", exc)
                work["api_runs"] = _unknown("api_runs_unreadable")
            else:
                total = handler + workers
                work["api_runs"] = (
                    _busy(total, {"handler": handler, "workers": workers,
                                  "source": "gateway-live"})
                    if total > 0 else _idle({"source": "gateway-live"})
                )
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission api unreadable: %s", exc)
        work["api_runs"] = _unknown("api_runs_unreadable")

    # Deferred executor workers.
    try:
        work["deferred_workers"] = _count_or_unknown(
            "deferred_workers", runner._active_deferred_agent_worker_count)
    except Exception:  # noqa: BLE001
        work["deferred_workers"] = _unknown("deferred_workers_unreadable")

    # Background delegations (process-local).
    try:
        from tools.async_delegation import active_count

        work["background_delegations"] = _count_or_unknown("background_delegations", active_count)
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission delegations unreadable: %s", exc)
        work["background_delegations"] = _unknown("background_delegations_unreadable")

    # Background terminal processes + pending completion watchers.
    try:
        from tools.process_registry import process_registry

        active = bool(process_registry.has_any_active())
        pending = len(getattr(process_registry, "pending_watchers", []) or [])
        total = (1 if active else 0) + (1 if pending else 0)
        if active or pending:
            work["background_processes"] = _busy(max(1, pending if pending else 1), {
                "source": "gateway-live", "has_active": active,
                "pending_watchers": pending,
            })
        else:
            work["background_processes"] = _idle({"source": "gateway-live"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission processes unreadable: %s", exc)
        work["background_processes"] = _unknown("background_processes_unreadable")

    # External worker completion: restart-safe worker pids still live.
    try:
        from cron.scheduler import get_running_job_details

        externals = [j for j in (get_running_job_details() or []) if j.get("worker_pid")]
        if externals:
            work["external_workers"] = _busy(len(externals), {
                "source": "gateway-live",
                "pending_completion": len(externals),
                "worker_pids": [j.get("worker_pid") for j in externals[:20]],
            })
        else:
            work["external_workers"] = _idle({"source": "gateway-live"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("admission external workers unreadable: %s", exc)
        work["external_workers"] = _unknown("external_workers_unreadable")

    computed_at = time.time()
    return {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "computed_at": datetime.fromtimestamp(computed_at, tz=timezone.utc).isoformat(),
        "pid": _runner_pid(runner),
        "work": work,
    }


def _runner_pid(runner: Any) -> int | None:
    import os

    try:
        pid = getattr(runner, "_pid", None)
        return int(pid) if pid is not None else int(os.getpid())
    except Exception:  # noqa: BLE001
        return None
