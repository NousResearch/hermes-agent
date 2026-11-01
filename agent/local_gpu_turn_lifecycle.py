"""Turn-boundary cleanup for local GPU admission leases."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def release_turn_local_gpu_admission(agent) -> None:
    """Release a turn lease unless Kanban completion owns its deferred lifetime."""
    if getattr(agent, "_local_gpu_admission_defer_release", False):
        return
    lease_ids = tuple(getattr(agent, "_local_gpu_admission_lease_ids", ()))
    if not lease_ids:
        return
    try:
        from hermes_cli.middleware import release_local_gpu_admission_leases

        scope = dict(getattr(agent, "_local_gpu_admission_scope", {}) or {})
        if release_local_gpu_admission_leases(lease_ids=lease_ids, scope=scope):
            agent._local_gpu_admission_lease_ids = ()
            agent._local_gpu_admission_scope = {}
            return
        logger.error(
            "Local GPU lease release failed; controller expiry/reconciliation is required",
            extra={"lease_ids": lease_ids, "task_run_id": scope.get("task_run_id")},
        )
    except Exception:
        logger.error(
            "Local GPU lease release failed; controller expiry/reconciliation is required",
            exc_info=True,
        )
