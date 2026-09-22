"""Managed worker bootstrap binding to the canonical board and origin receipt."""
from __future__ import annotations

import os
import logging

from agent.model_selection_store import get_receipt
from agent.model_selection_types import RoutingBlocked
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_db_dispatch import _set_worker_pid

logger = logging.getLogger(__name__)


def _enforce_kanban_routing_receipt(cli) -> bool:
    """Fence the claimed worker before task inference; unmanaged launches are unchanged."""
    receipt_id = os.environ.get("HERMES_KANBAN_ROUTING_RECEIPT", "").strip()
    if not receipt_id:
        return True
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    try:
        worker_run_id = int(raw_run_id) if raw_run_id else None
    except (TypeError, ValueError):
        worker_run_id = None
    if task_id and worker_run_id is not None:
        from hermes_cli import kanban_db as kb

        try:
            with connect_closing() as conn:
                live_run_id = kb._current_run_id(conn, task_id)
        except Exception:
            logger.error("guided-routing: cannot verify live claim; refusing worker startup")
            return False
        if live_run_id != worker_run_id:
            logger.error("guided-routing: stale/reclaimed worker claim")
            return False
    from agent.managed_route_runtime import enforce_worker_route
    from hermes_constants import get_hermes_home

    # Origin owns the immutable decision; worker home owns credentials.
    routing_home = os.environ.get("HERMES_KANBAN_ROUTING_ORIGIN_HOME", "").strip() or get_hermes_home()
    agent = cli.agent
    # Named custom providers share a transport family, not an admission identity.
    actual_provider = (getattr(agent, "requested_provider", "") or agent.provider or "").strip()
    try:
        register_managed_worker(routing_home, receipt_id, task_id, worker_run_id)
        enforce_worker_route(
            routing_home, receipt_id, actual_provider=actual_provider,
            actual_model=(agent.model or "").strip(),
            actual_endpoint=(getattr(agent, "base_url", None) or None),
            actual_reasoning=requested_effort_for_kanban_guard(cli),
        )
    except RoutingBlocked as exc:
        logger.error("guided-routing enforcement blocked this Kanban worker: %s", exc)
        return False
    _disable_inherited_fallback_for_managed_run(agent, receipt_id)
    agent._managed_routing_receipt_id = receipt_id
    agent._managed_routing_home = routing_home
    return True


def _disable_inherited_fallback_for_managed_run(agent, receipt_id: str) -> None:
    """A profile-wide fallback chain cannot authorize another managed route."""
    if not getattr(agent, "_fallback_chain", None):
        return
    logger.info("guided-routing: clearing inherited fallback chain for receipt=%s", receipt_id)
    agent._fallback_chain = []
    agent._fallback_index = 0
    agent._fallback_model = None


def requested_effort_for_kanban_guard(cli) -> str | None:
    from agent.reasoning_effort import requested_effort

    return requested_effort(getattr(cli, "reasoning_config", None))


def register_managed_worker(origin_home, receipt_id: str, task_id: str, run_id: int | None) -> None:
    """Register before inference, even if the spawning dispatcher died after Popen.

    The board CAS is the ownership authority; a valid but unlinked origin receipt
    cannot authorize a worker. Parent and child publication agree on run/PID.
    """
    decision = get_receipt(origin_home, receipt_id)
    requirements = decision["requirements"] if decision else {}
    if (not task_id or not run_id or requirements.get("execution_kind") != "kanban"
            or requirements.get("execution_id") != task_id
            or requirements.get("attempt_id") != str(run_id)
            or requirements.get("slot_id") != ""):
        raise RoutingBlocked("stale_or_revoked_decision", "receipt task/run binding mismatch")
    with connect_closing() as conn:
        registered = _set_worker_pid(
            conn, task_id, os.getpid(), expected_run_id=run_id,
            expected_receipt_id=receipt_id, expected_role=requirements.get("role"),
        )
    if not registered:
        raise RoutingBlocked("stale_or_revoked_decision", "canonical worker registration rejected")
