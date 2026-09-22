"""Managed worker bootstrap binding to the canonical board and origin receipt."""
from __future__ import annotations

import os

from agent.model_selection_store import get_receipt
from agent.model_selection_types import RoutingBlocked
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_db_dispatch import _set_worker_pid


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
