"""Project Usage dashboard plugin backend.

Mounted at /api/plugins/project_usage/ by the dashboard plugin loader.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from hermes_cli import project_usage_ledger

router = APIRouter()


@router.get("/summary")
def get_project_usage_summary(
    board: Optional[str] = Query(None, description="Optional board slug filter"),
    task_id: Optional[str] = Query(None, description="Optional task id drilldown"),
    refresh: bool = Query(True, description="Refresh ledger from source DBs before reading"),
):
    """Return per-board totals and per-task drilldown usage rows."""
    return project_usage_ledger.get_summary(board=board, task_id=task_id, refresh=refresh)


@router.post("/backfill")
def backfill_project_usage():
    """Force an idempotent ledger backfill."""
    return project_usage_ledger.backfill()
