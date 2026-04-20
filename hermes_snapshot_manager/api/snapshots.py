from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from hermes_snapshot_manager.services.operation_service import OperationService
from hermes_snapshot_manager.services.snapshot_service import SnapshotService

router = APIRouter(tags=["snapshots"])
service = SnapshotService()
operation_service = OperationService(service.paths)


@router.get("/api/snapshots")
def list_snapshots() -> list[dict]:
    return [snapshot.__dict__ for snapshot in service.list_snapshots()]


@router.post("/api/snapshots")
def create_snapshot(label: str | None = Form(default=None), trigger_type: str = Form(default="manual")) -> dict:
    snapshot = service.create_snapshot(label=label, trigger_type=trigger_type)
    return snapshot.__dict__


@router.post("/api/snapshots/start")
def start_snapshot(label: str | None = Form(default=None)) -> dict:
    return operation_service.start_snapshot(label=label)


@router.get("/api/operations/current")
def current_operation() -> dict:
    return operation_service.get_current_operation() or {}


@router.get("/api/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str) -> dict:
    return service.get_snapshot(snapshot_id)


@router.post("/api/operations/current/abort")
def abort_operation() -> dict:
    return operation_service.abort_operation() or {"status": "none"}


@router.post("/api/operations/current/clear")
def clear_stale_operation() -> dict:
    return operation_service.clear_stale_operation() or {"status": "none"}


@router.get("/api/snapshots/{snapshot_id}/verify")
def verify_snapshot(snapshot_id: str) -> dict:
    return service.verify_snapshot(snapshot_id)


@router.get("/api/snapshots/{snapshot_id}/diff-current")
def diff_snapshot_to_current(snapshot_id: str) -> dict:
    return service.diff_snapshot_to_current(snapshot_id)


@router.post("/snapshots/create")
def create_snapshot_from_ui(label: str | None = Form(default=None)) -> RedirectResponse:
    service.create_snapshot(label=label, trigger_type="manual")
    return RedirectResponse(url="/snapshots", status_code=303)


@router.post("/snapshots/{snapshot_id}/known-good")
def mark_known_good(snapshot_id: str) -> RedirectResponse:
    service.mark_known_good(snapshot_id, value=True)
    return RedirectResponse(url=f"/snapshots/{snapshot_id}", status_code=303)


@router.post("/snapshots/{snapshot_id}/verify")
def verify_snapshot_from_ui(snapshot_id: str) -> RedirectResponse:
    service.verify_snapshot(snapshot_id)
    return RedirectResponse(url=f"/snapshots/{snapshot_id}", status_code=303)
