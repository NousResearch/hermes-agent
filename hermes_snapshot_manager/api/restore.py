from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from hermes_snapshot_manager.services.operation_service import OperationService
from hermes_snapshot_manager.services.restore_service import RestoreService

router = APIRouter(tags=["restore"])
service = RestoreService()
operation_service = OperationService(service.paths)


@router.post("/api/snapshots/{snapshot_id}/restore")
def restore_snapshot(snapshot_id: str, notes: str | None = Form(default=None)) -> dict:
    return service.restore_snapshot(snapshot_id, notes=notes)


@router.post("/api/snapshots/{snapshot_id}/restore/start")
def start_restore_snapshot(snapshot_id: str, notes: str | None = Form(default=None)) -> dict:
    return operation_service.start_restore(snapshot_id, notes=notes)


@router.get("/api/restores")
def restore_history() -> list[dict]:
    return service.list_restore_history()


@router.post("/snapshots/{snapshot_id}/restore")
def restore_snapshot_from_ui(snapshot_id: str, notes: str | None = Form(default=None)) -> RedirectResponse:
    service.restore_snapshot(snapshot_id, notes=notes)
    return RedirectResponse(url=f"/snapshots/{snapshot_id}", status_code=303)
