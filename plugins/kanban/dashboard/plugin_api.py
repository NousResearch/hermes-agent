"""Kanban dashboard plugin — thin FastAPI adapter over hermes_cli.kanban_http."""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status as http_status
from pydantic import BaseModel, Field

from hermes_cli import kanban_db
from hermes_cli import kanban_http as kh

log = logging.getLogger(__name__)
router = APIRouter()

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]


def _check_ws_token(provided: Optional[str]) -> bool:
    if not provided:
        return False
    try:
        from hermes_cli import web_server as _ws
    except Exception:
        return True
    expected = getattr(_ws, "_SESSION_TOKEN", None)
    if not expected:
        return True
    return hmac.compare_digest(str(provided), str(expected))


def _map_error(exc: Exception) -> None:
    if isinstance(exc, kh.KanbanAPIError):
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc))
    raise exc


class CreateTaskBody(BaseModel):
    title: str
    body: Optional[str] = None
    assignee: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0
    workspace_kind: str = "scratch"
    workspace_path: Optional[str] = None
    parents: list[str] = Field(default_factory=list)
    triage: bool = False
    idempotency_key: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    skills: Optional[list[str]] = None


class UpdateTaskBody(BaseModel):
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    result: Optional[str] = None
    block_reason: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None


class CommentBody(BaseModel):
    body: str
    author: Optional[str] = "dashboard"


class LinkBody(BaseModel):
    parent_id: str
    child_id: str


class BulkTaskBody(BaseModel):
    ids: list[str]
    status: Optional[str] = None
    assignee: Optional[str] = None
    priority: Optional[int] = None
    archive: bool = False
    result: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    reclaim_first: bool = False


def _set_status_direct(conn: Any, task_id: str, status_value: str) -> bool:
    """Compatibility shim for tests/importers after moving logic to kanban_http."""
    return kh._set_status_direct(conn, task_id, status_value)


class ReclaimBody(BaseModel):
    reason: Optional[str] = None


class TerminateRunBody(BaseModel):
    reason: Optional[str] = None


class SpecifyBody(BaseModel):
    author: Optional[str] = None


class ReassignBody(BaseModel):
    profile: Optional[str] = None
    reclaim_first: bool = False
    reason: Optional[str] = None


class CreateBoardBody(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    switch: bool = False


class RenameBoardBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


class DescribeBody(BaseModel):
    description: str = ""


class DescribeAutoBody(BaseModel):
    force: bool = False


class DecomposeBody(BaseModel):
    max_children: Optional[int] = None


class OrchestrationSettingsBody(BaseModel):
    enabled: Optional[bool] = None
    max_spawn: Optional[int] = None
    dispatch_interval_seconds: Optional[float] = None


@router.get("/board")
def get_board(
    tenant: Optional[str] = Query(None),
    include_archived: bool = Query(False),
    board: Optional[str] = Query(None),
    workflow_template_id: Optional[str] = Query(None),
    current_step_key: Optional[str] = Query(None),
):
    try:
        return kh.http_get_board(tenant, include_archived, board, workflow_template_id, current_step_key)
    except Exception as exc:
        _map_error(exc)


@router.get("/tasks/{task_id}")
def get_task(
    task_id: str,
    board: Optional[str] = Query(None),
    run_state_type: Optional[str] = Query(None),
    run_state_name: Optional[str] = Query(None),
):
    try:
        return kh.http_get_task(task_id, board, run_state_type, run_state_name)
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks")
def create_task(payload: CreateTaskBody, board: Optional[str] = Query(None)):
    try:
        req = kh.CreateTaskRequest(**payload.model_dump(), created_by="dashboard")
        return kh.http_create_task(req, board)
    except Exception as exc:
        _map_error(exc)


@router.patch("/tasks/{task_id}")
def update_task(task_id: str, payload: UpdateTaskBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_update_task(task_id, kh.UpdateTaskRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.delete("/tasks/{task_id}")
def delete_task(task_id: str, board: Optional[str] = Query(None)):
    try:
        return kh.http_delete_task(task_id, board)
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/{task_id}/comments")
def add_comment(task_id: str, payload: CommentBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_add_comment(task_id, kh.CommentRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.post("/links")
def add_link(payload: LinkBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_add_link(kh.LinkRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.delete("/links")
def delete_link(
    parent_id: str = Query(...),
    child_id: str = Query(...),
    board: Optional[str] = Query(None),
):
    try:
        return kh.http_delete_link(parent_id, child_id, board)
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/bulk")
def bulk_update(payload: BulkTaskBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_bulk_update(kh.BulkTaskRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.get("/diagnostics")
def list_diagnostics(
    board: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
):
    try:
        return kh.http_list_diagnostics(board, severity)
    except Exception as exc:
        _map_error(exc)


@router.get("/workers/active")
def list_active_workers(board: Optional[str] = Query(None)):
    try:
        return kh.http_list_active_workers(board)
    except Exception as exc:
        _map_error(exc)


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: int, board: Optional[str] = Query(None)):
    try:
        return kh.http_get_run_endpoint(run_id, board)
    except Exception as exc:
        _map_error(exc)


@router.get("/runs/{run_id}/inspect")
def inspect_run_endpoint(run_id: int, board: Optional[str] = Query(None)):
    try:
        kh._psutil = _psutil  # type: ignore[attr-defined]
        return kh.http_inspect_run_endpoint(run_id, board)
    except Exception as exc:
        _map_error(exc)


@router.post("/runs/{run_id}/terminate")
def terminate_run_endpoint(
    run_id: int,
    payload: TerminateRunBody,
    board: Optional[str] = Query(None),
):
    try:
        resolved_board = kh.resolve_board(board)
        conn = kh._conn(board=resolved_board)
        try:
            run = kanban_db.get_run(conn, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"run {run_id} not found")
            if run.ended_at is not None:
                raise HTTPException(status_code=409, detail=f"run {run_id} already ended")
            ok = kanban_db.reclaim_task(conn, run.task_id, reason=payload.reason)
            if not ok:
                raise HTTPException(
                    status_code=409,
                    detail=f"run {run_id} task is not reclaimable",
                )
            return {"ok": True, "run_id": run_id, "task_id": run.task_id}
        finally:
            conn.close()
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/{task_id}/reclaim")
def reclaim_task_endpoint(task_id: str, payload: ReclaimBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_reclaim_task_endpoint(task_id, kh.ReclaimRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/{task_id}/specify")
def specify_task_endpoint(task_id: str, payload: SpecifyBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_specify_task_endpoint(task_id, kh.SpecifyRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/{task_id}/reassign")
def reassign_task_endpoint(task_id: str, payload: ReassignBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_reassign_task_endpoint(task_id, kh.ReassignRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.get("/config")
def get_config():
    return kh.http_get_config()


@router.get("/home-channels")
def get_home_channels(task_id: Optional[str] = Query(None), board: Optional[str] = Query(None)):
    return kh.http_get_home_channels(task_id, board)


@router.post("/tasks/{task_id}/home-subscribe/{platform}")
def subscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    try:
        return kh.http_subscribe_home(task_id, platform, board)
    except Exception as exc:
        _map_error(exc)


@router.delete("/tasks/{task_id}/home-subscribe/{platform}")
def unsubscribe_home(task_id: str, platform: str, board: Optional[str] = Query(None)):
    try:
        return kh.http_unsubscribe_home(task_id, platform, board)
    except Exception as exc:
        _map_error(exc)


@router.get("/stats")
def get_stats(board: Optional[str] = Query(None)):
    return kh.http_get_stats(board)


@router.get("/assignees")
def get_assignees(board: Optional[str] = Query(None)):
    return kh.http_get_assignees(board)


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str, board: Optional[str] = Query(None)):
    try:
        return kh.http_get_task_log(task_id, board)
    except Exception as exc:
        _map_error(exc)


@router.post("/dispatch")
def dispatch(
    dry_run: bool = Query(False),
    max_n: int = Query(8, alias="max"),
    board: Optional[str] = Query(None),
):
    try:
        return kh.http_dispatch(dry_run, max_n, board)
    except Exception as exc:
        _map_error(exc)


@router.get("/boards")
def list_boards(include_archived: bool = Query(False)):
    return kh.http_list_boards(include_archived)


@router.post("/boards")
def create_board_endpoint(payload: CreateBoardBody):
    try:
        return kh.http_create_board_endpoint(kh.CreateBoardRequest(**payload.model_dump()))
    except Exception as exc:
        _map_error(exc)


@router.patch("/boards/{slug}")
def rename_board(slug: str, payload: RenameBoardBody):
    try:
        return kh.http_rename_board(slug, kh.RenameBoardRequest(**payload.model_dump()))
    except Exception as exc:
        _map_error(exc)


@router.delete("/boards/{slug}")
def delete_board(slug: str, delete: bool = Query(False)):
    try:
        return kh.http_delete_board(slug, delete)
    except Exception as exc:
        _map_error(exc)


@router.post("/boards/{slug}/switch")
def switch_board(slug: str):
    try:
        return kh.http_switch_board(slug)
    except Exception as exc:
        _map_error(exc)


@router.get("/profiles")
def list_profile_roster():
    return kh.http_list_profile_roster()


@router.patch("/profiles/{profile_name}")
def update_profile_description(profile_name: str, payload: DescribeBody):
    try:
        return kh.http_update_profile_description(profile_name, kh.DescribeRequest(**payload.model_dump()))
    except Exception as exc:
        _map_error(exc)


@router.post("/profiles/{profile_name}/describe-auto")
def auto_describe_profile(profile_name: str, payload: DescribeAutoBody):
    try:
        return kh.http_auto_describe_profile(profile_name, kh.DescribeAutoRequest(**payload.model_dump()))
    except Exception as exc:
        _map_error(exc)


@router.post("/tasks/{task_id}/decompose")
def decompose_task_endpoint(task_id: str, payload: DecomposeBody, board: Optional[str] = Query(None)):
    try:
        return kh.http_decompose_task_endpoint(task_id, kh.DecomposeRequest(**payload.model_dump()), board)
    except Exception as exc:
        _map_error(exc)


@router.get("/orchestration")
def get_orchestration_settings():
    return kh.http_get_orchestration_settings()


@router.put("/orchestration")
def set_orchestration_settings(payload: OrchestrationSettingsBody):
    try:
        return kh.http_set_orchestration_settings(kh.OrchestrationSettingsRequest(**payload.model_dump()))
    except Exception as exc:
        _map_error(exc)


@router.websocket("/events")
async def stream_events(ws: WebSocket):
    token = ws.query_params.get("token")
    if not _check_ws_token(token):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    try:
        since_raw = ws.query_params.get("since", "0")
        try:
            cursor = int(since_raw)
        except ValueError:
            cursor = 0
        ws_board_raw = ws.query_params.get("board")
        try:
            ws_board = kh.kanban_db._normalize_board_slug(ws_board_raw) if ws_board_raw else None
        except ValueError:
            ws_board = None
        while True:
            cursor, events = await asyncio.to_thread(kh.fetch_events, cursor, ws_board)
            if events:
                await ws.send_json({"events": events, "cursor": cursor})
            await asyncio.sleep(kh.EVENT_POLL_SECONDS)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return
    except Exception as exc:
        log.warning("Kanban event stream error: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass
