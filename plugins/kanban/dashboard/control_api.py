"""Backend-only Kanban pickup control routes for future Office consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hermes_cli import kanban_control
from hermes_cli import kanban_db


router = APIRouter()


class ControlMutation(BaseModel):
    reason: Optional[str] = None
    board: Optional[str] = None


def _board_or_http(board: Optional[str]) -> Optional[str]:
    if board is None or board == "":
        return None
    try:
        normalized = kanban_db._normalize_board_slug(board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    active = {item["slug"] for item in kanban_db.list_boards(include_archived=False)}
    if normalized is None or normalized not in active:
        raise HTTPException(status_code=404, detail=f"board {board!r} does not exist")
    return normalized


def _unknown_503(result: dict):
    if result["state"] == "unknown":
        raise HTTPException(
            status_code=503,
            detail={"code": "control_state_unknown", "result": result},
        )
    return result


def _mutation_response(result: dict):
    if result["state"] == "unknown":
        return JSONResponse(
            status_code=503,
            content={"code": "control_state_unknown", "result": result},
        )
    if result["state"] == "partial":
        return JSONResponse(status_code=207, content=result)
    return result


def _request_actor(request: Request) -> str:
    session = getattr(request.state, "session", None)
    if session is not None and getattr(session, "user_id", None):
        return f"{getattr(session, 'provider', 'session')}:{session.user_id}"
    principal = getattr(request.state, "token_principal", None)
    if principal is not None and getattr(principal, "principal", None):
        return f"{getattr(principal, 'provider', 'token')}:{principal.principal}"
    native = getattr(request.state, "native_http_principal", None)
    if isinstance(native, dict):
        caller = native.get("user_id") or native.get("client_id") or native.get("profile_id")
        if caller:
            return f"native:{Path(str(caller)).name}"
    if getattr(request.app.state, "auth_required", False):
        raise HTTPException(status_code=401, detail="authenticated principal unavailable")
    return "dashboard-local"


@router.get("/control")
def get_control(board: Optional[str] = Query(None)):
    return _unknown_503(kanban_control.control_status(board=_board_or_http(board)))


@router.post("/control/pause")
def pause_control(payload: ControlMutation, request: Request):
    try:
        result = kanban_control.set_control_scope(
            "paused", changed_by=_request_actor(request), reason=payload.reason,
            board=_board_or_http(payload.board),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _mutation_response(result)


@router.post("/control/resume")
def resume_control(payload: ControlMutation, request: Request):
    try:
        result = kanban_control.set_control_scope(
            "running", changed_by=_request_actor(request), reason=payload.reason,
            board=_board_or_http(payload.board),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _mutation_response(result)
