"""Backend-only Kanban pickup control routes for future Office consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hermes_cli.kanban_control import control_status, set_control_scope


router = APIRouter()


class ControlMutation(BaseModel):
    reason: Optional[str] = None
    board: Optional[str] = None


def _board(board: Optional[str]) -> Optional[str]:
    """Preserve the control API's None = all-active-boards selection."""
    return board or None


def _selection_http(exc: ValueError) -> HTTPException:
    status_code = 404 if "does not exist or is archived" in str(exc) else 400
    return HTTPException(status_code=status_code, detail=str(exc))


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
    try:
        result = control_status(board=_board(board))
    except ValueError as exc:
        raise _selection_http(exc) from exc
    return _unknown_503(result)


def _set_control(desired_state: str, payload: ControlMutation, request: Request):
    try:
        result = set_control_scope(
            desired_state,
            changed_by=_request_actor(request),
            reason=payload.reason,
            board=_board(payload.board),
        )
    except ValueError as exc:
        raise _selection_http(exc) from exc
    return _mutation_response(result)


@router.post("/control/pause")
def pause_control(payload: ControlMutation, request: Request):
    return _set_control("paused", payload, request)


@router.post("/control/resume")
def resume_control(payload: ControlMutation, request: Request):
    return _set_control("running", payload, request)
