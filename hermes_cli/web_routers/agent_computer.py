"""Owner-authenticated REST surface for Agent Computers.

Mounted by web_server. Auth is the existing dashboard gate
(``_require_token`` / gated cookie). This router never exposes CDP,
cookies, or managed-profile filesystem paths.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from gateway.agent_computer import get_contract
from gateway.agent_computer.contract import error_payload, owner_principal
from gateway.agent_computer.errors import AgentComputerError
from hermes_cli.web_deps import late

router = APIRouter()
_require_token = late("_require_token")


class EnsureBody(BaseModel):
    profile_id: str


class IdentityCreateBody(BaseModel):
    ownership: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttachBody(BaseModel):
    identity_id: str


class LeaseBody(BaseModel):
    lease_id: str
    fencing_epoch: int
    kind: str = ""
    target: str = ""
    text: str = ""
    action_class: str = ""
    x: float | None = None
    y: float | None = None
    key: str = ""
    code: str = ""
    delta_x: float = 0
    delta_y: float = 0


class TakeoverBody(BaseModel):
    reason: str = ""


class ConnectBody(BaseModel):
    takeover_token: str


def _owner(request: Request) -> str:
    _require_token(request)
    return owner_principal()


def _http(exc: AgentComputerError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=error_payload(exc))


@router.get("/api/agent-computers")
def list_computers(request: Request):
    try:
        return get_contract().list_computers(_owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/ensure")
def ensure_computer(request: Request, body: EnsureBody):
    try:
        return get_contract().ensure(body.profile_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.get("/api/agent-computers/{computer_id}")
def get_computer(request: Request, computer_id: str):
    try:
        return get_contract().status(computer_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/wake")
def wake_computer(request: Request, computer_id: str):
    try:
        return get_contract().wake(computer_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/sleep")
def sleep_computer(request: Request, computer_id: str):
    try:
        return get_contract().sleep(computer_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/observe")
def observe_computer(request: Request, computer_id: str, body: LeaseBody):
    try:
        return get_contract().observe(
            computer_id,
            _owner(request),
            lease_id=body.lease_id,
            fencing_epoch=body.fencing_epoch,
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/act")
def act_computer(request: Request, computer_id: str, body: LeaseBody):
    try:
        return get_contract().act(
            computer_id,
            _owner(request),
            lease_id=body.lease_id,
            fencing_epoch=body.fencing_epoch,
            kind=body.kind,
            target=body.target,
            text=body.text,
            action_class=body.action_class,
            x=body.x,
            y=body.y,
            key=body.key,
            code=body.code,
            delta_x=body.delta_x,
            delta_y=body.delta_y,
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/takeover")
def takeover_computer(request: Request, computer_id: str, body: Optional[TakeoverBody] = None):
    try:
        return get_contract().request_takeover(
            computer_id, _owner(request), reason=(body.reason if body else "")
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/takeover/connect")
def connect_takeover(request: Request, computer_id: str, body: ConnectBody):
    try:
        return get_contract().connect_takeover(
            computer_id, _owner(request), takeover_token=body.takeover_token
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/give-back")
def give_back(request: Request, computer_id: str, body: LeaseBody):
    try:
        return get_contract().give_back(
            computer_id,
            _owner(request),
            lease_id=body.lease_id,
            fencing_epoch=body.fencing_epoch,
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/owner-disconnect")
def owner_disconnect(request: Request, computer_id: str):
    try:
        return get_contract().owner_disconnect(computer_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/agent-computers/{computer_id}/identities")
def attach_identity(request: Request, computer_id: str, body: AttachBody):
    try:
        return get_contract().attach_identity(computer_id, body.identity_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/browser-identities")
def create_identity(request: Request, body: IdentityCreateBody):
    try:
        return get_contract().create_identity(
            _owner(request), ownership=body.ownership, metadata=body.metadata
        )
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/browser-identities/{identity_id}/revoke")
def revoke_identity(request: Request, identity_id: str):
    try:
        return get_contract().revoke_identity(identity_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc


@router.post("/api/checkpoints/{checkpoint_id}/approve")
def approve_checkpoint(request: Request, checkpoint_id: str):
    try:
        return get_contract().approve_checkpoint(checkpoint_id, _owner(request))
    except AgentComputerError as exc:
        raise _http(exc) from exc
