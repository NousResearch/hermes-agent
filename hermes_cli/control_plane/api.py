"""Versioned HTTP API for the managed-node control plane."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .nodes import (
    ConcurrencyConflict,
    CredentialConflict,
    CredentialIssuance,
    IdempotencyConflict,
    InvalidTransition,
    NodeRegistry,
)

router = APIRouter(prefix="/api/control-plane/v1", tags=["control-plane"])
LifecycleState = Literal["enrolled", "active", "quarantined", "recovering", "retired"]


class NodeEnrollment(BaseModel):
    enrollment_key: str
    node_id: str | None = None
    role: str
    owner: str
    actor: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


class NodeTransition(BaseModel):
    state: str
    actor: str
    expected_revision: int
    reason: str


class NodeAuthentication(BaseModel):
    credential: str


class CredentialMutation(BaseModel):
    actor: str
    expected_credential_revision: int


class NodeView(BaseModel):
    id: str
    enrollment_key: str
    role: str
    owner: str
    state: LifecycleState
    capabilities: dict[str, Any]
    revision: int
    created_at: int
    updated_at: int
    credential_revision: int
    credential_status: Literal["active", "revoked"]
    credential_issued_at: int
    credential_rotated_at: int | None
    credential_revoked_at: int | None


class CredentialIssuanceView(BaseModel):
    node: NodeView
    credential: str | None = Field(
        description="Raw credential, present only at initial issuance or rotation"
    )


class AuthenticationResult(BaseModel):
    authenticated: Literal[True]


class NodeList(BaseModel):
    nodes: list[NodeView]


class NodeEventView(BaseModel):
    sequence: int
    node_id: str
    event_type: str
    actor: str
    from_state: LifecycleState | None
    to_state: LifecycleState
    node_revision: int
    occurred_at: int
    details: dict[str, Any]
    previous_hash: str | None
    event_hash: str


class NodeHistory(BaseModel):
    events: list[NodeEventView]


class AuditResult(BaseModel):
    valid: bool


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request"},
    401: {"model": ErrorResponse, "description": "Node authentication failed"},
    404: {"model": ErrorResponse, "description": "Managed node not found"},
    409: {"model": ErrorResponse, "description": "Lifecycle or revision conflict"},
}


def _registry() -> NodeRegistry:
    return NodeRegistry()


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _issuance(value: CredentialIssuance) -> dict[str, Any]:
    return {"node": asdict(value.node), "credential": value.credential}


@router.post("/nodes", response_model=CredentialIssuanceView, responses=ERROR_RESPONSES)
async def enroll_node(body: NodeEnrollment):
    """Enroll a node, or return its existing identity on an identical retry."""
    try:
        node = _registry().enroll(**body.model_dump())
    except IdempotencyConflict as exc:
        return _error(409, "enrollment_conflict", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))
    return _issuance(node)


@router.post(
    "/nodes/{node_id}/authenticate",
    response_model=AuthenticationResult,
    responses=ERROR_RESPONSES,
)
async def authenticate_node(node_id: str, body: NodeAuthentication):
    try:
        authenticated = _registry().authenticate(node_id, body.credential)
    except ValueError:
        authenticated = False
    if not authenticated:
        return _error(401, "invalid_node_credential", "node authentication failed")
    return {"authenticated": True}


@router.post(
    "/nodes/{node_id}/credential/rotate",
    response_model=CredentialIssuanceView,
    responses=ERROR_RESPONSES,
)
async def rotate_node_credential(node_id: str, body: CredentialMutation):
    try:
        issuance = _registry().rotate_credential(
            node_id,
            actor=body.actor,
            expected_credential_revision=body.expected_credential_revision,
        )
    except KeyError:
        return _error(404, "node_not_found", f"managed node not found: {node_id}")
    except CredentialConflict as exc:
        return _error(409, "credential_revision_conflict", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))
    return _issuance(issuance)


@router.post(
    "/nodes/{node_id}/credential/revoke",
    response_model=NodeView,
    responses=ERROR_RESPONSES,
)
async def revoke_node_credential(node_id: str, body: CredentialMutation):
    try:
        node = _registry().revoke_credential(
            node_id,
            actor=body.actor,
            expected_credential_revision=body.expected_credential_revision,
        )
    except KeyError:
        return _error(404, "node_not_found", f"managed node not found: {node_id}")
    except CredentialConflict as exc:
        return _error(409, "credential_revision_conflict", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))
    return asdict(node)


@router.get("/nodes", response_model=NodeList, responses=ERROR_RESPONSES)
async def list_nodes(state: str | None = Query(default=None)):
    try:
        nodes = _registry().list(state=state)
    except ValueError as exc:
        return _error(400, "invalid_state", str(exc))
    return {"nodes": [asdict(node) for node in nodes]}


@router.get("/nodes/{node_id}", response_model=NodeView, responses=ERROR_RESPONSES)
async def get_node(node_id: str):
    node = _registry().get(node_id)
    if node is None:
        return _error(404, "node_not_found", f"managed node not found: {node_id}")
    return asdict(node)


@router.post(
    "/nodes/{node_id}/transitions",
    response_model=NodeView,
    responses=ERROR_RESPONSES,
)
async def transition_node(node_id: str, body: NodeTransition):
    try:
        node = _registry().transition(
            node_id,
            body.state,
            actor=body.actor,
            expected_revision=body.expected_revision,
            reason=body.reason,
        )
    except KeyError:
        return _error(404, "node_not_found", f"managed node not found: {node_id}")
    except ConcurrencyConflict as exc:
        return _error(409, "revision_conflict", str(exc))
    except InvalidTransition as exc:
        return _error(409, "invalid_transition", str(exc))
    except ValueError as exc:
        return _error(400, "invalid_request", str(exc))
    return asdict(node)


@router.get(
    "/nodes/{node_id}/history",
    response_model=NodeHistory,
    responses=ERROR_RESPONSES,
)
async def node_history(node_id: str):
    registry = _registry()
    if registry.get(node_id) is None:
        return _error(404, "node_not_found", f"managed node not found: {node_id}")
    return {"events": [asdict(event) for event in registry.history(node_id)]}


@router.get("/audit", response_model=AuditResult)
async def verify_audit():
    return {"valid": _registry().verify_audit_chain()}
