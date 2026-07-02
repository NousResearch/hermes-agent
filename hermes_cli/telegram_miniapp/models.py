"""Typed public contracts for the Telegram Mini App M2 sidecar.

The implementation currently uses plain dictionaries for FastAPI responses to
keep M2 small; these aliases document the stable response shape for callers and
tests without introducing extra runtime behavior.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GatewayStatus(TypedDict):
    running: bool
    state: str
    busy: bool
    drainable: bool
    active_agents: int
    restart_requested: bool


class MiniAppStatus(TypedDict):
    mode: str
    actions_enabled: bool
    public_exposure: bool


class StatusSnapshot(TypedDict):
    ok: bool
    updated_at: str
    hermes_home: str
    gateway: GatewayStatus
    miniapp: MiniAppStatus


class PreviewSnapshotMeta(TypedDict):
    source: str
    source_label: str
    redaction: str
    contains_live_actions: bool


class CapabilityItem(TypedDict):
    id: str
    label: str
    enabled: bool
    mode: str
    reason: str


class CapabilitiesSnapshot(TypedDict):
    ok: bool
    meta: PreviewSnapshotMeta
    items: list[CapabilityItem]


class ApprovalItem(TypedDict):
    id: str
    title: str
    source: str
    risk: str
    summary: str
    requested_at: str
    status: str
    checks: list[str]


class ApprovalsSnapshot(TypedDict):
    ok: bool
    meta: PreviewSnapshotMeta
    items: list[ApprovalItem]


# Dormant M19 action-gate contracts. These document a future Phase 1
# approve/reject-once shape only; they do not create routes, handlers, gateway
# adapters, or frontend fetch calls.
ActionDecisionValue = Literal["approve_once", "reject_once"]


class ActionDecisionRequest(TypedDict):
    decision: ActionDecisionValue
    client_request_id: str
    snapshot_version: str


class ActionDecisionResponse(TypedDict):
    ok: bool
    decision_id: str
    status: str
    message: str


class ActionReadyApprovalItem(ApprovalItem, total=False):
    expires_at: str
    snapshot_version: str
    allowed_decisions: list[ActionDecisionValue]


class SessionPreviewItem(TypedDict):
    id: str
    agent: str
    state: str
    meta: str
    time: str
    tone: str


class SessionsSnapshot(TypedDict):
    ok: bool
    meta: PreviewSnapshotMeta
    items: list[SessionPreviewItem]


class LogPreviewItem(TypedDict):
    level: str
    message: str
    time: str


class LogsSnapshot(TypedDict):
    ok: bool
    meta: PreviewSnapshotMeta
    items: list[LogPreviewItem]


JsonDict = dict[str, Any]
