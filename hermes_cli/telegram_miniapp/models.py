"""Typed public contracts for the Telegram Mini App M2 sidecar.

The implementation currently uses plain dictionaries for FastAPI responses to
keep M2 small; these aliases document the stable response shape for callers and
tests without introducing extra runtime behavior.
"""

from __future__ import annotations

from typing import Any, TypedDict


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
    items: list[ApprovalItem]


class SessionPreviewItem(TypedDict):
    id: str
    agent: str
    state: str
    meta: str
    time: str
    tone: str


class SessionsSnapshot(TypedDict):
    ok: bool
    items: list[SessionPreviewItem]


class LogPreviewItem(TypedDict):
    level: str
    message: str
    time: str


class LogsSnapshot(TypedDict):
    ok: bool
    items: list[LogPreviewItem]


JsonDict = dict[str, Any]
