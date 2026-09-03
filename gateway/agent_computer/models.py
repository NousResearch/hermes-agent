"""Durable AgentComputer / BrowserIdentity domain objects.

AgentComputer and BrowserIdentity are separate objects on purpose:
a permanent Hermes profile owns one computer; that computer may attach
at most one BrowserIdentity at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


OWNER_PRINCIPAL = "owner"


def agent_principal(profile_id: str) -> str:
    return f"agent:{profile_id}"


def is_owner_principal(principal: str) -> bool:
    return principal == OWNER_PRINCIPAL or principal.startswith("owner:")


def is_agent_principal(principal: str, profile_id: str) -> bool:
    return principal == agent_principal(profile_id)


class Lifecycle(str, Enum):
    IDLE = "idle"
    WAKING = "waking"
    READY = "ready"
    BUSY = "busy"
    SLEEPING = "sleeping"


class ControlAuthority(str, Enum):
    AGENT_CONTROLLED = "AGENT_CONTROLLED"
    TAKEOVER_PENDING = "TAKEOVER_PENDING"
    YIELDING = "YIELDING"
    OWNER_CONTROLLED = "OWNER_CONTROLLED"
    RETURNING = "RETURNING"


class Controller(str, Enum):
    AGENT = "agent"
    OWNER = "owner"


class LeaseStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class CheckpointStatus(str, Enum):
    BLOCKED = "blocked"
    APPROVED = "approved"
    DENIED = "denied"
    CONSUMED = "consumed"


SENSITIVE_ACTION_CLASSES = frozenset(
    {
        "payment",
        "transfer",
        "credential_reveal",
        "account_change",
    }
)

AUDIT_EVENT_TYPES = frozenset(
    {
        "computer_provision",
        "runtime_start",
        "runtime_sleep",
        "browser_identity_attach",
        "browser_identity_detach",
        "identity_revoked",
        "input_accepted",
        "stale_controller_rejected",
        "takeover_requested",
        "agent_yielded",
        "owner_takeover_granted",
        "takeover_connected",
        "control_returned",
        "takeover_expired",
        "fencing_recovery",
        "recovery",
        "checkpoint_blocked",
        "checkpoint_approved",
        "owner_disconnect",
    }
)


@dataclass
class AgentComputer:
    id: str
    agent_profile_id: str
    backend: str
    persistence_ref: str
    lifecycle: Lifecycle = Lifecycle.IDLE
    control_authority: ControlAuthority = ControlAuthority.AGENT_CONTROLLED
    fencing_epoch: int = 0
    resume_observe_required: bool = False
    active_browser_identity_id: str | None = None
    workspace_url: str = ""
    workspace_title: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class BrowserIdentity:
    id: str
    profile_ref: str
    ownership: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    revoked: bool = False
    lock_computer_id: str | None = None
    lock_holder: str | None = None

    def allows(self, profile_id: str) -> bool:
        return profile_id in self.ownership


@dataclass
class ControlLease:
    lease_id: str
    computer_id: str
    controller: Controller
    fencing_epoch: int
    acquired_at: str
    expires_at: str | None = None
    status: LeaseStatus = LeaseStatus.ACTIVE


@dataclass
class TakeoverToken:
    token_id: str
    token_hash: str
    computer_id: str
    owner_principal: str
    fencing_epoch: int
    expires_at: str
    consumed: bool = False


@dataclass
class Checkpoint:
    id: str
    computer_id: str
    action_class: str
    status: CheckpointStatus
    created_at: str


@dataclass
class AuditEvent:
    event_type: str
    actor: str
    detail: dict[str, Any]
    created_at: str
    computer_id: str | None = None
    id: int | None = None


@dataclass
class Observation:
    url: str
    title: str
    text: str
    fencing_epoch: int
    controller: str
    observed_at: str
    screenshot_b64: str = ""
    screenshot_mime: str = ""
    screenshot_width: int = 0
    screenshot_height: int = 0
    viewport_width: int = 0
    viewport_height: int = 0
    device_pixel_ratio: float = 1.0


@dataclass
class InputReceipt:
    accepted: bool
    fencing_epoch: int
    url: str
    title: str
    text: str
