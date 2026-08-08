"""Executive Integration Layer — types module.

This module defines all frozen dataclasses and enums used by the
Executive Integration Layer. No logic, no I/O. Pure data.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple


SCHEMA_VERSION = "eil.v1"


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────


class RouteKind(str, Enum):
    """The 5 possible routes a user message can take."""

    CHAT = "chat"
    TOOL = "tool"
    EXECUTIVE = "executive"
    CLARIFY = "clarify"
    REJECT = "reject"


class LaunchStatus(str, Enum):
    """The status of an ExecutiveLaunchRequest."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTION_PREVIEW_READY = "execution_preview_ready"
    LAUNCHED = "launched"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SummaryKind(str, Enum):
    """The kind of user-facing summary."""

    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


# ──────────────────────────────────────────────────────────────────────
# Frozen dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObjectiveGatewayDecision:
    """Output of ObjectiveGateway. Immutable."""

    route_kind: RouteKind
    objective_id: Optional[str]
    confidence: float
    rationale: str
    matched_keywords: Tuple[str, ...]
    matched_intent: Optional[str]
    intent_routing_strategy: Optional[str]
    fallback_used: bool
    fingerprint: str
    created_at: str
    created_by: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_kind": self.route_kind.value,
            "objective_id": self.objective_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "matched_keywords": list(self.matched_keywords),
            "matched_intent": self.matched_intent,
            "intent_routing_strategy": self.intent_routing_strategy,
            "fallback_used": self.fallback_used,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ExecutiveLaunchRequest:
    """Structured payload for operator approval."""

    request_id: str
    objective_text: str
    objective_id: Optional[str]
    expected_phases: Tuple[str, ...]
    estimated_complexity: str
    risk_level: str
    risk_rationale: str
    requires_human_approval: bool
    keywords_matched: Tuple[str, ...]
    intent_routing_strategy: str
    gateway_decision_fingerprint: str
    user_summary: str
    approval_request_id: Optional[str]
    status: LaunchStatus
    created_at: str
    created_by: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "objective_text": self.objective_text,
            "objective_id": self.objective_id,
            "expected_phases": list(self.expected_phases),
            "estimated_complexity": self.estimated_complexity,
            "risk_level": self.risk_level,
            "risk_rationale": self.risk_rationale,
            "requires_human_approval": self.requires_human_approval,
            "keywords_matched": list(self.keywords_matched),
            "intent_routing_strategy": self.intent_routing_strategy,
            "gateway_decision_fingerprint": self.gateway_decision_fingerprint,
            "user_summary": self.user_summary,
            "approval_request_id": self.approval_request_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ExecutiveUserSummary:
    """Human-readable summary returned to the user."""

    request_id: str
    summary_kind: SummaryKind
    title: str
    body: str
    next_steps: Tuple[str, ...]
    warnings: Tuple[str, ...]
    details_url: Optional[str]
    fingerprint: str
    created_at: str
    created_by: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "summary_kind": self.summary_kind.value,
            "title": self.title,
            "body": self.body,
            "next_steps": list(self.next_steps),
            "warnings": list(self.warnings),
            "details_url": self.details_url,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
        }


@dataclass
class ExecutiveIntegrationMetrics:
    """Snapshot of EIL metrics. Updated by the EIL internally."""

    total_routes: int = 0
    chat_routes: int = 0
    tool_routes: int = 0
    executive_routes: int = 0
    clarify_routes: int = 0
    reject_routes: int = 0
    avg_route_confidence: float = 0.0
    avg_routing_latency_ms: float = 0.0
    launch_requests_created: int = 0
    launch_requests_approved: int = 0
    launch_requests_rejected: int = 0
    launch_requests_executed: int = 0
    launch_requests_failed: int = 0
    launch_requests_cancelled: int = 0
    per_intent_routing_strategy: Dict[str, int] = field(default_factory=dict)
    per_keyword_frequency: Dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


# ──────────────────────────────────────────────────────────────────────
# Deterministic helpers
# ──────────────────────────────────────────────────────────────────────


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_message(message: str) -> str:
    return " ".join((message or "").split()).strip()


def compute_decision_fingerprint(
    *,
    message: str,
    route_kind: RouteKind,
    matched_keywords: Tuple[str, ...],
    matched_intent: Optional[str],
    intent_routing_strategy: Optional[str],
) -> str:
    """sha256 of canonical inputs. Deterministic."""
    payload = {
        "message": _normalize_message(message),
        "route_kind": route_kind.value,
        "matched_keywords": sorted(matched_keywords),
        "matched_intent": matched_intent,
        "intent_routing_strategy": intent_routing_strategy,
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_launch_fingerprint(request: ExecutiveLaunchRequest) -> str:
    """sha256 of the launch request. Deterministic."""
    payload = {
        "request_id": request.request_id,
        "objective_text": request.objective_text,
        "objective_id": request.objective_id,
        "expected_phases": sorted(request.expected_phases),
        "estimated_complexity": request.estimated_complexity,
        "risk_level": request.risk_level,
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def new_request_id() -> str:
    """UUID4-based request id (deterministic when monkey-patched)."""
    return f"eil-{uuid.uuid4().hex}"


LAUNCH_FINGERPRINT_KEYS: Tuple[str, ...] = (
    "request_id",
    "objective_id",
    "estimated_complexity",
    "risk_level",
    "expected_phases",
)
