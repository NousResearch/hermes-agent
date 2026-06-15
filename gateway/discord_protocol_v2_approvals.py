"""Durable approvals for Discord Native Multi-Bot Protocol v2.

This module is deliberately small: it stores approval envelopes in the protocol
v2 SQLite store, renders/decodes opaque Discord component IDs, and performs
idempotent approve/deny transitions with append-only audit for first decisions.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Literal

from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.secret_refs import redact_sensitive_data

ApprovalDecision = Literal["approve", "deny"]
_APPROVAL_ID_RE = re.compile(r"^apv_[A-Za-z0-9_-]{16,64}$")
_COMPONENT_CUSTOM_ID_LIMIT = 100
_DECISION_TO_STATUS: dict[ApprovalDecision, str] = {
    "approve": "approved",
    "deny": "denied",
}
_SAFE_APPROVAL_ID_LABEL = "<redacted_structured_approval_id>"


@dataclass(frozen=True)
class DiscordProtocolV2ApprovalDecisionResult:
    ok: bool
    status: str
    message: str
    approval: dict[str, Any] | None = None
    duplicate: bool = False


def new_approval_id() -> str:
    """Return an opaque, stable-once-stored approval id suitable for Discord IDs."""

    return f"apv_{uuid.uuid4().hex}"


def is_valid_approval_id(approval_id: str) -> bool:
    return bool(_APPROVAL_ID_RE.fullmatch(str(approval_id or "")))


def safe_approval_id_label(approval_id: str) -> str:
    """Return a log/response-safe label for opaque v2 approval IDs."""

    # Approval IDs can be pasted from Discord custom IDs and are treated as
    # bearer-like opaque capabilities.  Never echo the raw value in responses or
    # audit payloads, including syntactically valid but unknown IDs.
    return _SAFE_APPROVAL_ID_LABEL


def create_component_custom_id(decision: ApprovalDecision, approval_id: str) -> str:
    if decision not in _DECISION_TO_STATUS:
        raise ValueError("decision must be approve or deny")
    if not is_valid_approval_id(approval_id):
        raise ValueError("invalid approval_id")
    custom_id = f"hermes_v2_approval:{decision}:{approval_id}"
    if len(custom_id) > _COMPONENT_CUSTOM_ID_LIMIT:
        raise ValueError("Discord component custom_id exceeds 100 characters")
    return custom_id


def parse_component_custom_id(custom_id: str) -> tuple[ApprovalDecision, str] | None:
    text = str(custom_id or "")
    if len(text) > _COMPONENT_CUSTOM_ID_LIMIT:
        return None
    parts = text.split(":", 2)
    if len(parts) != 3 or parts[0] != "hermes_v2_approval":
        return None
    decision = parts[1]
    approval_id = parts[2]
    if decision not in _DECISION_TO_STATUS or not is_valid_approval_id(approval_id):
        return None
    return decision, approval_id  # type: ignore[return-value]


def create_pending_approval(
    store: DiscordProtocolV2Store,
    *,
    topic_id: str,
    agent_id: str,
    requesting_event_id: str,
    approval_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a pending v2 approval envelope and return the stored row."""

    effective_approval_id = approval_id or new_approval_id()
    if not is_valid_approval_id(effective_approval_id):
        raise ValueError("approval_id must be opaque apv_[A-Za-z0-9_-]{16,64}")
    if not str(topic_id or ""):
        raise ValueError("topic_id is required")
    if not str(agent_id or ""):
        raise ValueError("agent_id is required")
    if not str(requesting_event_id or ""):
        raise ValueError("requesting_event_id is required")

    return store.upsert_approval(
        approval_id=effective_approval_id,
        target_agent_id=str(agent_id),
        agent_id=str(agent_id),
        topic_id=str(topic_id),
        requesting_event_id=str(requesting_event_id),
        status="pending",
        payload=_safe_payload(payload or {}),
    )


def decide_approval(
    store: DiscordProtocolV2Store,
    *,
    approval_id: str,
    decision: ApprovalDecision,
    actor_user_id: str | None = None,
    audit_payload: dict[str, Any] | None = None,
) -> DiscordProtocolV2ApprovalDecisionResult:
    """Approve or deny a durable approval exactly once.

    Replaying the same decision after restart is idempotent and does not append a
    duplicate success audit event. A conflicting second decision is a no-op.
    """

    if decision not in _DECISION_TO_STATUS:
        raise ValueError("decision must be approve or deny")
    safe_approval_id = safe_approval_id_label(approval_id)
    if not is_valid_approval_id(approval_id):
        return DiscordProtocolV2ApprovalDecisionResult(
            False, "unknown", f"Unknown approval action: {safe_approval_id}"
        )

    row = store.get_approval(approval_id)
    if row is None:
        return DiscordProtocolV2ApprovalDecisionResult(
            False, "unknown", f"Unknown approval action: {safe_approval_id}"
        )

    desired_status = _DECISION_TO_STATUS[decision]
    current_status = str(row.get("status") or "")
    if current_status != "pending":
        return DiscordProtocolV2ApprovalDecisionResult(
            current_status == desired_status,
            current_status,
            f"Approval action {safe_approval_id} is already {current_status}.",
            row,
            duplicate=current_status == desired_status,
        )

    now = _now()
    cur = store.conn.execute(
        """
        UPDATE approvals
        SET status = ?, updated_at = ?, version = version + 1
        WHERE approval_id = ? AND status = 'pending'
        """,
        (desired_status, now, approval_id),
    )
    if cur.rowcount != 1:
        store.conn.commit()
        refreshed = store.get_approval(approval_id)
        status = str((refreshed or {}).get("status") or "unknown")
        return DiscordProtocolV2ApprovalDecisionResult(
            status == desired_status,
            status,
            f"Approval action {safe_approval_id} is already {status}.",
            refreshed,
            duplicate=status == desired_status,
        )

    store.conn.execute(
        """
        INSERT INTO approval_audit_events (
            audit_event_id, approval_id, event_type, actor_user_id,
            status, payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            approval_id,
            "discord_v2_approval_approved" if decision == "approve" else "discord_v2_approval_denied",
            str(actor_user_id or "") or None,
            desired_status,
            json.dumps(_safe_payload(audit_payload or {}), sort_keys=True),
            now,
        ),
    )
    store.conn.commit()
    decided = store.get_approval(approval_id)
    return DiscordProtocolV2ApprovalDecisionResult(
        True,
        desired_status,
        f"Approval action {safe_approval_id} {desired_status}.",
        decided,
    )


def actor_is_primary_approver(
    *,
    user_id: str | None,
    role_ids: Iterable[Any] | None,
    owner_user_ids: Iterable[Any] | None,
    admin_role_ids: Iterable[Any] | None,
) -> bool:
    """Primary UI structured approvals are owner/admin only."""

    user = str(user_id or "")
    owners = {str(v) for v in (owner_user_ids or []) if str(v)}
    admins = {str(v) for v in (admin_role_ids or []) if str(v)}
    roles = {str(v) for v in (role_ids or []) if str(v)}
    return bool((user and user in owners) or (roles & admins))


def agent_has_capability_or_scope(
    store: DiscordProtocolV2Store,
    *,
    agent_id: str,
    capability: str | None = None,
    scope_key: str | None = None,
    scope_value: Any | None = None,
) -> bool:
    """Check the persisted identity registry without exposing token secrets."""

    identity = store.get_identity(agent_id)
    if identity is None or not int(identity.get("enabled") or 0):
        return False
    capabilities = _json_load(identity.get("capabilities_json"), [])
    scopes = _json_load(identity.get("scopes_json"), {})
    if capability and str(capability) not in {str(v) for v in capabilities or []}:
        return False
    if scope_key:
        if not isinstance(scopes, dict) or scope_key not in scopes:
            return False
        if scope_value is not None:
            value = scopes.get(scope_key)
            if isinstance(value, list):
                return str(scope_value) in {str(v) for v in value}
            return str(value) == str(scope_value)
    return True


def _safe_payload(value: Any) -> Any:
    # Reuse the central gateway/Hermes redactor so free-form string values under
    # innocuous keys (for example shell commands with Authorization headers or
    # sk-* tokens) are scrubbed before DB/audit persistence.
    return redact_sensitive_data(value)


def _json_load(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _now() -> str:
    return datetime.now(UTC).isoformat()
