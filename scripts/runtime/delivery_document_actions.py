"""Canonical helpers for tokenized document workspace actions.

Public document templates should use the same action vocabulary everywhere:
comments are direct, while approve/reject/sign require OTP verification through
recipient-bound delivery channels before the event is queued for trusted ingest.
"""
from __future__ import annotations

from typing import Any

CANONICAL_DOCUMENT_ACTIONS = {
    "comment": "commented",
    "commented": "commented",
    "approve": "approved",
    "approved": "approved",
    "reject": "rejected",
    "rejected": "rejected",
    "sign": "signed",
    "signed": "signed",
}

DOCUMENT_ACTION_EVENT_TYPES = {"commented", "approved", "rejected", "signed"}
OTP_REQUIRED_DOCUMENT_EVENT_TYPES = {"approved", "rejected", "signed"}
COMMENT_ONLY_DOCUMENT_EVENT_TYPES = {"commented"}


def normalize_document_action(value: Any) -> str | None:
    return CANONICAL_DOCUMENT_ACTIONS.get(str(value or "").strip().lower())


def document_action_requires_otp(event_type: Any) -> bool:
    return normalize_document_action(event_type) in OTP_REQUIRED_DOCUMENT_EVENT_TYPES


def document_action_allows_direct_post(event_type: Any) -> bool:
    return normalize_document_action(event_type) in COMMENT_ONLY_DOCUMENT_EVENT_TYPES


def clean_document_comment(value: Any, max_len: int = 4000) -> str | None:
    clean = str(value or "").strip()
    return clean[:max_len] if clean else None


def build_document_event(
    payload: dict[str, Any],
    *,
    token_ref: str | None,
    ip_address: str | None,
    user_agent: str | None,
    status: str = "pending_agent_ingest",
) -> dict[str, Any]:
    event_type = normalize_document_action(payload.get("event_type") or payload.get("action"))
    if not event_type:
        raise ValueError("invalid_document_action")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("invalid_metadata")
    return {
        "event_type": event_type,
        "deliverable_id": str(payload.get("deliverable_id") or "").strip(),
        "token_ref": token_ref,
        "actor_type": str(payload.get("actor_type") or "client").strip() or "client",
        "actor_ref": payload.get("actor_ref"),
        "ip_address": ip_address,
        "user_agent": user_agent,
        "comment": clean_document_comment(payload.get("comment")),
        "metadata": dict(metadata),
        "status": status,
    }
