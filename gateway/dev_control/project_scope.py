"""Shared helpers for Dev control project scoping."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_PROJECT_ID = "OrynWorkspace"


def normalize_project_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def resolve_project_id(*candidates: Any, default: str = DEFAULT_PROJECT_ID) -> str:
    for candidate in candidates:
        normalized = normalize_project_id(candidate)
        if normalized:
            return normalized
    return default


def project_id_from_payload(payload: Optional[dict[str, Any]]) -> str:
    if not isinstance(payload, dict):
        return DEFAULT_PROJECT_ID
    context = payload.get("project_context")
    context_id = None
    if isinstance(context, dict):
        context_id = context.get("project_id")
    return resolve_project_id(payload.get("project_id"), context_id)
