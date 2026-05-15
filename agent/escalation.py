"""Escalation publisher — thin wrapper over the Hermes Internal Message Bus.

All escalations go through hermes_event_bus.publish_event() which:
  1. Writes durably to SQLite FIRST
  2. Publishes to MQTT for real-time delivery

This module is kept as a compatibility shim so existing call sites
(credential_pool.py, auxiliary_client.py, etc.) continue to work
without modification.

Usage:
    from agent.escalation import escalate
    escalate(summary="OpenRouter 402", severity="high", category="payment")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _detect_agent() -> str:
    """Best-effort agent name detection."""
    explicit = os.getenv("HERMES_PROFILE", "").strip()
    if explicit:
        return explicit
    cwd = os.getcwd()
    if ".hermes/profiles/" in cwd:
        parts = cwd.split(".hermes/profiles/")
        if len(parts) > 1:
            return parts[1].split("/")[0]
    return "unknown"


def _severity_to_priority(severity: str) -> int:
    """Map escalation severity to bus priority."""
    mapping = {"critical": 1, "high": 2, "medium": 3, "low": 5}
    return mapping.get(severity.lower(), 3)


def _category_to_bus_category(category: str) -> str:
    """Map old escalation category to bus category."""
    mapping = {
        "payment": "system",
        "auth": "guardrail",
        "platform": "system",
        "model": "capability",
        "tool": "guardrail",
    }
    return mapping.get(category.lower(), "system")


def escalate(
    summary: str,
    *,
    severity: str = "medium",
    category: str = "unknown",
    details: str = "",
    user_request: str = "",
    attempted: Optional[List[str]] = None,
    log_path: str = "",
    skill_path: str = "",
    custom: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None,
) -> Optional[int]:
    """
    Publish an escalation to the Hermes Internal Message Bus.

    Returns the event row id on success, None on failure.
    """
    _agent = (agent or _detect_agent()).strip() or "unknown"

    payload: Dict[str, Any] = {
        "summary": summary,
        "details": details,
        "user_request": user_request,
        "attempted": attempted or [],
        "log_path": log_path,
        "skill_path": skill_path,
        "legacy_severity": severity,
        "legacy_category": category,
    }
    if custom:
        payload.update(custom)

    try:
        from hermes_event_bus import publish_event
        event_id = publish_event(
            category=_category_to_bus_category(category),
            event_type="escalation",
            source_agent=_agent,
            target_agent="phoenix",
            priority=_severity_to_priority(severity),
            payload=payload,
        )
        return event_id
    except Exception as exc:
        logger.warning("Escalation publish failed: %s", exc)
        return None


# Convenience wrappers

def escalate_payment(
    summary: str,
    details: str = "",
    attempted: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[int]:
    return escalate(summary, severity="high", category="payment", details=details, attempted=attempted, **kwargs)


def escalate_auth(
    summary: str,
    details: str = "",
    attempted: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[int]:
    return escalate(summary, severity="high", category="auth", details=details, attempted=attempted, **kwargs)


def escalate_platform(
    summary: str,
    details: str = "",
    attempted: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[int]:
    return escalate(summary, severity="critical", category="platform", details=details, attempted=attempted, **kwargs)


def escalate_model(
    summary: str,
    details: str = "",
    attempted: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[int]:
    return escalate(summary, severity="high", category="model", details=details, attempted=attempted, **kwargs)


def escalate_tool(
    summary: str,
    details: str = "",
    attempted: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[int]:
    return escalate(summary, severity="medium", category="tool", details=details, attempted=attempted, **kwargs)
