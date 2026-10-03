"""Scoped opted-in cron delivery handoff for an already-running turn.

Records trusted provenance at the moment a successful attach/mirror/seed
lands, keyed by exact ``session_id``. An accepted busy redirect moves the
pending brief onto the running agent as provider-facing reference context
(not a new user instruction). Fail-open: never raise.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pending: dict[str, list[dict[str, Any]]] = {}

_AGENT_SLOT = "_pending_cron_delivery_references"
_REFERENCE_MARKER = "[Reference context from a cron delivery attached to this session]"


def record_opted_in_cron_handoff(
    *,
    session_id: Optional[str],
    job: Optional[dict] = None,
    job_id: Optional[str] = None,
    brief_text: Optional[str] = None,
    provenance: str = "cron",
) -> None:
    """Record a scoped handoff after a successful opted-in persist. Never raises."""
    try:
        if not session_id:
            return
        if job is not None and job.get("attach_to_session") is not True:
            return
        text = (brief_text or "").strip()
        if not text:
            return
        resolved_job_id = job_id
        if not resolved_job_id and isinstance(job, dict):
            resolved_job_id = job.get("id")
        entry = {
            "job_id": str(resolved_job_id or "cron"),
            "brief_text": text,
            "provenance": provenance,
            "session_id": str(session_id),
        }
        key = str(session_id)
        with _lock:
            bucket = _pending.setdefault(key, [])
            if any(
                item.get("job_id") == entry["job_id"] and item.get("brief_text") == entry["brief_text"]
                for item in bucket
            ):
                return
            bucket.append(entry)
    except Exception:
        logger.debug("cron delivery handoff record failed", exc_info=True)


def take_opted_in_cron_handoffs(session_id: Optional[str]) -> List[dict[str, Any]]:
    """Consume pending handoffs for this exact session_id. Empty on any error."""
    try:
        if not session_id:
            return []
        with _lock:
            return list(_pending.pop(str(session_id), []) or [])
    except Exception:
        logger.debug("cron delivery handoff take failed", exc_info=True)
        return []


def attach_handoffs_to_agent(agent: Any, handoffs: Iterable[dict[str, Any]]) -> None:
    """Move consumed session handoffs onto the running agent. Fail-open."""
    try:
        items = [
            h for h in (handoffs or [])
            if isinstance(h, dict) and (h.get("brief_text") or "").strip()
        ]
        if agent is None or not items:
            return
        existing = getattr(agent, _AGENT_SLOT, None)
        if not isinstance(existing, list):
            existing = []
            setattr(agent, _AGENT_SLOT, existing)
        existing.extend(items)
    except Exception:
        logger.debug("cron delivery handoff attach failed", exc_info=True)


def drain_agent_delivery_references(agent: Any) -> List[dict[str, Any]]:
    """Consume agent-scoped delivery reference context. Empty on any error."""
    try:
        items = getattr(agent, _AGENT_SLOT, None)
        setattr(agent, _AGENT_SLOT, [])
        if not isinstance(items, list):
            return []
        return [h for h in items if isinstance(h, dict)]
    except Exception:
        logger.debug("cron delivery handoff drain failed", exc_info=True)
        return []


def format_delivery_reference_block(handoffs: Iterable[dict[str, Any]]) -> str:
    """Labeled provider-facing reference block; never a user instruction."""
    parts: List[str] = []
    for handoff in handoffs or []:
        if not isinstance(handoff, dict):
            continue
        brief = (handoff.get("brief_text") or "").strip()
        if not brief:
            continue
        job_id = handoff.get("job_id") or "cron"
        parts.append(f"{_REFERENCE_MARKER}\njob_id={job_id}\n{brief}")
    return "\n\n".join(parts)
