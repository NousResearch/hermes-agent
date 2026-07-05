"""Prefetch personalized gig progress into Chatwoot turns.

On gig-related member messages, resolves the authenticated CRWD user id and
injects a compact ``[CRWD gig context]`` block with per-gig ``stage`` and
``next_step`` so the coach answers from real membership/progress data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from plugins.platforms.chatwoot.coach_context import (
    cross_user_request_active,
    resolve_member_crwd_id,
)

logger = logging.getLogger(__name__)

_GIG_INTENT_PATTERNS = (
    re.compile(r"\bnext steps?\b", re.I),
    re.compile(r"\bwhat should i do\b", re.I),
    re.compile(r"\bwhat(?:'s| is) my status\b", re.I),
    re.compile(r"\bmy gigs?\b", re.I),
    re.compile(r"\bcurrent gigs?\b", re.I),
    re.compile(r"\bgigs? i('ve| have) joined\b", re.I),
    re.compile(r"\bwaitlist(?:ed)? gigs?\b", re.I),
    re.compile(r"\bpending approval\b", re.I),
    re.compile(r"\b(?:my )?receipt\b", re.I),
    re.compile(r"\b(?:my )?proof\b", re.I),
    re.compile(r"\b(?:my )?review\b", re.I),
    re.compile(r"\b(?:my )?payout\b", re.I),
    re.compile(r"\b(?:my )?payment\b", re.I),
    re.compile(r"\bhow(?:'s| is) .+ going\b", re.I),
    re.compile(r"\bgig details\b", re.I),
    re.compile(r"\btell me about .+ gig\b", re.I),
    re.compile(r"\bwhere am i\b", re.I),
    re.compile(r"\bwhat(?:'s| is) left\b", re.I),
)

_WAITLIST_PATTERNS = (
    re.compile(r"\bwaitlist(?:ed)?\b", re.I),
    re.compile(r"\bpending approval\b", re.I),
    re.compile(r"\bwaiting (?:for )?approval\b", re.I),
)

_AMBIGUOUS_FALLBACK = re.compile(
    r"\b(what now|help me|what do i do|i'm stuck|im stuck|status)\b",
    re.I,
)


def _is_chatwoot(platform: Any) -> bool:
    if str(platform or "").strip().lower() == "chatwoot":
        return True
    try:
        from gateway.session_context import get_session_env

        return (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower() == "chatwoot"
    except Exception:
        return False


def _matches(message: str, patterns) -> bool:
    return any(p.search(message) for p in patterns)


def should_prefetch_gig_context(user_message: str) -> bool:
    """Return True when the inbound message likely needs gig progress data."""
    msg = (user_message or "").strip()
    if not msg:
        return False
    if _matches(msg, _GIG_INTENT_PATTERNS):
        return True
    return bool(_AMBIGUOUS_FALLBACK.search(msg))


def _extract_gig_name(message: str) -> str:
    text = (message or "").strip()
    for prefix in (
        "next steps for ",
        "status for ",
        "tell me about ",
        "how is ",
        "how's ",
    ):
        if text.lower().startswith(prefix):
            name = text[len(prefix):].strip(" ?.")
            if name.lower() not in {"you", "yourself", "u", "me", "my gig", "my gigs"}:
                return name
    quoted = re.search(r'"([^"]+)"', text)
    if quoted:
        return quoted.group(1).strip()
    return ""


def build_gig_context_block(
    user_id: str,
    user_message: str = "",
    *,
    limit: int = 5,
) -> Optional[str]:
    """Fetch gig status and format the injection block, or None on failure."""
    if not user_id:
        return None
    try:
        from tools.crwd_db_tool import build_user_gig_status
    except Exception as exc:
        logger.debug("[crwd-gig-ctx] import failed: %s", exc)
        return None

    include_waitlisted = _matches(user_message, _WAITLIST_PATTERNS)
    gig_name = _extract_gig_name(user_message)

    try:
        payload = build_user_gig_status(
            user_id,
            gig_name=gig_name,
            include_waitlisted=include_waitlisted,
            limit=limit,
        )
    except Exception as exc:
        logger.debug("[crwd-gig-ctx] build_user_gig_status failed: %s", exc)
        return None

    items = payload.get("items") or []
    if not items and not include_waitlisted:
        # Ambiguous fallback: prefetch only when member has a small active set.
        if not _AMBIGUOUS_FALLBACK.search(user_message or ""):
            return None
        try:
            payload = build_user_gig_status(user_id, limit=limit)
            items = payload.get("items") or []
        except Exception:
            return None
        if not items or len(items) > 3:
            return None

    if not items:
        return None

    slim = {
        "active_gigs": [
            {
                "gig_id": row.get("gig_id"),
                "gig_name": row.get("gig_name"),
                "gig_type": row.get("gig_type"),
                "stage": row.get("stage"),
                "next_step": row.get("next_step"),
                "buy_link": row.get("buy_link"),
                "handoff_recommended": row.get("handoff_recommended"),
            }
            for row in items
        ],
        "count": len(items),
    }
    return "\n".join([
        "[CRWD gig context]",
        "Source: get_user_gig_status (crwd_staging). Answer from this data; "
        "do not give generic lifecycle steps when a next_step is present.",
        json.dumps(slim, indent=2, default=str),
    ])


def gig_context_hook(**kwargs: Any) -> Optional[Dict[str, str]]:
    """``pre_llm_call`` hook: inject personalized gig progress when relevant."""
    try:
        if not _is_chatwoot(kwargs.get("platform")):
            return None
        if not os.getenv("CRWD_MONGO_URI"):
            return None
        if cross_user_request_active():
            return None

        user_message = str(kwargs.get("user_message") or "")
        if not should_prefetch_gig_context(user_message):
            return None

        contact_id = str(kwargs.get("sender_id") or "").strip()
        if not contact_id:
            return None
        user_id = resolve_member_crwd_id(contact_id)
        if not user_id:
            return None

        block = build_gig_context_block(user_id, user_message)
        if not block:
            return None
        return {"context": block}
    except Exception as exc:
        logger.debug("[crwd-gig-ctx] hook failed: %s", exc)
        return None
