"""Intent router — auto-runs CRWD support queries before the LLM turn."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from ._utils import parse_object_id
from . import queries

logger = logging.getLogger(__name__)

_OBJECT_ID_RE = re.compile(r"\b[0-9a-fA-F]{24}\b")

_ACTIVE_GIGS_PATTERNS = (
    re.compile(r"\bactive gigs?\b", re.I),
    re.compile(r"\bopen gigs?\b", re.I),
    re.compile(r"\bavailable (gigs?|campaigns?)\b", re.I),
    re.compile(r"\bwhat can i join\b", re.I),
    re.compile(r"\bshow me gigs?\b", re.I),
)

_JOINED_GIGS_PATTERNS = (
    re.compile(r"\bmy joined gigs?\b", re.I),
    re.compile(r"\bgigs? i('ve| have) joined\b", re.I),
    re.compile(r"\bcurrent gigs?\b", re.I),
    re.compile(r"\bgigs? i am (in|on)\b", re.I),
)

_HISTORY_PATTERNS = (
    re.compile(r"\bgig history\b", re.I),
    re.compile(r"\bpast gigs?\b", re.I),
    re.compile(r"\bparticipation history\b", re.I),
    re.compile(r"\bmy gig history\b", re.I),
)

_PROFILE_PATTERNS = (
    re.compile(r"\bmy profile\b", re.I),
    re.compile(r"\buser profile\b", re.I),
    re.compile(r"\bprofile for\b", re.I),
)

_GIG_DETAILS_PATTERNS = (
    re.compile(r"\bgig details\b", re.I),
    re.compile(r"\btell me about\b", re.I),
    re.compile(r"\bdetails (for|about|on)\b", re.I),
    re.compile(r"\bwhat is (the )?gig\b", re.I),
)

_IDENTITY_PATTERNS = (
    re.compile(r"\btell me about (you|yourself|u)\b", re.I),
    re.compile(r"\bwho are you\b", re.I),
    re.compile(r"\bwhat are you\b", re.I),
    re.compile(r"\babout (you|yourself|u)\b", re.I),
)


def _matches(message: str, patterns) -> bool:
    return any(p.search(message) for p in patterns)


def _extract_object_id(message: str) -> Optional[str]:
    match = _OBJECT_ID_RE.search(message or "")
    return match.group(0) if match else None


def _extract_gig_name(message: str) -> Optional[str]:
    text = (message or "").strip()
    for prefix in ("tell me about ", "details about ", "details for ", "gig "):
        if text.lower().startswith(prefix):
            name = text[len(prefix):].strip(" ?.")
            if name.lower() in {"you", "yourself", "u", "me"}:
                return None
            return name
    quoted = re.search(r'"([^"]+)"', text)
    if quoted:
        return quoted.group(1).strip()
    return None


def route_intent(user_message: str, default_user_id: str) -> Optional[Dict[str, Any]]:
    """Return query result dict for a high-confidence intent, else None."""
    message = (user_message or "").strip()
    if not message:
        return None

    user_id = default_user_id or _extract_object_id(message) or ""

    try:
        if _matches(message, _IDENTITY_PATTERNS):
            return None

        if _matches(message, _ACTIVE_GIGS_PATTERNS):
            if user_id:
                return {"tool": "get_active_gigs", "result": queries.get_active_gigs(user_id)}

        if _matches(message, _JOINED_GIGS_PATTERNS):
            if user_id:
                return {"tool": "get_user_joined_gigs", "result": queries.get_user_joined_gigs(user_id)}

        if _matches(message, _HISTORY_PATTERNS):
            if user_id:
                return {"tool": "get_user_gig_history", "result": queries.get_user_gig_history(user_id)}

        if _matches(message, _PROFILE_PATTERNS):
            target_id = _extract_object_id(message) or user_id
            if target_id:
                return {"tool": "get_user_profile_by_id", "result": queries.get_user_profile_by_id(target_id)}

        oid = _extract_object_id(message)
        gig_name = _extract_gig_name(message)
        if oid and not _matches(message, _PROFILE_PATTERNS):
            try:
                parse_object_id(oid)
                result = queries.get_gig_details(gig_id=oid)
                if result.get("success"):
                    return {"tool": "get_gig_details", "result": result}
            except ValueError:
                pass

        if _matches(message, _GIG_DETAILS_PATTERNS) or gig_name:
            ref = gig_name or message
            result = queries.get_gig_details(name=ref)
            if result.get("success"):
                return {"tool": "get_gig_details", "result": result}

    except Exception as exc:
        logger.warning("app-chatbot intent router failed: %s", exc)
        return None

    return None


def format_router_context(
    user_message: str,
    *,
    default_user_id: str = "",
) -> str:
    """Build pre_llm_call context from intent router + user identity line."""
    lines = [
        "[Data access policy]",
        "- Fetch CRWD/gig/user data ONLY via: get_active_gigs, get_user_profile_by_id,",
        "  get_gig_details, get_user_gig_history, get_user_joined_gigs.",
        "- Do not attempt direct MongoDB queries for database data.",
        "- If no predefined tool covers the question, say so and ask the user to rephrase.",
    ]
    if default_user_id:
        lines.append(
            f"Current CLI user_id: {default_user_id} (from APP_CHATBOT_DEFAULT_USER_ID)"
        )

    routed = route_intent(user_message, default_user_id)
    if routed:
        tool_name = routed.get("tool", "unknown")
        payload = routed.get("result", {})
        lines.extend([
            "",
            "[Database Context]",
            f"Source: auto-prefetch via `{tool_name}` on crwd_staging.",
            "Answer from this data when sufficient. Call the same tool again if you need fresh or paginated results.",
            "",
            json.dumps(payload, indent=2, default=str),
        ])

    return "\n".join(lines).strip()
