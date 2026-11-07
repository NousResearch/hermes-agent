"""Leading-user-turn invariant for outbound API payloads (#131382).

``ensure_user_leads_api_messages`` runs in ``assemble_api_request`` on the
send-time API copy only — never the persisted ``messages`` — so nothing leaks
into session persistence or the SessionDB flush cursor, and the shrink-only
re-anchor repair contract (fe21a4d2f0) is untouched.
"""

from __future__ import annotations

from typing import Dict, List

# Bridge inserted ahead of a malformed leading assistant/tool turn so the
# payload opens with a genuine user turn. Kept short and framed as a
# continuation cue so weak local models don't treat it as fresh input; the
# real current query is still the last user message.
_LEADING_USER_BRIDGE = "(Conversation resumed — prior context follows below.)"

# Non-conversational rows skipped when scanning for the leading turn:
# provider envelopes (system) and Hermes transcript metadata (session_meta).
_LEAD_SKIP_ROLES = ("system", "session_meta")


def ensure_user_leads_api_messages(api_messages: List[Dict]) -> int:
    """Guarantee the first non-system message in an outbound payload is role=user.

    OpenAI-compatible chat templates — notably the Qwen3-derived templates that
    LM Studio and local gateways apply — resolve the "current user query" by
    walking the turns, and raise ``"No user query found in messages."`` when the
    conversation leads with an ``assistant``/``tool`` turn instead of a ``user``
    turn. Anthropic likewise rejects a first message that is not role=user. The
    shape reaches providers from a lineage whose history lost its opening user
    row (e.g. an in-memory history rebuilt after a mid-chat model switch,
    #131382) or whose persisted history opens with a context-compaction summary
    merged into an ``assistant(tool_calls)`` message.

    Operates on the API-call-time copy only — never the persisted ``messages`` —
    so nothing leaks into session persistence or the SessionDB flush cursor.
    Inserts a minimal ``user`` bridge BEFORE the offending turn, which preserves
    assistant->tool adjacency and every tool_call pairing. No-op on well-formed
    payloads, on system/meta-only payloads, and on payloads with no real user
    turn anywhere (a bridge there would fabricate one).

    Returns 1 if a bridge was inserted, else 0.
    """
    if not api_messages:
        return 0
    idx = 0
    n = len(api_messages)
    while idx < n and isinstance(api_messages[idx], dict) and api_messages[idx].get("role") in _LEAD_SKIP_ROLES:
        idx += 1
    if idx >= n:
        return 0  # nothing but system/meta rows — no turn to lead
    first = api_messages[idx]
    if not isinstance(first, dict) or first.get("role") == "user":
        return 0  # already well-formed
    if not any(
        isinstance(m, dict) and m.get("role") == "user" for m in api_messages[idx + 1:]
    ):
        return 0  # no real user turn anywhere — a bridge would fabricate one
    api_messages.insert(idx, {"role": "user", "content": _LEADING_USER_BRIDGE})
    return 1


__all__ = ["ensure_user_leads_api_messages"]
