"""``x-claude-code-session-id`` — session identity for Anthropic OAuth relays.

A relay that forwards Claude Code traffic (``capabilities.anthropic_oauth_proxy``)
usually fronts several subscriptions and spreads conversations across them. The
header Claude Code tags every request with is how such a relay tells one
conversation from another: without it every request looks like the same
conversation, so the relay keeps them all on the account it is currently using
and the remaining subscriptions sit idle.

The value only has to be opaque and stable per conversation, so it is derived
exactly like the other conversation-affinity hints Hermes already sends
(OpenCode's ``x-opencode-session``, OpenRouter's sticky ``session_id``): the
host-declared routing scope first, then the ambient conversation root, then the
physical session id — normalized through ``_cache_scope_from_session_id`` so
cron fires of one job share a scope.

Opt-in only. Native Anthropic and Anthropic-compatible providers that never
declared the proxy capability are left byte-identical, so no existing route
gains a header it did not ask for.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional

CLAUDE_CODE_SESSION_HEADER = "x-claude-code-session-id"

# Claude Code session ids are uuids; a relay reading the header validates its
# SHAPE rather than parsing it. Anything outside this alphabet (or longer than
# the 128 chars such validators allow) is replaced by a digest below, so a
# conversation whose id carries a slash or a space still gets a stable value
# instead of a header the relay silently discards.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def claude_code_session_id(session_id: Optional[str] = None) -> str:
    """Return the conversation-stable session id for the header, or ``""``."""
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        key = _cache_scope_from_session_id(
            get_affinity_scope() or get_conversation_context() or session_id
        )
    except Exception:
        key = str(session_id or "")
    key = key.strip()
    if not key:
        return ""
    if _SAFE_SESSION_ID.match(key):
        return key
    return "hermes_" + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:32]


def is_oauth_proxy_route(capabilities: Optional[Mapping[str, Any]]) -> bool:
    """True when the route declared ``capabilities.anthropic_oauth_proxy``."""
    return bool(isinstance(capabilities, Mapping) and capabilities.get("anthropic_oauth_proxy") is True)


def claude_code_session_headers(
    capabilities: Optional[Mapping[str, Any]],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"x-claude-code-session-id": <id>}`` for proxy routes, else ``{}``."""
    if not is_oauth_proxy_route(capabilities):
        return {}
    key = claude_code_session_id(session_id)
    return {CLAUDE_CODE_SESSION_HEADER: key} if key else {}


def merge_claude_code_session_headers(
    kwargs: dict[str, Any],
    capabilities: Optional[Mapping[str, Any]],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the session header into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved and
    the fast-mode ``anthropic-beta`` rebuild is never clobbered. Routes without
    the capability are left untouched.
    """
    headers = claude_code_session_headers(capabilities, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
