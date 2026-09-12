"""``x-opencode-session`` — OpenCode relay session-affinity header.

OpenCode (opencode.ai Zen/Go/free relay) pins requests that share an
``x-opencode-session`` value to the same upstream backend, which is what
keeps its prompt cache warm across the turns of one conversation. The value
only has to be opaque and consistent per conversation, so it is derived the
same way as the other conversation-affinity hints Hermes already sends
(OpenRouter's sticky ``session_id``, xAI's ``x-grok-conv-id``): the
host-declared routing scope first, then the ambient conversation root, then
the physical session id — normalized through ``_cache_scope_from_session_id``
so cron fires of one job share a scope.

Every OpenCode request — main turn on any transport, auxiliary calls
(compression, titles, vision, MoA) — goes through :func:`opencode_session_headers`
so the header cannot drift per code path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in opencode-zen/go/free providers, custom
    ``opencode-<family>-*`` providers, and any base_url hosted on opencode.ai.
    """
    try:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(provider) is not None:
            return True
    except Exception:
        pass
    try:
        from agent.anthropic_endpoints import _is_opencode_endpoint

        return _is_opencode_endpoint(str(base_url or ""))
    except Exception:
        return False


def resolve_affinity_key(session_id: Optional[str] = None) -> str:
    """Return the normalized rotation-stable conversation affinity key."""
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        return _cache_scope_from_session_id(
            get_affinity_scope() or get_conversation_context() or session_id
        )
    except Exception:
        return str(session_id or "")


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``."""
    if not is_opencode_target(provider, base_url):
        return {}
    key = resolve_affinity_key(session_id)
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def custom_provider_session_affinity_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
) -> dict[str, str]:
    """Return ``{<session_affinity_header>: <key>}`` for custom providers declaring one, else ``{}``."""
    try:
        from hermes_cli.config_providers import get_custom_provider_session_affinity_header

        header = get_custom_provider_session_affinity_header(
            base_url=base_url,
            provider=provider,
            custom_providers=custom_providers,
        )
        if not header:
            return {}
        key = resolve_affinity_key(session_id)
        return {header: key} if key else {}
    except Exception:
        return {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Merge OpenCode or custom provider affinity headers into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Non-OpenCode and non-affinity targets are left untouched.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    if not headers:
        headers = custom_provider_session_affinity_headers(
            provider, base_url, session_id, custom_providers=custom_providers
        )
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs


# Alias for callers naming the generalized capability
merge_session_affinity_headers = merge_opencode_session_headers
