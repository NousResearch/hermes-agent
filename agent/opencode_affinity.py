"""``x-opencode-session`` — OpenCode relay session-affinity header.

OpenCode (opencode.ai Zen/Go relay) pins requests that share an
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

from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"
_SENSITIVE_HEADER_NAMES = frozenset({OPENCODE_SESSION_HEADER, "authorization"})


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches the built-in opencode-zen/go providers, custom
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


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    *,
    cache_scope: Optional[str] = None,
    use_ambient: bool = True,
) -> dict[str, str]:
    """Return the OpenCode affinity header from an explicit safe scope.

    ``cache_scope`` is preferred because it survives physical-session rotation;
    ``session_id`` is the opaque physical fallback. Ambient affinity is retained
    only for legacy callers and can be disabled at isolation boundaries.
    """
    if not is_opencode_target(provider, base_url):
        return {}
    explicit_scope = str(cache_scope or "").strip()
    physical_id = str(session_id or "").strip()
    key = explicit_scope or physical_id
    if not key and use_ambient:
        try:
            from agent.portal_tags import get_affinity_scope, get_conversation_context

            key = str(get_affinity_scope() or get_conversation_context() or "").strip()
        except Exception:
            key = ""
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
    *,
    cache_scope: Optional[str] = None,
    use_ambient: bool = True,
) -> dict[str, Any]:
    """Merge OpenCode affinity into ``kwargs["extra_headers"]`` in place.

    For OpenCode targets, caller-supplied session and authorisation headers are
    removed case-insensitively. Non-sensitive headers survive, then the computed
    explicit affinity header is applied last. A missing safe scope never forwards
    a caller-provided sensitive value.
    """
    if not is_opencode_target(provider, base_url):
        return kwargs
    existing = kwargs.get("extra_headers")
    merged = {
        key: value for key, value in (existing.items() if isinstance(existing, dict) else ())
        if str(key).lower() not in _SENSITIVE_HEADER_NAMES
    }
    headers = opencode_session_headers(
        provider, base_url, session_id,
        cache_scope=cache_scope, use_ambient=use_ambient,
    )
    merged.update(headers)
    if merged:
        kwargs["extra_headers"] = merged
    else:
        kwargs.pop("extra_headers", None)
    return kwargs
