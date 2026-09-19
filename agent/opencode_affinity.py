"""Conversation-affinity request headers for session-aware relays and proxies.

Two sources, one merge point:

* ``x-opencode-session`` — OpenCode (opencode.ai Zen/Go relay) pins requests that share this
  value to the same upstream backend, which keeps its prompt cache warm across the turns of one
  conversation. Always sent to OpenCode targets.
* ``providers.<name>.session_affinity_header`` — an opt-in header NAME on a custom provider entry
  (default off). Session-aware proxies fronting a stateful backend (LiteLLM's ``x-litellm-session-id``,
  self-hosted Claude/OpenAI gateways) otherwise classify an agent-loop request whose last message is
  a ``tool_result`` as a new conversation and replay the whole history upstream (#86241, #104449).

The value only has to be opaque and consistent per conversation, so it is derived the same way as
the other affinity hints Hermes already sends (OpenRouter's sticky ``session_id``, xAI's
``x-grok-conv-id``): the host-declared routing scope first (a host that names its own conversation,
#96811), then the ambient conversation ROOT (stable across compaction rotation and delegate trees),
then the physical session id — normalized through ``_cache_scope_from_session_id`` so cron fires of
one job share a scope. Auxiliary calls (compression, titles, vision, MoA) have no session handle and
resolve the ambient value, so they stay on the conversation's backend too (#70820).

Every request — main turn on any transport, auxiliary calls — goes through
:func:`merge_session_affinity_headers` so the headers cannot drift per code path.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"


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


def resolve_affinity_key(session_id: Optional[str] = None) -> str:
    """Return the normalized rotation-stable conversation affinity key ("" when unknown)."""
    try:
        from agent.portal_tags import get_affinity_scope, get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        return _cache_scope_from_session_id(get_affinity_scope() or get_conversation_context() or session_id)
    except Exception:
        return str(session_id or "")


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``.

    OpenCode targets always get a key: when no conversation/session key resolves, an
    ephemeral ``oneshot-<hex>`` value is generated (OpenCode Go rejects requests without
    the header, #105841)."""
    if not is_opencode_target(provider, base_url):
        return {}
    key = resolve_affinity_key(session_id)
    if not key:
        # Stateless one-shot requests (commit messages, summaries, standalone prompts outside
        # a session) lack an ambient conversation or session id. OpenCode Go strictly requires
        # x-opencode-session on every request (HTTP 400 MissingSessionID if absent, #105841)
        # so generate an ephemeral session id fallback.
        key = f"oneshot-{uuid.uuid4().hex[:16]}"
    return {OPENCODE_SESSION_HEADER: key}


def custom_provider_session_affinity_headers(
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return ``{<session_affinity_header>: <key>}`` when the route's provider entry declares one, else ``{}``."""
    try:
        from hermes_cli.config import get_custom_provider_session_affinity_header

        header = get_custom_provider_session_affinity_header(str(base_url or ""))
    except Exception:
        return {}
    if not header:
        return {}
    key = resolve_affinity_key(session_id)
    return {header: key} if key else {}


def merge_session_affinity_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the affinity header(s) into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Targets with neither source configured are left untouched.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    headers.update(custom_provider_session_affinity_headers(base_url, session_id))
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
