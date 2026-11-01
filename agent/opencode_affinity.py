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

# Last conversation key resolved by a main turn, per Hermes home (#115000). Auxiliary threads (goal
# judge, compression, vision, MoA) don't inherit the parent's ``contextvars``, so
# ``resolve_affinity_key`` returns "". A fresh ``oneshot-`` would either lose cache locality or, on
# OpenCode Go, trip HTTP 400 ``MissingSessionID`` outright. The main turn's resolved key is the best
# stable fallback — the same conversation the auxiliary call belongs to.
#
# Slot key is ``hermes_home_key()`` because one process serves many profiles (multiplex gateway /
# Desktop ``serve``): a bare module global would hand profile A's conversation key to profile B's
# off-context auxiliary call (root ``AGENTS.md`` § Code Shape Rules — module globals hold the
# *launch* profile's state, so key slots by home).
_LAST_OPENCODE_SESSION_KEYS: dict[str, str] = {}

# The only auxiliary chains allowed to fall back to that cache. Membership means "this call belongs
# to the live conversation even though it sees no runtime context": the goal judge runs on a thread
# that inherits nothing (#115000), so the explicit ``session_id`` is always empty there. Everything
# else — a stateless one-shot (commit message, cron summary, standalone prompt), a subagent, a plugin
# task — has no conversation identity to claim, and reusing the last main-turn key would pin an
# unrelated request to another conversation's backend (#131047 review). Those keep a fresh
# ``oneshot-`` key, which the relays accept just like any other opaque value.
# ponytail: keyed by task name, so an off-context chain added later loses the borrowed key and lands
# on its own (still valid) backend. Add a key here only with evidence of a locality win.
CONVERSATION_BOUND_AUX_TASKS: frozenset[str] = frozenset({"goal_judge"})


def _profile_scope_key() -> str:
    """``hermes_home_key()`` of the calling profile; "" when the resolver is unavailable."""
    try:
        from hermes_constants import hermes_home_key

        return hermes_home_key()
    except Exception:
        return ""


def opencode_transport(provider: Optional[str], model: Optional[str], base_url: Optional[str]) -> tuple[Optional[str], str]:
    """``(api_mode, base_url)`` re-derived per model for an OpenCode relay target; ``(None, base_url)`` otherwise.

    OpenCode Zen/Go serve Responses-only (``gpt-*``, ``grok-*``, ``muse-spark``), Anthropic-wire
    (``minimax-*``, ``qwen*``, ``claude-*``) and chat/completions models behind one provider, so a
    provider-level or persisted ``api_mode`` is wrong for every model but the one it was saved for.
    The main runtime (``hermes_cli/runtime_provider.py``) always re-derives from the effective model;
    auxiliary resolution must agree or ``gpt-5.6-luna`` compression 500s on /chat/completions (#98799).
    Built-in families, custom entries named after one (``opencode-go-bridge``, #85589) and opencode.ai
    hosts all count.
    """
    from hermes_cli.models import normalize_opencode_base_url, normalize_opencode_model_id, opencode_model_api_mode
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider, _opencode_family_for_custom

    url = str(base_url or "")
    family = _opencode_family_for_custom(str(provider or ""), url)
    if family is None:
        return None, url
    # A custom entry that declares its own api_mode keeps it, exactly like the main runtime
    # (_resolve_named_custom_runtime only re-derives when the entry has none).
    if (_get_named_custom_provider(str(provider or "")) or {}).get("api_mode"):
        return None, url
    # ``<provider>/<model>`` config ids are stripped against the entry name before the family lookup.
    api_mode = opencode_model_api_mode(family, normalize_opencode_model_id(provider, model))
    return api_mode, normalize_opencode_base_url(provider, api_mode, url)


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
    *,
    reuse_cached_key: bool = False,
) -> dict[str, str]:
    """Return ``{"x-opencode-session": <key>}`` for OpenCode targets, else ``{}``.

    OpenCode targets always get a key: when no conversation/session key resolves, an
    ephemeral ``oneshot-<hex>`` value is generated (OpenCode Go rejects requests without
    the header, #105841). ``reuse_cached_key`` additionally lets a *conversation-bound* chain
    whose thread inherited no contextvars (the goal judge, ``CONVERSATION_BOUND_AUX_TASKS``)
    reuse the last key a main turn resolved (#115000). Callers that are stateless by
    construction leave it off so they keep their own ``oneshot-`` backend (#131047)."""
    if not is_opencode_target(provider, base_url):
        return {}
    key = resolve_affinity_key(session_id)
    slot = _profile_scope_key()
    if key:
        # Cache the resolved key so a later conversation-bound auxiliary thread with no ambient
        # contextvars (#115000) can reach the same conversation's backend.
        _LAST_OPENCODE_SESSION_KEYS[slot] = key
    elif reuse_cached_key and _LAST_OPENCODE_SESSION_KEYS.get(slot):
        key = _LAST_OPENCODE_SESSION_KEYS[slot]
    if not key:
        # Stateless one-shot requests (commit messages, summaries, standalone prompts outside
        # a session) lack an ambient conversation or session id. OpenCode Go strictly requires
        # x-opencode-session on every request (HTTP 400 MissingSessionID if absent, #105841)
        # so generate an ephemeral session id fallback.
        #
        # ponytail: reachable for every stateless call and for any profile whose main turn has not
        # resolved a key yet. A conversation-bound chain (``reuse_cached_key``) borrows that key
        # instead; the borrowed backend is warm but the value is opaque routing affinity, not auth.
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
    *,
    reuse_cached_key: bool = False,
) -> dict[str, Any]:
    """Merge the affinity header(s) into ``kwargs["extra_headers"]`` (in place).

    Existing per-request headers win, so a caller-pinned value is preserved.
    Targets with neither source configured are left untouched. ``reuse_cached_key`` is forwarded to
    :func:`opencode_session_headers` — pass it only for a chain that is conversation-bound by
    construction (``CONVERSATION_BOUND_AUX_TASKS``).
    """
    headers = opencode_session_headers(provider, base_url, session_id, reuse_cached_key=reuse_cached_key)
    headers.update(custom_provider_session_affinity_headers(base_url, session_id))
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
