"""Capability registry for Anthropic-compatible prompt caching.

Background
----------
Anthropic prompt caching (``cache_control`` breakpoints) is a contract layered
on top of the Anthropic Messages API. Native Anthropic obviously honours it.
But many third-party providers also speak the native Anthropic protocol
(``api_mode = 'anthropic_messages'``) and *some* of them also implement the
``cache_control`` contract for their own model families:

  - MiniMax (M2.x) — https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
  - LiteLLM Anthropic-proxy mode — passes through cache_control to the
    upstream provider, so caching works iff the upstream does.

Pre-#17332 the gate in ``_anthropic_prompt_cache_policy`` only enabled
caching for ``api_mode == 'anthropic_messages'`` when the model name
contained ``"claude"`` (i.e. third-party gateways serving Anthropic's
Claude family). That left providers using the native Anthropic protocol
for *their own* models (MiniMax-M2.7 etc.) silently re-billing the full
prompt on every turn.

Hardcoding ``provider == 'minimax'`` would fix MiniMax but leaves the
same regression waiting for the next provider that adds Anthropic-cache
support. This module is the longer-lived answer:

  1. ``ProviderConfig.extra`` carries an ``anthropic_cache: True`` flag
     and an optional ``anthropic_cache_hosts`` tuple (so URL-only callers
     who configure a custom endpoint pointing at MiniMax still get
     caching even though their ``provider`` reads ``"custom"``).

  2. ``provider_supports_anthropic_cache(provider, base_url)`` reads
     that registry. It also consults the user's ``agent.anthropic_cache_hosts``
     config list so operators can opt-in their own gateway without
     waiting for an upstream patch.

The actual decision in ``_anthropic_prompt_cache_policy`` keeps the
existing native-Anthropic / OpenRouter / Claude-on-third-party-gateway
branches; this module only adds the new capability-driven branch.

(issue #17332)
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from utils import base_url_hostname

logger = logging.getLogger(__name__)


# Hostnames known to honour the Anthropic ``cache_control`` contract for
# arbitrary (non-Claude) model families. Populated automatically from
# ``ProviderConfig.extra['anthropic_cache_hosts']`` at module-load time
# so we don't drift out of sync with the auth registry.
_REGISTRY_HOSTS: set = set()


def _refresh_registry_hosts() -> None:
    """Re-read ProviderConfig.extra entries. Safe to call multiple times."""
    _REGISTRY_HOSTS.clear()
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
    except Exception:  # pragma: no cover — circular-import safety
        return
    for pconfig in PROVIDER_REGISTRY.values():
        extra = getattr(pconfig, "extra", None) or {}
        if not extra.get("anthropic_cache"):
            continue
        hosts = extra.get("anthropic_cache_hosts") or ()
        for h in hosts:
            if isinstance(h, str) and h:
                _REGISTRY_HOSTS.add(h.lower())


_refresh_registry_hosts()


def _normalize_hosts(hosts):
    if not hosts:
        return set()
    out = set()
    for h in hosts:
        if isinstance(h, str) and h.strip():
            out.add(h.strip().lower())
    return out


def provider_supports_anthropic_cache(
    provider,
    base_url,
    *,
    user_configured_hosts=None,
):
    """Return True when this provider/base_url pair documents ``cache_control``
    support for its own model families on the native Anthropic transport.

    Resolution order:
      1. ProviderConfig.extra['anthropic_cache'] == True for this provider id
         (handles built-in providers like ``minimax`` / ``minimax-cn``).
      2. Hostname match against the union of registry-declared hosts and
         user-configured hosts (handles ``provider == 'custom'`` setups
         pointing at a known endpoint, and operator-curated additions).

    Returns False otherwise. Callers must still gate on
    ``api_mode == 'anthropic_messages'`` themselves — this helper says
    *nothing* about the wire protocol, only about the cache contract.
    """
    # 1. Provider id lookup (exact match against the registry).
    if provider:
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
        except Exception:
            PROVIDER_REGISTRY = {}
        key = provider.strip().lower() if isinstance(provider, str) else ""
        pconfig = PROVIDER_REGISTRY.get(key) if key and PROVIDER_REGISTRY else None
        if pconfig is not None:
            extra = getattr(pconfig, "extra", None) or {}
            if extra.get("anthropic_cache") is True:
                return True

    # 2. Hostname match — covers custom providers pointing at a known
    # endpoint, plus operator-configured additions.
    host = ""
    if base_url:
        host = (base_url_hostname(base_url) or "").lower()
    if host:
        configured = _normalize_hosts(user_configured_hosts)
        # _REGISTRY_HOSTS is intentionally reloaded lazily so tests that
        # mutate the registry inside a fixture don't get a stale snapshot.
        _refresh_registry_hosts()
        if host in _REGISTRY_HOSTS or host in configured:
            return True

    return False
