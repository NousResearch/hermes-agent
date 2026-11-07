"""The ``_primary_runtime`` snapshot: the route a fallback turn restores on the next turn.

Both builders (agent construction, and a deliberate model switch) record the custom entry that owns
the route's credential, resolved against the config the route runs on. Restoring scopes the
credential pool by that record, never by a live ``config.yaml`` lookup, so an entry edited or
removed mid-session can't hand the session another account's key.
"""

from __future__ import annotations

from typing import Any

from agent.custom_entry_pool import SNAPSHOT_OWNER_KEY, runtime_entry_owner


def _snapshot_primary_runtime(agent):
    # Per-turn restoration snapshot: after a fallback, the next turn restores these so the
    # preferred model gets a fresh attempt.
    _cc = agent.context_compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        SNAPSHOT_OWNER_KEY: runtime_entry_owner(agent.provider, agent.requested_provider, agent.base_url),
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        # The tier those overrides were pinned for (restored_request_overrides).
        "service_tier": getattr(agent, "service_tier", None),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Engine state _try_activate_fallback() overwrites (getattr: plugin engines may lack them).
        "compressor_model": getattr(_cc, "model", agent.model),
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(_cc, "api_key", ""),
        "compressor_provider": getattr(_cc, "provider", agent.provider),
        "compressor_context_length": _cc.context_length,
        "compressor_threshold_tokens": _cc.threshold_tokens,
    }
    if agent.api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })


def _build_primary_runtime_snapshot(agent, api_mode) -> dict[str, Any]:
    """The ``_primary_runtime`` record that persists a switch across turns."""
    cc = getattr(agent, "context_compressor", None) or None
    rt = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        SNAPSHOT_OWNER_KEY: runtime_entry_owner(agent.provider, agent.requested_provider, agent.base_url),
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_config": dict(agent.reasoning_config) if getattr(agent, "reasoning_config", None) else None,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Overrides must travel with the switched-to identity or a later recovery/restore resurrects
        # PRE-switch overrides from the stale init snapshot.
        # See #75091.
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        # The tier those overrides were pinned for (restored_request_overrides).
        "service_tier": getattr(agent, "service_tier", None),
        "runtime_capabilities": dict(getattr(agent, "runtime_capabilities", {}) or {}),
        "compressor_model": getattr(cc, "model", agent.model),
        "compressor_base_url": getattr(cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(cc, "api_key", ""),
        "compressor_provider": getattr(cc, "provider", agent.provider),
        "compressor_context_length": cc.context_length if cc else 0,
        "compressor_api_mode": getattr(cc, "api_mode", agent.api_mode),
        "compressor_threshold_tokens": cc.threshold_tokens if cc else 0,
    }
    if api_mode == "anthropic_messages":
        rt.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })
    return rt


def restored_request_overrides(agent, rt):
    """``request_overrides`` for restoring the snapshot ``rt``. The live tier wins over the snapshot's fast
    params, re-derived for ``rt``'s route: otherwise ``/fast off`` re-enables a paid tier. An unchanged
    non-static tier restores as is, so an explicit top-level value stays the caller's own."""
    overrides = dict(rt.get("request_overrides") or {})
    tier = getattr(agent, "service_tier", None)
    from agent.fast_mode import STATIC_TIERS
    if tier not in STATIC_TIERS and ("service_tier" not in rt or rt["service_tier"] == tier):
        return overrides
    from hermes_cli.models import resolve_fast_mode_overrides
    fast = resolve_fast_mode_overrides(
        rt.get("model"), provider=rt.get("provider"), base_url=rt.get("anthropic_base_url") or rt.get("base_url"), tier=tier,
    ) if tier in STATIC_TIERS else None
    # Only the values /fast pins are stale; any other service_tier word is the caller's own.
    kept = {k: v for k, v in overrides.items()
            if not ((k == "speed" and v == "fast") or (k == "service_tier" and v in STATIC_TIERS))}
    return {**kept, **(fast or {})}
