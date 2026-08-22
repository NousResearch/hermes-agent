"""Runtime adapter for main-orchestrator usage-aware routing.

The policy lives in :mod:`agent.orchestrator_usage_routing`; this module owns
cache reads and native runtime resolution. Selection never fetches usage
inline: ``build_usage_view`` reads the secret-free cache and schedules a
bounded, deduplicated background refresh when needed.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agent.delegation_routing import ProviderUsage
from agent.orchestrator_usage_routing import (
    decide_orchestrator_route,
    parse_orchestrator_routing_config,
)

logger = logging.getLogger(__name__)


def apply_orchestrator_usage_routing(
    *,
    model: str,
    runtime: dict[str, Any],
    current_model: Optional[str] = None,
    current_runtime: Optional[Mapping[str, Any]] = None,
    full_config: Optional[Mapping[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Return the model/runtime for the next turn without raising.

    ``model``/``runtime`` are the caller's base route. ``current_*`` may name
    an already-active fallback route; when omitted, the base route is current.
    Raw resolver/cache errors are never surfaced or logged.
    """
    try:
        if full_config is None:
            from hermes_cli.config import load_config_readonly

            full_config = load_config_readonly()
        config = parse_orchestrator_routing_config(full_config)
    except Exception as exc:
        logger.debug(
            "orchestrator usage routing config unavailable (%s)",
            type(exc).__name__,
        )
        return model, runtime

    if not config.enabled:
        return model, runtime

    try:
        from agent.delegation_usage_cache import build_usage_view

        usage_view = build_usage_view(
            [config.primary_provider],
            ttl_seconds=config.usage_ttl_seconds,
            stale_seconds=config.usage_stale_seconds,
            window_prefixes_by_provider={
                config.primary_provider: config.primary_usage_window_prefixes,
            },
        )
        usage = usage_view.entries.get(config.primary_provider) or ProviderUsage(
            provider=config.primary_provider
        )
    except Exception as exc:
        logger.debug(
            "orchestrator usage cache read failed (%s)", type(exc).__name__
        )
        return model, runtime

    managed_routes = {
        (config.primary_provider, config.primary_model),
        (config.fallback_provider, config.fallback_model),
    }
    base_route = (str(runtime.get("provider") or ""), str(model or ""))
    if base_route not in managed_routes:
        # The configured/session route was explicitly changed. It outranks a
        # cached agent from the previous turn (for example immediately after
        # `/model`) as well as the automatic two-route policy.
        return model, runtime

    active_runtime = dict(current_runtime or runtime)
    active_model = str(current_model or model)
    active_provider = str(active_runtime.get("provider") or "")
    if (active_provider, active_model) not in managed_routes:
        # Explicit `/model`, channel, or CLI selections outrank the automatic
        # two-route policy. Never normalize an unrelated model/provider back
        # to the configured primary or fallback.
        return active_model, active_runtime
    decision = decide_orchestrator_route(
        current_provider=active_provider,
        current_model=active_model,
        usage=usage,
        config=config,
    )

    if decision.provider == active_provider and decision.model == active_model:
        return active_model, active_runtime

    base_provider = str(runtime.get("provider") or "")
    if decision.provider == base_provider and decision.model == model:
        return model, runtime

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolved = resolve_runtime_provider(
            requested=decision.provider,
            target_model=decision.model,
        )
    except Exception as exc:
        logger.debug(
            "orchestrator usage target runtime unavailable for %s (%s)",
            decision.provider,
            type(exc).__name__,
        )
        return active_model, active_runtime

    selected_runtime = {
        "api_key": resolved.get("api_key"),
        "base_url": resolved.get("base_url"),
        "provider": resolved.get("provider", decision.provider),
        "requested_provider": resolved.get(
            "requested_provider", decision.provider
        ),
        "api_mode": resolved.get("api_mode"),
        "command": resolved.get("command"),
        "args": list(resolved.get("args") or []),
        "credential_pool": resolved.get("credential_pool"),
    }
    if "provider_project_id" in runtime:
        selected_runtime["provider_project_id"] = resolved.get("project_id")
    elif "project_id" in runtime:
        selected_runtime["project_id"] = resolved.get("project_id")
    if "max_tokens" in runtime or "max_tokens" in resolved:
        selected_runtime["max_tokens"] = resolved.get(
            "max_tokens", runtime.get("max_tokens")
        )
    logger.info("orchestrator usage routing: %s", decision.reason)
    return resolved.get("model") or decision.model, selected_runtime
