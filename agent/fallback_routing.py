"""Application-side fallback route facts projected through canonical provider routing."""

from __future__ import annotations

from typing import Any, Mapping

from agent.opencode_affinity import opencode_transport
from providers.routing import (
    InvocationRequest,
    InvocationRoute,
    canonicalize_api_mode,
    endpoint_api_mode,
    resolve_invocation_route,
)


def _nous_route_options(provider: str) -> dict[str, str]:
    if str(provider or "").strip().lower() not in {"nous", "nous-portal", "nousresearch"}:
        return {}
    try:
        from hermes_cli.config import load_config_readonly

        nous_cfg = load_config_readonly().get("nous") or {}
        return {"anthropic_wire": str(nous_cfg.get("anthropic_wire") or "chat")}
    except Exception:
        return {"anthropic_wire": "chat"}


def _configured_route_facts(provider: str) -> Mapping[str, Any]:
    try:
        from agent.configured_provider_resolution import get_configured_provider_entry

        return get_configured_provider_entry(provider) or {}
    except Exception:
        return {}


def resolve_fallback_invocation_route(
    provider: str,
    model: str,
    base_url: str = "",
    *,
    explicit_api_mode: str | None = None,
    route_base_url_hint: str = "",
) -> InvocationRoute:
    """Resolve one fallback route from selected provider/model plus app config facts.

    Config loading remains application-owned. Provider identity, endpoint normalization,
    wire selection, and runtime-kind policy remain owned by providers.routing.
    """

    configured = _configured_route_facts(provider)
    effective_base = str(base_url or configured.get("base_url") or "").strip()
    declared_mode = canonicalize_api_mode(configured.get("api_mode"))
    explicit_mode = canonicalize_api_mode(explicit_api_mode)
    hinted_mode = endpoint_api_mode(str(route_base_url_hint or "").strip()) or ""

    opencode_mode, opencode_base = opencode_transport(provider, model, effective_base)
    effective_base = str(opencode_base or effective_base).strip()
    route_mode = explicit_mode or hinted_mode or opencode_mode or declared_mode

    return resolve_invocation_route(
        InvocationRequest(
            provider=provider,
            model=model,
            base_url=effective_base,
            explicit_api_mode=route_mode,
            route_options=_nous_route_options(provider),
        )
    )


__all__ = ["resolve_fallback_invocation_route"]