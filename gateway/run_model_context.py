"""Gateway model route and context-window resolution."""

from __future__ import annotations

import dataclasses
from contextlib import suppress
from typing import Optional


@dataclasses.dataclass(frozen=True)
class _GatewayModelContext:
    """Effective gateway model route and context-window resolution."""

    model: str
    provider: str
    base_url: str
    context_length: int
    context_source: str


def _resolve_gateway_model_context(
    model: Optional[str] = None, route: Optional[dict] = None,
) -> _GatewayModelContext:
    """Resolve the configured gateway route and effective context window. Call off-loop (may block).

    ``route`` (``provider`` / ``base_url`` / ``api_key`` of a session-only /model switch) replaces the
    default runtime credentials so the window is looked up against the endpoint that actually serves
    ``model``; a ``model.context_length`` pin only survives when that route still matches the config.
    ``context_source`` is ``"default"`` only for a model unknown to the catalog that fell through to
    ``DEFAULT_FALLBACK_CONTEXT`` — a catalog-listed 256K model is ``"detected"``.
    """
    from agent.model_metadata import (
        DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT, _longest_key_match, get_model_context_length)
    from gateway.run import (
        _best_effort, _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs,
    )
    resolved_model = model or _resolve_gateway_model()
    config_context_length = provider = base_url = api_key = custom_providers = None
    configured_model = configured_provider = configured_base_url = None

    def _read_config() -> None:
        nonlocal config_context_length, provider, base_url, custom_providers
        nonlocal configured_model, configured_provider, configured_base_url
        data = _load_gateway_config()
        if not data:
            return
        model_cfg = data.get("model", {})
        if isinstance(model_cfg, dict):
            configured_model = model_cfg.get("default") or model_cfg.get("model")
            raw_ctx = model_cfg.get("context_length")
            if raw_ctx is not None:
                with suppress(TypeError, ValueError):
                    config_context_length = int(raw_ctx)
            configured_provider = provider = model_cfg.get("provider") or None
            configured_base_url = base_url = model_cfg.get("base_url") or None
        try:
            from hermes_cli.config import get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(data)
        except Exception:
            custom_providers = data.get("custom_providers")

    def _read_runtime() -> None:
        nonlocal provider, base_url, api_key
        if route and route.get("base_url"):
            # A session route with its own endpoint (a /model switch) replaces the default runtime
            # read; a route without one (persisted / SessionDB / plain config) still resolves the
            # default runtime credentials so a custom endpoint and its context_length pin survive.
            provider = route.get("provider") or provider
            base_url = route["base_url"]
            api_key = route.get("api_key")
            return
        runtime = _resolve_runtime_agent_kwargs()
        provider = runtime.get("provider") or provider
        base_url = runtime.get("base_url") or base_url
        api_key = runtime.get("api_key")

    def _pin_still_applies() -> bool:
        # Drop a configured context_length pin when the effective route no longer matches (or on error).
        from hermes_cli.route_identity import should_clear_context_pin
        return not should_clear_context_pin(
            configured_model, resolved_model, configured_base_url, base_url, configured_provider, provider)

    def _custom_ctx() -> Optional[int]:
        from hermes_cli.config import get_custom_provider_context_length
        return get_custom_provider_context_length(
            model=resolved_model, base_url=base_url, custom_providers=custom_providers)

    _best_effort(_read_config)
    _best_effort(_read_runtime)
    if config_context_length is not None and not _best_effort(_pin_still_applies):
        config_context_length = None
    if config_context_length is None and custom_providers and base_url:
        config_context_length = _best_effort(_custom_ctx) or None

    context_length = get_model_context_length(
        resolved_model, base_url=base_url or "", api_key=api_key or "",
        config_context_length=config_context_length, provider=provider or "",
        custom_providers=custom_providers)
    fell_through = (context_length == DEFAULT_FALLBACK_CONTEXT
                    and _longest_key_match(DEFAULT_CONTEXT_LENGTHS, str(resolved_model).lower()) is None)
    context_source = ("config" if config_context_length is not None
                      else "default" if fell_through else "detected")
    return _GatewayModelContext(
        model=resolved_model, provider=provider or "", base_url=base_url or "",
        context_length=context_length, context_source=context_source)
