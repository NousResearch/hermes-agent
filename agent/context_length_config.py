"""Shared config-driven context-length resolution helpers.

This module centralizes config override resolution so runtime and display
surfaces do not drift when `model.context_length` or `custom_providers`
provide explicit values.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


WarnFn = Callable[[str, str, Any], None]


def normalize_context_lookup_base_url(url: str) -> str:
    """Normalize base URLs before comparing config and runtime endpoints."""
    return (url or "").strip().rstrip("/").lower()


def coerce_positive_context_length(raw_value: Any) -> Optional[int]:
    """Return a positive integer context length, or ``None`` when invalid."""
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def get_compatible_custom_providers(
    agent_cfg: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return custom providers using config compatibility helpers when available."""
    cfg = agent_cfg if isinstance(agent_cfg, dict) else {}
    try:
        from hermes_cli.config import get_compatible_custom_providers as _get_compatible

        providers = _get_compatible(cfg)
    except Exception:
        providers = cfg.get("custom_providers")
        if not isinstance(providers, list):
            providers = []
    return [entry for entry in providers if isinstance(entry, dict)]


def custom_provider_match_score(
    entry: dict[str, Any], provider: str, base_url: str
) -> int:
    """Score how well a custom provider entry matches the requested runtime."""
    requested_provider = (provider or "").strip().lower()
    requested_base_url = normalize_context_lookup_base_url(base_url)

    provider_candidates: set[str] = set()
    entry_name = (entry.get("name") or "").strip()
    if entry_name:
        provider_candidates.add(entry_name.lower())
        try:
            from hermes_cli.providers import custom_provider_slug

            provider_candidates.add(custom_provider_slug(entry_name))
        except Exception:
            pass

    provider_key = (entry.get("provider_key") or "").strip().lower()
    if provider_key:
        provider_candidates.add(provider_key)

    if (
        requested_provider.startswith("custom:")
        and requested_provider in provider_candidates
    ):
        return 2

    entry_base_url = normalize_context_lookup_base_url(
        entry.get("base_url") or entry.get("url") or entry.get("api") or ""
    )
    if requested_base_url and entry_base_url and requested_base_url == entry_base_url:
        return 1

    return 0


def _warn_if_invalid(
    warn: WarnFn | None, scope: str, model: str, raw_value: Any
) -> None:
    """Call the warning hook when supplied."""
    if warn is not None:
        warn(scope, model, raw_value)


def resolve_custom_provider_context_length(
    *,
    model: str,
    provider: str,
    base_url: str,
    custom_providers: Optional[list[dict[str, Any]]],
    warn: WarnFn | None = None,
) -> Optional[int]:
    """Resolve per-model context length from the best matching custom provider."""
    best_score = 0
    best_context_length: Optional[int] = None

    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue

        score = custom_provider_match_score(entry, provider, base_url)
        if score == 0 or score < best_score:
            continue

        resolved: Optional[int] = None
        models = entry.get("models", {})
        if isinstance(models, dict):
            model_cfg = models.get(model, {})
            if isinstance(model_cfg, dict):
                raw_ctx = model_cfg.get("context_length")
                if raw_ctx is not None:
                    resolved = coerce_positive_context_length(raw_ctx)
                    if resolved is None:
                        _warn_if_invalid(warn, "custom_providers", model, raw_ctx)

        if resolved is None and str(entry.get("model") or "").strip() == model:
            raw_ctx = entry.get("context_length")
            if raw_ctx is not None:
                resolved = coerce_positive_context_length(raw_ctx)
                if resolved is None:
                    _warn_if_invalid(warn, "custom_providers", model, raw_ctx)

        if resolved is not None:
            best_score = score
            best_context_length = resolved

    return best_context_length


def resolve_config_context_length(
    *,
    model: str,
    provider: str,
    base_url: str,
    agent_cfg: Optional[dict[str, Any]],
    warn: WarnFn | None = None,
) -> Optional[int]:
    """Resolve the effective config-driven context length for a model runtime.

    Priority:
    1. matching custom provider per-model override
    2. top-level model.context_length
    3. None
    """
    cfg = agent_cfg if isinstance(agent_cfg, dict) else {}

    custom_ctx = resolve_custom_provider_context_length(
        model=model,
        provider=provider,
        base_url=base_url,
        custom_providers=get_compatible_custom_providers(cfg),
        warn=warn,
    )
    if custom_ctx is not None:
        return custom_ctx

    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        return None

    raw_ctx = model_cfg.get("context_length")
    if raw_ctx is None:
        return None

    value = coerce_positive_context_length(raw_ctx)
    if value is None:
        _warn_if_invalid(warn, "model", model, raw_ctx)
        return None
    return value


def resolve_display_context_length(
    *,
    model: str,
    provider: str,
    base_url: str,
    api_key: str,
    model_info: Any,
    agent_cfg: Optional[dict[str, Any]],
) -> Optional[int]:
    """Resolve the context length that `/model` surfaces should display."""
    config_ctx = resolve_config_context_length(
        model=model,
        provider=provider,
        base_url=base_url,
        agent_cfg=agent_cfg,
        warn=None,
    )
    if config_ctx is not None:
        return config_ctx

    if model_info is not None:
        context_window = getattr(model_info, "context_window", None)
        if isinstance(context_window, int) and context_window > 0:
            return context_window

    try:
        from agent.model_metadata import get_model_context_length

        return get_model_context_length(
            model,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
        )
    except Exception:
        return None
