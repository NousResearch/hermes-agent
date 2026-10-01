"""Canonical static catalogue identity facts for explicit selection."""

from __future__ import annotations

from typing import Any, Optional

from models import AmbiguousModelAliasError, MODEL_ALIASES, normalize_model_id, resolve_model_alias
from models.catalog_static import _PROVIDER_MODELS, _BORROWED_MODEL_PROVIDERS
from providers import is_aggregator, normalize_provider as _normalize_provider

def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = _normalize_provider(provider or "openrouter")
    return {k for k in (key, normalized) if k}

def _provider_catalog_names(provider: str) -> tuple[str, ...]:
    """Static picker models for *provider*."""
    return tuple(_PROVIDER_MODELS.get(provider, ()))

def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    for provider in providers:
        catalog = _provider_catalog_names(provider)
        if any(name_lower == model.lower() for model in catalog):
            return True
        normalized = normalize_model_id(provider, name_lower, known_ids=catalog)
        if normalized.lower() != name_lower and any(
            normalized.lower() == model.lower() for model in catalog
        ):
            return True
    return False

def _resolve_static_model_alias(
    name_lower: str, current_keys: set[str]) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    if name_lower not in MODEL_ALIASES:
        return None

    def _match(provider: str) -> Optional[str]:
        catalog = _PROVIDER_MODELS.get(provider, ())
        try:
            return resolve_model_alias(name_lower, provider, catalog)
        except AmbiguousModelAliasError as exc:
            # Alias matching belongs to the model domain; provider auto-selection still owns
            # its historical deterministic choice among static candidates.
            matched = {candidate.lower() for candidate in exc.candidates}
            return next((model for model in catalog if model.lower() in matched), None)

    # Current provider first, then native vendors, then aggregators / borrow-list providers the user
    # is already on — so `sonnet` resolves to anthropic before any re-exposing provider.
    aggregators = {p for p in _PROVIDER_MODELS if is_aggregator(p)}
    skip = current_keys | aggregators | _BORROWED_MODEL_PROVIDERS
    candidates = [
        *current_keys, *(p for p in _PROVIDER_MODELS if p not in skip),
        *(p for p in aggregators if p in current_keys),
        *(p for p in _BORROWED_MODEL_PROVIDERS if p in current_keys)]
    for provider in candidates:
        if matched := _match(provider):
            return provider, matched
    return None

def _static_catalog_matches(name: str, current_provider: str):
    """Yield every ``(provider_id, name)`` whose static catalog lists *name*, in ladder order.

    Several first-party providers list the same slug (``gpt-5.6-luna`` on ``openai-api`` AND
    ``openai-codex``); the first is only a guess, so callers that gate on credentials need the
    siblings too (#102775)."""
    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)
    # Step 1: direct static-catalog match. Aggregators list other vendors' models — never
    # auto-switch TO them. A custom endpoint (custom / custom:*) is never auto-switched away
    # from: the user configured it deliberately and may serve the same model name there.
    if current_provider != "custom" and not current_provider.startswith("custom:"):
        for pid in _PROVIDER_MODELS:
            if pid in current_keys or is_aggregator(pid) or pid in _BORROWED_MODEL_PROVIDERS:
                continue
            if _model_in_provider_catalog(name_lower, {pid}):
                yield (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every native-vendor
    # catalog, and only when one is the current provider.
    for pid in _BORROWED_MODEL_PROVIDERS:
        if pid not in current_keys and _model_in_provider_catalog(name_lower, {pid}):
            yield (pid, name)

_SKIP = frozenset({"", "auto", "openrouter", "custom"})

def current_provider_owns_vendor(model_name: str, current_provider: str) -> bool:
    """True when *model_name* belongs to the vendor a single-vendor first-party provider natively
    serves (``gpt-6-astra`` on ``openai-codex``, ``grok-4.6`` on ``xai-oauth``).

    A first-party session plus that vendor's own id is a selection, not a guess: when the live
    catalog could not confirm the id (fetch failed, static fallback lags an early-access rollout)
    the answer is "stay and let the vendor accept or reject it", never "a reseller lists it, so
    switch there". Aggregators, custom endpoints and multi-vendor resellers (nvidia, alibaba, ...)
    have no single native vendor and are skipped."""
    from providers import is_aggregator, normalize_provider, vendor_for_model

    provider = (current_provider or "").strip().lower()
    if provider in _SKIP or provider.startswith("custom:"):
        return False
    normalized = normalize_provider(provider)
    if normalized in _SKIP or is_aggregator(normalized):
        return False
    vendor = vendor_for_model(model_name or "")
    if not vendor:
        return False
    # An id the classifier cannot place (Bedrock ``us.anthropic.claude-…``) is evidence the
    # provider is NOT single-vendor; only a fully classified, single-vendor catalog owns the name.
    native = {vendor_for_model(mid) for mid in _PROVIDER_MODELS.get(normalized, ())}
    return native == {vendor}


def resolve_declared_provider_prefix(model_name: str, configured: set[str]) -> Optional[tuple[str, str]]:
    """Route an explicit ``vendor/model`` prefix (``nous/deepseek-v4-pro``, ``ollama/qwen3.5:4b``) to
    a provider the user defined in ``providers:`` (by raw name or alias) instead of the default.

    ``nous/deepseek-v4-pro`` or ``ollama/qwen3.5:4b`` should route to the named provider instead of falling
    back to the configured default (which silently sends non-default models to the wrong endpoint, #87189).
    """
    if "/" not in model_name:
        return None
    vendor, model = model_name.split("/", 1)
    vendor, model = vendor.strip().lower(), model.strip()
    if not vendor or not model:
        return None
    # An explicitly named provider block (``ollama:``) wins over the alias table, which may
    # canonicalize the same name elsewhere (``ollama`` → ``custom``).
    for candidate in dict.fromkeys((vendor, _normalize_provider(vendor))):
        if candidate in configured:
            return (candidate, model)
    return None

def find_openrouter_slug(model_name: str, ids: list[str]) -> Optional[str]:
    """Full OpenRouter slug for a bare or partial model name (exact slug first, then bare part)."""
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None
    return (
        next((mid for mid in ids if name_lower == mid.lower()), None)
        or next((mid for mid in ids if "/" in mid and name_lower == mid.split("/", 1)[1].lower()), None)
    )
