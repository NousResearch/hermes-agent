"""Pure curated catalogue projections from caller-supplied live facts."""

from __future__ import annotations

from typing import Any, Optional

from models.metadata.pricing import _zero_priced

def _openrouter_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("prompt", "completion"), "0")

def _openrouter_model_supports_tools(item: Any) -> bool:
    """True when ``supported_parameters`` advertises ``tools`` (hermes-agent is tool-calling-first).
    Permissive when the field is absent/malformed: some OpenRouter-compatible gateways (Nous Portal,
    private mirrors) don't populate it, and the picker must not silently empty for them.

    Ported from Kilo-Org/kilocode#9068.
    """
    params = item.get("supported_parameters") if isinstance(item, dict) else None
    return "tools" in params if isinstance(params, list) else True

def _ai_gateway_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("input", "output"), "0")


def project_openrouter_catalog(fallback, live_by_id, silent_default):
    curated = []
    for preferred_id, _ in fallback:
        live_item = live_by_id.get(preferred_id)
        if live_item is None or not _openrouter_model_supports_tools(live_item):
            continue
        desc = "default" if preferred_id == silent_default else (
            "free" if _openrouter_model_is_free(live_item.get("pricing")) else "")
        curated.append((preferred_id, desc))
    if curated and not curated[0][1]:
        curated[0] = (curated[0][0], "recommended")
    return curated


def project_ai_gateway_catalog(fallback, live_by_id):
    curated = [(mid, "free" if _ai_gateway_model_is_free(live_by_id[mid].get("pricing")) else "")
               for mid, _ in fallback if mid in live_by_id]
    if not curated:
        return []
    free_moonshot = next((mid for mid, item in live_by_id.items()
                         if mid.startswith("moonshotai/") and _ai_gateway_model_is_free(item.get("pricing"))), None)
    if free_moonshot:
        return [(free_moonshot, "recommended")] + [(mid, desc) for mid, desc in curated if mid != free_moonshot]
    curated[0] = (curated[0][0], "recommended")
    return curated


_MODEL_SEARCH_ALIASES: dict[str, tuple[str, ...]] = {
    "k3": ("kimi-k3", "kimi"),
    # OpenCode Zen serves the "Ox Alpha" stealth model under an opaque
    # preview slug; let users find it by its public codename.
    "x-preview-f-free": ("ox-alpha", "ox")}

_MODEL_ALIAS_CANONICAL: dict[str, str] = {
    wire_id: aliases[0].lower() for wire_id, aliases in _MODEL_SEARCH_ALIASES.items() if aliases}

def model_alias_canonical(model: str) -> str:
    """Return the canonical public slug for a bare wire-id alias."""
    key = (model or "").strip().lower()
    return _MODEL_ALIAS_CANONICAL.get(key, key)


from models import catalog_static

_OPENCODE_FREE_EXCLUDED_MODELS = frozenset(
    {"ox-alpha-free", "deepseek-v4-flash-free", "x-preview-f-free"}
)

def _merge_unique(primary: list[str], secondary: list[str], key=lambda m: str(m).lower()) -> list[str]:
    """``primary`` verbatim, then ``secondary`` entries whose ``key`` is new (deduped as it goes)."""
    merged, seen = list(primary), {key(m) for m in primary}
    for m in secondary:
        k = key(m)
        if k not in seen:
            seen.add(k)
            merged.append(m)
    return merged

def _model_dedup_key(model_id: str) -> str:
    return model_alias_canonical(str(model_id).strip().lower())

def _drop_delisted_opencode_models(normalized: str, rows: Optional[list[str]]) -> Optional[list[str]]:
    """The relay still LISTS delisted ids it no longer serves, and the curated floor (merged back in
    as the secondary half, or served alone when there is no key) carries retired ids too. Filter the
    FINAL rows for the live-first Zen/Go pickers so no path can offer a slug that 401s (#111749,
    #115496)."""
    if rows and normalized in catalog_static._LIVE_FIRST_PICKER_PROVIDERS:
        return type(rows)(m for m in rows if str(m).lower() not in _OPENCODE_FREE_EXCLUDED_MODELS)
    return rows


def merge_profile_models(provider: str, live: list[str], curated: list[str]) -> list[str]:
    primary, secondary = (live, curated) if provider in catalog_static._LIVE_FIRST_PICKER_PROVIDERS else (curated, live)
    return _drop_delisted_opencode_models(provider, _merge_unique(primary, secondary, key=_model_dedup_key))
