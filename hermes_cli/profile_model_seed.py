"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.config import _expand_env_vars, find_provider_entry, is_provider_enabled
from hermes_cli.config_providers import _normalize_custom_provider_entry
from hermes_cli.providers import custom_provider_aliases
from hermes_cli.runtime_provider_custom import (
    _get_named_custom_provider,
    _normalize_custom_provider_name,
    _shadowed_by_builtin,
)


def launch_model_seed(source_cfg: dict) -> dict:
    """The config a fresh profile needs to run the launch profile's model: its ``model`` block plus
    the endpoint definition it points at. ``model.provider: custom:lab`` alone is "Unknown provider"
    on the first turn. The selected ``providers:`` or ``custom_providers:`` entry is copied as
    written (``${VAR}`` refs stay refs), and an exact ``providers.<built-in>`` / ``providers.custom``
    key travels for its tuning even without a URL. ``{}`` when the launch profile has no model."""
    model_cfg = source_cfg.get("model")
    if not model_cfg:
        return {}
    seed = {"model": model_cfg}
    provider = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    if not isinstance(provider, str):
        return seed
    # Runtime expands ``model.provider: ${VAR}`` before resolving; select with the same value.
    provider = str(_expand_env_vars(provider))
    requested = _normalize_custom_provider_name(provider)
    providers = source_cfg.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    selected = _get_named_custom_provider(
        provider, config=source_cfg, resolve_credentials=False
    )
    if selected is None:
        exact = providers.get(provider)
        # Bare ``custom`` runs as provider ``custom`` and reads ``providers.custom`` tuning too.
        if (
            isinstance(exact, dict)
            and requested not in {"", "auto"}
            and (requested == "custom" or _shadowed_by_builtin(requested))
        ):
            seed["providers"] = {provider: exact}
        return seed
    key, entry = find_provider_entry(providers, selected.get("provider_key"))
    if entry is not None:
        seed["providers"] = {key: entry}
        return seed
    custom_providers = source_cfg.get("custom_providers")
    for entry in custom_providers if isinstance(custom_providers, list) else ():
        normalized = _normalize_custom_provider_entry(entry)
        if normalized and (normalized["name"], normalized["base_url"]) == (
            selected["name"],
            selected["base_url"],
        ):
            seed["custom_providers"] = [entry]
            return seed
    return seed


def _identity(entry, key: str = "") -> frozenset[str]:
    """Aliases runtime can match *entry* by: empty for non-dicts, disabled entries and entries
    the shared normalizer rejects (no usable URL; ``baseUrl`` counts, as it does at runtime)."""
    if not isinstance(entry, dict) or not is_provider_enabled(entry):
        return frozenset()
    if _normalize_custom_provider_entry(entry, provider_key=key) is None:
        return frozenset()
    key = key or str(entry.get("provider_key") or "")
    return custom_provider_aliases(str(entry.get("name") or key), key)


def merge_endpoint_seed(dst_cfg: dict, seed: dict) -> dict:
    """Return a copy of *dst_cfg* with the seeded endpoint written first. Runtime takes the first
    alias match (keyed ``providers:`` before ``custom_providers:``), so destination entries with a
    URL that claim the same identity would shadow the inherited endpoint and are dropped; a
    same-key destination entry is replaced whole. Entries runtime cannot match (no URL, disabled)
    never shadow: such a seed replaces only its own key, and such destination entries are kept."""
    seed_providers = seed.get("providers") or {}
    seed_custom = seed.get("custom_providers") or []
    claimed: set[str] = set()
    for key, entry in seed_providers.items():
        claimed |= _identity(entry, str(key))
    for entry in seed_custom:
        claimed |= _identity(entry)

    merged = dict(dst_cfg)
    dst_providers = dst_cfg.get("providers")
    if isinstance(dst_providers, dict) or seed_providers:
        kept = {
            k: v for k, v in (dst_providers if isinstance(dst_providers, dict) else {}).items()
            if k not in seed_providers and not claimed & _identity(v, str(k))
        }
        merged["providers"] = {**seed_providers, **kept}
    dst_custom = dst_cfg.get("custom_providers")
    if isinstance(dst_custom, list) or seed_custom:
        kept_list = [
            e for e in (dst_custom if isinstance(dst_custom, list) else [])
            if not claimed & _identity(e)
        ]
        merged["custom_providers"] = seed_custom + kept_list
    return merged
