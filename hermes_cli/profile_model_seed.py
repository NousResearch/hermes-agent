"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.config import _expand_env_vars, find_provider_entry
from hermes_cli.config_providers import _normalize_custom_provider_entry
from hermes_cli.providers import custom_provider_aliases
from hermes_cli.runtime_provider_custom import (
    _entry_url,
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


def merge_endpoint_seed(dst_cfg: dict, seed: dict) -> dict:
    """*dst_cfg* with the seeded endpoint written ahead of, and instead of, destination entries
    claiming the same identity. Runtime takes the first alias match (keyed ``providers:`` before
    ``custom_providers:``), so a colliding entry already in the profile would shadow the inherited
    endpoint. URL-less tuning seeds only replace their own key."""
    seed_providers = seed.get("providers") or {}
    seed_custom = seed.get("custom_providers") or []
    claimed: frozenset[str] = frozenset().union(
        *(custom_provider_aliases(str(e.get("name") or k), str(k))
          for k, e in seed_providers.items() if _entry_url(e)),
        *(custom_provider_aliases(str(e.get("name") or ""), str(e.get("provider_key") or ""))
          for e in seed_custom),
    )

    def unclaimed(entry, key="") -> bool:
        if not isinstance(entry, dict):
            return True
        key = key or str(entry.get("provider_key") or "")
        return not claimed & custom_provider_aliases(str(entry.get("name") or key), key)

    merged = dict(dst_cfg)
    existing = dst_cfg.get("providers")
    if isinstance(existing, dict) or seed_providers:
        merged["providers"] = {**seed_providers, **{
            k: v for k, v in (existing if isinstance(existing, dict) else {}).items()
            if k not in seed_providers and unclaimed(v, str(k))}}
    existing = dst_cfg.get("custom_providers")
    if isinstance(existing, list) or seed_custom:
        merged["custom_providers"] = seed_custom + [
            e for e in (existing if isinstance(existing, list) else []) if unclaimed(e)]
    return merged
