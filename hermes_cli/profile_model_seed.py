"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.config import _expand_env_vars, find_provider_entry
from hermes_cli.config_providers import _normalize_custom_provider_entry
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
