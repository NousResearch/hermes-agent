"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.config import (
    get_compatible_custom_providers,
    is_provider_enabled,
    providers_dict_to_custom_providers,
)
from hermes_cli.providers import custom_provider_aliases
from hermes_cli.runtime_provider_custom import _shadowed_by_builtin


def _matches_custom_provider(requested: str, entries: list[dict]) -> bool:
    return bool(
        entries
        and requested
        in custom_provider_aliases(
            entries[0]["name"], entries[0].get("provider_key", "")
        )
    )


def launch_model_seed(source_cfg: dict) -> dict:
    """Keep raw credential references and the original provider schema in the new profile."""
    model_cfg = source_cfg.get("model")
    if not model_cfg:
        return {}
    seed = {"model": model_cfg}
    provider = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    if not isinstance(provider, str):
        return seed
    requested = provider.strip().lower().replace(" ", "-")
    providers = source_cfg.get("providers")
    providers = providers if isinstance(providers, dict) else {}
    exact = providers.get(provider)
    # Exact config keys also carry built-in provider tuning, without an endpoint URL.
    if isinstance(exact, dict) and is_provider_enabled(exact):
        seed["providers"] = {provider: exact}
        return seed
    if not requested or requested == "auto" or _shadowed_by_builtin(requested):
        return seed
    for key, entry in providers.items():
        if _matches_custom_provider(
            requested, providers_dict_to_custom_providers({key: entry})
        ):
            seed["providers"] = {key: entry}
            return seed
    custom_providers = source_cfg.get("custom_providers")
    for entry in custom_providers if isinstance(custom_providers, list) else ():
        if _matches_custom_provider(
            requested, get_compatible_custom_providers({"custom_providers": [entry]})
        ):
            seed["custom_providers"] = [entry]
            return seed
    return seed
