"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.config import (
    find_provider_entry,
    get_compatible_custom_providers,
)
from hermes_cli.runtime_provider_custom import (
    _get_named_custom_provider,
    _shadowed_by_builtin,
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
    selected = _get_named_custom_provider(
        provider, config=source_cfg, resolve_credentials=False
    )
    if selected is None:
        # Exact built-in keys carry provider tuning even without an endpoint URL.
        exact = providers.get(provider)
        if (
            isinstance(exact, dict)
            and requested not in {"", "auto"}
            and _shadowed_by_builtin(requested)
        ):
            seed["providers"] = {provider: exact}
        return seed
    key, entry = find_provider_entry(providers, selected.get("provider_key"))
    if entry is not None:
        seed["providers"] = {key: entry}
        return seed
    custom_providers = source_cfg.get("custom_providers")
    for entry in custom_providers if isinstance(custom_providers, list) else ():
        normalized = get_compatible_custom_providers({"custom_providers": [entry]})
        if normalized and (normalized[0]["name"], normalized[0]["base_url"]) == (
            selected["name"],
            selected["base_url"],
        ):
            seed["custom_providers"] = [entry]
            return seed
    return seed
