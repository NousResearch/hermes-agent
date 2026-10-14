"""Copy the model and its selected custom endpoint when creating an independent profile."""

from hermes_cli.providers import custom_provider_aliases
from hermes_cli.config import is_provider_enabled


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
    for key, entry in providers.items() if isinstance(providers, dict) else ():
        if (
            isinstance(entry, dict)
            and is_provider_enabled(entry)
            and (entry.get("api") or entry.get("url") or entry.get("base_url"))
            and requested in custom_provider_aliases(entry.get("name") or key, key)
        ):
            seed["providers"] = {key: entry}
            return seed
    custom_providers = source_cfg.get("custom_providers")
    for entry in custom_providers if isinstance(custom_providers, list) else ():
        if isinstance(entry, dict) and requested in custom_provider_aliases(
            entry.get("name", ""), entry.get("provider_key", "")
        ):
            seed["custom_providers"] = [entry]
            return seed
    return seed
