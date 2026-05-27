from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    for provider_config in _get_provider_configs(config, provider_id):
        model_config = _get_model_config(provider_config, model)
        if model_config is not None:
            timeout = _coerce_timeout(model_config.get("timeout_seconds"))
            if timeout is not None:
                return timeout

        timeout = _coerce_timeout(provider_config.get("request_timeout_seconds"))
        if timeout is not None:
            return timeout

    return None


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    for provider_config in _get_provider_configs(config, provider_id):
        model_config = _get_model_config(provider_config, model)
        if model_config is not None:
            timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
            if timeout is not None:
                return timeout

        timeout = _coerce_timeout(provider_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return None


def _get_provider_configs(
    config: dict[str, object], provider_id: str
) -> list[dict[str, object]]:
    provider_configs: list[dict[str, object]] = []
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if isinstance(providers, dict):
        provider_config = providers.get(provider_id)
        if isinstance(provider_config, dict):
            provider_configs.append(provider_config)

    if not provider_id.startswith("custom:"):
        return provider_configs

    custom_name = provider_id.split(":", 1)[1].strip().casefold()
    custom_config = _get_custom_provider_config(config, custom_name)
    if custom_config is not None:
        provider_configs.append(custom_config)

    if isinstance(providers, dict):
        provider_config = providers.get("custom")
        if isinstance(provider_config, dict):
            provider_configs.append(provider_config)
    return provider_configs


def _get_custom_provider_config(
    config: dict[str, object], custom_name: str
) -> dict[str, object] | None:
    custom_providers = config.get("custom_providers", []) if isinstance(config, dict) else []
    if isinstance(custom_providers, dict):
        custom_providers = list(custom_providers.values())
    if not isinstance(custom_providers, list):
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("name", "")).strip().casefold()
        if entry_name == custom_name:
            return entry
    return None


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
