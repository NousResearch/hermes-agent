from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _provider_config_keys(provider_id: str, providers: dict) -> list[str]:
    """Provider keys to try: named custom aliases first, then bare provider_id.

    For bare "custom", tries every named custom provider's config key (the durable slug
    under providers.<key>) before falling back to providers.custom, so a user setting
    stale_timeout_seconds under providers.sglgateway has it honoured when the runtime
    provider is "custom".
    """
    if not provider_id or provider_id.strip().lower() != "custom":
        return [provider_id]
    if not isinstance(providers, dict):
        return ["custom"]
    candidates: list[str] = []
    for key, entry in providers.items():
        if not isinstance(entry, dict):
            continue
        base_url = entry.get("base_url") or entry.get("url") or entry.get("api")
        if not (isinstance(base_url, str) and base_url.strip()):
            continue
        enabled = entry.get("enabled")
        if enabled is False:
            continue
        candidates.append(str(key))
    return candidates + ["custom"]


def _configured_timeout(
    provider_id: str, model: str | None, model_key: str, provider_key: str
) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``."""
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None
    for key in _provider_config_keys(provider_id, providers):
        provider_config = providers.get(key, {})
        if not isinstance(provider_config, dict):
            continue
        model_config = _get_model_config(provider_config, model)
        if model_config is not None:
            timeout = _coerce_timeout(model_config.get(model_key))
            if timeout is not None:
                return timeout
        timeout = _coerce_timeout(provider_config.get(provider_key))
        if timeout is not None:
            return timeout
    return None


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _configured_timeout(
        provider_id, model, "timeout_seconds", "request_timeout_seconds"
    )


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _configured_timeout(
        provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds"
    )


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None
