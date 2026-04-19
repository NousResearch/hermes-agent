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
        from hermes_cli.config import load_config
    except ImportError:
        return None

    config = load_config()
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    if model:
        models = provider_config.get("models", {})
        model_config = models.get(model, {}) if isinstance(models, dict) else {}
        if isinstance(model_config, dict):
            timeout = _coerce_timeout(model_config.get("timeout_seconds"))
            if timeout is not None:
                return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))
