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

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


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

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def get_provider_stale_fallback_threshold(
    provider_id: str, model: str | None = None
) -> int:
    """Return the consecutive stale-stream kill count that triggers fallback.

    Resolution order (first match wins):
      1. providers.<id>.models.<model>.stale_fallback_threshold
      2. providers.<id>.stale_fallback_threshold
      3. top-level stale_fallback_threshold
      4. default of 2

    Returns 0 to disable stale-stream fallback escalation entirely.
    Negative or non-integer values are clamped/ignored and fall through
    to the next source.
    """
    default = 2

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return default

    if not isinstance(config, dict):
        return default

    def _coerce(raw: object) -> int | None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    if provider_id:
        providers = config.get("providers", {})
        provider_config = (
            providers.get(provider_id, {}) if isinstance(providers, dict) else {}
        )
        if isinstance(provider_config, dict):
            model_config = _get_model_config(provider_config, model)
            if model_config is not None:
                threshold = _coerce(model_config.get("stale_fallback_threshold"))
                if threshold is not None:
                    return threshold
            threshold = _coerce(provider_config.get("stale_fallback_threshold"))
            if threshold is not None:
                return threshold

    threshold = _coerce(config.get("stale_fallback_threshold"))
    if threshold is not None:
        return threshold

    return default


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
