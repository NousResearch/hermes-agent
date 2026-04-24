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

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


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


def get_provider_session_id_header_name(
    provider_id: str | None, model: str | None = None
) -> str | None:
    """Return the configured session-ID header name for the given provider/model.

    Reads ``providers.<provider_id>.session_id_header_name`` with optional
    model-level override under
    ``providers.<provider_id>.models.<model>.session_id_header_name``.
    Returns ``None`` when not configured for this provider.
    """
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

    # Model-level override first (most specific wins)
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        val = model_config.get("session_id_header_name")
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Provider-level fallback
    val = provider_config.get("session_id_header_name")
    if isinstance(val, str) and val.strip():
        return val.strip()

    return None
