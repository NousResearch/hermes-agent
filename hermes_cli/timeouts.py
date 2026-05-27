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


def get_provider_header_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider header timeout in seconds, if any."""
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
        timeout = _coerce_timeout(model_config.get("header_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("header_timeout_seconds"))


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


def build_provider_timeout(provider_id: str, model: str | None = None):
    """Build an httpx.Timeout (or float) for the given provider/model.
    Returns httpx.Timeout if header_timeout_seconds is configured, float if only
    request_timeout_seconds, or None if neither."""
    req_timeout = get_provider_request_timeout(provider_id, model)
    hdr_timeout = get_provider_header_timeout(provider_id, model)
    if hdr_timeout is None:
        return req_timeout
    try:
        from httpx import Timeout
    except ImportError:
        return req_timeout
    overall = req_timeout if (req_timeout is not None and req_timeout > 0) else 1800.0
    return Timeout(timeout=overall, connect=10.0, read=float(hdr_timeout))
