from __future__ import annotations


def _coerce_timeout(value: object) -> float | None:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _coerce_optional_timeout(value: object) -> float | None:
    """Coerce timeout values where 0 is a valid explicit override.

    Used by opt-in watchdogs so ``0`` can mean "disabled" at the caller.
    Negative values are still treated as invalid/missing.
    """
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout < 0:
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


def get_provider_content_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured stream visible-output stale timeout, if any.

    This is separate from ``stale_timeout_seconds``: the normal stale timeout
    tracks wire activity, while this timeout tracks real assistant output
    (visible text or tool-call deltas) so reasoning-only streams can be cut
    off without penalizing healthy streams that are emitting content.
    """
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

    keys = ("content_stale_timeout_seconds", "stream_content_stale_timeout_seconds")
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        for key in keys:
            timeout = _coerce_optional_timeout(model_config.get(key))
            if timeout is not None:
                return timeout

    for key in keys:
        timeout = _coerce_optional_timeout(provider_config.get(key))
        if timeout is not None:
            return timeout
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
