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


def _coerce_bool(raw: object) -> bool:
    """Best-effort truthiness coercion for a config value.

    Returns True only for clear truthy primitives (True, non-zero numbers,
    the strings 'true'/'yes'/'on'/'1', case-insensitive). Anything else
    (None, '', 'false', 0, list, dict, garbage) is False. Never raises.
    """
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "on", "1")
    return False


def get_provider_follow_redirects(
    provider_id: str, model: str | None = None
) -> bool:
    """Return whether the LLM API client should follow HTTP redirects for a provider.

    Read from ``providers.<id>.follow_redirects`` in config.yaml, with an optional
    ``providers.<id>.models.<model>.follow_redirects`` override. Default is False
    (httpx's native default) so behavior is unchanged unless the user opts in.

    TheGrid.ai and similar providers occasionally issue a 307/308 to a different
    path on the same host; without this opt-in, the first request fails with a
    redirect response instead of being followed.
    """
    if not provider_id:
        return False

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return False

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return False

    model_config = _get_model_config(provider_config, model)
    if model_config is not None and "follow_redirects" in model_config:
        return _coerce_bool(model_config.get("follow_redirects"))

    return _coerce_bool(provider_config.get("follow_redirects"))
