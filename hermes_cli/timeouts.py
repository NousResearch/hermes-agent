from __future__ import annotations


def _coerce_optional_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


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


def get_provider_streaming_enabled(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> bool | None:
    """Return provider/model streaming override, if configured.

    ``None`` means no explicit override; callers should use the default runtime
    streaming policy. ``False`` disables stream=True for the matching provider or
    model route, which is useful for proxies/models that expose non-streaming
    Chat Completions only. ``base_url`` lets custom endpoints resolve settings
    from their named ``providers`` / ``custom_providers`` entry even when the
    runtime provider id is the generic ``custom``.
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
        provider_config = {}

    streaming = _get_streaming_from_provider_config(provider_config, model)
    if streaming is not None:
        return streaming

    if base_url:
        custom_config = _get_custom_provider_config_by_base_url(config, base_url)
        if custom_config is not None:
            streaming = _get_streaming_from_provider_config(custom_config, model)
            if streaming is not None:
                return streaming
    return None


def _get_streaming_from_provider_config(
    provider_config: dict[str, object], model: str | None
) -> bool | None:
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        streaming = _coerce_optional_bool(model_config.get("streaming"))
        if streaming is not None:
            return streaming
    return _coerce_optional_bool(provider_config.get("streaming"))


def _get_custom_provider_config_by_base_url(
    config: dict[str, object], base_url: str
) -> dict[str, object] | None:
    target = str(base_url or "").rstrip("/").lower()
    if not target:
        return None

    try:
        from hermes_cli.config import get_compatible_custom_providers

        entries = get_compatible_custom_providers(config)
    except Exception:
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_url = str(entry.get("base_url") or "").rstrip("/").lower()
        if entry_url == target:
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
