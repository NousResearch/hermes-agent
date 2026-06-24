from __future__ import annotations

from typing import Any


def _coerce_bool(raw: object) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
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


def _coerce_positive_int(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _coerce_string_list(raw: object) -> list[str] | None:
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        values = [str(item).strip() for item in raw]
    else:
        return None
    values = [item for item in values if item]
    return values


def _get_provider_config(provider_id: str) -> dict[str, Any] | None:
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
    if isinstance(provider_config, dict):
        return provider_config
    return None


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
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

    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def get_provider_disable_streaming(
    provider_id: str, model: str | None = None
) -> bool:
    """Return whether streaming is disabled for a provider/model."""
    if not provider_id:
        return False

    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
        return False

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        model_value = _coerce_bool(model_config.get("disable_streaming"))
        if model_value is not None:
            return model_value

    provider_value = _coerce_bool(provider_config.get("disable_streaming"))
    return bool(provider_value) if provider_value is not None else False


def get_provider_max_tokens(
    provider_id: str, model: str | None = None
) -> int | None:
    """Return a configured output-token cap for a provider/model."""
    if not provider_id:
        return None

    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        for key in ("max_output_tokens", "max_tokens"):
            value = _coerce_positive_int(model_config.get(key))
            if value is not None:
                return value

    for key in ("max_output_tokens", "max_tokens"):
        value = _coerce_positive_int(provider_config.get(key))
        if value is not None:
            return value
    return None


def get_provider_system_prompt_char_limit(
    provider_id: str, model: str | None = None
) -> int | None:
    """Return a configured system prompt character cap for a provider/model."""
    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        value = _coerce_positive_int(model_config.get("system_prompt_char_limit"))
        if value is not None:
            return value

    return _coerce_positive_int(provider_config.get("system_prompt_char_limit"))


def get_provider_tool_allowlist(
    provider_id: str, model: str | None = None
) -> list[str] | None:
    """Return configured tool names allowed for a provider/model."""
    provider_config = _get_provider_config(provider_id)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        values = _coerce_string_list(model_config.get("tool_allowlist"))
        if values is not None:
            return values

    return _coerce_string_list(provider_config.get("tool_allowlist"))


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
