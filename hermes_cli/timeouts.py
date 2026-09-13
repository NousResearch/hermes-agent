from __future__ import annotations

import os


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _configured_timeout(provider_id: str, model: str | None, model_key: str, provider_key: str) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``."""
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider_config, dict):
        return None
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get(model_key))
        if timeout is not None:
            return timeout
    return _coerce_timeout(provider_config.get(provider_key))


def get_provider_request_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "timeout_seconds", "request_timeout_seconds")


def get_provider_stale_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds")


def resolve_bedrock_sdk_timeout(model: str | None = None) -> float:
    """Socket read timeout for every Bedrock transport (Converse, AnthropicBedrock, Mantle).

    ``providers.bedrock.request_timeout_seconds`` (or a per-model ``timeout_seconds``)
    wins; otherwise ``HERMES_API_TIMEOUT``; otherwise 1800s. This is the value that
    must reach boto3 ``read_timeout`` / httpx — compression worker budgets do not.
    """
    configured = get_provider_request_timeout("bedrock", model)
    if configured is not None:
        return configured
    env_timeout = _coerce_timeout(os.getenv("HERMES_API_TIMEOUT"))
    return env_timeout if env_timeout is not None else 1800.0


def _get_model_config(provider_config: dict[str, object], model: str | None) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None
