from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderStallRecoveryConfig:
    enabled: bool = True
    health_probe_enabled: bool = False
    health_probe_timeout_seconds: float = 5.0
    same_provider_retries: int = 1


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_provider_stall_recovery_config() -> ProviderStallRecoveryConfig:
    """Return the bounded, fail-safe provider stall recovery policy."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        config = {}

    agent = config.get("agent", {}) if isinstance(config, dict) else {}
    raw = agent.get("provider_stall_recovery", {}) if isinstance(agent, dict) else {}
    if not isinstance(raw, dict):
        raw = {}

    probe_timeout = _coerce_timeout(raw.get("health_probe_timeout_seconds")) or 5.0
    try:
        retries = int(raw.get("same_provider_retries", 1))
    except (TypeError, ValueError):
        retries = 1

    return ProviderStallRecoveryConfig(
        enabled=raw.get("enabled", True) is not False,
        health_probe_enabled=raw.get("health_probe_enabled", False) is True,
        health_probe_timeout_seconds=min(30.0, max(1.0, probe_timeout)),
        same_provider_retries=min(1, max(0, retries)),
    )


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
