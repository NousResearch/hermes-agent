"""Gateway-side provider/fallback validation for agent turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.provider_errors import (
    HermesFallbackConfigError,
    HermesProviderConfigError,
    redacted_url_class,
)


@dataclass(frozen=True)
class GatewayAgentTurnValidation:
    provider: str
    model: str
    base_url_class: str
    auth_present: bool
    fallback_status: str
    fallback_count: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _identity(provider: str, model: str, base_url: str) -> tuple[str, str, str]:
    return (
        _clean(provider).lower(),
        _clean(model).lower(),
        _clean(base_url).rstrip("/").lower(),
    )


def _raw_fallback_entries(config: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    entries: list[dict[str, Any]] = []
    saw_key = False
    for key in ("fallback_providers", "fallback_model"):
        if key not in config:
            continue
        saw_key = True
        raw = config.get(key)
        if raw is None:
            raise HermesFallbackConfigError(
                f"{key} is null; remove it or configure provider/model entries"
            )
        if isinstance(raw, dict):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = raw
        else:
            raise HermesFallbackConfigError(
                f"{key} must be a fallback object or list, got {type(raw).__name__}"
            )
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise HermesFallbackConfigError(
                    f"{key}[{idx}] must be a fallback object, got {type(item).__name__}"
                )
            provider = _clean(item.get("provider"))
            model = _clean(item.get("model"))
            if not provider or not model:
                raise HermesFallbackConfigError(
                    f"{key}[{idx}] must include non-empty provider and model"
                )
            entries.append(dict(item, provider=provider, model=model))
    return entries, saw_key


def _provider_factory_known(config: dict[str, Any], provider: str) -> bool:
    if provider in {"custom", "openai"}:
        return True
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        if provider in PROVIDER_REGISTRY:
            return True
    except Exception:
        pass

    custom_providers = config.get("custom_providers")
    if isinstance(custom_providers, list):
        for entry in custom_providers:
            if isinstance(entry, dict) and _clean(entry.get("name")).lower() == provider.lower():
                return True
    return False


def validate_gateway_agent_turn_config(
    config: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
) -> GatewayAgentTurnValidation:
    """Validate the provider/fallback shape without exposing credentials."""
    cfg = config if isinstance(config, dict) else {}
    rt = runtime if isinstance(runtime, dict) else {}

    provider = _clean(rt.get("provider"))
    model = _clean(rt.get("model"))
    base_url = _clean(rt.get("base_url"))
    api_key = rt.get("api_key")
    command = rt.get("command")
    auth_present = bool(api_key or command)

    if not provider:
        raise HermesProviderConfigError("provider name is required")
    if not model:
        raise HermesProviderConfigError("model is required")
    if not base_url and not command:
        raise HermesProviderConfigError("provider endpoint/base_url is required")
    if not _provider_factory_known(cfg, provider):
        raise HermesProviderConfigError(f"unknown provider factory: {provider}")
    if not auth_present:
        raise HermesProviderConfigError(
            f"provider {provider} has no configured auth material"
        )

    warnings: list[str] = []
    if provider == "openai-codex" and redacted_url_class(base_url) == "chatgpt.com":
        warnings.append(
            "openai-codex chatgpt.com backend is not the stable OpenAI Responses API path"
        )

    entries, _saw_fallback_key = _raw_fallback_entries(cfg)
    if not entries:
        fallback_status = "no_valid_fallback_configured"
    else:
        primary_id = _identity(provider, model, base_url)
        valid: list[dict[str, Any]] = []
        invalid_same = 0
        for entry in entries:
            entry_id = _identity(entry.get("provider"), entry.get("model"), entry.get("base_url"))
            allow_same = bool(entry.get("allow_same_provider_fallback"))
            if entry_id == primary_id and not allow_same:
                invalid_same += 1
                continue
            valid.append(entry)
        if not valid:
            raise HermesFallbackConfigError(
                "fallback_status=no_valid_fallback_configured; "
                "fallback entries resolve to the current provider/model/base_url"
            )
        fallback_status = "valid_fallback_configured"
        if invalid_same:
            warnings.append("same-provider fallback entry ignored")

    return GatewayAgentTurnValidation(
        provider=provider,
        model=model,
        base_url_class=redacted_url_class(base_url),
        auth_present=auth_present,
        fallback_status=fallback_status,
        fallback_count=len(entries),
        warnings=tuple(warnings),
    )
