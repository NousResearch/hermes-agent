"""Gateway /model resolution over canonical selection and routing domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from providers import get_provider_profile, match_configured_provider

from application_model_switch_enrichment import ModelValidationError, enrich_model_switch
from gateway.session_model_resolution import resolve_session_model


class ModelSwitchResolutionError(ValueError):
    pass


@dataclass(slots=True)
class GatewayModelSwitchResult:
    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    api_mode: str = ""
    runtime_kind: str = ""
    request_overrides: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    warning_message: str = ""
    provider_label: str = ""
    resolved_via_alias: str = ""
    capabilities: Any = None
    runtime_capabilities: dict[str, bool] = field(default_factory=dict)
    model_info: Any = None
    is_global: bool = False


def _configured(config: Mapping[str, Any], provider: str):
    return match_configured_provider(
        provider,
        providers=(
            config.get("providers")
            if isinstance(config.get("providers"), Mapping)
            else None
        ),
        custom_providers=(
            config.get("custom_providers")
            if isinstance(config.get("custom_providers"), list)
            else None
        ),
    )


def _provider_only_model(
    config: Mapping[str, Any],
    provider: str,
    current_base_url: str,
) -> str:
    configured = _configured(config, provider)
    if configured is not None and configured.model:
        return configured.model
    profile = get_provider_profile(provider)
    if profile is not None and profile.fallback_models:
        return str(profile.fallback_models[0])
    base_url = (
        configured.base_url
        if configured is not None
        else profile.base_url
        if profile is not None
        else current_base_url
    )
    if base_url:
        try:
            from models.catalog_probe import detect_single_openai_model

            if model := detect_single_openai_model(base_url):
                return str(model)
        except Exception:
            pass
    raise ModelSwitchResolutionError(
        f"No model detected for provider '{provider}'. Specify the model explicitly."
    )


def resolve_model_switch(
    *,
    config: Mapping[str, Any],
    raw_input: str,
    current_provider: str,
    current_model: str,
    current_base_url: str = "",
    current_api_key: str = "",
    explicit_provider: str = "",
    is_global: bool = False,
) -> GatewayModelSwitchResult:
    del current_model
    target = str(raw_input or "").strip()
    if explicit_provider and not target:
        target = _provider_only_model(config, explicit_provider, current_base_url)
    if not target:
        raise ModelSwitchResolutionError("No model specified.")

    try:
        resolved = resolve_session_model(
            config=config,
            raw_model=target,
            explicit_provider=explicit_provider,
            current_provider=current_provider,
            current_base_url=current_base_url,
            current_api_key=current_api_key,
        )
        result = GatewayModelSwitchResult(
            success=True,
            new_model=resolved.model,
            target_provider=resolved.provider,
            provider_changed=resolved.provider_changed,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            api_mode=resolved.api_mode,
            runtime_kind=resolved.runtime_kind,
            is_global=is_global,
        )
        return enrich_model_switch(result, config)
    except ModelValidationError as exc:
        raise ModelSwitchResolutionError(str(exc)) from exc
    except ModelSwitchResolutionError:
        raise
    except Exception as exc:
        raise ModelSwitchResolutionError(
            f"Could not resolve model '{target}'."
        ) from exc


__all__ = [
    "GatewayModelSwitchResult",
    "ModelSwitchResolutionError",
    "resolve_model_switch",
]
