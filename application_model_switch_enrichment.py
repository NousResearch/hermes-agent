"""Application enrichment for a canonically resolved application model switch."""

from __future__ import annotations

from typing import Any, Mapping

from agent.model_warnings import nous_hermes_non_agentic_warning
from agent.native_compaction import resolve_native_compaction_capabilities
from providers import get_provider_label, match_configured_provider


class ModelValidationError(ValueError):
    pass


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


def validate_model_switch(result, config: Mapping[str, Any]) -> str:
    try:
        from models.catalog_chat import is_known_non_chat_model

        if is_known_non_chat_model(result.new_model):
            raise ModelValidationError(
                f"{result.new_model} is a generation model and cannot be used for chat."
            )
    except ModelValidationError:
        raise
    except Exception:
        pass

    try:
        from hermes_cli.models_validate import validate_requested_model

        verdict = validate_requested_model(
            result.new_model,
            result.target_provider,
            api_key=result.api_key,
            base_url=result.base_url,
            api_mode=result.api_mode or None,
        )
    except Exception as exc:
        verdict = {
            "accepted": False,
            "message": f"Could not validate {result.new_model}: {exc}",
        }
    if verdict.get("accepted"):
        return str(verdict.get("message") or "")

    configured = _configured(config, result.target_provider)
    declared = False
    if configured is not None:
        raw = configured.raw
        values = raw.get("models") if isinstance(raw, Mapping) else None
        if isinstance(values, Mapping):
            declared = result.new_model in values
        elif isinstance(values, (list, tuple)):
            declared = result.new_model in values
        declared = declared or result.new_model in {
            str(raw.get("model") or ""),
            str(raw.get("default_model") or ""),
        }
    if not declared:
        raise ModelValidationError(
            str(verdict.get("message") or f"Invalid model {result.new_model}")
        )
    return str(verdict.get("message") or "")


def enrich_model_switch(result, config: Mapping[str, Any]):
    validation_warning = validate_model_switch(result, config)
    configured = _configured(config, result.target_provider)
    result.provider_label = (
        configured.name
        if configured is not None and configured.name
        else get_provider_label(result.target_provider)
    )
    if configured is not None and configured.extra_body:
        result.request_overrides = {"extra_body": dict(configured.extra_body)}

    from agent.models_dev import get_model_info, query_model_metadata

    result.capabilities = query_model_metadata(
        result.target_provider, result.new_model, allow_network=True, config=dict(config)
    )
    result.model_info = get_model_info(
        result.target_provider, result.new_model, allow_network=True, config=dict(config)
    )
    result.runtime_capabilities = resolve_native_compaction_capabilities(
        model=result.new_model,
        base_url=result.base_url,
        provider=result.target_provider,
        is_codex_backend=result.target_provider.strip().lower() == "openai-codex",
    )
    warnings = [
        warning
        for warning in (
            validation_warning,
            nous_hermes_non_agentic_warning(result.new_model),
        )
        if warning
    ]
    result.warning_message = " | ".join(warnings)
    return result


__all__ = ["ModelValidationError", "enrich_model_switch", "validate_model_switch"]
