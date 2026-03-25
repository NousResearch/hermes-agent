"""Shared model-switching logic for CLI and gateway /model commands.

Both the CLI (cli.py) and gateway (gateway/run.py) /model handlers
share the same core pipeline:

  parse_model_input → is_custom detection → auto-detect provider
  → credential resolution → validate model → return result

This module extracts that shared pipeline into pure functions that
return result objects. The callers handle all platform-specific
concerns: state mutation, config persistence, output formatting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSwitchResult:
    """Result of a model switch attempt."""

    success: bool
    new_model: str = ""
    target_provider: str = ""
    provider_changed: bool = False
    api_key: str = ""
    base_url: str = ""
    persist: bool = False
    error_message: str = ""
    warning_message: str = ""
    is_custom_target: bool = False
    provider_label: str = ""


@dataclass
class CustomAutoResult:
    """Result of switching to bare 'custom' provider with auto-detect."""

    success: bool
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    error_message: str = ""


def switch_model(
    raw_input: str,
    current_provider: str,
    current_base_url: str = "",
    current_api_key: str = "",
) -> ModelSwitchResult:
    """Core model-switching pipeline shared between CLI and gateway.

    Handles parsing, provider detection, credential resolution, and
    model validation.  Does NOT handle config persistence, state
    mutation, or output formatting — those are caller responsibilities.

    Args:
        raw_input: The user's model input (e.g. "claude-sonnet-4",
            "zai:glm-5", "custom:local:qwen").
        current_provider: The currently active provider.
        current_base_url: The currently active base URL (used for
            is_custom detection).
        current_api_key: The currently active API key.

    Returns:
        ModelSwitchResult with all information the caller needs to
        apply the switch and format output.
    """
    from hermes_cli.models import (
        parse_model_input,
        detect_provider_for_model,
        validate_requested_model,
        _PROVIDER_LABELS,
    )
    from hermes_cli.runtime_provider import resolve_runtime_provider

    # Step 1: Parse provider:model syntax
    target_provider, new_model = parse_model_input(raw_input, current_provider)

    # Step 2: Detect if we're currently on a custom endpoint
    _base = current_base_url or ""
    is_custom = current_provider == "custom" or (
        "localhost" in _base or "127.0.0.1" in _base
    )

    # Step 3: Auto-detect provider when no explicit provider:model syntax
    # was used.  Skip for custom providers — the model name might
    # coincidentally match a known provider's catalog.
    if target_provider == current_provider and not is_custom:
        detected = detect_provider_for_model(new_model, current_provider)
        if detected:
            target_provider, new_model = detected

    provider_changed = target_provider != current_provider

    # Step 4: Resolve credentials for target provider
    api_key = current_api_key
    base_url = current_base_url
    if provider_changed:
        try:
            runtime = resolve_runtime_provider(requested=target_provider)
            api_key = runtime.get("api_key", "")
            base_url = runtime.get("base_url", "")
        except Exception as e:
            provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
            if target_provider == "custom":
                return ModelSwitchResult(
                    success=False,
                    target_provider=target_provider,
                    error_message=(
                        "No custom endpoint configured. Set model.base_url "
                        "in config.yaml, or set OPENAI_BASE_URL in .env, "
                        "or run: hermes setup → Custom OpenAI-compatible endpoint"
                    ),
                )
            return ModelSwitchResult(
                success=False,
                target_provider=target_provider,
                error_message=(
                    f"Could not resolve credentials for provider "
                    f"'{provider_label}': {e}"
                ),
            )
    else:
        # Gateway also resolves for unchanged provider to get accurate
        # base_url for validation probing.
        try:
            runtime = resolve_runtime_provider(requested=current_provider)
            api_key = runtime.get("api_key", "")
            base_url = runtime.get("base_url", "")
        except Exception:
            pass

    # Step 5: Validate the model
    try:
        validation = validate_requested_model(
            new_model,
            target_provider,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception:
        validation = {
            "accepted": True,
            "persist": True,
            "recognized": False,
            "message": None,
        }

    if not validation.get("accepted"):
        msg = validation.get("message", "Invalid model")
        return ModelSwitchResult(
            success=False,
            new_model=new_model,
            target_provider=target_provider,
            error_message=msg,
        )

    # Step 6: Build result
    provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
    is_custom_target = target_provider == "custom" or (
        base_url
        and "openrouter.ai" not in (base_url or "")
        and ("localhost" in (base_url or "") or "127.0.0.1" in (base_url or ""))
    )

    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        provider_changed=provider_changed,
        api_key=api_key,
        base_url=base_url,
        persist=bool(validation.get("persist")),
        warning_message=validation.get("message") or "",
        is_custom_target=is_custom_target,
        provider_label=provider_label,
    )


def switch_to_custom_provider() -> CustomAutoResult:
    """Handle bare '/model custom' — resolve endpoint and auto-detect model.

    Returns a result object; the caller handles persistence and output.
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        _auto_detect_local_model,
    )

    try:
        runtime = resolve_runtime_provider(requested="custom")
    except Exception as e:
        return CustomAutoResult(
            success=False,
            error_message=f"Could not resolve custom endpoint: {e}",
        )

    cust_base = runtime.get("base_url", "")
    cust_key = runtime.get("api_key", "")

    if not cust_base or "openrouter.ai" in cust_base:
        return CustomAutoResult(
            success=False,
            error_message=(
                "No custom endpoint configured. "
                "Set model.base_url in config.yaml, or set OPENAI_BASE_URL "
                "in .env, or run: hermes setup → Custom OpenAI-compatible endpoint"
            ),
        )

    detected_model = _auto_detect_local_model(cust_base)
    if not detected_model:
        return CustomAutoResult(
            success=False,
            base_url=cust_base,
            api_key=cust_key,
            error_message=(
                f"Custom endpoint at {cust_base} is reachable but no single "
                f"model was auto-detected. Specify the model explicitly: "
                f"/model custom:<model-name>"
            ),
        )

    return CustomAutoResult(
        success=True,
        model=detected_model,
        base_url=cust_base,
        api_key=cust_key,
    )
