"""Top-level ``hermes model`` command orchestration."""

from __future__ import annotations

import sys
from typing import Callable


def _clear_model_cache() -> None:
    try:
        from hermes_cli.models import clear_provider_models_cache

        clear_provider_models_cache()
        print("  Cleared model picker cache.")
    except Exception:
        pass


def _run_noninteractive(provider: str, model_id: str) -> None:
    from cli import save_config_value
    from hermes_cli.config import get_compatible_custom_providers, load_config
    from hermes_cli.model_switch import switch_model

    config = load_config() or {}
    raw_model_config = config.get("model")
    if isinstance(raw_model_config, dict):
        model_config = raw_model_config
    elif isinstance(raw_model_config, str) and raw_model_config.strip():
        model_config = {"default": raw_model_config.strip()}
    else:
        model_config = {}

    result = switch_model(
        raw_input=model_id,
        current_provider=str(model_config.get("provider") or "auto"),
        current_model=str(model_config.get("default") or ""),
        current_base_url=str(model_config.get("base_url") or ""),
        current_api_key=str(model_config.get("api_key") or ""),
        is_global=True,
        explicit_provider=provider,
        user_providers=(
            config.get("providers")
            if isinstance(config.get("providers"), dict)
            else None
        ),
        custom_providers=get_compatible_custom_providers(config),
    )
    if not result.success:
        print(f"Error: {result.error_message}", file=sys.stderr)
        raise SystemExit(1)
    if not result.model_verified:
        print(
            f"Error: model '{model_id}' could not be verified for provider "
            f"'{provider}'; configuration was not changed.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    save_config_value("model.base_url", result.base_url or None)
    save_config_value("model.api_mode", result.api_mode or None)
    print(
        f"Default model set to: {result.new_model} "
        f"(via {result.provider_label or result.target_provider})"
    )


def run_model_command(
    args,
    *,
    require_tty: Callable[[str], None],
    interactive_select: Callable,
) -> None:
    """Run flag-driven selection when complete, otherwise retain the picker."""
    provider = str(getattr(args, "provider", None) or "").strip()
    model_id = str(getattr(args, "model_id", None) or "").strip()
    if bool(provider) != bool(model_id):
        print(
            "Error: --provider and --model must be supplied together.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if getattr(args, "refresh", False):
        _clear_model_cache()
    if provider:
        _run_noninteractive(provider, model_id)
        return

    require_tty("model")
    from hermes_cli.setup import run_setup_action_with_navigation

    run_setup_action_with_navigation(
        "Model & Provider",
        lambda: interactive_select(args=args),
        cancelled_message="No change.",
    )
