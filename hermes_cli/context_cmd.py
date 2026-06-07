"""Helpers for the model context-length override command."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


ONE_M_CONTEXT_LENGTH = 1_000_000

_STATUS_ALIASES = {"", "status", "show", "current"}
_AUTO_ALIASES = {"auto", "off", "disable", "disabled", "clear", "reset", "0"}
_ONE_M_ALIASES = {
    "1m",
    "1m-context",
    "1million",
    "million",
    "on",
    "enable",
    "enabled",
}


def _parse_token_count(raw: str) -> Optional[int]:
    text = raw.strip().lower().replace(",", "").replace("_", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([km])?", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix == "m":
        number *= 1_000_000
    elif suffix == "k":
        number *= 1_000
    value = int(number)
    return value if value > 0 else None


def parse_context_length_action(raw: str) -> Tuple[str, Optional[int]]:
    """Parse a context command argument into ``status``, ``auto``, or ``set``."""
    text = (raw or "").strip().lower()
    if text in _STATUS_ALIASES:
        return "status", None
    if text in _AUTO_ALIASES:
        return "auto", None
    if text in _ONE_M_ALIASES:
        return "set", ONE_M_CONTEXT_LENGTH

    value = _parse_token_count(text)
    if value is not None:
        return "set", value

    raise ValueError("Usage: context [1m|auto|status]")


def get_model_context_length_override(config: Dict[str, Any]) -> Optional[int]:
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get("context_length")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def set_model_context_length_override(
    config: Dict[str, Any],
    context_length: Optional[int],
) -> Dict[str, Any]:
    """Mutate ``config`` so ``model.context_length`` is set or cleared."""
    model = config.get("model")
    if context_length is None and not isinstance(model, dict):
        return config

    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        config["model"] = model

    if context_length and context_length > 0:
        model["context_length"] = int(context_length)
    else:
        model.pop("context_length", None)
    return config


def save_model_context_length_override(context_length: Optional[int]) -> Dict[str, Any]:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    set_model_context_length_override(config, context_length)
    save_config(config)
    return config


def format_context_length(context_length: Optional[int]) -> str:
    return f"{int(context_length):,}" if context_length else "auto-detect"


def describe_model_context_length_override(config: Optional[Dict[str, Any]] = None) -> str:
    from hermes_cli.config import load_config

    config = config if config is not None else load_config()
    override = get_model_context_length_override(config)
    if override:
        return (
            f"Context length override: {format_context_length(override)} tokens "
            "(model.context_length)."
        )
    return "Context length override: auto-detect (model.context_length unset)."


def apply_context_length_override_to_agent(
    agent: Any,
    context_length: Optional[int],
) -> Optional[int]:
    """Apply a config override to a live agent's context compressor."""
    if agent is None:
        return None

    setattr(agent, "_config_context_length", context_length)
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return None

    model = getattr(agent, "model", "") or ""
    base_url = getattr(agent, "base_url", "") or ""
    api_key = getattr(agent, "api_key", "") or ""
    provider = getattr(agent, "provider", "") or ""
    api_mode = getattr(agent, "api_mode", "") or ""

    resolved = context_length
    if resolved is None:
        try:
            from agent.model_metadata import get_model_context_length

            resolved = get_model_context_length(
                model,
                base_url=base_url,
                api_key=api_key,
                provider=provider or None,
                custom_providers=getattr(agent, "_custom_providers", None),
                config_context_length=None,
            )
        except Exception:
            resolved = None

    try:
        resolved_int = int(resolved) if resolved else None
    except (TypeError, ValueError):
        resolved_int = None
    if not resolved_int or resolved_int <= 0:
        return None

    if hasattr(compressor, "update_model"):
        compressor.update_model(
            model,
            resolved_int,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )
    else:
        setattr(compressor, "context_length", resolved_int)
    return resolved_int


def run_context_config_command(raw_args: str, *, agent: Any = None) -> str:
    action, context_length = parse_context_length_action(raw_args)
    if action == "status":
        return describe_model_context_length_override()

    save_model_context_length_override(context_length if action == "set" else None)
    applied = apply_context_length_override_to_agent(
        agent,
        context_length if action == "set" else None,
    )

    if action == "set":
        lines = [
            f"Context length override set to {format_context_length(context_length)} tokens.",
            "Saved as model.context_length in config.yaml.",
        ]
    else:
        lines = [
            "Context length override cleared.",
            "Hermes will auto-detect the model context window.",
        ]

    if agent is not None:
        if applied:
            lines.append(f"Current session context window is now {format_context_length(applied)} tokens.")
        else:
            lines.append("Current session will refresh the context window on the next model/session init.")
    return "\n".join(lines)
