"""Helpers for platform-scoped LLM request option overrides."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from hermes_constants import parse_reasoning_effort


def merge_request_overrides(*layers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge request override layers, with later layers taking precedence.

    ``extra_body`` is shallow-merged so platform-scoped fields and explicit
    caller fields can coexist without replacing the whole object.
    """
    merged: Dict[str, Any] = {}
    merged_extra: Dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key, value in layer.items():
            if key == "extra_body":
                if isinstance(value, dict):
                    merged_extra.update(value)
                continue
            merged[key] = value
    if merged_extra:
        merged["extra_body"] = merged_extra
    return merged


def _request_options_from_platform_mapping(mapping: Any) -> Dict[str, Any]:
    """Return request options declared by a platform override mapping."""
    if not isinstance(mapping, dict):
        return {}

    nested = mapping.get("request_overrides")
    options: Dict[str, Any] = dict(nested) if isinstance(nested, dict) else {}
    for key in ("extra_body", "reasoning_effort", "reasoning_config", "service_tier"):
        if key in mapping:
            options[key] = mapping[key]
    return options


def scoped_request_options_for_platform(
    config: Any,
    platform: str | None,
) -> Dict[str, Any]:
    """Resolve ``platform_request_overrides.<platform>`` from config."""
    if not platform or not isinstance(config, dict):
        return {}
    raw = config.get("platform_request_overrides")
    if not isinstance(raw, dict):
        return {}
    return _request_options_from_platform_mapping(raw.get(str(platform)))


def reasoning_config_from_request_options(
    options: Any,
    *,
    fallback: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract Hermes' normalized reasoning config from request options."""
    if not isinstance(options, dict):
        return dict(fallback) if isinstance(fallback, dict) else fallback

    raw_config = options.get("reasoning_config")
    if isinstance(raw_config, dict):
        return dict(raw_config)

    if "reasoning_effort" in options:
        parsed = parse_reasoning_effort(options.get("reasoning_effort"))
        if parsed is not None:
            return parsed

    return dict(fallback) if isinstance(fallback, dict) else fallback


def strip_internal_request_options(options: Any) -> Dict[str, Any]:
    """Remove config-only keys before passing overrides to transport kwargs."""
    if not isinstance(options, dict):
        return {}
    return {
        key: value
        for key, value in options.items()
        if key not in {"reasoning_effort", "reasoning_config"}
    }


def first_service_tier(layers: Iterable[Any]) -> Optional[str]:
    """Return the highest-precedence non-empty service tier in *layers*."""
    chosen: Optional[str] = None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        raw = layer.get("service_tier")
        if isinstance(raw, str) and raw.strip():
            chosen = raw.strip()
    return chosen
