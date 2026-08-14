"""Shared helpers for provider-specific HTTP headers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def get_model_custom_headers(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return sanitized ``model.custom_headers`` from config.yaml."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            return {}

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    if not isinstance(model_cfg, dict):
        return {}

    raw_headers = model_cfg.get("custom_headers")
    if not isinstance(raw_headers, dict):
        return {}

    headers: Dict[str, str] = {}
    for key, value in raw_headers.items():
        if not isinstance(key, str):
            continue
        clean_key = key.strip()
        if not clean_key:
            continue
        if isinstance(value, str):
            clean_value = value.strip()
        else:
            clean_value = str(value).strip() if value is not None else ""
        if clean_value:
            headers[clean_key] = clean_value
    return headers


def merge_default_headers(*header_sets: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge header dictionaries in order, skipping empty inputs."""
    merged: Dict[str, str] = {}
    for header_set in header_sets:
        if header_set:
            merged.update(header_set)
    return merged
