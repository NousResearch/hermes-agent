"""Shared application ownership of persisted /model selection shape."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from providers import normalize_route_base_url
from utils import atomic_roundtrip_yaml_update


def route_changed(model_cfg: Any, result) -> bool:
    cfg = model_cfg if isinstance(model_cfg, dict) else {}
    if str(cfg.get("provider") or "").strip().lower() != str(
        result.target_provider or ""
    ).strip().lower():
        return True
    return normalize_route_base_url(cfg.get("base_url")) != normalize_route_base_url(
        result.base_url
    )


def model_selection_config_updates(result, current_model_cfg: Any) -> dict[str, Any]:
    cfg = current_model_cfg if isinstance(current_model_cfg, dict) else {}
    updates: dict[str, Any] = {
        "default": result.new_model,
        "provider": result.target_provider,
        "base_url": result.base_url or None,
        "api_mode": result.api_mode or None,
    }
    changed = route_changed(cfg, result)
    configured_model = str(cfg.get("default") or cfg.get("model") or "").strip()
    if "context_length" in cfg and (
        (configured_model and configured_model != str(result.new_model or "").strip())
        or changed
    ):
        updates["context_length"] = None

    target = str(result.target_provider or "").strip().lower()
    stale = ["api_key", "api"] if (not target.startswith("custom") or changed) else []
    if changed:
        stale += ["key_env", "api_key_env"]
    for key in stale:
        if key in cfg:
            updates[key] = None
    return updates


def apply_model_selection(model_cfg: Any, result) -> dict[str, Any]:
    """Apply the shared global model shape to an already-loaded config block."""
    updated = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    for key, value in model_selection_config_updates(result, updated).items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    return updated


def persist_model_selection(result, config_path: Any) -> None:
    from hermes_cli.config import read_user_config_raw

    path = Path(config_path)
    raw = read_user_config_raw(path)
    for key, value in model_selection_config_updates(result, raw.get("model")).items():
        atomic_roundtrip_yaml_update(path, f"model.{key}", value)
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


__all__ = ["apply_model_selection", "model_selection_config_updates", "persist_model_selection", "route_changed"]
