"""Session-row route identity helpers (CLI/TUI/Desktop sticky resume).

Pure extraction of opaque provider+model (and optional endpoint knobs) from a
``sessions`` table row so resume paths can re-apply the route the conversation
actually used — without importing the TUI gateway server.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def parse_model_config(raw: Any) -> dict[str, Any]:
    """Return a dict model_config from JSON text, dict, or empty."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def route_from_session_row(row: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract sticky route fields from a SessionDB ``get_session`` row.

    Returns only non-empty string fields among: model, provider, base_url,
    api_mode. Never includes secrets. Missing/invalid rows yield {}.
    """
    if not isinstance(row, Mapping):
        return {}

    cfg = parse_model_config(row.get("model_config"))
    model = str(row.get("model") or cfg.get("model") or "").strip()
    provider = str(cfg.get("provider") or "").strip()
    # billing_provider is a billing bucket, not always a routable identity
    # (e.g. bare "custom"). Prefer explicit provider from model_config only.
    base_url = str(cfg.get("base_url") or "").strip()
    api_mode = str(cfg.get("api_mode") or "").strip()

    out: dict[str, str] = {}
    if model:
        out["model"] = model
    if provider and provider.lower() != "custom":
        # bare "custom" is non-routable without entry identity; skip it so
        # resume falls back to configured provider rather than OpenRouter.
        out["provider"] = provider
    elif provider.lower() == "custom" and base_url:
        out["provider"] = "custom"
    if base_url:
        out["base_url"] = base_url
    if api_mode:
        out["api_mode"] = api_mode
    return out


def model_config_for_route(
    *,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    api_mode: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a persistable model_config for create_session / update_session_meta."""
    cfg: dict[str, Any] = dict(extra or {})
    if model:
        cfg["model"] = str(model).strip()
    if provider:
        cfg["provider"] = str(provider).strip()
    if base_url:
        cfg["base_url"] = str(base_url).strip()
    elif "base_url" in cfg and not cfg.get("base_url"):
        cfg.pop("base_url", None)
    if api_mode:
        cfg["api_mode"] = str(api_mode).strip()
    elif "api_mode" in cfg and not cfg.get("api_mode"):
        cfg.pop("api_mode", None)
    return cfg
