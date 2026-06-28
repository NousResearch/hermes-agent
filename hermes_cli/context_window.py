"""Helpers for user-configured context-window caps.

``context_length`` is the historical config key. ``max_context_length`` is a
clearer alias for the common case where a provider advertises a very large
window and the user wants Hermes to behave as though the window were smaller.
"""

from __future__ import annotations

from typing import Any


_CONTEXT_KEYS = ("context_length", "max_context_length")


def parse_context_window_cap(value: Any) -> int | None:
    """Parse a positive integer context-window cap.

    ``None``, empty strings, and ``"auto"`` mean no explicit cap. Booleans are
    rejected even though ``bool`` is an ``int`` subclass.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "auto":
            return None
        value = text.replace(",", "").replace("_", "")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def model_config_context_length(model_cfg: Any) -> tuple[int | None, str | None, Any]:
    """Return ``(cap, key, raw_value)`` from a model config dict.

    ``context_length`` takes precedence for backward compatibility; only when it
    is absent do we consult ``max_context_length``.
    """
    if not isinstance(model_cfg, dict):
        return None, None, None
    for key in _CONTEXT_KEYS:
        if key in model_cfg and model_cfg.get(key) is not None:
            raw = model_cfg.get(key)
            cap = parse_context_window_cap(raw)
            if cap is None and isinstance(raw, str) and raw.strip().lower() in {"", "auto"}:
                return None, None, raw
            return cap, key, raw
    return None, None, None


def route_matches_model_config(
    model_cfg: Any,
    *,
    model: str,
    provider: str = "",
    base_url: str = "",
) -> bool:
    """Return whether a route is the primary route described by ``model_cfg``.

    This keeps a flat ``model.max_context_length`` from leaking into unrelated
    session-only /model switches. Provider and model must match when configured;
    base_url is checked only when both sides have one.
    """
    if not isinstance(model_cfg, dict):
        return False
    cfg_model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip().lower()
    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
    cfg_base = str(model_cfg.get("base_url") or "").strip().rstrip("/").lower()
    route_model = str(model or "").strip().lower()
    route_provider = str(provider or "").strip().lower()
    route_base = str(base_url or "").strip().rstrip("/").lower()
    if cfg_model and route_model and cfg_model != route_model:
        return False
    if cfg_provider and route_provider and cfg_provider != route_provider:
        return False
    if cfg_base and route_base and cfg_base != route_base:
        return False
    return bool(cfg_model or cfg_provider or cfg_base)


def scoped_model_config_context_length(
    config: dict[str, Any] | None,
    *,
    model: str,
    provider: str = "",
    base_url: str = "",
) -> int | None:
    """Return the flat model cap only when it belongs to this route."""
    model_cfg = (config or {}).get("model") if isinstance(config, dict) else None
    cap, _key, _raw = model_config_context_length(model_cfg)
    if cap is None:
        return None
    if route_matches_model_config(model_cfg, model=model, provider=provider, base_url=base_url):
        return cap
    return None


def entry_context_length(entry: Any) -> int | None:
    """Return a cap from a fallback/custom-provider entry dict."""
    cap, _key, _raw = model_config_context_length(entry)
    return cap
