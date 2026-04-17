"""Setup-time defaults for chat providers that also serve TTS / image / vision / video / music."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Tool categories that get config defaults written at setup time.
# Vision is runtime-only (no config persistence) so it is not listed here.
_CONFIG_CATEGORIES = ("tts", "image_gen", "video_gen", "music_gen")

# Existing config values considered "not explicitly set by the user".
_OVERRIDABLE = {"", "edge", "auto", "fal"}


def _api_host(config):
    url = (config.get("model") or {}).get("base_url", "")
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _is_native_provider(config):
    return "minimax" in _api_host(config)


def _provider_label(config):
    model = config.get("model") or {}
    if isinstance(model, dict):
        return str(model.get("provider") or "").strip().lower()
    return ""


def _safe_load_config():
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _category_display(cat):
    if cat == "tts":
        return "Text-to-speech"
    return cat.replace("_gen", " generation").replace("_", " ").capitalize()


def active_provider_api_root(config):
    """API root derived from ``model.base_url``, stripping ``/anthropic``."""
    model = config.get("model")
    if not isinstance(model, dict):
        return ""
    base = str(model.get("base_url") or "").strip().rstrip("/")
    if not base:
        return ""
    return base[: -len("/anthropic")] if base.endswith("/anthropic") else base


def native_api_url(subpath, config=None):
    """Full URL for a native API subpath, or ``""`` if not a native provider."""
    cfg = config if config is not None else _safe_load_config()
    if not _is_native_provider(cfg):
        return ""
    root = active_provider_api_root(cfg).rstrip("/")
    return f"{root}{subpath}" if root else ""


def native_credential(config=None):
    """API key matching the active provider's region, or ``""``."""
    import os
    cfg = config if config is not None else _safe_load_config()
    host = _api_host(cfg)
    if "minimaxi.com" in host:
        return (os.environ.get("MINIMAX_CN_API_KEY", "").strip()
                or os.environ.get("MINIMAX_API_KEY", "").strip())
    return (os.environ.get("MINIMAX_API_KEY", "").strip()
            or os.environ.get("MINIMAX_CN_API_KEY", "").strip())


def get_native_tools(config):
    """Tool categories served natively by the active provider, or ``()``."""
    if not _is_native_provider(config):
        return ()
    return _CONFIG_CATEGORIES + ("vision",)


def provider_has_native_tool(tool, config):
    if tool not in get_native_tools(config):
        return False
    try:
        from tools.tool_backend_helpers import prefers_gateway
        if prefers_gateway(tool):
            return False
    except Exception:
        pass
    return True


def apply_provider_native_tool_defaults(config):
    """Wire config defaults for providers that serve tools natively."""
    if not _is_native_provider(config):
        return set()
    label = _provider_label(config)
    if not label:
        return set()
    changed = set()
    for cat in _CONFIG_CATEGORIES:
        cfg = config.setdefault(cat, {})
        if isinstance(cfg, dict) and str(cfg.get("provider") or "").strip().lower() in _OVERRIDABLE:
            cfg["provider"] = label
            changed.add(cat)
    if changed:
        aux = config.get("auxiliary") if isinstance(config.get("auxiliary"), dict) else {}
        vis = aux.get("vision") if isinstance(aux.get("vision"), dict) else {}
        if str(vis.get("provider") or "").strip().lower() in _OVERRIDABLE:
            changed.add("vision")
    return changed


def describe_changes(changed, config):
    """Bullet-list summary used by the setup wizard."""
    items = sorted(changed)
    if not items:
        return "No changes \u2014 existing tool choices were preserved."
    return "\n".join(f"  \u2022 {_category_display(k)}" for k in items)
