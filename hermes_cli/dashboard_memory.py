"""Dashboard memory optimization settings (config.yaml ``dashboard.memory``)."""

from __future__ import annotations

import os
from typing import Any, Dict

_DEFAULTS: Dict[str, Any] = {
    "allocator": "auto",
    "lazy_mcp": True,
    "trim": True,
    "trim_idle_seconds": 60,
    "trim_cooldown_seconds": 120,
}


def get_dashboard_memory_config() -> Dict[str, Any]:
    """Return merged ``dashboard.memory`` settings."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        dash = cfg.get("dashboard") if isinstance(cfg.get("dashboard"), dict) else {}
        mem = dash.get("memory") if isinstance(dash.get("memory"), dict) else {}
    except Exception:
        mem = {}

    out = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in mem and mem[key] is not None:
            out[key] = mem[key]
    return out


def lazy_mcp_enabled() -> bool:
    mem = get_dashboard_memory_config()
    return bool(mem.get("lazy_mcp", True))


def memory_trim_enabled() -> bool:
    mem = get_dashboard_memory_config()
    return bool(mem.get("trim", True))


def memory_trim_idle_seconds() -> int:
    mem = get_dashboard_memory_config()
    try:
        return max(10, int(mem.get("trim_idle_seconds", 60)))
    except (TypeError, ValueError):
        return 60


def memory_trim_cooldown_seconds() -> int:
    mem = get_dashboard_memory_config()
    try:
        return max(30, int(mem.get("trim_cooldown_seconds", 120)))
    except (TypeError, ValueError):
        return 120


def allocator_mode() -> str:
    """Return ``auto``, ``mimalloc``, or ``system``."""
    mem = get_dashboard_memory_config()
    mode = str(mem.get("allocator") or "auto").strip().lower()
    if mode not in ("auto", "mimalloc", "system"):
        return "auto"
    return mode


def should_use_mimalloc_preload() -> bool:
    mode = allocator_mode()
    if mode == "system":
        return False
    if mode == "mimalloc":
        return True
    # auto: desktop backends on macOS/Linux benefit most
    if os.environ.get("HERMES_DESKTOP") != "1":
        return False
    if os.name == "nt":
        return False
    return True
