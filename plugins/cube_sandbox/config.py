"""Cube sandbox plugin configuration."""

from __future__ import annotations

import os

from tools.cube_split import is_cube_split_enabled


def is_plugin_active() -> bool:
    """True when this plugin should register overrides (``SANDBOX_TYPE=cube``)."""
    return is_cube_split_enabled()


def select_tier(task_id: str | None = None) -> str:
    """Return Cube snapshot tier: ``task`` (default) or ``session``."""
    tier = os.getenv("CUBE_DEFAULT_TIER", "task").strip().lower() or "task"
    if tier not in {"task", "session"}:
        return "task"
    return tier
