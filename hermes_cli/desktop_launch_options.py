"""Shared normalization for Desktop launch configuration."""

from __future__ import annotations


def normalize_desktop_disable_gpu(value: object) -> str:
    """Normalize ``desktop.disable_gpu`` to Electron's supported values."""

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return "1"
        if value in {"0", "false", "no", "off"}:
            return "0"

    return "auto"