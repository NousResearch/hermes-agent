"""Shared timeout configuration helpers for platform adapters."""

import os


def env_float(name: str, default: float) -> float:
    """Read a float from an environment variable, with fallback.

    Used by platform adapters to make timeouts configurable without
    changing defaults. See HERMES_*_TIMEOUT env vars.
    """
    val = os.environ.get(name)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def env_int(name: str, default: int) -> int:
    """Read an int from an environment variable, with fallback."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default
