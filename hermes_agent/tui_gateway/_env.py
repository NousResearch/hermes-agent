"""Tolerant env-var knob parsing shared by the gateway entry points: a bare
``float(os.environ[...])`` would raise at import on a typo (``...POLL_S=2s``) and kill a
worker before it serves a command; these fall back to ``default`` on absent/empty/malformed."""

from __future__ import annotations

import os


def _env_number(cast, name: str, default):
    try:
        return cast(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    return _env_number(float, name, default)


def env_int(name: str, default: int) -> int:
    return _env_number(int, name, default)
