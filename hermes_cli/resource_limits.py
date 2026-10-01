"""Frozen-updater compatibility facade for resource-limit policy.

Current Hermes code loads configuration at the caller and passes it to
``runtime.resource_limits``. This module remains because an already-running old
updater may import ``configured_nofile_soft_limit`` after replacing the checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from runtime import resource_limits as _runtime

DEFAULT_NOFILE_SOFT_LIMIT = _runtime.DEFAULT_NOFILE_SOFT_LIMIT


def _loaded_config(config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if config is not None:
        return config
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly()
    except Exception:
        return None


def configured_nofile_soft_limit(
    config: Mapping[str, Any] | None = None,
) -> int | None:
    """Compatibility wrapper preserving the historical optional-config API."""
    return _runtime.configured_nofile_soft_limit(_loaded_config(config))


def apply_nofile_soft_limit(config: Mapping[str, Any] | None = None) -> bool:
    """Compatibility wrapper preserving the historical optional-config API."""
    return _runtime.apply_nofile_soft_limit(_loaded_config(config))


__all__ = [
    "DEFAULT_NOFILE_SOFT_LIMIT",
    "apply_nofile_soft_limit",
    "configured_nofile_soft_limit",
]
