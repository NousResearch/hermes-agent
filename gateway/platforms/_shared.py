"""Cross-adapter helpers shared by gateway/platforms/* and plugins/platforms/*.

Kept dependency-light (stdlib + ``agent.secret_scope``) so every adapter can
import it at module top level without cycles.
"""

from __future__ import annotations

import os
from typing import Any

# Profile-scoped secret reader for multiplexing support (PR #50094)
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret


def get_scoped_secret(name: str, default: Any = None) -> Any:
    """Scope-aware credential read with the default-profile startup fallback.

    An installed profile secret scope is authoritative: a scoped miss returns
    ``default`` (never borrow another profile's value from ``os.environ``).
    The DEFAULT profile constructs and sends *unscoped* under multiplexing,
    where a bare ``get_secret`` raises ``UnscopedSecretError``; there
    ``os.environ`` is that profile's own value, so fall back to it.
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


def profile_scoped() -> bool:
    # --------------------------------------------------------------------------- YAML → env config bridge
    # (apply_yaml_config_fn, #25443)
    # ---------------------------------------------------------------------------
    """True when running inside a multiplexed secondary profile's scope.

    Secondary-profile adapters are constructed/connected inside
    ``_profile_runtime_scope`` (secret scope installed + multiplex active).
    The DEFAULT profile under multiplexing runs unscoped and keeps the legacy
    ``os.environ`` precedence, so YAML->env bridges must skip only when True.
    """
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        return bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        return False


def coerce_port(value: Any, default: int) -> int:
    """``int(value)`` or ``default`` when unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
