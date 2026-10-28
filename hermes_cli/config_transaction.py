"""Guarded user-layer read/modify/write for config editors and plugins."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional


def read_user_config_raw(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Read a user ``config.yaml`` EXACTLY as written (no defaults/overlay/expansion, no cache).
    ONLY legal for write-back round-trips and raw-file diagnostics — behavioral reads must use
    load_config()/load_config_readonly()."""
    from hermes_cli import config

    if config_path is None:
        config_path = config.get_config_path()
    try:
        data = config.read_config_doc(config_path) or {}
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def update_user_config(mutate: Callable[[dict[str, Any]], None]) -> None:
    """Mutate the active profile's raw user layer, then commit via the config backend.

    The callback runs once under core's in-process config lock. It must edit only
    its owned settings and must not change profile scope or perform nested writes.
    An exception aborts the write. Defaults, managed overlays and env expansion
    are never persisted. Missing layers start empty; unreadable layers fail closed.

    This is not a cross-process transaction or a gateway RPC. The file backend
    atomically publishes the document, but other processes/non-cooperating writers
    can race; remote backends retain their own conflict/lock handling. See the
    developer guide's Config Backend Seam for the concurrency contract.
    """
    from hermes_cli import config

    with config._CONFIG_LOCK:
        if config.is_managed():
            raise RuntimeError(config.format_managed_message("change settings"))
        path = config.get_config_path()
        data = config.require_readable_config_before_write(path)
        mutate(data)
        config.atomic_config_write(path, data)
