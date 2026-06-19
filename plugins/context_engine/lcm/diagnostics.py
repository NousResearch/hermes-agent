"""Shared read-only diagnostic helpers for LCM tools and commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def state_db_path_for_engine(engine: Any) -> Path:
    """Return the Hermes state database path for an LCM engine.

    The path is read-only diagnostic input. When ``LCM_HERMES_BASE_DIR`` is
    configured, enforce the same containment guard for all diagnostic surfaces.
    """
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if hermes_home:
        resolved = Path(hermes_home).expanduser().resolve() / "state.db"
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                resolved.relative_to(allowed_base)
            except ValueError:
                raise ValueError(
                    f"hermes_home {hermes_home} resolves to {resolved} which is not within allowed base {allowed_base}"
                )
        return resolved
    db_path = Path(getattr(engine._store, "db_path", Path.home() / ".hermes" / "lcm.db"))
    return db_path.parent / "state.db"


def has_lifecycle_fragmentation(stats: dict[str, Any]) -> bool:
    """Return whether lifecycle diagnostics should be treated as warning evidence."""
    direct_mismatch_keys = (
        "lifecycle_current_missing_in_lcm_any",
        "lifecycle_last_finalized_missing_in_lcm_any",
        "lifecycle_current_missing_in_state",
        "lifecycle_last_finalized_missing_in_state",
        "lcm_message_sessions_missing_in_state",
        "lcm_node_sessions_missing_in_state",
    )
    lifecycle_rows = int(stats.get("lifecycle_rows", 0) or 0)
    missing_lifecycle_reference_keys = (
        "message_sessions_without_lifecycle_reference",
        "node_sessions_without_lifecycle_reference",
    )
    return (
        any(int(stats.get(key, 0) or 0) > 0 for key in direct_mismatch_keys)
        or (
            lifecycle_rows > 0
            and any(int(stats.get(key, 0) or 0) > 0 for key in missing_lifecycle_reference_keys)
        )
        or (bool(stats.get("state_db_checked")) and bool(stats.get("state_db_error")))
    )


# Backward-compatible private aliases for existing command/tool internals and tests.
_state_db_path_for_engine = state_db_path_for_engine
_has_lifecycle_fragmentation = has_lifecycle_fragmentation
