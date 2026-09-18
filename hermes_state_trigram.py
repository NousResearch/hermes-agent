"""Profile-scoped configuration gate for the optional trigram FTS index."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from hermes_cli.config_effective import ConfigResolutionError, resolve_effective_config_value
from hermes_state_common import FTS_TRIGRAM_STALE_KEY

logger = logging.getLogger("hermes_state")


def trigram_fts_enabled_from_config(db_path: Path) -> Optional[bool]:
    """Resolve ``sessions.trigram_fts`` from the config beside *db_path*.

    ``None`` means a present source was broken; the caller must consult the durable
    quarantine marker instead of treating the key as absent/default-on.
    """
    config_path = Path(db_path).expanduser().parent / "config.yaml"
    try:
        value = resolve_effective_config_value(
            config_path, "sessions", "trigram_fts", default=True,
        )
    except ConfigResolutionError as exc:
        logger.warning(
            "Could not resolve %s for trigram FTS; deferring to the on-disk quarantine marker: %s",
            config_path, exc,
        )
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "off", "no", "0"}:
            return False
        if normalized in {"true", "on", "yes", "1"}:
            return True
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    logger.warning(
        "Invalid sessions.trigram_fts value in %s; deferring to the on-disk quarantine marker",
        config_path,
    )
    return None


def trigram_enabled_after_config_resolution(
    conn: sqlite3.Connection, configured: Optional[bool],
) -> bool:
    """Honor durable quarantine after a failed config read; fresh stores default on."""
    if configured is not None:
        return configured
    try:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'state_meta' LIMIT 1"
        ).fetchone() is None:
            return True
        stale = conn.execute(
            "SELECT 1 FROM state_meta WHERE key = ? LIMIT 1", (FTS_TRIGRAM_STALE_KEY,),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("Could not inspect the trigram quarantine marker; keeping trigram disabled: %s", exc)
        return False
    if stale is not None:
        logger.warning("Preserving the existing trigram quarantine because config resolution failed")
        return False
    return True