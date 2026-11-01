"""
Self Evolution Plugin — Strategy Store
========================================

Manages strategy rules with version history and rollback support.

Strategies stored at ~/.hermes/self_evolution/strategies.json
Archives at ~/.hermes/self_evolution/archive/strategies_v{N}.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from self_evolution.paths import DATA_DIR as STRATEGIES_DIR, STRATEGIES_FILE, ARCHIVE_DIR


class StrategyStore:
    """Load, save, and version strategy rules."""

    def load(self) -> dict:
        """Load current strategies."""
        if not STRATEGIES_FILE.exists():
            return {"version": 0, "rules": []}
        try:
            return json.loads(STRATEGIES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"version": 0, "rules": []}

    def save(self, data: dict):
        """Save strategies to file."""
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
        STRATEGIES_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def archive(self, version: int):
        """Archive current strategies for rollback."""
        if not STRATEGIES_FILE.exists():
            return
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"strategies_v{version}.json"
        archive_path.write_text(
            STRATEGIES_FILE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        logger.info("Archived strategies version %d", version)

    def load_archive(self, version: int) -> Optional[dict]:
        """Load an archived version."""
        archive_path = ARCHIVE_DIR / f"strategies_v{version}.json"
        if not archive_path.exists():
            return None
        try:
            return json.loads(archive_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def restore(self, data: dict):
        """Restore strategies from an archive."""
        self.save(data)
        logger.info("Restored strategies from archive")

    def get_version(self) -> int:
        """Get current version number."""
        return self.load().get("version", 0)
