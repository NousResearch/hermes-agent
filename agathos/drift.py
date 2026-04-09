"""
ARGUS config drift detection — monitor config/credential/skill changes at runtime.

Tracks mtime + sha256 of key files between poll cycles. On change, logs the
drift and records in agathos.db for audit. Does NOT restart — running sessions
use their loaded config. Alerts notify operators of changes.

Monitored files:
  ~/.hermes/config.yaml       — main hermes config
  ~/.hermes/.env              — credentials
  ~/.hermes/skills/           — installed skills (mtime + count)
  ~/.hermes/agathos/directives.yaml — ARGUS prime directives
  agathos/watcher_schema.sql  — AGATHOS DB schema
"""

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agathos.drift")

# Files to monitor: (label, path_resolver)
# path_resolver returns Path or None if not applicable


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except ImportError:
        return Path.home() / ".hermes"


def _monitored_files() -> List[Tuple[str, Path]]:
    """Return list of (label, path) for files to monitor."""
    home = _hermes_home()
    files = [
        ("config.yaml", home / "config.yaml"),
        (".env", home / ".env"),
        ("directives.yaml", home / "agathos" / "directives.yaml"),
    ]

    # Skills directory — monitor mtime and file count
    skills_dir = home / "skills"
    if skills_dir.is_dir():
        files.append(("skills/", skills_dir))

    # watcher_schema.sql — relative to agathos module
    schema_path = Path(__file__).parent / "watcher_schema.sql"
    if schema_path.exists():
        files.append(("watcher_schema.sql", schema_path))

    return files


def _hash_file(path: Path) -> Optional[str]:
    """Compute sha256 of a file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def _hash_dir(path: Path) -> Optional[str]:
    """Compute a hash representing directory state (file count + latest mtime)."""
    try:
        files = list(path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        latest_mtime = max((f.stat().st_mtime for f in files if f.is_file()), default=0)
        state = "%d:%.0f" % (file_count, latest_mtime)
        return hashlib.sha256(state.encode()).hexdigest()[:16]
    except Exception:
        return None


def _file_state(path: Path) -> Dict[str, Any]:
    """Get current state of a file or directory."""
    if not path.exists():
        return {"exists": False, "hash": None, "mtime": None, "size": None}

    try:
        stat = path.stat()
        if path.is_dir():
            file_hash = _hash_dir(path)
            files = list(path.rglob("*"))
            size = sum(1 for f in files if f.is_file())
        else:
            file_hash = _hash_file(path)
            size = stat.st_size

        return {
            "exists": True,
            "hash": file_hash,
            "mtime": stat.st_mtime,
            "size": size,
        }
    except Exception:
        return {"exists": False, "hash": None, "mtime": None, "size": None}


class DriftDetector:
    """Tracks config file state between poll cycles."""

    def __init__(self):
        self._previous_states: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Capture current state of all monitored files."""
        states = {}
        for label, path in _monitored_files():
            states[label] = _file_state(path)
        return states

    def check(self) -> List[Dict[str, Any]]:
        """Compare current state to previous. Returns list of changes.

        On first call, initializes baseline and returns empty list.
        """
        current = self.snapshot()

        if not self._initialized:
            self._previous_states = current
            self._initialized = True
            logger.info(
                "Drift detector initialized — monitoring %d files", len(current)
            )
            return []

        changes = []
        for label, current_state in current.items():
            prev_state = self._previous_states.get(label, {})

            # Detect changes
            if current_state.get("hash") != prev_state.get("hash"):
                change = {
                    "file": label,
                    "change_type": "modified",
                    "old_hash": prev_state.get("hash"),
                    "new_hash": current_state.get("hash"),
                    "old_size": prev_state.get("size"),
                    "new_size": current_state.get("size"),
                }

                # Determine specific change type
                if not prev_state.get("exists") and current_state.get("exists"):
                    change["change_type"] = "created"
                elif prev_state.get("exists") and not current_state.get("exists"):
                    change["change_type"] = "deleted"

                changes.append(change)
                logger.info("Drift detected: %s %s", label, change["change_type"])

        self._previous_states = current
        return changes

    def record_changes(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        changes: List[Dict[str, Any]],
    ) -> None:
        """Record detected changes in the database."""
        for change in changes:
            try:
                cursor.execute(
                    """
                    INSERT INTO config_changes
                    (file_path, change_type, old_hash, new_hash)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        change["file"],
                        change["change_type"],
                        change["old_hash"],
                        change["new_hash"],
                    ),
                )
            except Exception as e:
                logger.error("Failed to record drift for %s: %s", change["file"], e)

        if changes:
            try:
                conn.commit()
            except Exception as e:
                logger.error("Failed to commit drift records: %s", e)


def ensure_table(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    """Create config_changes table if it doesn't exist."""
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                change_type TEXT NOT NULL,
                old_hash TEXT,
                new_hash TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error("Failed to create config_changes table: %s", e)
