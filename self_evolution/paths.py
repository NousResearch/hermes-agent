"""
Self Evolution Plugin — Centralized Path Definitions
=====================================================

Single source of truth for all filesystem paths used by the plugin.
"""

from pathlib import Path

HERMES_HOME = Path.home() / ".hermes"
DATA_DIR = HERMES_HOME / "self_evolution"
DB_PATH = DATA_DIR / "evolution.db"
STRATEGIES_FILE = DATA_DIR / "strategies.json"
ARCHIVE_DIR = DATA_DIR / "archive"
SKILLS_DIR = HERMES_HOME / "skills" / "learned"
MEMORIES_DIR = HERMES_HOME / "memories"
CRON_DIR = HERMES_HOME / "cron"
