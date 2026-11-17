"""Retention ordering and pruning shared by complete and salvage backup archives."""

import logging
from pathlib import Path

logger = logging.getLogger("hermes_cli.backup")


def newest_first(root: Path, keep_entry) -> list[Path]:
    """Entries of *root* passing ``keep_entry``, newest first; empty when missing."""
    if not root.exists():
        return []
    return sorted(filter(keep_entry, root.iterdir()), key=lambda p: p.name, reverse=True)


def prune_oldest(entries: list[Path], keep: int, remove, what: str) -> int:
    """Remove every entry after the retention limit; return successful removals."""
    deleted = 0
    for path in entries[keep:]:
        try:
            remove(path)
            deleted += 1
        except OSError as exc:
            logger.warning("Failed to prune %s %s: %s", what, path.name, exc)
    return deleted
