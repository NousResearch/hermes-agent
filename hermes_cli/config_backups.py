"""Point-in-time copies of ``config.yaml``: one directory, one naming scheme, bounded count.

Every writer that wants a "before" copy of the user's config (setup wizard, corrupt-file
snapshot, model migrations) goes through :func:`backup_config`. Copies live in
``<HERMES_HOME>/backups/config/`` — ``backups/`` is already excluded from full backups, so they
never nest — as ``config.yaml.<reason>.<YYYYMMDD-HHMMSS>``. A copy identical to the newest one
for the same reason is skipped, and only the newest ``keep`` per reason survive, so repeated
``hermes setup`` runs or a gateway restarting against broken YAML cannot litter the home dir.
"""

from __future__ import annotations

import filecmp
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BACKUPS_SUBDIR = Path("backups") / "config"
DEFAULT_KEEP = 5

# Names earlier code wrote next to config.yaml (setup wizard, corrupt snapshot, xai migration).
# Moved into the backups dir on first use so they stop accumulating in the home root; hand-named
# copies (``config.yaml.bak-my-note``) are the user's and are never touched.
_LEGACY_SIBLING_GLOBS = ("config.yaml.bak.[0-9]*", "config.yaml.corrupt.*.bak", "config.yaml.bak-pre-migrate-*")


def backups_dir(config_path: Path) -> Path:
    return config_path.parent / BACKUPS_SUBDIR


def list_config_backups(config_path: Path, reason: Optional[str] = None) -> list[Path]:
    """Existing backups, newest first; filtered to one *reason* when given."""
    root = backups_dir(config_path)
    if not root.is_dir():
        return []
    prefix = f"{config_path.name}.{reason}." if reason else f"{config_path.name}."
    return sorted((p for p in root.iterdir() if p.is_file() and p.name.startswith(prefix)),
                  key=lambda p: p.name, reverse=True)


def backup_config(config_path: Path, reason: str, *, keep: int = DEFAULT_KEEP) -> Optional[Path]:
    """Copy *config_path* to the backups dir; return the new path, or None when skipped/failed.

    Skips when the file is missing/empty, or when the newest backup for *reason* already holds
    identical bytes. Never raises: a failed backup must not block the write it precedes.
    """
    try:
        if not config_path.is_file() or config_path.stat().st_size == 0:
            return None
        root = backups_dir(config_path)
        root.mkdir(parents=True, exist_ok=True)
        _sweep_legacy_siblings(config_path, root)
        existing = list_config_backups(config_path, reason)
        # Dedupe against the newest backup we can actually READ: an unreadable candidate (mode 000,
        # a root-owned file, a broken mount) must not turn "cannot compare" into "cannot back up".
        for candidate in existing:
            try:
                identical = filecmp.cmp(config_path, candidate, shallow=False)
            except OSError as exc:
                logger.debug("Could not compare %s with backup %s: %s", config_path, candidate, exc)
                continue
            if identical:
                return None
            break  # newest readable copy differs → a fresh backup is warranted
        dest = root / f"{config_path.name}.{reason}.{time.strftime('%Y%m%d-%H%M%S')}"
        if dest.is_symlink() or dest.exists():  # never write through a planted link
            return None
        shutil.copy2(config_path, dest)
        for stale in [dest, *existing][keep:]:
            stale.unlink(missing_ok=True)
        return dest
    except OSError as exc:
        # Never WARNING from inside a config load: the record is rendered by handlers whose
        # formatter reads config again (RedactingFormatter -> agent.redact._redact_enabled, which
        # calls load_config_readonly), and that re-entry is the recursion the load guard exists to
        # stop. Outside a load a failed backup is worth surfacing: it silently disables
        # last-known-good recovery.
        from hermes_cli.config import _config_load_in_progress
        log = logger.debug if _config_load_in_progress() else logger.warning
        log("Could not back up %s (%s): %s", config_path, reason, exc)
        return None


def load_newest_good_backup(config_path: Path) -> Optional[dict]:
    """Parse the newest READABLE ``good`` backup (the file as it was at the last successful load).

    Returns the raw mapping, or None when there is no usable copy. A copy we cannot open (mode 000,
    another writer holding it, a broken mount) is skipped in favour of the next readable one — one
    unreadable file must not silently disable last-known-good recovery. A copy that *parses* badly
    ends the search instead: that file was damaged after the fact, and guessing further back would
    serve a config the user never saw as current.
    """
    from utils import fast_safe_load
    for candidate in list_config_backups(config_path, "good"):
        try:
            with candidate.open(encoding="utf-8") as f:
                data = fast_safe_load(f)
        except OSError as exc:
            logger.debug("Skipping unreadable last-known-good backup %s: %s", candidate, exc)
            continue
        except Exception as exc:
            logger.warning("Last-known-good backup %s is unreadable: %s", candidate, exc)
            return None
        if not isinstance(data, dict):
            logger.debug("Last-known-good backup %s is not a mapping: %r", candidate, type(data).__name__)
            return None
        return data
    return None


def _sweep_legacy_siblings(config_path: Path, root: Path) -> None:
    for pattern in _LEGACY_SIBLING_GLOBS:
        for old in config_path.parent.glob(pattern):
            if not old.is_file() or old.is_symlink():
                continue
            try:
                old.replace(root / old.name)
            except OSError as exc:
                logger.debug("Could not move legacy backup %s: %s", old, exc)
