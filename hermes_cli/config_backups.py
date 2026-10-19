"""Point-in-time copies of ``config.yaml``: one directory, one naming scheme, bounded count.

Every writer that wants a "before" copy of the user's config (setup wizard, corrupt-file
snapshot, model migrations) goes through :func:`backup_config`. Copies live in
``<HERMES_HOME>/backups/config/`` — ``backups/`` is already excluded from full backups, so they
never nest — as ``config.yaml.<reason>.<YYYYMMDD-HHMMSS>``. A copy identical to the newest one
for the same reason is skipped, and only the newest ``keep`` per reason survive, so repeated
``hermes setup`` runs or a gateway restarting against broken YAML cannot litter the home dir.
"""

from __future__ import annotations

import logging
import os
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
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _sweep_legacy_siblings(config_path, root)
        existing = list_config_backups(config_path, reason)
        # Not filecmp.cmp: its process-wide cache keys on (size, mtime), so an equal-size edit
        # within one mtime tick of an earlier comparison is reported identical and never backed up.
        if existing and config_path.read_bytes() == existing[0].read_bytes():
            return None
        dest = root / f"{config_path.name}.{reason}.{time.strftime('%Y%m%d-%H%M%S')}"
        if dest.is_symlink() or dest.exists():  # never write through a planted link
            return None
        _copy_private(config_path, dest)
        _tighten_legacy(root)
        for stale in [dest, *existing][keep:]:
            stale.unlink(missing_ok=True)
        return dest
    except OSError as exc:
        logger.warning("Could not back up %s (%s): %s", config_path, reason, exc)
        return None


def load_newest_good_backup(config_path: Path) -> Optional[dict]:
    """Parse the newest ``good`` backup (the file as it was at the last successful load).

    Returns the raw mapping, or None when there is no usable copy. Older ``good`` copies are not
    tried: a backup that fails to parse means the on-disk copy was damaged after the fact, and
    guessing further back would serve a config the user never saw as current.
    """
    newest = list_config_backups(config_path, "good")[:1]
    if not newest:
        return None
    try:
        from utils import fast_safe_load
        with newest[0].open(encoding="utf-8") as f:
            data = fast_safe_load(f)
    except Exception as exc:
        logger.warning("Last-known-good backup %s is unreadable: %s", newest[0], exc)
        return None
    return data if isinstance(data, dict) else None


def _copy_private(src: Path, dest: Path) -> None:
    """Byte-copy *src* to *dest* with the private mode set at creation.

    ``config.yaml`` commonly holds live provider keys, but ``shutil.copy2`` inherits the
    source's mode, so a 0644 config (or a permissive umask) stamped every backup 0644 —
    world-readable copies of secrets, in a dir nothing else guards. The mode is fixed in
    ``os.open`` (not a later chmod) so no world-readable window exists between creation and
    tightening, the same invariant ``auth.py`` keeps for token files. mtime still carries
    over (what ``copy2`` was bought for); the mode deliberately does not.
    """
    st = src.stat()
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out, open(src, "rb") as f:
        shutil.copyfileobj(f, out)
    os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))


def _tighten_legacy(root: Path) -> None:
    """Narrow pre-fix group/world bits on the backup dir and its copies on next write.

    One-time repair per file: backups written before the mode-at-creation fix keep whatever
    mode the old ``copy2`` gave them until this runs. Bounded by the same ``keep`` rotation
    as the copies themselves."""
    try:
        if root.stat().st_mode & 0o077:
            root.chmod(0o700)
    except OSError:
        pass
    for p in root.iterdir():
        try:
            if p.is_file() and not p.is_symlink() and p.stat().st_mode & 0o077:
                p.chmod(0o600)
        except OSError:
            pass


def _sweep_legacy_siblings(config_path: Path, root: Path) -> None:
    for pattern in _LEGACY_SIBLING_GLOBS:
        for old in config_path.parent.glob(pattern):
            if not old.is_file() or old.is_symlink():
                continue
            try:
                old.replace(root / old.name)
            except OSError as exc:
                logger.debug("Could not move legacy backup %s: %s", old, exc)
