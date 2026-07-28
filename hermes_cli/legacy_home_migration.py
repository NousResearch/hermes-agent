"""One-shot migration of a legacy ``~/.hermes`` home into the Windows-native
default ``%LOCALAPPDATA%\\hermes``.

The Windows default Hermes home moved from ``~/.hermes`` to
``%LOCALAPPDATA%\\hermes`` (see ``_get_platform_default_hermes_home`` in
``hermes_constants.py``).  Nothing carried existing state across that move:
the installer seeds the new home from *templates*, so a user upgrading in
place came up with fresh config and silently lost everything the templates
don't cover — most painfully ``auth.json`` (provider credentials) and
``cron/jobs.json`` (all scheduled jobs), which made cron jobs die with
"No Codex credentials stored" until the files were copied by hand.

This module closes that gap with a *fill-gap* copy: every state file or
directory present in the legacy home and absent in the new home is copied
over; anything that already exists in the new home is left untouched (the
new home may have newer state — it wins).  Rather than a curated
include-list (the bug class that caused the incident), the walk copies
everything and excludes only regeneratable or process-local entries:
caches, logs, locks, pid files, sandboxes, and managed tool installs.

The migration is one-shot: a marker file is written to the new home after a
completed pass and skips all future runs.  A failed pass writes no marker,
so the (idempotent) copy simply retries on the next start.

Callers invoke :func:`maybe_migrate_legacy_windows_home` as early as
possible — before anything reads config from the home.  The function never
raises.
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import _get_platform_default_hermes_home, get_hermes_home

logger = logging.getLogger(__name__)

# Marker written to the NEW home after a completed migration pass.
MARKER_NAME = ".legacy_home_migrated.json"

# Top-level or nested entry names that are never copied.  Everything else in
# the legacy home is state worth carrying over.  Keep this an EXCLUDE list:
# the incident this module fixes was caused by an include-list that missed
# ``auth.json`` and ``cron/jobs.json``.
_EXCLUDED_NAMES = frozenset(
    {
        # Regeneratable caches / transient dirs
        "cache",
        "bootstrap-cache",
        "image_cache",
        "audio_cache",
        "media_cache",
        "document_cache",
        "tmp",
        "logs",
        "sandboxes",
        "trash",
        # Process-local runtime state — must reflect the live process
        "gateway.lock",
        "gateway.pid",
        ".update_check",
        # Managed tooling installs — large and re-downloadable
        "node",
        "git",
        "bin",
        "venv",
        "node_modules",
        # The Windows installer places the source checkout inside the home
        "hermes-agent",
        # Service wrappers bake in old-home paths; `hermes setup` regenerates
        "gateway-service",
    }
)

# How many per-file entries the marker records before truncating (a large
# sessions/ tree can hold thousands of files; counts stay exact).
_MARKER_LIST_CAP = 500


def _is_excluded(name: str) -> bool:
    if name in _EXCLUDED_NAMES:
        return True
    # Lock/pid files are process-local; SQLite sidecars are copied only
    # together with their database (see _copy_file), never standalone.
    return name.endswith((".lock", ".pid", "-wal", "-shm", "-journal"))


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _looks_like_hermes_home(path: Path) -> bool:
    """A directory qualifies as a legacy home when it holds real state."""
    markers = ("config.yaml", "auth.json", "state.db", ".env", "cron")
    try:
        return path.is_dir() and any((path / m).exists() for m in markers)
    except OSError:
        return False


def _copy_file(src: Path, dst: Path, copied: list, errors: list) -> None:
    try:
        shutil.copy2(src, dst)
        copied.append(str(dst))
    except OSError as exc:
        errors.append(f"{src}: {exc}")
        return
    # A ``<name>-wal`` sibling marks a SQLite database in WAL mode.  The WAL
    # holds committed-but-not-checkpointed transactions, so copy it together
    # with a freshly copied db (the destination db did not exist, so there is
    # nothing to clobber).  ``-shm`` is scratch shared memory — SQLite
    # recreates it; never copy it.
    wal = src.parent / (src.name + "-wal")
    dst_wal = dst.parent / (dst.name + "-wal")
    if wal.exists() and not dst_wal.exists():
        try:
            shutil.copy2(wal, dst_wal)
            copied.append(str(dst_wal))
        except OSError as exc:
            errors.append(f"{wal}: {exc}")


def _fill_gap_copy(
    src_dir: Path, dst_dir: Path, copied: list, skipped: list, errors: list
) -> None:
    """Recursively copy entries missing from *dst_dir*; never overwrite."""
    try:
        entries = sorted(src_dir.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        errors.append(f"{src_dir}: {exc}")
        return
    for entry in entries:
        name = entry.name
        if _is_excluded(name):
            continue
        dst = dst_dir / name
        try:
            if entry.is_symlink():
                # Don't chase links out of the legacy home.
                continue
            if entry.is_dir():
                if dst.exists() and not dst.is_dir():
                    skipped.append(str(dst))
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                _fill_gap_copy(entry, dst, copied, skipped, errors)
            elif entry.is_file():
                if dst.exists():
                    skipped.append(str(dst))
                    continue
                _copy_file(entry, dst, copied, errors)
        except OSError as exc:
            errors.append(f"{entry}: {exc}")


def maybe_migrate_legacy_windows_home(
    new_home: Path | None = None,
    legacy_home: Path | None = None,
    platform: str | None = None,
) -> dict | None:
    """Fill state gaps in the native Windows home from a legacy ``~/.hermes``.

    Runs at most once (marker-gated), only on Windows, and only when the
    active home *is* the platform default — a custom ``HERMES_HOME``
    pointing elsewhere is respected and left alone.  Existing files in the
    new home are never overwritten.

    Returns a summary dict when a migration pass ran, ``None`` otherwise.
    Never raises.
    """
    try:
        if (platform or sys.platform) != "win32":
            return None

        new_home = Path(new_home) if new_home else get_hermes_home()
        if _norm(new_home) != _norm(_get_platform_default_hermes_home()):
            return None

        legacy_home = (
            Path(legacy_home) if legacy_home else Path.home() / ".hermes"
        )
        if _norm(legacy_home) == _norm(new_home):
            return None
        if not _looks_like_hermes_home(legacy_home):
            return None
        try:
            if new_home.exists() and os.path.samefile(legacy_home, new_home):
                # One is a junction/symlink onto the other — nothing to do.
                return None
        except OSError:
            pass

        marker_path = new_home / MARKER_NAME
        if marker_path.exists():
            return None

        new_home.mkdir(parents=True, exist_ok=True)

        copied: list = []
        skipped: list = []
        errors: list = []
        _fill_gap_copy(legacy_home, new_home, copied, skipped, errors)

        summary = {
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "legacy_home": str(legacy_home),
            "new_home": str(new_home),
            "copied_count": len(copied),
            "skipped_existing_count": len(skipped),
            "error_count": len(errors),
            "copied": copied[:_MARKER_LIST_CAP],
            "errors": errors[:_MARKER_LIST_CAP],
        }

        if errors:
            # No marker: the fill-gap copy is idempotent, so the next start
            # retries whatever failed (e.g. a file locked by a stale gateway).
            logger.warning(
                "Legacy Hermes home migration incomplete: copied %d item(s) "
                "from %s, %d error(s) — will retry on next start. First "
                "error: %s",
                len(copied),
                legacy_home,
                len(errors),
                errors[0],
            )
            return summary

        try:
            marker_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "Legacy home migration could not write marker %s: %s",
                marker_path,
                exc,
            )

        if copied:
            logger.info(
                "Migrated legacy Hermes home %s -> %s: copied %d item(s), "
                "kept %d existing item(s). Details: %s",
                legacy_home,
                new_home,
                len(copied),
                len(skipped),
                marker_path,
            )
        return summary
    except Exception as exc:  # noqa: BLE001 — must never block startup
        logger.warning("Legacy Hermes home migration skipped: %s", exc)
        return None
