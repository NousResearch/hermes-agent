"""Local paste store: collapsed large pastes live in ``$HERMES_HOME/pastes`` and expire.

The composer collapses a large paste into ``[Pasted text #N: L lines → <path>]`` and keeps the
text in ``pastes/paste_<n>_<HHMMSS>.txt``. Submitting expands the placeholder back into the file's
contents (``cli_stream_mixin._expand_paste_references``), so the file has exactly one job: carry
the text from the paste to the send. After that it is copy-paste input, not a backup — the text is
already in the session transcript and in whatever app it was copied from.

Nothing used to delete those files, so they accumulated forever: a privacy problem, not a feature.
They now expire. ``paste_retention_minutes`` (default 60; 0 keeps them forever) bounds their age,
and the sweep runs when a paste is written and once per surface start.

Never touched here: ``pastes/pending.json`` — that belongs to ``hermes debug share`` (public
paste.rs uploads) and has its own expiry in ``hermes_cli.debug``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

PASTE_FILE_PREFIX = "paste_"
PASTE_FILE_SUFFIX = ".txt"
PASTE_DIRNAME = "pastes"
#: ``hermes debug share`` state — lives in the same directory, expires on its own schedule.
DEBUG_PENDING_NAME = "pending.json"

DEFAULT_RETENTION_MINUTES = 60
#: A paste is swept once per minute per process, not on every write.
SWEEP_INTERVAL_SECONDS = 60.0

_sweep_lock = threading.Lock()
_last_sweep_at = 0.0

#: Pastes written by this process. Deleted when the process exits so a whole session's pasted text
#: does not outlive the session; the age sweep is the safety net for anything a crash leaves behind.
_session_pastes: set[Path] = set()
_atexit_hooked = False


def _install_atexit_cleanup() -> None:
    global _atexit_hooked
    if _atexit_hooked:
        return
    _atexit_hooked = True
    import atexit
    atexit.register(clear_session_pastes)


def clear_session_pastes() -> int:
    """Delete every paste this process wrote → count removed. Honors retention 0 (keep forever)."""
    with _sweep_lock:
        pending = list(_session_pastes)
        _session_pastes.clear()
    if not pending:
        return 0
    if retention_minutes() <= 0:
        return 0
    removed = 0
    for path in pending:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue  # already swept by age, or a permissions problem; the sweep will retry
    return removed


def paste_dir(home: Optional[Path] = None) -> Path:
    """The profile's paste directory (``<hermes home>/pastes``)."""
    if home is None:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return Path(home) / PASTE_DIRNAME


def is_paste_file(name: str) -> bool:
    """True for a collapsed-paste payload; False for ``pending.json`` and anything else."""
    return name.startswith(PASTE_FILE_PREFIX) and name.endswith(PASTE_FILE_SUFFIX)


def retention_minutes(config: Optional[Mapping] = None) -> float:
    """Configured retention in minutes. ``0`` (or negative) means "never delete"."""
    raw = None
    if config is not None:
        raw = config.get("paste_retention_minutes", DEFAULT_RETENTION_MINUTES)
    else:
        try:
            from hermes_cli.config import load_config_readonly
            raw = (load_config_readonly() or {}).get("paste_retention_minutes",
                                                     DEFAULT_RETENTION_MINUTES)
        except Exception:
            logger.debug("Could not read paste_retention_minutes; using default", exc_info=True)
            return float(DEFAULT_RETENTION_MINUTES)
    try:
        minutes = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_RETENTION_MINUTES)
    if minutes != minutes or minutes < 0:  # NaN or negative → default
        return float(DEFAULT_RETENTION_MINUTES)
    return minutes


def write_paste(text: str, counter: int, *, home: Optional[Path] = None) -> Path:
    """Write a collapsed paste and return its path.

    Names match the historical scheme (``paste_<n>_<HHMMSS>.txt``); when two processes write the
    same counter in the same second the second one gets a numeric suffix instead of clobbering the
    first paste.
    """
    directory = paste_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{PASTE_FILE_PREFIX}{counter}_{datetime.now().strftime('%H%M%S')}"
    path = directory / f"{stem}{PASTE_FILE_SUFFIX}"
    attempt = 1
    while True:
        try:
            with open(path, "x", encoding="utf-8") as fh:
                fh.write(text)
            break
        except FileExistsError:
            path = directory / f"{stem}_{attempt}{PASTE_FILE_SUFFIX}"
            attempt += 1
    _install_atexit_cleanup()
    with _sweep_lock:
        _session_pastes.add(path)
    return path


def _paste_files(directory: Path) -> list[os.DirEntry]:
    try:
        with os.scandir(directory) as it:
            return [e for e in it if is_paste_file(e.name)]
    except OSError:
        return []


def sweep_pastes(
    now: Optional[float] = None,
    *,
    minutes: Optional[float] = None,
    home: Optional[Path] = None,
    purge_all: bool = False,
) -> tuple[int, int]:
    """Delete expired collapsed pastes → ``(deleted, remaining)``.

    ``purge_all=True`` ignores age and removes every paste file (explicit user request only).
    Best-effort: an undeletable file stays for the next sweep rather than raising.
    """
    directory = paste_dir(home)
    entries = _paste_files(directory)
    if not entries:
        return (0, 0)
    limit = minutes if minutes is not None else retention_minutes()
    cutoff_seconds = max(limit, 0.0) * 60.0
    current = time.time() if now is None else now
    keep_forever = (not purge_all) and limit <= 0

    deleted = 0
    remaining = 0
    for entry in entries:
        if keep_forever:
            remaining += 1
            continue
        try:
            age = current - entry.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue  # vanished between scan and stat
        if not purge_all and age < cutoff_seconds:
            remaining += 1
            continue
        try:
            os.unlink(entry.path)
            deleted += 1
        except OSError:
            remaining += 1
    if deleted:
        logger.info("Swept %d expired paste file(s) from %s", deleted, directory)
    return (deleted, remaining)


def maybe_sweep_pastes(*, config: Optional[Mapping] = None, home: Optional[Path] = None) -> None:
    """Sweep at most once per :data:`SWEEP_INTERVAL_SECONDS` per process; never raises."""
    global _last_sweep_at
    now = time.monotonic()
    with _sweep_lock:
        if now - _last_sweep_at < SWEEP_INTERVAL_SECONDS:
            return
        _last_sweep_at = now
    try:
        sweep_pastes(minutes=retention_minutes(config), home=home)
    except Exception:
        logger.debug("Paste sweep failed", exc_info=True)


def list_pastes(home: Optional[Path] = None) -> list[Path]:
    """Current paste files, newest first."""
    entries = _paste_files(paste_dir(home))
    paths: list[Path] = []
    for entry in entries:
        try:
            paths.append(entry.path if isinstance(entry.path, Path) else Path(entry.path))
        except OSError:
            continue
    return sorted(paths, key=lambda p: _mtime(p), reverse=True)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
