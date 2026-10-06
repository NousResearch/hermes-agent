"""Desktop composer-images cleanup and reference management utilities.

Usage guide:
  Library module - not executed directly. Public entry points:
    - get_composer_images_dir()         : resolve Electron userData/composer-images cross-platform
    - extract_composer_image_paths()    : extract @image: absolute paths from one persisted message
    - collect_active_composer_refs()    : collect the union of composer refs across all live sessions
    - cleanup_composer_for_deletion()   : remove composer images referenced exclusively by deleted sessions
    - cleanup_orphaned_composer_images(): hourly sweep: drop stale files with no surviving session ref
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_state")


def _env_dir(var: str, fallback: Path) -> Path:
    """Path($var) when set, else *fallback*."""
    return Path(value) if (value := os.environ.get(var)) else fallback


def desktop_userdata_dir() -> Path:
    """Electron ``app.getPath('userData')`` for app "Hermes" on each platform.

    Honors ``HERMES_DESKTOP_USER_DATA_DIR`` (full override) and
    ``HERMES_DATA_DIR_SUFFIX`` (suffix appended to the default path), mirroring
    the desktop's ``resolveDesktopUserData`` in ``data-paths.mjs``.
    """
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / "Hermes"
    elif sys.platform == "win32":
        base = _env_dir("APPDATA", home / "AppData" / "Roaming") / "Hermes"
    else:
        base = _env_dir("XDG_CONFIG_HOME", home / ".config") / "Hermes"

    override = os.environ.get("HERMES_DESKTOP_USER_DATA_DIR")
    if override:
        return Path(override)

    suffix = os.environ.get("HERMES_DATA_DIR_SUFFIX") or ""
    if suffix:
        return Path(str(base) + suffix)
    return base


def get_composer_images_dir() -> Path:
    """Return the Electron userData/composer-images directory (platform-neutral).

    The directory is not required to exist; callers create it only when writing.
    """
    return desktop_userdata_dir() / "composer-images"


_IMAGE_REF_RE = re.compile(r"@image:(`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|\S+)")


def _unquote_ref(value: str) -> str:
    """Strip one pair of matching surrounding quotes from an @image: ref value."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("`", '"', "'"):
        return value[1:-1]
    return value


def extract_composer_image_paths(content: Any, composer_dir: Path) -> list[str]:
    """Extract absolute paths of composer-images referenced in one persisted message.

    *content* may be a plain string, a structured parts ``list``, or a structured
    ``dict`` — mirrors the shapes stored in the ``messages.content`` column.
    Only paths under *composer_dir* (resolved real-path when possible) are returned;
    ``@image:`` refs pointing elsewhere (paste dir, gateway media caches, HTTP) are
    skipped. The returned paths are the raw strings stored (symlinks not resolved on
    the way out, so the set matches the on-disk filename used in ref counting).
    """
    if content is None:
        return []

    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, dict):
        t = content.get("text") or content.get("content")
        if isinstance(t, str):
            texts.append(t)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                t = part.get("text") or part.get("content")
                if isinstance(t, str):
                    texts.append(t)

    if not texts:
        return []

    composer_str = str(composer_dir)
    found: list[str] = []
    for text in texts:
        for match in _IMAGE_REF_RE.finditer(text):
            raw = _unquote_ref(match.group(1))
            if not raw:
                continue
            try:
                candidate = Path(raw)
            except (OSError, ValueError):
                continue
            if not candidate.is_absolute():
                continue
            # Fast prefix check against the textual composer-dir path; tolerate
            # both Windows separator forms since content may have been written on
            # another host (rare) — Path comparison handles that on each platform.
            try:
                candidate.relative_to(composer_dir)
                under = True
            except ValueError:
                under = False
            if under:
                # The stored string is the canonical ref key (matching the on-disk
                # write from the Electron side), not a resolved form.
                found.append(raw)
            elif raw.startswith(composer_str + os.sep) or (
                os.altsep and raw.startswith(composer_str + os.altsep)
            ):
                found.append(raw)
    return found


def collect_active_composer_refs(
    conn: sqlite3.Connection,
    composer_dir: Path,
    exclude_session_ids: set[str] | None = None,
) -> set[str]:
    """Scan every active ``messages.content`` row and return the set of composer-image
    absolute paths still referenced by *any* session (optionally excluding a set of
    session ids whose deletion is in-flight).

    The scan is a full-table pass over the (usually small) messages content column;
    we only fetch rows that contain ``@image:`` via LIKE to avoid decoding unrelated
    payloads. ``@image`` is never stored without the colon in persisted text so the
    LIKE filter is selective enough to be cheap on real databases.
    """
    refs: set[str] = set()
    exclude = exclude_session_ids or set()
    try:
        if exclude:
            placeholders = ",".join("?" * len(exclude))
            cursor = conn.execute(
                f"SELECT session_id, content FROM messages WHERE content LIKE '%@image:%' "
                f"AND active = 1 AND session_id NOT IN ({placeholders})",
                tuple(exclude),
            )
        else:
            cursor = conn.execute(
                "SELECT session_id, content FROM messages WHERE content LIKE '%@image:%' AND active = 1"
            )
    except sqlite3.OperationalError as exc:
        logger.debug("composer ref scan skipped: %s", exc)
        return refs

    for _sid, content in cursor:
        for path in extract_composer_image_paths(content, composer_dir):
            refs.add(path)
    return refs


def _safe_unlink_many(paths: list[Path]) -> int:
    removed = 0
    for path in paths:
        with contextlib.suppress(OSError):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def collect_composer_refs_for_sessions(
    conn: sqlite3.Connection,
    session_ids: list[str],
) -> set[str]:
    """Return the set of composer-image absolute paths referenced by *any* message
    in ``session_ids``. Used *before* deleting those session rows so the refs can
    be compared against the surviving set after commit."""
    if not session_ids:
        return set()
    composer_dir = get_composer_images_dir()
    refs: set[str] = set()
    try:
        placeholders = ",".join("?" * len(session_ids))
        cursor = conn.execute(
            f"SELECT content FROM messages WHERE content LIKE '%@image:%' "
            f"AND session_id IN ({placeholders})",
            tuple(session_ids),
        )
    except sqlite3.OperationalError as exc:
        logger.debug("composer refs-for-sessions scan skipped: %s", exc)
        return refs
    for (content,) in cursor:
        for path in extract_composer_image_paths(content, composer_dir):
            refs.add(path)
    return refs


def cleanup_composer_for_deletion(
    conn: sqlite3.Connection,
    deleted_refs: set[str],
) -> int:
    """Given the set of composer refs collected from the rows about to be (or just
    been) deleted, remove files no longer referenced by any surviving session.

    Callers should run ``deleted_refs = collect_composer_refs_for_sessions(...)``
    inside the write transaction (before DELETE), then invoke this helper after
    the DELETE has committed so the surviving-ref scan sees the post-delete DB
    state. Returns the count of files unlinked.
    """
    if not deleted_refs:
        return 0
    composer_dir = get_composer_images_dir()
    if not composer_dir.is_dir():
        return 0
    surviving = collect_active_composer_refs(conn, composer_dir)
    orphans = deleted_refs - surviving
    return _safe_unlink_many([Path(p) for p in orphans])


def cleanup_orphaned_composer_images(
    conn: sqlite3.Connection | None = None,
    *,
    max_age_hours: int = 168,
) -> int:
    """Hourly-sweep cleanup for the desktop composer-images folder.

    Removes files older than *max_age_hours* (default 7 days) that are no longer
    referenced by any active session row. Grace period: a file newer than the
    window is never removed even if unreferenced — in-flight drafts or a just-pasted
    image that hasn't been flushed yet shouldn't be reaped between paste and send.
    Returns the count of files unlinked.

    If *conn* is omitted, opens a temporary read-only connection against the
    current profile's ``state.db`` (``get_hermes_home() / "state.db"``) and closes
    it before returning. Callers that already hold a SessionDB connection (e.g.
    ``SessionDB._conn`` or ``hermes_state_sessions.delete_session``'s post-commit
    writer) may pass it in directly to save an open.
    """
    composer_dir = get_composer_images_dir()
    if not composer_dir.is_dir():
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    old_files: list[Path] = []
    try:
        for entry in composer_dir.iterdir():
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    old_files.append(entry)
            except OSError:
                continue
    except OSError as exc:
        logger.debug("composer-images dir scan skipped: %s", exc)
        return 0
    if not old_files:
        return 0

    borrowed = conn is not None
    db_conn = conn
    try:
        if db_conn is None:
            from hermes_constants import get_hermes_home  # noqa: PLC0415

            db_path = get_hermes_home() / "state.db"
            if not db_path.is_file():
                # No session DB yet → all old files are eligible since nothing can
                # reference them. Wide sweep with the same grace window is safe.
                return _safe_unlink_many(old_files)
            try:
                db_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            except sqlite3.OperationalError as exc:
                logger.debug("composer orphan sweep: DB open failed: %s", exc)
                return 0
        active_refs = collect_active_composer_refs(db_conn, composer_dir)
        active_abs: set[str] = set()
        for ref in active_refs:
            active_abs.add(ref)
            with contextlib.suppress(OSError):
                active_abs.add(str(Path(ref).resolve()))

        candidates: list[Path] = []
        for f in old_files:
            f_str = str(f)
            if f_str in active_abs:
                continue
            with contextlib.suppress(OSError):
                if str(f.resolve()) in active_abs:
                    continue
            candidates.append(f)
        return _safe_unlink_many(candidates)
    finally:
        if db_conn is not None and not borrowed:
            with contextlib.suppress(Exception):
                db_conn.close()
