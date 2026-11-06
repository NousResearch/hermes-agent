"""Explicit session -> project binding (``sessions.project_id``).

A session started in (or moved into) a project carries that project's id; ``cwd`` stays the
working folder. Membership readers treat ``project_id`` as authoritative and fall back to cwd
inference only for unbound rows ("In this folder"). The project rows themselves live in the
per-profile ``projects.db`` next to ``state.db`` (``hermes_cli/projects_db.py``).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# state_meta marker: the one-time v1 backfill ran (or the store was created already binding).
SESSION_PROJECT_BACKFILL_DONE = "session_project_backfill_v1"
# Surfaces whose sessions are created or moved deliberately by a client (desktop / Otto / TUI),
# so a cwd that IS a project's primary folder means "filed under that project". CLI sessions run
# from a subfolder stay unbound.
BACKFILL_SOURCES = ("api_server", "desktop", "tui")


def _path_key(path: str) -> str:
    """Comparison key matching ``projects_db._primary_path_key`` (absolute, case/sep-normalized)."""
    expanded = os.path.abspath(os.path.expanduser(str(path).strip()))
    return os.path.normcase(expanded.rstrip("/\\") or expanded)


def _primary_paths(projects_db: Path) -> dict[str, str]:
    """``{primary-path key: project id}`` for every active project, oldest project first wins.
    Read-only: a missing projects.db is never created here."""
    if not projects_db.is_file():
        return {}
    conn = sqlite3.connect(f"file:{projects_db}?mode=ro", uri=True, timeout=5)
    try:
        rows = conn.execute(
            "SELECT p.id, COALESCE(p.primary_path, (SELECT f.path FROM project_folders f "
            "WHERE f.project_id = p.id ORDER BY f.is_primary DESC, f.added_at ASC LIMIT 1)) "
            "FROM projects p WHERE p.archived = 0 ORDER BY p.created_at ASC").fetchall()
    finally:
        conn.close()
    by_path: dict[str, str] = {}
    for pid, primary in rows:
        if primary:
            by_path.setdefault(_path_key(primary), pid)
    return by_path


def backfill_session_projects(cursor: sqlite3.Cursor, projects_db: Path) -> int:
    """Bind legacy client sessions whose cwd equals a project's primary folder (one-time).

    Only ``source``/``created_source`` in :data:`BACKFILL_SOURCES` and only NULL ``project_id``
    rows qualify; an explicit binding is never rewritten. Returns the number of rows bound.
    """
    try:
        by_path = _primary_paths(projects_db)
    except sqlite3.Error:
        logger.warning("session project backfill: projects.db unreadable, skipped", exc_info=True)
        return 0
    if not by_path:
        return 0
    marks = ", ".join("?" for _ in BACKFILL_SOURCES)
    rows = cursor.execute(
        f"SELECT id, cwd FROM sessions WHERE project_id IS NULL AND COALESCE(cwd, '') != '' "
        f"AND (source IN ({marks}) OR created_source IN ({marks}))",
        (*BACKFILL_SOURCES, *BACKFILL_SOURCES)).fetchall()
    updates = [(pid, row[0]) for row in rows if (pid := by_path.get(_path_key(row[1])))]
    if updates:
        cursor.executemany("UPDATE sessions SET project_id = ? WHERE id = ? AND project_id IS NULL", updates)
    return len(updates)


def run_session_project_backfill_once(store: Any, cursor: sqlite3.Cursor) -> None:
    """Run :func:`backfill_session_projects` once per store, recorded in ``state_meta``.

    ``projects.db`` is the per-profile sibling of ``state.db``; a store without one (tests, ad-hoc
    copies, a profile that never created a project) is marked done with nothing to bind.
    """
    if cursor.execute("SELECT 1 FROM state_meta WHERE key = ?", (SESSION_PROJECT_BACKFILL_DONE,)).fetchone():
        return
    try:
        bound = backfill_session_projects(cursor, Path(store.db_path).parent / "projects.db")
    except sqlite3.OperationalError:
        logger.warning("session project backfill failed; will retry on next open", exc_info=True)
        return
    if bound:
        logger.info("session project backfill: bound %d session(s) to their project", bound)
    store.set_meta(SESSION_PROJECT_BACKFILL_DONE, "1", cursor=cursor)


def set_session_project(db: Any, session_id: str, project_id: Optional[str]) -> bool:
    """Bind (or with ``None`` unbind) a session and its whole compression lineage: list readers
    project roots forward to their tip, so a tip-only write would let the root disagree."""
    return db._set_lineage_column("project_id", session_id, project_id or None)


def session_run_cwd(row: Optional[dict]) -> str:
    """The working folder a turn for this session row runs in: the stored ``cwd`` (which a project
    binding sets to the project's folder), ``""`` when the row has none."""
    return str((row or {}).get("cwd") or "").strip()
