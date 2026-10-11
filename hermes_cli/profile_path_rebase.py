"""Rebase absolute paths stored under a renamed profile directory (#136430).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/``, so every absolute path that
pointed inside the old directory now names a directory that no longer exists. ``projects.db``
(``projects.primary_path``, ``project_folders.path``, ``discovered_repos.root``) and ``state.db``
(``sessions.cwd``, ``sessions.git_repo_root``, ACP's ``model_config.cwd``) persist such paths; left
alone, moving a session into a project under the profile fails with ``working directory does not
exist``. Kanban task workspaces and cron job workdirs can also name profile paths and are not
rebased here.

Matching is by whole path component: ``profiles/old`` rebases ``profiles/old`` and
``profiles/old/x`` but never the sibling ``profiles/old2``. It is case-insensitive where the
platform is (``os.path.normcase``, as ``projects_db`` compares primary paths), and the stored suffix
keeps its own spelling. Only the spellings from :func:`prefix_pairs` are recognised: the path as
built and its symlink-resolved form. Idempotent: a rebased path no longer starts with the old
prefix, so a retry (``hermes profile migrate-identity``) finds nothing to do.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Callable, Optional, Sequence

PrefixPairs = Sequence[tuple[str, str]]


def prefix_pairs(old_dir: Path, new_dir: Path) -> list[tuple[str, str]]:
    """``(old, new)`` prefixes to rebase: the path as built plus its symlink-resolved form (e.g.
    macOS ``/var`` → ``/private/var``), since stored paths may carry either spelling. Resolution goes
    through the parent because ``old_dir`` itself no longer exists after the move."""
    pairs = [(str(old_dir), str(new_dir))]
    try:
        parent = new_dir.parent.resolve()
    except (OSError, RuntimeError):  # RuntimeError: symlink loop on Python < 3.13
        return pairs
    resolved = (str(parent / old_dir.name), str(parent / new_dir.name))
    if resolved != pairs[0]:
        pairs.append(resolved)
    return pairs


def rebase_path(
    path: object, pairs: PrefixPairs, normcase: Callable[[str], str] = os.path.normcase,
) -> Optional[str]:
    """*path* moved under the new prefix, or None when it does not live under any old prefix.

    *normcase* must preserve length (``posixpath``/``ntpath.normcase`` only fold case and
    separators), so the suffix after the matched prefix is spliced from the original *path*.
    """
    if not isinstance(path, str) or not path:
        return None
    folded = normcase(path)
    for old, new in pairs:
        old = old.rstrip("/\\")
        old_folded = normcase(old)
        if folded == old_folded:
            return new
        if folded.startswith(old_folded) and path[len(old)] in "/\\":
            return new + path[len(old):]
    return None


def _rebase_keyed_rows(
    conn: sqlite3.Connection, table: str, column: str, key: str, pairs: PrefixPairs,
) -> tuple[int, set]:
    """Rebase *column* in a table whose primary key includes it. A row whose rebased value already
    exists (a retry, or a folder added under the new path) is merged into that row. Returns the
    number of rows changed and the *key* values of the rows that were merged."""
    where_key = f" AND {key} = ?" if key else ""
    select_key = f", {key}" if key else ""
    changed, merged = 0, set()
    for row in conn.execute(f"SELECT {column}{select_key} FROM {table}").fetchall():
        new = rebase_path(row[0], pairs)
        if new is None:
            continue
        params = (row[0], *row[1:])
        moved = conn.execute(
            f"UPDATE OR IGNORE {table} SET {column} = ? WHERE {column} = ?{where_key}", (new, *params)).rowcount
        if not moved:  # the rebased value already exists: drop the stale duplicate
            conn.execute(f"DELETE FROM {table} WHERE {column} = ?{where_key}", params)
            if key:
                merged.add(row[1])
        changed += 1
    return changed, merged


def _rederive_primary_flags(conn: sqlite3.Connection, project_ids: set) -> None:
    """A merged folder row may have carried the primary flag; re-derive it for those projects only,
    comparing paths the way ``projects_db`` does (normalized, case-folded where the OS is)."""
    from hermes_cli.projects_db import _primary_path_key

    for project_id in project_ids:
        row = conn.execute("SELECT primary_path FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None or not row[0]:
            continue
        primary_key = _primary_path_key(row[0])
        for (path,) in conn.execute(
                "SELECT path FROM project_folders WHERE project_id = ?", (project_id,)).fetchall():
            conn.execute("UPDATE project_folders SET is_primary = ? WHERE project_id = ? AND path = ?",
                         (int(_primary_path_key(path) == primary_key), project_id, path))


def rebase_projects_db(db_path: Path, pairs: PrefixPairs) -> int:
    """Rebase every project/folder/discovered-repo path in *db_path*; returns rows changed."""
    from hermes_cli import projects_db
    from hermes_cli.sqlite_util import write_txn

    if not db_path.exists():
        return 0
    with projects_db.connect_closing(db_path) as conn, write_txn(conn):
        changed = 0
        for project_id, primary in conn.execute("SELECT id, primary_path FROM projects").fetchall():
            new = rebase_path(primary, pairs)
            if new is not None:
                conn.execute("UPDATE projects SET primary_path = ? WHERE id = ?", (new, project_id))
                changed += 1
        folders_changed, merged_projects = _rebase_keyed_rows(conn, "project_folders", "path", "project_id", pairs)
        repos_changed, _ = _rebase_keyed_rows(conn, "discovered_repos", "root", "", pairs)
        _rederive_primary_flags(conn, merged_projects)
        return changed + folders_changed + repos_changed
