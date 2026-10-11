"""Rebase absolute paths stored under a renamed profile directory (#136430).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/``, so every absolute path that
pointed inside the old directory now names a directory that no longer exists. ``projects.db``
(``projects.primary_path``, ``project_folders.path``, ``discovered_repos.root``) and ``state.db``
(``sessions.cwd``, ``sessions.git_repo_root``) persist such paths; left alone, moving a session into
a project under the profile fails with ``working directory does not exist``.

Matching is by whole path component: ``profiles/old`` rebases ``profiles/old`` and
``profiles/old/x`` but never the sibling ``profiles/old2``. Idempotent: a rebased path no longer
starts with the old prefix, so a retry (``hermes profile migrate-identity``) finds nothing to do.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Sequence

PrefixPairs = Sequence[tuple[str, str]]


def prefix_pairs(old_dir: Path, new_dir: Path) -> list[tuple[str, str]]:
    """``(old, new)`` prefixes to rebase: the path as built plus its symlink-resolved form (e.g.
    macOS ``/var`` → ``/private/var``), since stored paths may carry either spelling. Resolution goes
    through the parent because ``old_dir`` itself no longer exists after the move."""
    pairs = [(str(old_dir), str(new_dir))]
    try:
        parent = new_dir.parent.resolve()
    except OSError:
        return pairs
    resolved = (str(parent / old_dir.name), str(parent / new_dir.name))
    if resolved != pairs[0]:
        pairs.append(resolved)
    return pairs


def rebase_path(path: object, pairs: PrefixPairs) -> Optional[str]:
    """*path* moved under the new prefix, or None when it does not live under any old prefix."""
    if not isinstance(path, str) or not path:
        return None
    for old, new in pairs:
        old = old.rstrip("/\\")
        if path == old:
            return new
        if path.startswith(old) and path[len(old)] in "/\\":
            return new + path[len(old):]
    return None


def _rebase_keyed_rows(conn: sqlite3.Connection, table: str, column: str, key: str, pairs: PrefixPairs) -> int:
    """Rebase *column* in a table whose primary key includes it. A row whose rebased value already
    exists (a retry, or a folder added under the new path) is merged into that row."""
    where_key = f" AND {key} = ?" if key else ""
    select_key = f", {key}" if key else ""
    changed = 0
    for row in conn.execute(f"SELECT {column}{select_key} FROM {table}").fetchall():
        new = rebase_path(row[0], pairs)
        if new is None:
            continue
        params = (row[0], *row[1:])
        conn.execute(f"UPDATE OR IGNORE {table} SET {column} = ? WHERE {column} = ?{where_key}", (new, *params))
        conn.execute(f"DELETE FROM {table} WHERE {column} = ?{where_key}", params)  # collided: merged above
        changed += 1
    return changed


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
        changed += _rebase_keyed_rows(conn, "project_folders", "path", "project_id", pairs)
        changed += _rebase_keyed_rows(conn, "discovered_repos", "root", "", pairs)
        if changed:
            # A merged folder row may have dropped the primary flag; re-derive it from primary_path.
            conn.execute(
                "UPDATE project_folders SET is_primary = (path = "
                "(SELECT primary_path FROM projects WHERE projects.id = project_folders.project_id)) "
                "WHERE project_id IN (SELECT id FROM projects WHERE primary_path IS NOT NULL)")
        return changed
