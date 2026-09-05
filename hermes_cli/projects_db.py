"""Per-profile first-class Project store.

The schema is small and additive: column additions go through ``add_column_if_missing`` so
opening an old DB is always safe.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing, write_txn
from hermes_constants import get_hermes_home


def projects_db_path() -> Path:
    """The per-profile projects DB path (``$HERMES_HOME/projects.db``)."""
    return get_hermes_home() / "projects.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    slug          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    description   TEXT,
    icon          TEXT,
    color         TEXT,
    board_slug    TEXT,
    primary_path  TEXT,
    created_at    INTEGER NOT NULL,
    archived      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_folders (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    label       TEXT,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    added_at    INTEGER NOT NULL,
    PRIMARY KEY (project_id, path)
);

CREATE INDEX IF NOT EXISTS idx_project_folders_path
    ON project_folders(path);

CREATE TABLE IF NOT EXISTS project_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

-- Git repos found by scanning the filesystem (desktop "repo-first" discovery).
-- Cached here so the overview is instant after the first scan instead of
-- re-walking the disk every time the Projects view opens.
CREATE TABLE IF NOT EXISTS discovered_repos (
    root          TEXT PRIMARY KEY,
    label         TEXT,
    last_seen     INTEGER NOT NULL
);
"""

# Lowercase alphanumerics, hyphens, underscores; 1-64 chars; no leading separator. Strict enough to
# stop traversal/path separators, loose enough for kebab-case. Display formatting lives in ``name``.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")
# Deterministic branch slug: lowercase, separators collapsed, capped.
_BRANCH_SAFE_RE = re.compile(r"[^a-z0-9._-]+")
_INITIALIZED_PATHS: set[str] = set()
# TEXT columns added to `projects` after v1; re-applied idempotently on every open so a legacy DB
# upgrades in place.
_OPTIONAL_PROJECT_COLUMNS = ("board_slug", "primary_path", "icon", "color")
# Nullable TEXT columns that may be absent from a legacy row.
_OPTIONAL_ROW_FIELDS = ("description", "icon", "color", "board_slug", "primary_path")
_ACTIVE_META_KEY = "active_id"
_DISCOVERY_POLICY_META_KEY = "repo_discovery_policy"


def _slugify(name: str) -> str:
    """Derive a slug candidate from a human name (best-effort)."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-_")
    return s[:64].strip("-_") or "project"


def normalize_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    s = str(slug).strip().lower() if slug is not None else ""
    if not s:
        return None
    if not _SLUG_RE.match(s):
        raise ValueError(
            f"invalid project slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with "
            f"'-' or '_'"
        )
    return s


def _now() -> int:
    return int(time.time())


def _normalize_path(path: str) -> str:
    """Absolute, user-expanded, separator-normalized path (no trailing sep)."""
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    return p.rstrip("/\\") or p


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the per-profile projects DB.

    WAL with DELETE fallback for network filesystems (``hermes_state`` helper). Schema init is
    idempotent (``CREATE TABLE IF NOT EXISTS`` + additive migrations) and cached per-path per-process.
    """
    path = db_path if db_path is not None else projects_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        from hermes_state_wal import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="projects.db")
        conn.execute("PRAGMA foreign_keys=ON")
        if resolved not in _INITIALIZED_PATHS:
            conn.executescript(SCHEMA_SQL)
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(projects)")}
            for col in _OPTIONAL_PROJECT_COLUMNS:
                if col not in cols:
                    _add_column_if_missing(conn, "projects", col, f"{col} TEXT")
            _INITIALIZED_PATHS.add(resolved)
    except Exception:
        conn.close()
        raise
    return conn


@contextlib.contextmanager
def connect_closing(db_path: Optional[Path] = None):
    """Open a projects DB connection and close it on exit (sqlite3's own context manager only
    commits/rollbacks, so long-lived gateway/dashboard processes would leak fds otherwise)."""
    conn = connect(db_path=db_path)
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.close()


@dataclass
class ProjectFolder:
    path: str
    label: Optional[str] = None
    is_primary: bool = False
    added_at: int = 0

    def to_dict(self) -> dict:
        return {"path": self.path, "label": self.label, "is_primary": bool(self.is_primary), "added_at": self.added_at}


@dataclass
class Project:
    id: str
    slug: str
    name: str
    created_at: int
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    board_slug: Optional[str] = None
    primary_path: Optional[str] = None
    archived: bool = False
    folders: List[ProjectFolder] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("id", "slug", "name", *_OPTIONAL_ROW_FIELDS)}
        return {**d, "archived": bool(self.archived), "created_at": self.created_at, "folders": [f.to_dict() for f in self.folders]}


def _load_project(conn: sqlite3.Connection, row: sqlite3.Row) -> Project:
    """Materialize a ``projects`` row together with its folders."""
    keys = row.keys()
    folders = conn.execute(
        "SELECT path, label, is_primary, added_at FROM project_folders WHERE project_id = ? ORDER BY is_primary DESC, added_at ASC",
        (row["id"],),
    ).fetchall()
    return Project(
        id=row["id"], slug=row["slug"], name=row["name"], created_at=row["created_at"],
        archived=bool(row["archived"]) if "archived" in keys else False,
        folders=[ProjectFolder(r["path"], r["label"], bool(r["is_primary"]), r["added_at"]) for r in folders],
        **{f: row[f] for f in _OPTIONAL_ROW_FIELDS if f in keys},
    )


def _unique_slug(conn: sqlite3.Connection, candidate: str) -> str:
    """Return ``candidate`` or ``candidate-2``, ``-3`` ... if taken."""
    n, slug = 1, candidate
    while conn.execute("SELECT 1 FROM projects WHERE slug = ?", (slug,)).fetchone() is not None:
        n += 1
        slug = candidate[: 64 - len(f"-{n}")].rstrip("-_") + f"-{n}"
    return slug


def _primary_path_key(path: str) -> str:
    """Comparison key for primary-path dedup (absolute + case/sep-normalized)."""
    return os.path.normcase(_normalize_path(path))


def find_by_primary_path(conn: sqlite3.Connection, path: str, *, include_archived: bool = False) -> Optional[Project]:
    """The first (oldest) project whose primary path matches ``path`` (separator/case normalized so
    equivalent Windows spellings don't slip past the dedup check), else None."""
    key = _primary_path_key(path)
    for proj in list_projects(conn, include_archived=include_archived) if key else ():
        primary = proj.primary_path or next(
            (f.path for f in proj.folders if f.is_primary), proj.folders[0].path if proj.folders else None
        )
        if primary and _primary_path_key(primary) == key:
            return proj
    return None


def create_project(
    conn: sqlite3.Connection, *, name: str, slug: Optional[str] = None, folders: Optional[Iterable[str]] = None,
    primary_path: Optional[str] = None, description: Optional[str] = None, icon: Optional[str] = None,
    color: Optional[str] = None, board_slug: Optional[str] = None, allow_duplicate_path: bool = False,
) -> str:
    """Create a project and return its id. ``folders`` are normalized to absolute paths; ``primary_path``
    is added to the folder set (if absent) and marked primary, else the first folder becomes primary."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("project name must not be empty")
    slug_candidate = normalize_slug(slug) if slug else _slugify(name)
    pid = "p_" + secrets.token_hex(4)
    now = _now()
    folder_paths = list(dict.fromkeys(p for p in map(_normalize_path, folders or []) if p))
    primary = _normalize_path(primary_path) if primary_path else None
    if primary and primary not in folder_paths:
        folder_paths.insert(0, primary)
    if primary is None and folder_paths:
        primary = folder_paths[0]
    existing = find_by_primary_path(conn, primary) if primary and not allow_duplicate_path else None
    if existing is not None:
        raise ValueError(
            f"folder already belongs to project '{existing.slug}' ({existing.id}); "
            "switch to it instead of creating a duplicate"
        )
    with write_txn(conn):
        conn.execute(
            "INSERT INTO projects (id, slug, name, description, icon, color, board_slug,  primary_path, created_at, archived) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (pid, _unique_slug(conn, slug_candidate), name, description, icon, color,
             normalize_slug(board_slug) if board_slug else None, primary, now),
        )
        conn.executemany(
            "INSERT INTO project_folders (project_id, path, label, is_primary, added_at) VALUES (?, ?, ?, ?, ?)",
            [(pid, path, None, 1 if path == primary else 0, now) for path in folder_paths],
        )
    return pid


def list_projects(conn: sqlite3.Connection, *, include_archived: bool = False) -> List[Project]:
    sql = "SELECT * FROM projects" + ("" if include_archived else " WHERE archived = 0") + " ORDER BY created_at ASC"
    return [_load_project(conn, r) for r in conn.execute(sql).fetchall()]


def get_project(conn: sqlite3.Connection, id_or_slug: str) -> Optional[Project]:
    """Look up a project by id first, then by slug."""
    row = (
        conn.execute("SELECT * FROM projects WHERE id = ?", (id_or_slug,)).fetchone()
        or conn.execute("SELECT * FROM projects WHERE slug = ?", (str(id_or_slug).lower(),)).fetchone()
    )
    return None if row is None else _load_project(conn, row)


def update_project(
    conn: sqlite3.Connection, project_id: str, *, name: Optional[str] = None, description: Optional[str] = None,
    icon: Optional[str] = None, color: Optional[str] = None, board_slug: Optional[str] = None,
) -> bool:
    """Patch top-level project fields; only provided (non-None) fields change. ``icon``, ``color`` and
    ``board_slug`` take ``""`` to clear (store NULL) — ``None`` leaves the field untouched."""
    if name is not None:
        name = str(name).strip()
        if not name:
            raise ValueError("project name must not be empty")
    if board_slug is not None:
        board_slug = normalize_slug(board_slug) if board_slug.strip() else ""
    # (column, provided value, stored value) — "" clears icon/color/board_slug to NULL.
    fields = [
        (col, given, stored) for col, given, stored in (
            ("name", name, name), ("description", description, description), ("icon", icon, icon or None),
            ("color", color, color or None), ("board_slug", board_slug, board_slug or None),
        ) if given is not None
    ]
    if not fields:
        return False
    sets = ", ".join(f"{col} = ?" for col, _, _ in fields)
    return _execute_rowcount(conn, f"UPDATE projects SET {sets} WHERE id = ?", [f[2] for f in fields] + [project_id]) > 0


def _execute_rowcount(conn: sqlite3.Connection, sql: str, params) -> int:
    """Run one write statement in its own txn and return the affected row count."""
    with write_txn(conn):
        cur = conn.execute(sql, params)
    return cur.rowcount


def add_folder(conn: sqlite3.Connection, project_id: str, path: str, *, label: Optional[str] = None, is_primary: bool = False) -> str:
    """Add a folder to a project. Returns the normalized path."""
    norm = _normalize_path(path)
    if not norm:
        raise ValueError("folder path must not be empty")
    if get_project(conn, project_id) is None:
        raise ValueError(f"no such project: {project_id}")
    with write_txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO project_folders (project_id, path, label, is_primary, added_at) VALUES (?, ?, ?, 0, ?)",
            (project_id, norm, label, _now()),
        )
        if label is not None:
            conn.execute("UPDATE project_folders SET label = ? WHERE project_id = ? AND path = ?", (label, project_id, norm))
        # An explicit primary, or the first folder of an empty project, becomes primary.
        if is_primary or conn.execute(
            "SELECT 1 FROM project_folders WHERE project_id = ? AND is_primary = 1", (project_id,)
        ).fetchone() is None:
            _set_primary_locked(conn, project_id, norm)
    return norm


def remove_folder(conn: sqlite3.Connection, project_id: str, path: str) -> bool:
    """Remove a folder from a project. Repoints primary if it was primary."""
    norm = _normalize_path(path)
    with write_txn(conn):
        was_primary = conn.execute(
            "SELECT is_primary FROM project_folders WHERE project_id = ? AND path = ?", (project_id, norm)
        ).fetchone()
        cur = conn.execute("DELETE FROM project_folders WHERE project_id = ? AND path = ?", (project_id, norm))
        if was_primary is not None and was_primary["is_primary"]:
            nxt = conn.execute(
                "SELECT path FROM project_folders WHERE project_id = ? ORDER BY added_at ASC LIMIT 1", (project_id,)
            ).fetchone()
            if nxt and nxt["path"]:
                _set_primary_locked(conn, project_id, nxt["path"])
            else:
                conn.execute("UPDATE projects SET primary_path = NULL WHERE id = ?", (project_id,))
    return cur.rowcount > 0


def _set_primary_locked(conn: sqlite3.Connection, project_id: str, path: str) -> None:
    """Set the primary folder (caller already holds a write txn)."""
    conn.execute("UPDATE project_folders SET is_primary = 0 WHERE project_id = ?", (project_id,))
    conn.execute("UPDATE project_folders SET is_primary = 1 WHERE project_id = ? AND path = ?", (project_id, path))
    conn.execute("UPDATE projects SET primary_path = ? WHERE id = ?", (path, project_id))


def set_primary(conn: sqlite3.Connection, project_id: str, path: str) -> bool:
    norm = _normalize_path(path)
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM project_folders WHERE project_id = ? AND path = ?", (project_id, norm)).fetchone() is None:
            return False
        _set_primary_locked(conn, project_id, norm)
    return True


def archive_project(conn: sqlite3.Connection, project_id: str) -> bool:
    return _execute_rowcount(conn, "UPDATE projects SET archived = 1 WHERE id = ?", (project_id,)) > 0


def restore_project(conn: sqlite3.Connection, project_id: str) -> bool:
    return _execute_rowcount(conn, "UPDATE projects SET archived = 0 WHERE id = ?", (project_id,)) > 0


def delete_project(conn: sqlite3.Connection, project_id: str) -> bool:
    """Hard-delete a project and its folders (cascade)."""
    return _execute_rowcount(conn, "DELETE FROM projects WHERE id = ?", (project_id,)) > 0


# --- Active-project pointer + discovery policy (project_meta KV) --------------

def _upsert_meta_locked(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a project_meta row (caller already holds a write txn)."""
    conn.execute(
        "INSERT INTO project_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM project_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_active(conn: sqlite3.Connection, project_id: Optional[str]) -> None:
    """Set (or clear, when ``None``) the active project pointer."""
    with write_txn(conn):
        if project_id is None:
            conn.execute("DELETE FROM project_meta WHERE key = ?", (_ACTIVE_META_KEY,))
        else:
            _upsert_meta_locked(conn, _ACTIVE_META_KEY, project_id)


def get_active_id(conn: sqlite3.Connection) -> Optional[str]:
    return _get_meta(conn, _ACTIVE_META_KEY)


def get_discovery_policy_key(conn: sqlite3.Connection) -> Optional[str]:
    return _get_meta(conn, _DISCOVERY_POLICY_META_KEY)


def _clear_repos_locked(conn: sqlite3.Connection, clear: bool, policy_key: Optional[str]) -> None:
    """Optionally wipe the scan cache, then record the policy key when given (caller holds a write txn)."""
    if clear:
        conn.execute("DELETE FROM discovered_repos")
    if policy_key is not None:
        _upsert_meta_locked(conn, _DISCOVERY_POLICY_META_KEY, policy_key)


def reconcile_discovered_repos_policy(conn: sqlite3.Connection, policy_key: str, *, preserve_unversioned: bool = False) -> bool:
    """Clear cached scan rows when their discovery policy changes; pre-policy rows are retained only
    for the backward-compatible default policy. Returns whether rows were cleared."""
    current = get_discovery_policy_key(conn)
    if current == policy_key:
        return False
    cleared = current is not None or not preserve_unversioned
    with write_txn(conn):
        _clear_repos_locked(conn, cleared, policy_key)
    return cleared


def clear_discovered_repos(conn: sqlite3.Connection, *, policy_key: Optional[str] = None) -> None:
    with write_txn(conn):
        _clear_repos_locked(conn, True, policy_key)


def record_discovered_repos(
    conn: sqlite3.Connection, repos: Iterable[tuple[str, Optional[str]]], *, replace: bool = False,
    policy_key: Optional[str] = None,
) -> int:
    """Persist scanned ``(root, label)`` repo roots (normalized; label falls back to basename) and
    return the row count. ``replace`` = authoritative fresh scan: stale rows are deleted first so old
    eval/worktree noise doesn't live forever."""
    now = _now()
    rows = [
        (norm, label or os.path.basename(norm) or norm, now)
        for norm, label in ((_normalize_path(root), label) for root, label in repos) if norm
    ]
    with write_txn(conn):
        if replace:
            conn.execute("DELETE FROM discovered_repos")
        if rows:
            conn.executemany(
                "INSERT INTO discovered_repos (root, label, last_seen) VALUES (?, ?, ?) "
                "ON CONFLICT(root) DO UPDATE SET label = excluded.label, last_seen = excluded.last_seen",
                rows,
            )
        _clear_repos_locked(conn, False, policy_key)
    return len(rows)


def list_discovered_repos(conn: sqlite3.Connection) -> List[dict]:
    """All cached discovered repo roots, most-recently-seen first."""
    return [dict(r) for r in conn.execute("SELECT root, label, last_seen FROM discovered_repos ORDER BY last_seen DESC").fetchall()]


def project_for_path(conn: sqlite3.Connection, path: str, *, include_archived: bool = False) -> Optional[Project]:
    """Return the project owning ``path``: a folder owns it when equal or an ancestor, and the longest
    folder wins so nested projects resolve to the innermost one."""
    if not str(path or "").strip():
        return None
    target = _normalize_path(path)
    sql = "SELECT pf.project_id AS pid, pf.path AS folder FROM project_folders pf JOIN projects p ON p.id = pf.project_id"
    if not include_archived:
        sql += " WHERE p.archived = 0"

    def owns(folder: str) -> bool:
        stem = folder.rstrip("/\\")
        return target == folder or target.startswith(stem + os.sep) or target.startswith(stem + "/")

    owners = [row for row in conn.execute(sql).fetchall() if owns(row["folder"])]
    return get_project(conn, max(owners, key=lambda r: len(r["folder"]))["pid"]) if owners else None


def branch_name_for(project: Project, task_id: str, *, title: str = "") -> str:
    """Deterministic ``<project-slug>/<task-id>[-<title-slug>]`` branch name for a project-linked kanban
    task (stable and human-meaningful, replacing the random ``wt/<task-id>`` fallback)."""
    base = f"{project.slug or _slugify(project.name)}/{task_id}"
    tslug = _BRANCH_SAFE_RE.sub("-", str(title).strip().lower()).strip("-")[:40].strip("-") if title else ""
    return f"{base}-{tslug}" if tslug else base
