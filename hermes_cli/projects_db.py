"""Per-profile first-class Project store.

A **Project** is a human-named, multi-folder workspace. Unlike the desktop's
old inferred "workspaces" (derived from each session's ``cwd`` + a git probe)
and unlike kanban's self-generated worktrees, a Project is an explicit,
persisted entity the user creates and names. It anchors:

- **Desktop session grouping** — a session belongs to a project when its
  ``cwd`` lives under one of the project's folders (longest-prefix match).
- **Kanban task worktrees** — a task linked to a project creates its worktree
  under the project's primary repo with a deterministic branch name, instead
  of the random ``wt/<task-id>`` fallback.

Scope: **per-profile**, stored at ``$HERMES_HOME/projects.db`` (resolved via
``get_hermes_home()``), mirroring sessions / config / cron. This deliberately
differs from kanban, whose board DB is root-anchored and shared across
profiles. A Project may *bind* a kanban board (``board_slug``) so the two
systems agree on the repo + branch convention without merging their stores.

The schema is intentionally small and additive: column additions go through
:func:`_add_column_if_missing` so opening an old DB is always safe.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def projects_db_path() -> Path:
    """The per-profile projects DB path (``$HERMES_HOME/projects.db``).

    Profile-aware: ``get_hermes_home()`` already points at the active profile's
    home. Tests pass an explicit ``db_path`` to :func:`connect`.
    """
    return get_hermes_home() / "projects.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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

CREATE TABLE IF NOT EXISTS project_session_assignments (
    session_id   TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    assigned_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_project_session_assignments_project
    ON project_session_assignments(project_id);
"""


# ---------------------------------------------------------------------------
# Slug + id helpers
# ---------------------------------------------------------------------------

# Lowercase alphanumerics, hyphens, underscores; 1-64 chars; no leading
# separator. Strict enough to stop traversal and path separators, loose enough
# for kebab-case names like ``hermes-agent``. Display formatting (spaces,
# emoji, capitalisation) lives in ``name``; the slug is just a stable handle.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _slugify(name: str) -> str:
    """Derive a slug candidate from a human name (best-effort)."""
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-_")
    s = s[:64].strip("-_")
    return s or "project"


def normalize_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    if slug is None:
        return None
    s = str(slug).strip().lower()
    if not s:
        return None
    if not _SLUG_RE.match(s):
        raise ValueError(
            f"invalid project slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with "
            f"'-' or '_'"
        )
    return s


def _new_project_id() -> str:
    return "p_" + secrets.token_hex(4)


def _now() -> int:
    return int(time.time())


def _normalize_path(path: str) -> str:
    """Absolute, user-expanded, separator-normalized path (no trailing sep)."""
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    return p.rstrip("/\\") or p


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the per-profile projects DB.

    WAL with DELETE fallback for network filesystems (shared helper from
    ``hermes_state``). Schema init is idempotent (``CREATE TABLE IF NOT
    EXISTS`` + additive migrations) and cached per-path per-process.
    """
    path = db_path if db_path is not None else projects_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="projects.db")
        conn.execute("PRAGMA foreign_keys=ON")
        if resolved not in _INITIALIZED_PATHS:
            conn.executescript(SCHEMA_SQL)
            _migrate_add_optional_columns(conn)
            _INITIALIZED_PATHS.add(resolved)
    except Exception:
        conn.close()
        raise
    return conn


@contextlib.contextmanager
def connect_closing(db_path: Optional[Path] = None):
    """Open a projects DB connection and guarantee it is closed on exit.

    sqlite3's connection context manager only commits/rollbacks; it does NOT
    close the file descriptor. Long-lived processes (gateway, dashboard) route
    many project operations through ``connect()``; without closing, FDs to
    ``projects.db`` accumulate. Mirrors ``kanban_db.connect_closing``.
    """
    conn = connect(db_path=db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# TEXT columns added to `projects` after v1; re-applied idempotently on every
# open so a legacy DB upgrades in place.
_OPTIONAL_PROJECT_COLUMNS = ("board_slug", "primary_path", "icon", "color")


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after v1 to legacy DBs (safe on every open)."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(projects)")}
    for col in _OPTIONAL_PROJECT_COLUMNS:
        if col not in cols:
            _add_column_if_missing(conn, "projects", col, f"{col} TEXT")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProjectFolder:
    path: str
    label: Optional[str] = None
    is_primary: bool = False
    added_at: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "label": self.label,
            "is_primary": bool(self.is_primary),
            "added_at": self.added_at,
        }


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
        return {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "board_slug": self.board_slug,
            "primary_path": self.primary_path,
            "archived": bool(self.archived),
            "created_at": self.created_at,
            "folders": [f.to_dict() for f in self.folders],
        }


def _project_from_row(row: sqlite3.Row) -> Project:
    keys = row.keys()
    return Project(
        id=row["id"],
        slug=row["slug"],
        name=row["name"],
        created_at=row["created_at"],
        description=row["description"] if "description" in keys else None,
        icon=row["icon"] if "icon" in keys else None,
        color=row["color"] if "color" in keys else None,
        board_slug=row["board_slug"] if "board_slug" in keys else None,
        primary_path=row["primary_path"] if "primary_path" in keys else None,
        archived=bool(row["archived"]) if "archived" in keys else False,
    )


def _load_folders(conn: sqlite3.Connection, project_id: str) -> List[ProjectFolder]:
    rows = conn.execute(
        "SELECT path, label, is_primary, added_at FROM project_folders "
        "WHERE project_id = ? ORDER BY is_primary DESC, added_at ASC",
        (project_id,),
    ).fetchall()
    return [
        ProjectFolder(
            path=r["path"],
            label=r["label"],
            is_primary=bool(r["is_primary"]),
            added_at=r["added_at"],
        )
        for r in rows
    ]


def _attach_folders(conn: sqlite3.Connection, project: Project) -> Project:
    project.folders = _load_folders(conn, project.id)
    return project


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _unique_slug(conn: sqlite3.Connection, candidate: str) -> str:
    """Return ``candidate`` or ``candidate-2``, ``-3`` ... if taken."""
    base = candidate
    n = 1
    slug = base
    while conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (slug,)
    ).fetchone() is not None:
        n += 1
        suffix = f"-{n}"
        slug = (base[: 64 - len(suffix)]).rstrip("-_") + suffix
    return slug


def create_project(
    conn: sqlite3.Connection,
    *,
    name: str,
    slug: Optional[str] = None,
    folders: Optional[Iterable[str]] = None,
    primary_path: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    board_slug: Optional[str] = None,
) -> str:
    """Create a project and return its id.

    ``folders`` are normalized to absolute paths. If ``primary_path`` is given
    it is added to the folder set (if not already present) and marked primary;
    otherwise the first folder becomes primary.
    """
    name = str(name or "").strip()
    if not name:
        raise ValueError("project name must not be empty")

    slug_candidate = normalize_slug(slug) if slug else _slugify(name)
    pid = _new_project_id()
    now = _now()

    folder_paths: List[str] = []
    for f in folders or []:
        norm = _normalize_path(f)
        if norm and norm not in folder_paths:
            folder_paths.append(norm)

    primary = _normalize_path(primary_path) if primary_path else None
    if primary and primary not in folder_paths:
        folder_paths.insert(0, primary)
    if primary is None and folder_paths:
        primary = folder_paths[0]

    with write_txn(conn):
        unique = _unique_slug(conn, slug_candidate)
        conn.execute(
            "INSERT INTO projects "
            "(id, slug, name, description, icon, color, board_slug, "
            " primary_path, created_at, archived) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                pid,
                unique,
                name,
                description,
                icon,
                color,
                normalize_slug(board_slug) if board_slug else None,
                primary,
                now,
            ),
        )
        for path in folder_paths:
            conn.execute(
                "INSERT INTO project_folders "
                "(project_id, path, label, is_primary, added_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, path, None, 1 if path == primary else 0, now),
            )
    return pid


def list_projects(
    conn: sqlite3.Connection, *, include_archived: bool = False
) -> List[Project]:
    sql = "SELECT * FROM projects"
    if not include_archived:
        sql += " WHERE archived = 0"
    sql += " ORDER BY created_at ASC"
    rows = conn.execute(sql).fetchall()
    return [_attach_folders(conn, _project_from_row(r)) for r in rows]


def get_project(
    conn: sqlite3.Connection, id_or_slug: str
) -> Optional[Project]:
    """Look up a project by id first, then by slug."""
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ?", (id_or_slug,)
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM projects WHERE slug = ?", (str(id_or_slug).lower(),)
        ).fetchone()
    if row is None:
        return None
    return _attach_folders(conn, _project_from_row(row))


def update_project(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    board_slug: Optional[str] = None,
) -> bool:
    """Patch top-level project fields. Only provided fields change.

    ``icon``, ``color``, and ``board_slug`` accept an empty string to clear
    (store NULL) — passing ``None`` leaves the field untouched, so callers that
    want to clear must send ``""``.
    """
    sets: List[str] = []
    params: List[object] = []
    if name is not None:
        n = str(name).strip()
        if not n:
            raise ValueError("project name must not be empty")
        sets.append("name = ?")
        params.append(n)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if icon is not None:
        sets.append("icon = ?")
        params.append(icon or None)
    if color is not None:
        sets.append("color = ?")
        params.append(color or None)
    if board_slug is not None:
        sets.append("board_slug = ?")
        params.append(normalize_slug(board_slug) if board_slug.strip() else None)
    if not sets:
        return False
    params.append(project_id)
    with write_txn(conn):
        cur = conn.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", params
        )
    return cur.rowcount > 0


def add_folder(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
    *,
    label: Optional[str] = None,
    is_primary: bool = False,
) -> str:
    """Add a folder to a project. Returns the normalized path.

    When ``is_primary`` is set, the folder becomes the project's primary repo
    (the previous primary is demoted, and ``projects.primary_path`` updates).
    """
    norm = _normalize_path(path)
    if not norm:
        raise ValueError("folder path must not be empty")
    if get_project(conn, project_id) is None:
        raise ValueError(f"no such project: {project_id}")
    now = _now()
    with write_txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO project_folders "
            "(project_id, path, label, is_primary, added_at) "
            "VALUES (?, ?, ?, 0, ?)",
            (project_id, norm, label, now),
        )
        if label is not None:
            conn.execute(
                "UPDATE project_folders SET label = ? "
                "WHERE project_id = ? AND path = ?",
                (label, project_id, norm),
            )
        if is_primary:
            _set_primary_locked(conn, project_id, norm)
        else:
            # First folder of an empty project becomes primary implicitly.
            existing_primary = conn.execute(
                "SELECT 1 FROM project_folders "
                "WHERE project_id = ? AND is_primary = 1",
                (project_id,),
            ).fetchone()
            if existing_primary is None:
                _set_primary_locked(conn, project_id, norm)
    return norm


def remove_folder(conn: sqlite3.Connection, project_id: str, path: str) -> bool:
    """Remove a folder from a project. Repoints primary if it was primary."""
    norm = _normalize_path(path)
    with write_txn(conn):
        was_primary = conn.execute(
            "SELECT is_primary FROM project_folders "
            "WHERE project_id = ? AND path = ?",
            (project_id, norm),
        ).fetchone()
        cur = conn.execute(
            "DELETE FROM project_folders WHERE project_id = ? AND path = ?",
            (project_id, norm),
        )
        if was_primary is not None and was_primary["is_primary"]:
            nxt = conn.execute(
                "SELECT path FROM project_folders WHERE project_id = ? "
                "ORDER BY added_at ASC LIMIT 1",
                (project_id,),
            ).fetchone()
            new_primary = nxt["path"] if nxt else None
            if new_primary:
                _set_primary_locked(conn, project_id, new_primary)
            else:
                conn.execute(
                    "UPDATE projects SET primary_path = NULL WHERE id = ?",
                    (project_id,),
                )
    return cur.rowcount > 0


def _set_primary_locked(
    conn: sqlite3.Connection, project_id: str, path: str
) -> None:
    """Set the primary folder (caller already holds a write txn)."""
    conn.execute(
        "UPDATE project_folders SET is_primary = 0 WHERE project_id = ?",
        (project_id,),
    )
    conn.execute(
        "UPDATE project_folders SET is_primary = 1 "
        "WHERE project_id = ? AND path = ?",
        (project_id, path),
    )
    conn.execute(
        "UPDATE projects SET primary_path = ? WHERE id = ?",
        (path, project_id),
    )


def set_primary(conn: sqlite3.Connection, project_id: str, path: str) -> bool:
    norm = _normalize_path(path)
    with write_txn(conn):
        exists = conn.execute(
            "SELECT 1 FROM project_folders WHERE project_id = ? AND path = ?",
            (project_id, norm),
        ).fetchone()
        if exists is None:
            return False
        _set_primary_locked(conn, project_id, norm)
    return True


def archive_project(conn: sqlite3.Connection, project_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE projects SET archived = 1 WHERE id = ?", (project_id,)
        )
    return cur.rowcount > 0


def restore_project(conn: sqlite3.Connection, project_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE projects SET archived = 0 WHERE id = ?", (project_id,)
        )
    return cur.rowcount > 0


def delete_project(conn: sqlite3.Connection, project_id: str) -> bool:
    """Hard-delete a project and its folders (cascade)."""
    with write_txn(conn):
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cur.rowcount > 0


def assign_session(conn: sqlite3.Connection, project_id: str, session_id: str) -> None:
    """Pin an existing session to a project, overriding cwd-based grouping."""
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id required")
    if get_project(conn, project_id) is None:
        raise ValueError(f"no such project: {project_id}")
    with write_txn(conn):
        conn.execute(
            "INSERT INTO project_session_assignments (session_id, project_id, assigned_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "project_id = excluded.project_id, assigned_at = excluded.assigned_at",
            (sid, project_id, _now()),
        )


def unassign_session(conn: sqlite3.Connection, session_id: str) -> bool:
    """Remove an explicit project assignment so cwd inference applies again."""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM project_session_assignments WHERE session_id = ?", (sid,)
        )
    return cur.rowcount > 0


def list_session_assignments(conn: sqlite3.Connection) -> dict[str, str]:
    """Current explicit session -> project assignments for non-archived projects."""
    rows = conn.execute(
        "SELECT psa.session_id, psa.project_id "
        "FROM project_session_assignments psa "
        "JOIN projects p ON p.id = psa.project_id "
        "WHERE p.archived = 0"
    ).fetchall()
    return {str(r["session_id"]): str(r["project_id"]) for r in rows}


# ---------------------------------------------------------------------------
# Active-project pointer (project_meta KV)
# ---------------------------------------------------------------------------


_ACTIVE_META_KEY = "active_id"


def set_active(conn: sqlite3.Connection, project_id: Optional[str]) -> None:
    """Set (or clear, when ``None``) the active project pointer."""
    with write_txn(conn):
        if project_id is None:
            conn.execute("DELETE FROM project_meta WHERE key = ?", (_ACTIVE_META_KEY,))
        else:
            conn.execute(
                "INSERT INTO project_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_ACTIVE_META_KEY, project_id),
            )


def get_active_id(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM project_meta WHERE key = ?", (_ACTIVE_META_KEY,)
    ).fetchone()
    return row["value"] if row else None


# ---------------------------------------------------------------------------
# Discovered repos (filesystem scan cache)
# ---------------------------------------------------------------------------


def record_discovered_repos(
    conn: sqlite3.Connection,
    repos: Iterable[tuple[str, Optional[str]]],
    *,
    replace: bool = False,
) -> int:
    """Persist scanned git repo roots into the cache.

    ``repos`` is an iterable of ``(root, label)``. Roots are normalized; the
    label falls back to the basename. Returns the number of rows written.

    When ``replace`` is true, this is the authoritative result of a fresh disk
    scan: delete stale rows first so old eval/worktree noise disappears instead
    of living forever in the cache.
    """
    now = _now()
    rows = []
    for root, label in repos:
        norm = _normalize_path(root)
        if not norm:
            continue
        rows.append((norm, (label or os.path.basename(norm) or norm), now))

    with write_txn(conn):
        if replace:
            conn.execute("DELETE FROM discovered_repos")
        if rows:
            conn.executemany(
                "INSERT INTO discovered_repos (root, label, last_seen) VALUES (?, ?, ?) "
                "ON CONFLICT(root) DO UPDATE SET label = excluded.label, "
                "last_seen = excluded.last_seen",
                rows,
            )
    return len(rows)


def list_discovered_repos(conn: sqlite3.Connection) -> List[dict]:
    """All cached discovered repo roots, most-recently-seen first."""
    rows = conn.execute(
        "SELECT root, label, last_seen FROM discovered_repos ORDER BY last_seen DESC"
    ).fetchall()
    return [
        {"root": r["root"], "label": r["label"], "last_seen": r["last_seen"]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Resolution + naming
# ---------------------------------------------------------------------------


def project_for_path(
    conn: sqlite3.Connection, path: str, *, include_archived: bool = False
) -> Optional[Project]:
    """Return the project owning ``path`` (longest-prefix folder match).

    A folder owns ``path`` when ``path`` equals the folder or is nested under
    it. The most specific (longest) folder wins, so nested projects resolve to
    the innermost one.
    """
    if not str(path or "").strip():
        return None
    target = _normalize_path(path)
    sql = (
        "SELECT pf.project_id AS pid, pf.path AS folder "
        "FROM project_folders pf JOIN projects p ON p.id = pf.project_id"
    )
    if not include_archived:
        sql += " WHERE p.archived = 0"
    best_pid: Optional[str] = None
    best_len = -1
    for row in conn.execute(sql).fetchall():
        folder = row["folder"]
        if target == folder or target.startswith(folder.rstrip("/\\") + os.sep) or \
                target.startswith(folder.rstrip("/\\") + "/"):
            if len(folder) > best_len:
                best_len = len(folder)
                best_pid = row["pid"]
    if best_pid is None:
        return None
    return get_project(conn, best_pid)


# Deterministic branch slug: lowercase, separators collapsed, capped.
_BRANCH_SAFE_RE = re.compile(r"[^a-z0-9._-]+")


def branch_name_for(project: Project, task_id: str, *, title: str = "") -> str:
    """Deterministic branch name for a project-linked kanban task.

    Shape: ``<project-slug>/<task-id>`` (optionally ``-<title-slug>``). Stable
    and human-meaningful, replacing the random ``wt/<task-id>`` fallback.
    """
    slug = project.slug or _slugify(project.name)
    base = f"{slug}/{task_id}"
    if title:
        tslug = _BRANCH_SAFE_RE.sub("-", str(title).strip().lower()).strip("-")
        tslug = tslug[:40].strip("-")
        if tslug:
            base = f"{base}-{tslug}"
    return base
