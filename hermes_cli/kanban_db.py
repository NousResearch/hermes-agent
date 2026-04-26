"""SQLite-backed Kanban board for multi-profile collaboration.

The board lives at ``$HERMES_HOME/kanban.db`` (profile-agnostic on purpose:
multiple profiles on the same machine all see the same board, which IS the
coordination primitive).

Schema is intentionally small: tasks, task_links, task_comments,
task_events.  The ``workspace_kind`` field decouples coordination from git
worktrees so that research / ops / digital-twin workloads work alongside
coding workloads.  See ``docs/hermes-kanban-v1-spec.pdf`` for the full
design specification.

Concurrency strategy: WAL mode + ``BEGIN IMMEDIATE`` for write
transactions + compare-and-swap (CAS) updates on ``tasks.status`` and
``tasks.claim_lock``.  SQLite serializes writers via its WAL lock, so at
most one claimer can win any given task.  Losers observe zero affected
rows and move on -- no retry loops, no distributed-lock machinery.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"todo", "ready", "running", "blocked", "done", "archived"}
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}

# A running task's claim is valid for 15 minutes; after that the next
# dispatcher tick reclaims it.  Workers that outlive this window should call
# ``heartbeat_claim(task_id)`` periodically.  In practice most kanban
# workloads either finish within 15m or set a longer claim explicitly.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def kanban_db_path() -> Path:
    """Return the path to ``kanban.db`` inside the active HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "kanban.db"


def workspaces_root() -> Path:
    """Return the directory under which ``scratch`` workspaces are created."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "kanban" / "workspaces"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    result: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"] if "tenant" in row.keys() else None,
            result=row["result"] if "result" in row.keys() else None,
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    body           TEXT,
    assignee       TEXT,
    status         TEXT NOT NULL,
    priority       INTEGER DEFAULT 0,
    created_by     TEXT,
    created_at     INTEGER NOT NULL,
    started_at     INTEGER,
    completed_at   INTEGER,
    workspace_kind TEXT NOT NULL DEFAULT 'scratch',
    workspace_path TEXT,
    claim_lock     TEXT,
    claim_expires  INTEGER,
    tenant         TEXT,
    result         TEXT
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_tenant          ON tasks(tenant);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    time but keeps the code robust if the DB file is ever re-created.
    """
    path = db_path or kanban_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> Path:
    """Create the schema if it doesn't exist; return the path used."""
    path = db_path or kanban_db_path()
    with contextlib.closing(connect(path)) as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate_add_optional_columns(conn)
    return path


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after v1 release to legacy DBs.

    Called by ``init_db`` so opening an old DB is always safe.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "tenant" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN tenant TEXT")
    if "result" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN result TEXT")


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.).  A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_task_id() -> str:
    """Generate a short, URL-safe, human-readable task id.

    Format: ``t_<4 hex chars>``.  Space is 65k values; collisions are
    rare but handled by a one-shot retry in ``create_task``.
    """
    return "t_" + secrets.token_hex(2)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Task creation / mutation
# ---------------------------------------------------------------------------

def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
) -> str:
    """Create a new task and optionally link it under parent tasks.

    Returns the new task id.  Status is ``ready`` when there are no
    parents (or all parents already ``done``), otherwise ``todo``.
    """
    if not title or not title.strip():
        raise ValueError("title is required")
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    parents = tuple(p for p in parents if p)

    now = int(time.time())

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            with write_txn(conn):
                # Determine initial status from parent status.
                initial_status = "ready"
                if parents:
                    missing = _find_missing_parents(conn, parents)
                    if missing:
                        raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                    # If any parent is not yet done, we're todo.
                    rows = conn.execute(
                        "SELECT status FROM tasks WHERE id IN "
                        "(" + ",".join("?" * len(parents)) + ")",
                        parents,
                    ).fetchall()
                    if any(r["status"] != "done" for r in rows):
                        initial_status = "todo"

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        tenant
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        assignee,
                        initial_status,
                        priority,
                        created_by,
                        now,
                        workspace_kind,
                        workspace_path,
                        tenant,
                    ),
                )
                for pid in parents:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                        (pid, task_id),
                    )
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": initial_status,
                        "parents": list(parents),
                        "tenant": tenant,
                    },
                )
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    parents = list(parents)
    if not parents:
        return []
    placeholders = ",".join("?" * len(parents))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        parents,
    ).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in parents if p not in present]


def get_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return Task.from_row(row) if row else None


def list_tasks(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    include_archived: bool = False,
    limit: Optional[int] = None,
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(assignee)
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign or reassign a task.  Returns True on success.

    Refuses to reassign a task that's currently running (claim_lock set).
    Reassign after the current run completes if needed.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _append_event(conn, task_id, "assigned", {"assignee": profile})
        return True


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _find_missing_parents(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(
                f"linking {parent_id} -> {child_id} would create a cycle"
            )
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        # If child was ready but parent is not yet done, demote child to todo.
        parent_status = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (parent_id,)
        ).fetchone()["status"]
        if parent_status != "done":
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                (child_id,),
            )
        _append_event(
            conn, child_id, "linked",
            {"parent": parent_id, "child": child_id},
        )


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        return cur.rowcount > 0


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
        (task_id,),
    ).fetchall()
    return [r["child_id"] for r in rows]


def parent_results(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Return ``(parent_id, result)`` for every done parent of ``task_id``."""
    rows = conn.execute(
        """
        SELECT t.id AS id, t.result AS result
        FROM tasks t
        JOIN task_links l ON l.parent_id = t.id
        WHERE l.child_id = ? AND t.status = 'done'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [(r["id"], r["result"]) for r in rows]


# ---------------------------------------------------------------------------
# Comments & events
# ---------------------------------------------------------------------------

def add_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str
) -> int:
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author.strip(), body.strip(), now),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


def list_comments(conn: sqlite3.Connection, task_id: str) -> list[Comment]:
    rows = conn.execute(
        "SELECT * FROM task_comments WHERE task_id = ? ORDER BY created_at ASC",
        (task_id,),
    ).fetchall()
    return [
        Comment(
            id=r["id"],
            task_id=r["task_id"],
            author=r["author"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def list_events(conn: sqlite3.Connection, task_id: str) -> list[Event]:
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(
            Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
            )
        )
    return out


def _append_event(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[dict] = None,
) -> None:
    """Record an event row.  Called from within an already-open txn."""
    now = int(time.time())
    pl = json.dumps(payload, ensure_ascii=False) if payload else None
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, kind, pl, now),
    )


# ---------------------------------------------------------------------------
# Dependency resolution (todo -> ready)
# ---------------------------------------------------------------------------

def recompute_ready(conn: sqlite3.Connection) -> int:
    """Promote ``todo`` tasks to ``ready`` when all parents are ``done``.

    Returns the number of tasks promoted.  Safe to call inside or outside
    an existing transaction; it opens its own IMMEDIATE txn.
    """
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id FROM tasks WHERE status = 'todo'"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if all(p["status"] == "done" for p in parents):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                    (task_id,),
                )
                _append_event(conn, task_id, "ready", None)
                promoted += 1
    return promoted


# ---------------------------------------------------------------------------
# Claim / complete / block
# ---------------------------------------------------------------------------

def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + int(ttl_seconds)
    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'ready'
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        _append_event(conn, task_id, "claimed", {"lock": lock, "expires": expires})
        return get_task(conn, task_id)


def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.
    """
    expires = int(time.time()) + int(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?",
            (expires, task_id, lock),
        )
        return cur.rowcount == 1


def release_stale_claims(conn: sqlite3.Connection) -> int:
    """Reset any ``running`` task whose claim has expired.

    Returns the number of stale claims reclaimed.  Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    with write_txn(conn):
        stale = conn.execute(
            "SELECT id, claim_lock FROM tasks "
            "WHERE status = 'running' AND claim_expires IS NOT NULL AND claim_expires < ?",
            (now,),
        ).fetchall()
        for row in stale:
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL "
                "WHERE id = ? AND status = 'running'",
                (row["id"],),
            )
            _append_event(
                conn, row["id"], "reclaimed",
                {"stale_lock": row["claim_lock"]},
            )
            reclaimed += 1
    return reclaimed


def complete_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
) -> bool:
    """Transition ``running|ready -> done`` and record ``result``.

    Accepts a task that's merely ``ready`` too, so a manual CLI
    completion (``hermes kanban complete <id>``) works without requiring
    a claim/start/complete sequence.
    """
    now = int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status       = 'done',
                   result       = ?,
                   completed_at = ?,
                   claim_lock   = NULL,
                   claim_expires= NULL
             WHERE id = ?
               AND status IN ('running', 'ready', 'blocked')
            """,
            (result, now, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "completed",
            {"result_len": len(result) if result else 0},
        )
    # Recompute ready status for dependents (separate txn so children see done).
    recompute_ready(conn)
    return True


def block_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
) -> bool:
    """Transition ``running -> blocked``."""
    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status       = 'blocked',
                   claim_lock   = NULL,
                   claim_expires= NULL
             WHERE id = ?
               AND status IN ('running', 'ready')
            """,
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        _append_event(conn, task_id, "blocked", {"reason": reason})
        return True


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``blocked -> ready``."""
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'blocked'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        _append_event(conn, task_id, "unblocked", None)
        return True


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived' WHERE id = ? AND status != 'archived'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        _append_event(conn, task_id, "archived", None)
        return True


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def resolve_workspace(task: Task) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    - ``scratch``: a fresh dir under ``$HERMES_HOME/kanban/workspaces/<id>/``.
    - ``dir:<path>``: the path stored in ``workspace_path``.  Created if missing.
    - ``worktree``: a git worktree at ``workspace_path``.  Not created
      automatically in v1 -- the kanban-worker skill documents
      ``git worktree add`` as a worker-side step.  Returns the intended path.

    Persist the resolved path back to the task row via ``set_workspace_path``
    so subsequent runs reuse the same directory.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "scratch":
        if task.workspace_path:
            p = Path(task.workspace_path).expanduser()
        else:
            p = workspaces_root() / task.id
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "dir":
        if not task.workspace_path:
            raise ValueError(
                f"task {task.id} has workspace_kind=dir but no workspace_path"
            )
        p = Path(task.workspace_path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "worktree":
        if not task.workspace_path:
            # Default: .worktrees/<id>/ under CWD.  Worker skill creates it.
            return Path.cwd() / ".worktrees" / task.id
        return Path(task.workspace_path).expanduser()
    raise ValueError(f"unknown workspace_kind: {kind}")


def set_workspace_path(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(path), task_id),
        )


# ---------------------------------------------------------------------------
# Dispatcher (one-shot pass)
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass."""

    reclaimed: int = 0
    promoted: int = 0
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """List of ``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
) -> DispatchResult:
    """Run one dispatcher tick.

    Steps:
      1. Reclaim stale running tasks.
      2. Promote todo -> ready where all parents are done.
      3. For each ready task with an assignee, atomically claim and call
         ``spawn_fn(task, workspace_path)``.

    ``spawn_fn`` defaults to ``_default_spawn`` which invokes
    ``hermes -p <profile> chat -q "..."`` in the background.  Tests pass
    a stub.
    """
    result = DispatchResult()
    result.reclaimed = release_stale_claims(conn)
    result.promoted = recompute_ready(conn)

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    spawned = 0
    for row in ready_rows:
        if max_spawn is not None and spawned >= max_spawn:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        if dry_run:
            result.spawned.append((row["id"], row["assignee"], ""))
            continue
        claimed = claim_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        workspace = resolve_workspace(claimed)
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if spawn_fn is None:
            spawn_fn = _default_spawn
        try:
            spawn_fn(claimed, str(workspace))
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
        except Exception as exc:
            # Spawn failed: release the claim so the next tick can retry.
            with write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL WHERE id = ? AND status = 'running'",
                    (claimed.id,),
                )
                _append_event(
                    conn, claimed.id, "spawn_failed",
                    {"error": str(exc)[:500]},
                )
    return result


def _default_spawn(task: Task, workspace: str) -> None:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    We don't wait for the child; its completion is observed by polling
    the board ``complete``/``block`` transitions that the worker writes.
    """
    import subprocess
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    prompt = f"work kanban task {task.id}"
    env = dict(os.environ)
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace

    cmd = [
        "hermes",
        "-p", task.assignee,
        "chat",
        "-q", prompt,
    ]
    # Use Popen with DEVNULL stdin so the child doesn't inherit our tty.
    # Redirect output to a per-task log under HERMES_HOME/kanban/logs/.
    from hermes_constants import get_hermes_home
    log_dir = get_hermes_home() / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")
    try:
        subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.


# ---------------------------------------------------------------------------
# Worker context builder (what a spawned worker sees)
# ---------------------------------------------------------------------------

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the full text a worker should read to understand its task.

    Order (per design spec §8):
      1. Task title (mandatory).
      2. Task body (optional opening post).
      3. Every comment on the task, chronologically, with authors.
      4. Completion results of every done parent task.
    """
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")

    lines: list[str] = []
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    lines.append("")

    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(task.body.strip())
        lines.append("")

    parents = parent_results(conn, task_id)
    if parents:
        lines.append("## Parent task results")
        for pid, result in parents:
            lines.append(f"### {pid}")
            lines.append((result or "(no result recorded)").strip())
            lines.append("")

    comments = list_comments(conn, task_id)
    if comments:
        lines.append("## Comment thread")
        for c in comments:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.created_at))
            lines.append(f"**{c.author}** ({ts}):")
            lines.append(c.body.strip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
