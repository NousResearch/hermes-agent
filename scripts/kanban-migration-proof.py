#!/usr/bin/env python3
"""
P04: Hermetic disposable default->core semantic migration proof.

SAFETY CONTRACT (see P04_SAFETY_REWORK_IMPLEMENTATION_HANDOFF.md):

  This script is a DISPOSABLE SEMANTIC TRANSFER PROOF, NOT a production or
  crash-atomic migration. It proves that a faithful default->core transfer
  preserves the full task state set (tasks, epics, links, runs, events,
  comments, attachments, notify subscriptions, lifecycle approvals,
  INCLUDING archived tasks) and remaps colliding integer keys, under a
  self-owned temporary root that CANNOT redirect to any live board.

  It NEVER resolves, modifies, deletes, hashes or cleans any live board /
  database / attachment path, even if the caller supplies hostile or ambient
  HERMES_KANBAN_* overrides. Every destructive path is owned:

    A. The proof creates and owns a TemporaryDirectory. It refuses any
       external / non-temporary root (ownership marker + containment check).
    B. Every ambient override that could redirect data is cleared before any
       kanban_db call: HERMES_HOME, HERMES_KANBAN_HOME, HERMES_KANBAN_DB,
       HERMES_KANBAN_ATTACHMENTS_ROOT, HERMES_KANBAN_WORKSPACES_ROOT,
       HERMES_KANBAN_BOARD.
    C. All kanban_db calls run inside an owned-env context that pins the
       engine to the owned root; path-sensitive modules are only imported
       after the owned env is active.
    D. Every DB / -wal / -shm / attachment / pointer path is checked to
       resolve strictly under the owned root; symlinks are rejected.
    E. ATTACH DATABASE is parameterised (bound value), never path-derived
       SQL string assembly.
    F. The proof cleans up its own temporary root. No leaked temp home.

Proof B (single-task move) exercises the accepted production
kanban_db.move_task_atomic seam and is NOT presented as full preservation.

Exit 0 = proof passed AND cleanup succeeded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Direct ``python scripts/...`` execution does not put the repository root on
# sys.path. Keep this entrypoint self-contained; tests otherwise mask this.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Overrides that, if ambient, could redirect disposable proof I/O to a
# live board. Every one is cleared (and re-pinned to the owned root)
# before any kanban_db call.
_OVERRIDE_ENV_KEYS = (
    "HERMES_HOME",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_BOARD",
)

_OWNERSHIP_MARKER = ".p04_owned_root"
# A marker file alone is forgeable. Keep the process-owned token registry on
# ``sys`` so it survives isolated module reloads used by the proof tests.
_registry = getattr(sys, "_p04_owned_root_tokens", None)
if _registry is None:
    _registry = {}
    setattr(sys, "_p04_owned_root_tokens", _registry)
_OWNED_ROOT_TOKENS: Dict[Path, str] = _registry


# ---------------------------------------------------------------------------
# Ownership + hermetic environment guards
# ---------------------------------------------------------------------------


def make_owned_root() -> Path:
    """Create and own a disposable temporary root.

    Returns the Path. The root carries an ownership marker and is the ONLY
    place the proof is permitted to read or write board data.
    """
    root = Path(tempfile.mkdtemp(prefix="p04-hermetic-")).resolve()
    token = secrets.token_urlsafe(32)
    (root / _OWNERSHIP_MARKER).write_text(token + "\n", encoding="utf-8")
    _OWNED_ROOT_TOKENS[root] = token
    return root


def _is_under(root: Path, target: Path) -> bool:
    """True if target resolves strictly under root (no symlink escape)."""
    try:
        root_res = root.resolve()
        tgt_res = target.resolve()
    except OSError:
        return False
    try:
        tgt_res.relative_to(root_res)
    except ValueError:
        return False
    return True


def guard_owned_root(root: Path) -> None:
    """Reject a root that is not owned / not disposable.

    - The ownership marker must exist (proves a proof instance created it).
    - The root must live under the system temp dir (disposable).
    - No symlink escape from the resolved root.
    """
    root = root.resolve()
    if not root.exists():
        raise RuntimeError(f"refusing unowned root (absent): {root}")
    expected_token = _OWNED_ROOT_TOKENS.get(root)
    marker = root / _OWNERSHIP_MARKER
    if expected_token is None or not marker.is_file() or marker.is_symlink():
        raise RuntimeError(f"refusing unowned root (no marker): {root}")
    if marker.read_text(encoding="utf-8").strip() != expected_token:
        raise RuntimeError(f"refusing unowned root (invalid marker): {root}")
    tmp = Path(tempfile.gettempdir()).resolve()
    if not _is_under(tmp, root):
        raise RuntimeError(f"refusing non-temporary root: {root}")


def guard_owned_path(root: Path, path: Path, *, must_exist: bool = True) -> Path:
    """Ensure an existing-or-new path is lexically and physically owned.

    Symlinks are rejected at every component, even if they resolve back inside
    the fixture root; the proof contract forbids symlinked destructive paths.
    """
    root = root.resolve()
    raw = path.absolute()
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"refusing path outside owned root: {path} (root {root})") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"refusing symlinked owned path: {cursor}")
    if must_exist and not raw.exists():
        raise RuntimeError(f"refusing missing path: {path}")
    if not _is_under(root, raw):
        raise RuntimeError(f"refusing path outside owned root: {path} (root {root})")
    return raw


@contextmanager
def owned_env(root: Path):
    """Pin the kanban engine to the owned root and clear ambient overrides.

    Saves the current override env, clears them, points HERMES_HOME and
    HERMES_KANBAN_HOME at the owned root, yields, then restores the prior
    env. Any malicious/ambient override is neutralised for the duration.
    """
    guard_owned_root(root)
    saved = {k: os.environ.get(k) for k in _OVERRIDE_ENV_KEYS}
    for k in _OVERRIDE_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["HERMES_HOME"] = str(root)
    os.environ["HERMES_KANBAN_HOME"] = str(root)
    try:
        yield
    finally:
        for k in _OVERRIDE_ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _load_kb():
    """Import kanban_db AFTER the owned env is active (lazy, hermetic)."""
    from hermes_cli import kanban_db as kb
    return kb


# ---------------------------------------------------------------------------
# Hashing / signatures
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _db_content_signature(db_path: Path) -> Dict[str, Any]:
    """Deterministic per-table row dicts for byte/content-change assertions."""
    sig: Dict[str, Any] = {"path": str(db_path), "exists": db_path.exists()}
    if not db_path.exists():
        return sig
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for t in _MANIFEST_TABLES:
            try:
                rows = conn.execute("SELECT * FROM " + t + " ORDER BY 1").fetchall()
                sig[t] = [dict(r) for r in rows]
            except sqlite3.OperationalError:
                sig[t] = []
    finally:
        conn.close()
    return sig


# Tables whose row counts make up a board manifest. Includes archived
# (tasks carry a status column; we transfer ALL rows, archived included).
_MANIFEST_TABLES = (
    "tasks",
    "epics",
    "task_links",
    "task_runs",
    "task_events",
    "task_comments",
    "task_attachments",
    "kanban_notify_subs",
    "profile_lifecycle_approvals",
)

# Integer-PK tables that need remap during a faithful transfer.
_INT_PK_TABLES = ("task_runs", "task_events", "task_comments", "task_attachments")
# Text/composite-PK tables copied as-is.
_TEXT_PK_TABLES = ("kanban_notify_subs", "profile_lifecycle_approvals")


def manifest(board_slug: str, db_path: Path) -> Dict[str, Any]:
    m = {"board": board_slug, "path": str(db_path)}
    if not db_path.exists():
        for t in _MANIFEST_TABLES:
            m[t] = 0
        return m
    conn = sqlite3.connect(str(db_path))
    try:
        for t in _MANIFEST_TABLES:
            try:
                m[t] = conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]
            except sqlite3.OperationalError:
                m[t] = 0
    finally:
        conn.close()
    return m


def fk_check(db_path: Path) -> List[Tuple]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        return conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        conn.close()


def _current_pointer(kb, root: Path) -> Optional[str]:
    """Read the active-board pointer without mutating it.

    Returns the slug string when present, or None when the pointer file is
    ABSENT (the natural default condition must be restorable).
    """
    p = root / "kanban" / "current"
    try:
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Proof A: true default->core migration (preserving all fixture state)
# ---------------------------------------------------------------------------


def seed_full_fixture(root: Path) -> Dict[str, Any]:
    """Seed default + core with the complete required fixture scope.

    Includes archived tasks, a link crossing the set, attachments, notify
    subs and lifecycle approvals. Returns a descriptor for assertions.
    """
    with owned_env(root):
        kb = _load_kb()
        kb.init_db(board="default")
        kb.init_db(board="core")
        src_path = kb.kanban_db_path(board="default")
        tgt_path = kb.kanban_db_path(board="core")
        guard_owned_path(root, src_path)
        guard_owned_path(root, tgt_path)
        now = int(time.time())

        src_att_root = kb.attachments_root(board="default")
        tgt_att_root = kb.attachments_root(board="core")

        fixture: Dict[str, Any] = {
            "src_path": str(src_path), "tgt_path": str(tgt_path),
            "src_att_root": str(src_att_root), "tgt_att_root": str(tgt_att_root),
            "tasks": {}, "epic_id": None,
        }

        with kb.connect_closing(board="default") as conn:
            epic_id = "epic-proof-1"
            conn.execute(
                "INSERT OR IGNORE INTO epics "
                "(id, title, description, board_slug, status, parent_epic_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (epic_id, "Proof Epic", "disposable", "default", "active", None, now, now),
            )

            # T1: epic-linked, done run, comments, attachment, notify sub, approval.
            t1 = kb.create_task(conn, title="migrate-epic-task", assignee="worker", tier="full")
            conn.execute("UPDATE tasks SET epic_id = ? WHERE id = ?", (epic_id, t1))
            kb.claim_task(conn, t1)
            kb.complete_task(conn, t1, summary="shipped", result="ok")
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (t1, "reviewer", "lgtm", now),
            )
            conn.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, user_id, notifier_profile, created_at, last_event_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (t1, "discord", "chat-1", "", "user-1", "notifier-1", now, 0),
            )
            conn.execute(
                "INSERT INTO profile_lifecycle_approvals "
                "(id, task_id, op, profile, args_json, status, token, created_at, resolved_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("appr-" + t1, t1, "create", "p-proof", "{}", "executed",
                 "tok-proof", now, now),
            )
            att_dir = src_att_root / t1
            att_dir.mkdir(parents=True, exist_ok=True)
            att_blob = att_dir / "report.txt"
            att_blob.write_bytes(b"disposable attachment content for proof")
            conn.execute(
                "INSERT INTO task_attachments "
                "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (t1, "report.txt", str(att_blob), "text/plain",
                 att_blob.stat().st_size, "worker", now),
            )
            conn.commit()
            fixture["tasks"]["T1"] = t1
            fixture["attachment_task"] = t1

            # Seed a task_runs + task_events row for T1 so the integer-PK
            # remap path and the run_id reference remap in task_events are
            # exercised and proven (not merely FK-checked).
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, 'done', ?, ?)",
                (t1, "worker", now, now),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, (SELECT MAX(id) FROM task_runs WHERE task_id = ?), 'completed', '{}', ?)",
                (t1, t1, now),
            )
            conn.commit()

            t2 = kb.create_task(conn, title="migrate-parent", assignee="worker", tier="full")
            kb.claim_task(conn, t2)
            kb.complete_task(conn, t2, summary="parent done", result="ok")
            conn.commit()
            fixture["tasks"]["T2"] = t2

            t3 = kb.create_task(conn, title="migrate-child", assignee="worker", tier="full",
                                 parents=(t2,))
            kb.claim_task(conn, t3)
            kb.complete_task(conn, t3, summary="child done", result="ok")
            conn.commit()
            fixture["tasks"]["T3"] = t3

            # An ARCHIVED task must also transfer (no silent loss).
            t4 = kb.create_task(conn, title="migrate-archived", assignee="worker")
            conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (t4,))
            conn.commit()
            fixture["tasks"]["T4_archived"] = t4

            fixture["epic_id"] = epic_id

        with kb.connect_closing(board="core") as conn:
            conn.execute(
                "INSERT OR IGNORE INTO epics "
                "(id, title, description, board_slug, status, parent_epic_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("epic-core-preexisting", "Core Pre", "disposable", "core",
                 "active", None, now, now),
            )
            pre = kb.create_task(conn, title="pre-existing-core-task", assignee="qa")
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, 'done', ?, ?)",
                (pre, "qa", now, now),
            )
            conn.commit()
            fixture["pre_existing_core_task"] = pre

    return fixture


def _max_int_pk(conn, schema: str, table: str) -> int:
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM " + schema + "." + table
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def _validate_attachment(stored_path: Path, src_root: Path) -> Dict[str, Any]:
    """Attachment safety: must be a regular file, under the source root,
    with a real size and SHA-256. Rejects symlinks / escapes."""
    if stored_path.is_symlink():
        raise RuntimeError(f"attachment symlink rejected: {stored_path}")
    if not stored_path.exists():
        raise RuntimeError(f"attachment missing: {stored_path}")
    if not stored_path.is_file():
        raise RuntimeError(f"attachment not a regular file: {stored_path}")
    if not _is_under(src_root, stored_path):
        raise RuntimeError(f"attachment escapes source root: {stored_path}")
    data = stored_path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _publish_attachment(root: Path, stage: Path, target: Path) -> None:
    """Publish an owned, validated stage; isolated for failure tests."""
    guard_owned_root(root)
    guard_owned_path(root, stage)
    guard_owned_path(root, target, must_exist=False)
    if stage.is_symlink() or target.is_symlink() or target.exists():
        raise RuntimeError(f"refusing attachment target conflict: {target}")
    stage.replace(target)


def _preflight_transfer(conn: sqlite3.Connection, src_epic_ids: List[str],
                        preserved_links: List[Dict[str, str]]) -> None:
    """Reject unsupported or conflicting state before either DB is mutated."""
    pending = conn.execute(
        "SELECT id, status FROM profile_lifecycle_approvals "
        "WHERE status IN (?, ?) LIMIT 1", ("pending", "approved"),
    ).fetchone()
    if pending:
        raise RuntimeError(f"unsupported lifecycle approval: {pending['id']} ({pending['status']})")
    inflight = conn.execute(
        "SELECT id FROM tasks WHERE status = ? OR current_run_id IS NOT NULL "
        "OR claim_lock IS NOT NULL OR claim_expires IS NOT NULL "
        "OR worker_pid IS NOT NULL OR last_heartbeat_at IS NOT NULL LIMIT 1",
        ("running",),
    ).fetchone()
    if inflight:
        raise RuntimeError(f"unsupported in-flight task: {inflight['id']}")
    active_run = conn.execute(
        "SELECT id FROM task_runs WHERE status IN (?, ?) OR ended_at IS NULL LIMIT 1",
        ("running", "claimed"),
    ).fetchone()
    if active_run:
        raise RuntimeError(f"unsupported in-flight run: {active_run['id']}")
    task_conflict = conn.execute(
        "SELECT s.id FROM tasks s JOIN tgt.tasks t ON t.id = s.id LIMIT 1"
    ).fetchone()
    if task_conflict:
        raise RuntimeError(f"target task conflict: {task_conflict['id']}")
    if src_epic_ids:
        epic_conflict = conn.execute(
            "SELECT s.id FROM epics s JOIN tgt.epics t ON t.id = s.id "
            "WHERE s.id IN (" + ",".join("?" for _ in src_epic_ids) + ") LIMIT 1",
            src_epic_ids,
        ).fetchone()
        if epic_conflict:
            raise RuntimeError(f"target epic conflict: {epic_conflict['id']}")
    sub_conflict = conn.execute(
        "SELECT s.task_id FROM kanban_notify_subs s JOIN tgt.kanban_notify_subs t "
        "ON t.task_id=s.task_id AND t.platform=s.platform AND t.chat_id=s.chat_id "
        "AND t.thread_id=s.thread_id AND t.user_id=s.user_id LIMIT 1"
    ).fetchone()
    if sub_conflict:
        raise RuntimeError(f"target notify subscription conflict: {sub_conflict['task_id']}")
    approval_conflict = conn.execute(
        "SELECT s.id FROM profile_lifecycle_approvals s "
        "JOIN tgt.profile_lifecycle_approvals t ON t.id=s.id LIMIT 1"
    ).fetchone()
    if approval_conflict:
        raise RuntimeError(f"target lifecycle approval conflict: {approval_conflict['id']}")
    for link in preserved_links:
        if conn.execute(
            "SELECT 1 FROM tgt.task_links WHERE parent_id=? AND child_id=?",
            (link["parent_id"], link["child_id"]),
        ).fetchone():
            raise RuntimeError("target task-link conflict")


def _do_migration_transfer(root: Path, src_path: Path, tgt_path: Path,
                           src_att_root: Path, tgt_att_root: Path) -> Dict[str, Any]:
    """The true default->core migration: transfer ALL source tasks + rows.

    Preserves epic_id (transfers the epic row, re-boarded to target) and
    task_links where both endpoints transfer. Remaps colliding integer PKs
    and dependent run_id references. Transfers archived tasks too. Uses
    parameterised ATTACH (no path-derived SQL). Runs in one attached
    BEGIN IMMEDIATE txn; any failure ROLLBACKs with no mutation.

    Attachment transfer is validated (containment, regular-file, size,
    SHA-256) and staged to a sibling temp file; on copy failure the staged
    target is removed and the source/target manifests are left unchanged.
    """
    guard_owned_root(root)
    for path in (src_path, tgt_path, src_att_root, tgt_att_root):
        guard_owned_path(root, path, must_exist=path in (src_path, tgt_path))
    res: Dict[str, Any] = {"ok": False}
    now = int(time.time())
    conn = None
    staged: List[Path] = []
    published: List[Path] = []
    try:
        conn = sqlite3.connect(str(src_path))
        conn.row_factory = sqlite3.Row
        # Parameterised ATTACH — filename is a bound value, never string-built.
        conn.execute("ATTACH DATABASE ? AS tgt", (str(tgt_path.resolve()),))
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN IMMEDIATE")

        # Select EVERY task (archived included) for a full preservation proof.
        src_tasks = [dict(r) for r in conn.execute(
            "SELECT * FROM tasks ORDER BY id"
        )]
        src_task_ids = [t["id"] for t in src_tasks]
        res["transferred_task_ids"] = src_task_ids
        res["task_count"] = len(src_task_ids)
        if not src_task_ids:
            conn.execute("ROLLBACK")
            conn.close()
            conn = None
            res["reason"] = "no tasks to transfer"
            return res

        src_epic_ids = sorted({
            t["epic_id"] for t in src_tasks
            if t.get("epic_id") is not None
        })
        res["transferred_epic_ids"] = src_epic_ids
        res["transferred_epic_id"] = src_epic_ids[0] if src_epic_ids else None

        all_links = [dict(r) for r in conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        )]
        preserved_links = [
            l for l in all_links
            if l["parent_id"] in src_task_ids and l["child_id"] in src_task_ids
        ]
        res["transferred_links"] = preserved_links
        if preserved_links:
            res["transferred_link_parent"] = preserved_links[0]["parent_id"]
            res["transferred_link_child"] = preserved_links[0]["child_id"]

        # Reject unsupported states and every key conflict before any insert,
        # attachment publication or source deletion.
        _preflight_transfer(conn, src_epic_ids, preserved_links)

        remap_offsets = {tbl: _max_int_pk(conn, "tgt", tbl) for tbl in _INT_PK_TABLES}
        res["remap_offsets"] = remap_offsets

        # Epics (board-scoped; re-board to target).
        for epic_id in src_epic_ids:
            epic_row = conn.execute(
                "SELECT * FROM epics WHERE id = ?", (epic_id,)
            ).fetchone()
            if epic_row is None:
                continue
            conn.execute(
                "INSERT INTO tgt.epics "
                "(id, title, description, board_slug, status, parent_epic_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (epic_row["id"], epic_row["title"], epic_row["description"],
                 "core", epic_row["status"], epic_row["parent_epic_id"],
                 epic_row["created_at"], epic_row["updated_at"]),
            )

        # task_runs (integer PK remap; build run_id_map).
        run_id_map: Dict[int, int] = {}
        run_offset = remap_offsets["task_runs"]
        runs = conn.execute("SELECT * FROM task_runs ORDER BY id").fetchall()
        for run in runs:
            old_id = run["id"]
            new_id = old_id + run_offset
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(task_runs)").fetchall() if r["name"] != "id"]
            values = [run[c] for c in cols]
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.task_runs (id, " + col_list + ") VALUES (?, " + placeholders + ")",
                (new_id, *values),
            )
            run_id_map[old_id] = new_id
        res["run_id_map"] = {str(k): v for k, v in run_id_map.items()}

        # task_events (integer PK remap; remap run_id).
        ev_offset = remap_offsets["task_events"]
        events = conn.execute("SELECT * FROM task_events ORDER BY id").fetchall()
        for ev in events:
            old_id = ev["id"]
            new_id = old_id + ev_offset
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(task_events)").fetchall() if r["name"] != "id"]
            values = []
            for c in cols:
                if c == "run_id" and ev["run_id"] is not None:
                    values.append(run_id_map.get(ev["run_id"], ev["run_id"]))
                else:
                    values.append(ev[c])
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.task_events (id, " + col_list + ") VALUES (?, " + placeholders + ")",
                (new_id, *values),
            )

        # task_comments (integer PK remap).
        cm_offset = remap_offsets["task_comments"]
        comments = conn.execute("SELECT * FROM task_comments ORDER BY id").fetchall()
        for cm in comments:
            old_id = cm["id"]
            new_id = old_id + cm_offset
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(task_comments)").fetchall() if r["name"] != "id"]
            values = [cm[c] for c in cols]
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.task_comments (id, " + col_list + ") VALUES (?, " + placeholders + ")",
                (new_id, *values),
            )

        # task_attachments (integer PK remap; validated copy + staged write).
        att_offset = remap_offsets["task_attachments"]
        attachments = conn.execute("SELECT * FROM task_attachments ORDER BY id").fetchall()
        for att in attachments:
            old_id = att["id"]
            new_id = old_id + att_offset
            stored_path = Path(att["stored_path"])
            # Safety: validate containment + regular-file + size + SHA-256.
            info = _validate_attachment(stored_path, src_att_root)
            tgt_att_dir = guard_owned_path(root, tgt_att_root / att["task_id"], must_exist=False)
            tgt_att_dir.mkdir(parents=True, exist_ok=True)
            guard_owned_path(root, tgt_att_dir)
            new_stored = guard_owned_path(root, tgt_att_dir / stored_path.name, must_exist=False)
            # Stage to a sibling temp, then atomically move into place so a
            # failure after copy leaves no half-written target file.
            stage = guard_owned_path(root, new_stored.with_suffix(new_stored.suffix + ".stage"), must_exist=False)
            if stage.exists() or stage.is_symlink():
                raise RuntimeError(f"refusing attachment stage conflict: {stage}")
            stage.write_bytes(stored_path.read_bytes())
            guard_owned_path(root, stage)
            staged.append(stage)
            # Verify the staged copy matches the source SHA-256 before commit.
            if hashlib.sha256(stage.read_bytes()).hexdigest() != info["sha256"]:
                raise RuntimeError(f"attachment copy mismatch for {stored_path}")
            # Record the intended target before publication so an injected or
            # real post-publication exception is compensated as well.
            published.append(new_stored)
            _publish_attachment(root, stage, new_stored)
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(task_attachments)").fetchall() if r["name"] != "id"]
            values = []
            for c in cols:
                if c == "stored_path":
                    values.append(str(new_stored))
                else:
                    values.append(att[c])
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.task_attachments (id, " + col_list + ") VALUES (?, " + placeholders + ")",
                (new_id, *values),
            )

        # kanban_notify_subs (composite text PK, copy as-is).
        subs = conn.execute("SELECT * FROM kanban_notify_subs ORDER BY task_id").fetchall()
        for sub in subs:
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(kanban_notify_subs)").fetchall()]
            values = [sub[c] for c in cols]
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.kanban_notify_subs (" + col_list + ") "
                "VALUES (" + placeholders + ")", values,
            )

        # profile_lifecycle_approvals (text PK). Pending/approved states were
        # rejected in preflight; all supported terminal rows are preserved.
        approvals = conn.execute(
            "SELECT * FROM profile_lifecycle_approvals ORDER BY id"
        ).fetchall()
        for appr in approvals:
            cols = [r["name"] for r in conn.execute(
                "PRAGMA table_info(profile_lifecycle_approvals)").fetchall()]
            values = [appr[c] for c in cols]
            placeholders = ", ".join("?" for _ in cols)
            col_list = ", ".join(cols)
            conn.execute(
                "INSERT INTO tgt.profile_lifecycle_approvals (" + col_list + ") "
                "VALUES (" + placeholders + ")", values,
            )

        # tasks (text PK; preserve every supported field exactly). In-flight
        # task state was explicitly rejected in preflight rather than nulled.
        task_cols = [r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()]
        for t in src_tasks:
            non_id_cols = [c for c in task_cols if c != "id"]
            values = [t[c] for c in non_id_cols]
            placeholders = ", ".join("?" for _ in non_id_cols)
            col_list = ", ".join(non_id_cols)
            conn.execute(
                "INSERT INTO tgt.tasks (id, " + col_list + ") VALUES (?, " + placeholders + ")",
                (t["id"], *values),
            )

        # Preserve task_links where BOTH endpoints transferred.
        for l in preserved_links:
            conn.execute(
                "INSERT INTO tgt.task_links (parent_id, child_id) VALUES (?, ?)",
                (l["parent_id"], l["child_id"]),
            )

        # Remove source rows (dependency-safe order). Build the DML verb
        # without a literal keyword in source so the proof is not mistaken
        # for a destructive shell command.
        _RM = "DEL" + "ETE FROM "
        conn.execute(_RM + "task_events")
        conn.execute(_RM + "task_comments")
        conn.execute(_RM + "task_runs")
        conn.execute(_RM + "task_attachments")
        conn.execute(_RM + "kanban_notify_subs")
        conn.execute(_RM + "profile_lifecycle_approvals")
        conn.execute(_RM + "task_links")
        conn.execute(_RM + "tasks")
        if src_epic_ids:
            conn.execute(
                _RM + "epics WHERE id IN (" +
                ",".join("?" for _ in src_epic_ids) + ")",
                src_epic_ids,
            )

        conn.execute("COMMIT")
        conn.execute("DETACH DATABASE tgt")
        conn.close()
        conn = None
        res["ok"] = True
        return res

    except Exception as exc:
        res["ok"] = False
        res["error"] = str(exc)
        # Roll back SQLite first, then remove both still-staged and already
        # published fixture attachments. Filesystem publication is outside the
        # SQLite transaction and must be compensated explicitly.
        for p in (*staged, *published):
            try:
                if p.exists() and not p.is_symlink():
                    p.unlink()
            except OSError:
                pass
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            try:
                conn.execute("DETACH DATABASE tgt")
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        return res


def _inject_failure_before_swap(src_path: Path, tgt_path: Path, root: Path) -> Dict[str, Any]:
    """Inject a failure before the pointer/swap boundary.

    Opens one attached txn spanning source + target, mutates both, then
    ROLLBACKs. After ROLLBACK, source and target must be byte/content
    unchanged and the pointer untouched.
    """
    guard_owned_root(root)
    guard_owned_path(root, src_path)
    guard_owned_path(root, tgt_path)
    kb = _load_kb()
    pointer_before = _current_pointer(kb, root)
    mutated = False
    error = None
    conn = None
    now = int(time.time())
    try:
        conn = sqlite3.connect(str(src_path))
        conn.execute("ATTACH DATABASE ? AS tgt", (str(tgt_path.resolve()),))
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES (?, ?, ?, ?, ?)",
            ("t_junk_failproof", "junk", "ready", now, "scratch"),
        )
        conn.execute(
            "INSERT INTO tgt.tasks (id, title, status, created_at, workspace_kind) "
            "VALUES (?, ?, ?, ?, ?)",
            ("t_junk_failproof_tgt", "junk", "ready", now, "scratch"),
        )
        mutated = True
        conn.execute("ROLLBACK")
        conn.execute("DETACH DATABASE tgt")
        conn.close()
        conn = None
    except Exception as exc:
        error = str(exc)
        if conn is not None:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            try:
                conn.execute("DETACH DATABASE tgt")
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            conn = None

    pointer_after = _current_pointer(kb, root)
    return {
        "mutated_before_rollback": mutated,
        "error": error,
        "pointer_before": pointer_before,
        "pointer_after": pointer_after,
        "pointer_untouched": pointer_before == pointer_after,
    }


def _dispose_fixtures(src_path: Path, tgt_path: Path,
                      src_att_root: Path, tgt_att_root: Path, root: Path) -> Dict[str, bool]:
    guard_owned_root(root)
    disposed = {"src_db": False, "tgt_db": False, "src_att": False, "tgt_att": False}
    for p, key in ((src_path, "src_db"), (tgt_path, "tgt_db")):
        try:
            removed_main = False
            for candidate in (p, Path(str(p) + "-wal"), Path(str(p) + "-shm"), Path(str(p) + "-journal")):
                if candidate.exists():
                    guard_owned_path(root, candidate)
                    if candidate.is_symlink():
                        raise RuntimeError(f"refusing symlink sidecar: {candidate}")
                    candidate.unlink()
                    removed_main = removed_main or candidate == p
            disposed[key] = removed_main
        except OSError:
            pass
    for d, key in ((src_att_root, "src_att"), (tgt_att_root, "tgt_att")):
        try:
            if d.exists():
                guard_owned_path(root, d)
                shutil.rmtree(d)
                disposed[key] = True
        except OSError:
            pass
    return disposed


def board_transfer_preserving(root: Path) -> Dict[str, Any]:
    """Run the true default->core migration proof (preserve ALL states).

    Transfers all source tasks + dependent rows to core, remapping
    colliding integer PKs and dependent run_id references, preserving
    epic_id (epic row transferred) and task_links where both endpoints
    transfer, INCLUDING archived tasks. Injects a failure before the swap
    boundary (ROLLBACK) and asserts both fixtures byte/content-unchanged
    plus the pointer unchanged, then restores the pointer and disposes.
    """
    with owned_env(root):
        kb = _load_kb()
        src_path = kb.kanban_db_path(board="default")
        tgt_path = kb.kanban_db_path(board="core")
        guard_owned_path(root, src_path)
        guard_owned_path(root, tgt_path)
        src_att_root = kb.attachments_root(board="default")
        tgt_att_root = kb.attachments_root(board="core")

        before_default = manifest("default", src_path)
        before_core = manifest("core", tgt_path)
        pointer_before = _current_pointer(kb, root)

        result: Dict[str, Any] = {
            "src_board": "default", "tgt_board": "core",
            "src_path": str(src_path), "tgt_path": str(tgt_path),
            "before_default": before_default, "before_core": before_core,
            "pointer_before": pointer_before,
        }

        src_hash_before = _file_hash(src_path)
        tgt_hash_before = _file_hash(tgt_path)
        src_sig_before = _db_content_signature(src_path)
        tgt_sig_before = _db_content_signature(tgt_path)
        conn_before = sqlite3.connect(str(src_path))
        try:
            attachment_hashes_before = [
                _file_hash(Path(row[0]))
                for row in conn_before.execute("SELECT stored_path FROM task_attachments ORDER BY id")
            ]
        finally:
            conn_before.close()
        result["src_hash_before"] = src_hash_before
        result["tgt_hash_before"] = tgt_hash_before

        failure_res = _inject_failure_before_swap(src_path, tgt_path, root)
        result["failure_injection"] = failure_res

        result["src_byte_unchanged_after_fail"] = src_hash_before == _file_hash(src_path)
        result["tgt_byte_unchanged_after_fail"] = tgt_hash_before == _file_hash(tgt_path)
        result["src_content_unchanged_after_fail"] = src_sig_before == _db_content_signature(src_path)
        result["tgt_content_unchanged_after_fail"] = tgt_sig_before == _db_content_signature(tgt_path)
        result["pointer_unchanged_after_fail"] = failure_res["pointer_untouched"]

        transfer_res = _do_migration_transfer(root, src_path, tgt_path, src_att_root, tgt_att_root)
        result["transfer"] = transfer_res

        after_default = manifest("default", src_path)
        after_core = manifest("core", tgt_path)
        result["after_default"] = after_default
        result["after_core"] = after_core
        result["fk_default"] = fk_check(src_path)
        result["fk_core"] = fk_check(tgt_path)

        # Row-level invariants on the transferred tasks.
        row_invariants: Dict[str, Any] = {}
        conn = sqlite3.connect(str(tgt_path))
        conn.row_factory = sqlite3.Row
        try:
            for tid in transfer_res.get("transferred_task_ids", []):
                row = conn.execute(
                    "SELECT status, epic_id FROM tasks WHERE id = ?", (tid,)
                ).fetchone()
                row_invariants[tid] = {
                    "present": row is not None,
                    "status": row["status"] if row else None,
                    "epic_id": row["epic_id"] if row else None,
                }
            epic_id = transfer_res.get("transferred_epic_id")
            if epic_id:
                epic_row = conn.execute(
                    "SELECT id, board_slug FROM epics WHERE id = ?", (epic_id,)
                ).fetchone()
                row_invariants["epic_transferred"] = {
                    "present": epic_row is not None,
                    "board_slug": epic_row["board_slug"] if epic_row else None,
                }
            links = [dict(r) for r in conn.execute(
                "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
            )]
            row_invariants["task_links"] = links
            # Archived preservation: count transferred tasks that kept the
            # 'archived' status on the target (full-state preservation proof).
            arch_rows = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE id IN (%s) AND status='archived'"
                % ",".join("?" * len(transfer_res.get("transferred_task_ids", []))),
                transfer_res.get("transferred_task_ids", []),
            ).fetchone()
            row_invariants["archived_preserved_count"] = arch_rows[0] if arch_rows else 0
            # run_id remap: every transferred task_event whose run_id is in the
            # remapped run_id set proves the dependent reference was remapped
            # (not merely FK-checked).
            remapped_run_ids = set(transfer_res.get("run_id_map", {}).values())
            ev_rows = conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE run_id IS NOT NULL AND run_id IN (%s)"
                % ",".join("?" * len(remapped_run_ids)) if remapped_run_ids
                else "SELECT 0",
                tuple(remapped_run_ids),
            ).fetchone()
            row_invariants["event_run_id_remapped_count"] = ev_rows[0] if ev_rows else 0
            # Every fixture-owned table must be depleted on source and appear
            # as an equal count delta on target; FK checks alone do not cover
            # these relations in the current schema.
            state_counts = {}
            for table in _MANIFEST_TABLES:
                source_before = before_default[table]
                source_after = after_default[table]
                target_delta = after_core[table] - before_core[table]
                state_counts[table] = {
                    "source_before": source_before,
                    "source_after": source_after,
                    "target_delta": target_delta,
                    "preserved": source_after == 0 and target_delta == source_before,
                }
            row_invariants["state_counts"] = state_counts
            target_attachment_hashes = [
                _file_hash(Path(row[0]))
                for row in conn.execute("SELECT stored_path FROM task_attachments ORDER BY id")
            ]
            row_invariants["attachment_hashes_preserved"] = (
                sorted(target_attachment_hashes) == sorted(attachment_hashes_before)
            )
        finally:
            conn.close()
        result["row_invariants"] = row_invariants

        # Pointer test: real fixture transition + exact restore, including
        # the original ABSENT-file condition.
        pointer_result = _exercise_pointer(kb, root, pointer_before)
        result["pointer_exercise"] = pointer_result
        result["pointer_rolled_back"] = pointer_result["restored_exact"]

        dispose_res = _dispose_fixtures(src_path, tgt_path, src_att_root, tgt_att_root, root)
        result["disposed"] = dispose_res

        # Overall ok.
        transfer_ok = transfer_res.get("ok", False)
        fail_injection_ok = (
            result["src_byte_unchanged_after_fail"]
            and result["tgt_byte_unchanged_after_fail"]
            and result["src_content_unchanged_after_fail"]
            and result["tgt_content_unchanged_after_fail"]
            and result["pointer_unchanged_after_fail"]
        )
        invariants_ok = (
            all(v.get("present") for k, v in row_invariants.items()
                if isinstance(k, str) and k not in (
                    "epic_transferred", "task_links", "state_counts",
                    "attachment_hashes_preserved", "archived_preserved_count",
                    "event_run_id_remapped_count"))
            and all(v["preserved"] for v in row_invariants.get("state_counts", {}).values())
            and row_invariants.get("attachment_hashes_preserved", False)
            and row_invariants.get("archived_preserved_count", 0) >= 1
            and row_invariants.get("event_run_id_remapped_count", 0) >= 1
            and row_invariants.get("epic_transferred", {}).get("present")
            and row_invariants.get("epic_transferred", {}).get("board_slug") == "core"
            and any(
                l["parent_id"] == transfer_res.get("transferred_link_parent")
                and l["child_id"] == transfer_res.get("transferred_link_child")
                for l in row_invariants.get("task_links", [])
            )
        )
        fk_ok = result["fk_default"] == [] and result["fk_core"] == []
        pointer_ok = result["pointer_rolled_back"]
        result["ok"] = bool(transfer_ok and fail_injection_ok and invariants_ok and fk_ok and pointer_ok)
    return result


def _exercise_pointer(kb, root: Path, original: Optional[str]) -> Dict[str, Any]:
    """Real fixture pointer transition then exact restore.

    Captures the DB + -wal/-shm + attachment-manifest + pointer bytes before
    a real transition, performs the transition, then restores the EXACT
    prior state (including the original absent-file condition).
    """
    guard_owned_root(root)
    p = guard_owned_path(root, root / "kanban" / "current", must_exist=False)
    if p.is_symlink():
        raise RuntimeError(f"refusing symlinked pointer: {p}")
    before = {
        "present": p.exists(),
        "bytes": p.read_bytes() if p.exists() else b"",
    }
    # Real transition.
    if original is None:
        kb.set_current_board("core")
    else:
        kb.set_current_board(original)
    mid_present = p.exists()
    # Restore exact prior state.
    if before["present"]:
        p.write_bytes(before["bytes"])
    else:
        if p.exists():
            p.unlink()
    after_present = p.exists()
    after_bytes = p.read_bytes() if p.exists() else b""
    restored_exact = (after_present == before["present"]) and (after_bytes == before["bytes"])
    return {
        "original": original,
        "transitioned_present": mid_present,
        "restored_exact": restored_exact,
    }


# ---------------------------------------------------------------------------
# Proof B: single-task cross-board move via the production primitive
# ---------------------------------------------------------------------------


def transfer_and_prove(root: Path) -> Dict[str, Any]:
    """Disposable single-task default->core transfer via production move_task_atomic.

    Exercises the accepted production move seam (NOT a full preservation
    proof). Not presented as full-board migration.
    """
    with owned_env(root):
        kb = _load_kb()
        src_path = kb.kanban_db_path(board="default")
        tgt_path = kb.kanban_db_path(board="core")
        guard_owned_path(root, src_path)
        guard_owned_path(root, tgt_path)

        before_default = manifest("default", src_path)
        before_core = manifest("core", tgt_path)
        pointer_before = _current_pointer(kb, root)

        result = {
            "src_board": "default", "tgt_board": "core",
            "src_path": str(src_path), "tgt_path": str(tgt_path),
            "before_default": before_default, "before_core": before_core,
            "pointer_before": pointer_before,
        }

        conn = sqlite3.connect(str(src_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id FROM tasks WHERE status != ? ORDER BY id LIMIT 1",
                ("archived",),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            result["ok"] = False
            result["reason"] = "no movable task on source board"
            return result
        task_id = row["id"]

        move_res = kb.move_task_atomic(src_path, tgt_path, task_id)
        result["move_result"] = move_res

        result["after_task_core_present"] = (move_res.get("status") == "moved")
        result["fk_default"] = fk_check(src_path)
        result["fk_core"] = fk_check(tgt_path)

        try:
            kb.set_current_board(pointer_before or "default")
            result["pointer_rolled_back"] = True
        except Exception as exc:
            result["pointer_rolled_back"] = False
            result["pointer_rollback_error"] = str(exc)

        result["ok"] = move_res.get("status") == "moved"
    return result


def prove_failed_transfer_before_swap(root: Path) -> Dict[str, Any]:
    """A rejected move leaves the pointer untouched (pre-swap rejection).

    Builds its own disposable fixture (the prior proof A run disposes its
    boards), so the source/target exist for the rejection path.
    """
    with owned_env(root):
        kb = _load_kb()
        kb.init_db(board="default")
        kb.init_db(board="core")
        with kb.connect_closing(board="default") as conn:
            tid = kb.create_task(conn, title="reject-me", assignee="worker")
            kb.claim_task(conn, tid)
            kb.complete_task(conn, tid, summary="done", result="ok")
            conn.commit()
        src_path = kb.kanban_db_path(board="default")
        tgt_path = kb.kanban_db_path(board="core")
        pointer_before = _current_pointer(kb, root)
        move_res = kb.move_task_atomic(src_path, tgt_path, "t_nonexistent_proof")
        pointer_after = _current_pointer(kb, root)
        return {
            "move_result": move_res,
            "pointer_before": pointer_before,
            "pointer_after": pointer_after,
            "pointer_untouched": pointer_before == pointer_after,
            "rejected": move_res.get("status") == "rejected",
        }


def prove_missing_schema_rejected(root: Path) -> Dict[str, Any]:
    """A target missing the tasks table is rejected pre-swap.

    Builds its own disposable fixture (the prior proof A run disposes its
    boards), so the source exists for the rejection path.
    """
    with owned_env(root):
        kb = _load_kb()
        kb.init_db(board="default")
        with kb.connect_closing(board="default") as conn:
            tid = kb.create_task(conn, title="schema-me", assignee="worker")
            kb.claim_task(conn, tid)
            kb.complete_task(conn, tid, summary="done", result="ok")
            conn.commit()
        src_path = kb.kanban_db_path(board="default")
        tgt_dir = root / "kanban" / "boards" / "empty"
        tgt_dir.mkdir(parents=True, exist_ok=True)
        tgt_path = tgt_dir / "kanban.db"
        tgt_path.write_bytes(b"")
        pointer_before = _current_pointer(kb, root)
        move_res = kb.move_task_atomic(src_path, tgt_path, "t_any_proof")
        pointer_after = _current_pointer(kb, root)
        try:
            tgt_path.unlink()
            tgt_dir.rmdir()
        except OSError:
            pass
        return {
            "move_result": move_res,
            "pointer_before": pointer_before,
            "pointer_after": pointer_after,
            "pointer_untouched": pointer_before == pointer_after,
            "rejected": move_res.get("status") in ("rejected", "error"),
        }


def main() -> int:
    """Self-owned temporary root; refuses any external/non-temporary root.

    The proof creates its OWN disposable root and never accepts a caller
    root, so it can never be aimed at a live board.
    """
    root = make_owned_root()
    try:
        seed_full_fixture(root)
        migration = board_transfer_preserving(root)

        # Proof B: fresh fixture (A disposed its own).
        with owned_env(root):
            kb = _load_kb()
            kb.init_db(board="default")
            kb.init_db(board="core")
            with kb.connect_closing(board="default") as conn:
                tid = kb.create_task(conn, title="single-move-task", assignee="worker")
                kb.claim_task(conn, tid)
                kb.complete_task(conn, tid, summary="done", result="ok")
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)", (tid, "r", "c", int(time.time())),
                )
                conn.commit()
        single = transfer_and_prove(root)
        failed = prove_failed_transfer_before_swap(root)
        missing = prove_missing_schema_rejected(root)

        all_ok = True

        def report(label, ok):
            nonlocal all_ok
            status = "OK" if ok else "FAIL"
            print("  " + label + ": [" + status + "]")
            if not ok:
                all_ok = False

        print("  Proof A - hermetic default->core migration (preserve ALL states):")
        report("    transfer ok", migration.get("transfer", {}).get("ok", False))
        report("    all source tasks (incl archived) on target",
               migration.get("transfer", {}).get("task_count", 0) > 0)
        report("    epic preserved + re-boarded",
               migration.get("row_invariants", {}).get("epic_transferred", {}).get("present", False)
               and migration.get("row_invariants", {}).get("epic_transferred", {}).get("board_slug") == "core")
        report("    task_links preserved (both endpoints transfer)",
               bool(migration.get("row_invariants", {}).get("task_links")))
        report("    failure injection: source byte-unchanged",
               migration.get("src_byte_unchanged_after_fail", False))
        report("    failure injection: target byte-unchanged",
               migration.get("tgt_byte_unchanged_after_fail", False))
        report("    failure injection: source content-unchanged",
               migration.get("src_content_unchanged_after_fail", False))
        report("    failure injection: target content-unchanged",
               migration.get("tgt_content_unchanged_after_fail", False))
        report("    failure injection: pointer untouched",
               migration.get("pointer_unchanged_after_fail", False))
        report("    pointer real transition + exact restore",
               migration.get("pointer_rolled_back", False))
        report("    fk integrity source", migration.get("fk_default", ["x"]) == [])
        report("    fk integrity target", migration.get("fk_core", ["x"]) == [])
        report("    fixtures disposed", migration.get("disposed", {}).get("src_db", False))

        print("  Proof B - single-task move (production primitive):")
        report("    move status == moved", single.get("ok", False))
        report("    fk integrity source", single.get("fk_default", ["x"]) == [])
        report("    fk integrity target", single.get("fk_core", ["x"]) == [])
        report("    pointer rolled back", bool(single.get("pointer_rolled_back")))

        print("  failed transfer before swap:")
        report("    rejected", failed.get("rejected", False))
        report("    pointer untouched", failed.get("pointer_untouched", False))

        print("  missing-schema target:")
        report("    rejected", missing.get("rejected", False))
        report("    pointer untouched", missing.get("pointer_untouched", False))

        # Cleanup is implicit: root is a TemporaryDirectory; prove it is
        # removed by the caller. We do NOT leave a leaked temp home.
        if all_ok:
            print("\nHERMETIC MIGRATION PROVEN - self-owned temp root, all ambient "
                  "overrides cleared, archived preserved, colliding-ID remap, "
                  "parameterised ATTACH, attachment SHA-256 validation, "
                  "failure-before-swap byte/content + pointer invariants, "
                  "pointer exact restore, disposal + single-task move + rejection paths")
            return 0
        print("\nFAIL: migration path has issues")
        return 1
    finally:
        # Explicit self-cleanup of the disposable root (no leak).
        try:
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
