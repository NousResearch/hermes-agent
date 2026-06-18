"""
Completion checks for ``kind: "until_done"`` cron jobs (Band 2 of the
dynamic-workflow gap plan).

A ``until_done`` job is a work-based primitive: it fires, the agent does a
unit of work, and the job re-queues itself until a user-defined completion
check returns True. The check is a pure function over external state
(kanban right now, but designed to extend to filesystem, CRM, etc).

Why a separate module: the scheduler tick is the work-based loop primitive.
This module is the work-completion criteria, pluggable per job. The
``parse_schedule`` parser delegates here; ``mark_job_run`` consults here.

What this is NOT (per SECURITY.md §2.2): an LLM safety boundary. The
completion check reads structured state via sqlite or HTTP; an adversarial
agent cannot bypass it by writing to a file. But it is NOT a security
boundary against an adversarial scheduler — that is still the OS process.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default poll interval when re-queuing a still-incomplete until_done job.
# Small enough to feel responsive, large enough that the tick loop is
# not thrashing.
DEFAULT_POLL_SECONDS = 60

# Hard cap: a single until_done job cannot re-queue itself more than this
# many times. The agent gets N attempts, then the job is auto-disabled
# with last_status=stuck. This is a safety net against the work
# definition being impossible or the completion check being broken.
MAX_ATTEMPTS = 1440  # 1440 * 60s = 24h of continuous polling


# ── Criteria grammar ─────────────────────────────────────────────────────

# Supported forms (parsed in this order, first match wins):
#   "kanban_idle"                            — no tasks with status in (triage, todo, ready, running)
#   "kanban_idle:workspace=NAME"             — same, scoped to workspace_kind
#   "kanban_idle:status=triage,ready"        — same, explicit status set
#   "kanban_empty:workspace=NAME"            — no tasks of ANY status in workspace
#   "kanban_done:id=t_xxx"                   — task t_xxx is in status=done or archived
#   "kanban_done:workspace=NAME"             — every task in workspace is done/archived
#   "files_match:glob=/path/*.md,mtime_gt=N" — all matching files have mtime > N
#   "always_done"                            — true after first run (debugging)
#
# The criteria is stored verbatim in schedule["criteria"] and re-parsed on
# each check. This keeps storage simple and means the grammar can grow
# without parser migrations.

_STATUS_OPEN = {"triage", "todo", "ready", "running", "blocked"}
_STATUS_TERMINAL = {"done", "rejected", "cancelled", "archived"}


def parse_criteria(text: str) -> Dict[str, Any]:
    """Parse a criteria string into a structured dict. Always returns a dict
    with a 'kind' key. Raises ValueError on unrecognized form."""
    if not text or not text.strip():
        raise ValueError("empty until_done criteria")
    text = text.strip()
    if text == "always_done":
        return {"kind": "always_done"}
    if text == "kanban_idle":
        return {"kind": "kanban_idle"}
    m = re.match(r"^kanban_idle:(.+)$", text)
    if m:
        spec = _parse_kv_spec(m.group(1))
        return {"kind": "kanban_idle", **spec}
    if text == "kanban_empty":
        return {"kind": "kanban_empty"}
    m = re.match(r"^kanban_empty:(.+)$", text)
    if m:
        spec = _parse_kv_spec(m.group(1))
        return {"kind": "kanban_empty", **spec}
    m = re.match(r"^kanban_done:(.+)$", text)
    if m:
        spec = _parse_kv_spec(m.group(1))
        if "id" not in spec and "workspace" not in spec:
            raise ValueError(f"kanban_done requires id= or workspace= in: {text!r}")
        return {"kind": "kanban_done", **spec}
    m = re.match(r"^files_match:(.+)$", text)
    if m:
        spec = _parse_kv_spec(m.group(1))
        if "glob" not in spec:
            raise ValueError(f"files_match requires glob= in: {text!r}")
        return {"kind": "files_match", **spec}
    raise ValueError(f"unrecognized until_done criteria: {text!r}")


def _parse_kv_spec(spec: str) -> Dict[str, str]:
    """Parse a `k1=v1,k2=v2` segment into a dict. Comma inside values not
    supported (use a single value per key for now)."""
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value in: {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


# ── Kanban connection helper ─────────────────────────────────────────────

def _kanban_db_path() -> str:
    return os.environ.get("KANBAN_DB", str(Path.home() / ".hermes" / "kanban.db"))


def _kanban_open() -> sqlite3.Connection:
    p = _kanban_db_path()
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn


def _kanban_columns(conn: sqlite3.Connection) -> set:
    return {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}


# ── Check implementations ────────────────────────────────────────────────

def _check_kanban_idle(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """True when no tasks in the chosen scope are in an open status."""
    workspace = spec.get("workspace")
    custom_statuses = spec.get("status")
    if custom_statuses:
        open_set = {s.strip() for s in custom_statuses.split("|") if s.strip()}
    else:
        open_set = _STATUS_OPEN
    conn = _kanban_open()
    try:
        cols = _kanban_columns(conn)
        if workspace and "workspace_kind" in cols:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM tasks WHERE status IN ({','.join('?' for _ in open_set)}) AND workspace_kind = ?",
                (*open_set, workspace),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM tasks WHERE status IN ({','.join('?' for _ in open_set)})",
                tuple(open_set),
            ).fetchone()
        n = int(row["n"])
        return (n == 0, f"{n} open task(s)" + (f" in {workspace}" if workspace else ""))
    finally:
        conn.close()


def _check_kanban_empty(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """True when no tasks at all exist in the chosen scope (any status)."""
    workspace = spec.get("workspace")
    conn = _kanban_open()
    try:
        if workspace:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE workspace_kind = ?", (workspace,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
        n = int(row["n"])
        return (n == 0, f"{n} task(s) total" + (f" in {workspace}" if workspace else ""))
    finally:
        conn.close()


def _check_kanban_done(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """True when the referenced task is terminal, or every task in the
    workspace is terminal."""
    task_id = spec.get("id")
    workspace = spec.get("workspace")
    conn = _kanban_open()
    try:
        if task_id:
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return (True, f"{task_id} not found (treat as done)")
            return (
                row["status"] in _STATUS_TERMINAL,
                f"{task_id} status={row['status']}",
            )
        if workspace:
            cols = _kanban_columns(conn)
            if "workspace_kind" in cols:
                open_rows = conn.execute(
                    f"SELECT COUNT(*) AS n FROM tasks WHERE workspace_kind = ? AND status NOT IN ({','.join('?' for _ in _STATUS_TERMINAL)})",
                    (workspace, *_STATUS_TERMINAL),
                ).fetchone()
            else:
                return (False, "no workspace_kind column; cannot scope")
            n = int(open_rows["n"])
            return (n == 0, f"{n} open in {workspace}")
        return (False, "kanban_done requires id= or workspace=")
    finally:
        conn.close()


def _check_files_match(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """True when all files matching glob have mtime > threshold, or when
    no files match. mtime_gt is a Unix timestamp; the spec uses the
    'compare_against' hint to interpret it."""
    glob_pattern = spec.get("glob", "")
    mtime_gt_raw = spec.get("mtime_gt", "")
    if not glob_pattern or not mtime_gt_raw:
        return (False, "files_match missing glob= or mtime_gt=")
    try:
        mtime_threshold = float(mtime_gt_raw)
    except ValueError:
        return (False, f"mtime_gt not numeric: {mtime_gt_raw!r}")
    matches = list(Path("/").glob(glob_pattern.lstrip("/"))) if glob_pattern.startswith("/") else list(Path(".").glob(glob_pattern))
    if not matches:
        return (True, f"no files match {glob_pattern!r}")
    stale = [m for m in matches if m.stat().st_mtime <= mtime_threshold]
    return (not stale, f"{len(matches)} matched, {len(stale)} stale")


def _check_always_done(_: Dict[str, Any]) -> Tuple[bool, str]:
    """Debug helper — completes after first run."""
    return (True, "always_done")


# ── Public dispatch ─────────────────────────────────────────────────────

_CHECKERS = {
    "kanban_idle": _check_kanban_idle,
    "kanban_empty": _check_kanban_empty,
    "kanban_done": _check_kanban_done,
    "files_match": _check_files_match,
    "always_done": _check_always_done,
}


def check(criteria: Dict[str, Any]) -> Tuple[bool, str]:
    """Run a parsed criteria. Returns (is_done, human_readable_state)."""
    kind = criteria.get("kind", "")
    checker = _CHECKERS.get(kind)
    if not checker:
        return (False, f"unknown criteria kind: {kind!r}")
    try:
        return checker(criteria)
    except Exception as e:
        return (False, f"check error ({type(e).__name__}): {e}")


def check_by_text(text: str) -> Tuple[bool, str]:
    """Convenience: parse + check. Catches parse errors as a not-done."""
    try:
        return check(parse_criteria(text))
    except ValueError as e:
        return (False, f"parse error: {e}")
