from __future__ import annotations

import contextlib
import contextvars
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator, Optional


class IntakePolicyError(ValueError):
    """The enrolled board's intake policy cannot be trusted."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXEMPT_CREATION: contextvars.ContextVar[tuple[int, str] | None] = contextvars.ContextVar(
    "kanban_intake_exempt_creation", default=None,
)
_REVIEWING: contextvars.ContextVar[tuple[int, str, str, str] | None] = contextvars.ContextVar(
    "kanban_intake_reviewing", default=None,
)


def _main_path(conn: sqlite3.Connection) -> Optional[Path]:
    try:
        row = next((row for row in conn.execute("PRAGMA database_list") if row[1] == "main"), None)
        return Path(row[2]).resolve() if row and row[2] else None
    except (sqlite3.Error, OSError, TypeError):
        return None


def _board_policy(conn: sqlite3.Connection) -> tuple[bool, Optional[str]]:
    from hermes_cli import kanban_db as kb

    db_path = _main_path(conn)
    if db_path is None:
        return False, None
    candidates = [(kb.DEFAULT_BOARD, kb.kanban_home() / "kanban.db")]
    root = kb.boards_root()
    try:
        candidates.extend(
            (entry.name, entry / "kanban.db") for entry in root.iterdir()
            if entry.is_dir() and entry.name != kb.DEFAULT_BOARD
        )
    except OSError:
        pass
    matches = [slug for slug, path in candidates if path.resolve() == db_path]
    if not matches:
        return False, None
    if len(matches) != 1:
        raise IntakePolicyError("Kanban database maps to multiple board identities")
    slug = matches[0]
    metadata_path = kb.board_metadata_path(slug)
    if not metadata_path.exists():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakePolicyError(f"invalid intake policy metadata for board {slug!r}") from exc
    if not isinstance(metadata, dict):
        raise IntakePolicyError(f"invalid intake policy metadata for board {slug!r}")
    if "intake_profile" not in metadata:
        return False, None
    profile = metadata["intake_profile"]
    if not isinstance(profile, str):
        raise IntakePolicyError("board intake_profile must be a string")
    profile = profile.strip()
    return (bool(profile), profile or None)


@contextlib.contextmanager
def exempt_creation(conn: sqlite3.Connection, *, reason: str) -> Iterator[None]:
    """Mark tasks created by trusted native workflows as explicitly exempt."""
    reason = str(reason).strip()
    if not reason:
        raise ValueError("an exemption reason is required")
    token = _EXEMPT_CREATION.set((id(conn), reason))
    try:
        yield
    finally:
        _EXEMPT_CREATION.reset(token)


@contextlib.contextmanager
def reviewing(
    conn: sqlite3.Connection, task_id: str, *, profile: str, source_fingerprint: str,
) -> Iterator[None]:
    """Scope trusted planner output to one task and a validated source digest."""
    if not conn.in_transaction:
        raise IntakePolicyError("planner review requires an active native write transaction")
    profile = str(profile).strip()
    fingerprint = str(source_fingerprint).strip().lower()
    if not _SHA256_RE.fullmatch(fingerprint):
        raise IntakePolicyError("source_fingerprint must be a SHA256 hex digest")
    row = conn.execute(
        "SELECT intake_state, intake_profile FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None:
        raise IntakePolicyError("review target does not exist")
    if row[0] not in (None, "pending"):
        raise IntakePolicyError("review target is already admitted or exempt")
    _, configured = _board_policy(conn)
    expected = row[1] or configured
    if expected is None or profile != expected or (row[1] and configured and row[1] != configured):
        raise IntakePolicyError("reviewer does not match the persisted board intake profile")
    token = _REVIEWING.set((id(conn), task_id, profile, fingerprint))
    try:
        yield
    finally:
        _REVIEWING.reset(token)


def _review_context(conn: sqlite3.Connection, task_id: str) -> Optional[tuple[str, str]]:
    context = _REVIEWING.get()
    if context is None or context[:2] != (id(conn), task_id):
        return None
    _, _, profile, fingerprint = context
    row = conn.execute("SELECT intake_profile FROM tasks WHERE id = ?", (task_id,)).fetchone()
    persisted = row[0] if row else None
    _, expected = _board_policy(conn)
    if not _SHA256_RE.fullmatch(fingerprint) or profile != (persisted or expected):
        raise IntakePolicyError("active planner review no longer matches board policy")
    return profile, fingerprint


def review(conn: sqlite3.Connection, task_id: str) -> Optional[tuple[str, str]]:
    """Return the validated planner receipt active for this task, if any."""
    return _review_context(conn, task_id)


def creation_admission(conn: sqlite3.Connection) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(state, profile, exemption_reason)`` for one new task."""
    exemption = _EXEMPT_CREATION.get()
    if exemption is not None and exemption[0] == id(conn):
        return "exempt", None, exemption[1]
    enrolled, profile = _board_policy(conn)
    return ("pending", profile, None) if enrolled else (None, None, None)


def creation_status(
    conn: sqlite3.Connection, requested_status: str, resolved_status: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    state, profile, reason = creation_admission(conn)
    status = "triage" if state == "pending" and requested_status != "blocked" else resolved_status
    return status, state, profile, reason


def record_exemption(conn: sqlite3.Connection, task_id: str, reason: Optional[str]) -> None:
    if reason:
        from hermes_cli.kanban_db import _append_event

        _append_event(conn, task_id, "intake_exempt", {"reason": reason})


def is_admitted(conn: sqlite3.Connection, task_id: str) -> bool:
    """Whether a task may enter worker or completion paths."""
    row = conn.execute("SELECT intake_state FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row is None or row[0] in (None, "reviewed", "inherited", "exempt")


def require_admitted(conn: sqlite3.Connection, task_id: str) -> bool:
    """Admission predicate for write paths; unknown IDs retain their old result."""
    return is_admitted(conn, task_id)


def is_pending(conn: sqlite3.Connection, task_id: str) -> bool:
    row = conn.execute("SELECT intake_state FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return bool(row and row[0] == "pending")


def record_review(conn: sqlite3.Connection, task_id: str) -> Optional[tuple[str, str]]:
    """Persist the scoped review receipt inside the caller's graph transaction."""
    receipt = review(conn, task_id)
    if receipt is None:
        return None
    profile, fingerprint = receipt
    row = conn.execute("SELECT intake_state FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None or row[0] not in (None, "pending"):
        raise IntakePolicyError("review target changed before native admission")
    if row[0] == "pending":
        updated = conn.execute(
            "UPDATE tasks SET intake_state = 'reviewed' WHERE id = ? AND intake_state = 'pending'",
            (task_id,),
        )
        if updated.rowcount != 1:
            raise IntakePolicyError("review target changed before native admission")
        conn.execute(
            "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'intake_reviewed', ?, ?)",
            (task_id, json.dumps({"profile": profile, "source_fingerprint": fingerprint}), int(time.time())),
        )
    return receipt


def record_inherited(conn: sqlite3.Connection, task_id: str, profile: str, fingerprint: str) -> None:
    """Record a planner-created child as admitted within its atomic graph write."""
    conn.execute(
        "UPDATE tasks SET intake_state = 'inherited', intake_profile = ? WHERE id = ?",
        (profile, task_id),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'intake_inherited', ?, ?)",
        (task_id, json.dumps({"profile": profile, "source_fingerprint": fingerprint}), int(time.time())),
    )
