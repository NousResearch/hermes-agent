"""Durable mobile approval journal for gateway-hosted Group Chats."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MAX_PENDING_APPROVALS = 8
MAX_APPROVAL_TEXT_CHARS = 512
PENDING_APPROVAL_TTL_SECONDS = 24 * 60 * 60
COMMAND_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_PENDING_COMMANDS_PER_ROOM = MAX_PENDING_APPROVALS
MAX_PENDING_COMMANDS_TOTAL = 512
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_APPROVAL_SCOPE_FIELDS = (
    "room_id",
    "authority_gateway_id",
    "authority_epoch",
    "member_id",
    "task_id",
    "execution_generation",
    "request_id",
)


class MessagingApprovalError(ValueError):
    """A messaging approval was malformed, stale, or reused inconsistently."""


class MessagingApprovalTerminalError(RuntimeError):
    """An exact approval can no longer be applied and must not be retried."""


class MessagingApprovalObservationStale(RuntimeError):
    """A worker observation lost its exact room lease before mutation."""


def _identifier(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise MessagingApprovalError(f"invalid {label}")
    return normalized


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:MAX_APPROVAL_TEXT_CHARS]


def _display_text(value: Any, *, limit: int) -> str:
    """Keep untrusted labels readable without letting them impersonate controls."""

    return _text(value)[:limit].translate(
        str.maketrans({
            "@": "＠",
            "\\": "＼",
            "`": "｀",
            "*": "＊",
            "_": "＿",
            "{": "｛",
            "}": "｝",
            "[": "［",
            "]": "］",
            "#": "＃",
            "|": "｜",
            ">": "＞",
            "~": "～",
        })
    )


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (Group Chat approvals)")
    _initialize(conn)
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_pending_approvals (
               room_id TEXT NOT NULL,
               authority_gateway_id TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               member_id TEXT NOT NULL,
               task_id TEXT NOT NULL,
               execution_generation INTEGER NOT NULL,
               request_id TEXT NOT NULL,
               profile TEXT NOT NULL,
               session_id TEXT NOT NULL,
               observer_generation TEXT NOT NULL,
               observer_lease_generation INTEGER NOT NULL,
               description TEXT NOT NULL,
               command_text TEXT NOT NULL,
               updated_at REAL NOT NULL,
               PRIMARY KEY (room_id, member_id)
           )"""
    )
    pending_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(hosted_room_pending_approvals)")
    }
    if not {"observer_generation", "observer_lease_generation"} <= pending_columns:
        try:
            conn.execute("BEGIN IMMEDIATE")
            pending_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(hosted_room_pending_approvals)"
                )
            }
            if "observer_generation" not in pending_columns:
                conn.execute(
                    """ALTER TABLE hosted_room_pending_approvals
                       ADD COLUMN observer_generation TEXT NOT NULL DEFAULT 'legacy'"""
                )
            if "observer_lease_generation" not in pending_columns:
                conn.execute(
                    """ALTER TABLE hosted_room_pending_approvals
                       ADD COLUMN observer_lease_generation INTEGER NOT NULL DEFAULT 0"""
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_messaging_approval_commands (
               command_id TEXT PRIMARY KEY,
               room_id TEXT NOT NULL,
               authority_gateway_id TEXT NOT NULL,
               authority_epoch INTEGER NOT NULL,
               member_id TEXT NOT NULL,
               task_id TEXT NOT NULL,
               execution_generation INTEGER NOT NULL,
               request_id TEXT NOT NULL,
               choice TEXT NOT NULL CHECK (choice IN ('once', 'deny')),
               state TEXT NOT NULL CHECK (state IN ('pending', 'completed')),
               result_text TEXT,
               created_at REAL NOT NULL,
               updated_at REAL NOT NULL,
               UNIQUE (
                   room_id, authority_gateway_id, authority_epoch,
                   member_id, task_id,
                   execution_generation, request_id
               )
           )"""
    )


def _prune_locked(conn: sqlite3.Connection, *, now: float) -> None:
    conn.execute(
        "DELETE FROM hosted_room_pending_approvals WHERE updated_at<?",
        (now - PENDING_APPROVAL_TTL_SECONDS,),
    )
    conn.execute(
        """DELETE FROM hosted_room_messaging_approval_commands
            WHERE (state='pending' AND updated_at<?)
               OR (state='completed' AND updated_at<?)""",
        (
            now - PENDING_APPROVAL_TTL_SECONDS,
            now - COMMAND_RETENTION_SECONDS,
        ),
    )


def _require_observer_lease(
    conn: sqlite3.Connection,
    observation: Mapping[str, Any],
    *,
    now: float,
) -> None:
    observer = str(observation.get("observer_generation") or "legacy")
    if observer == "legacy":
        return
    lease_generation = int(observation.get("observer_lease_generation") or 0)
    if lease_generation < 1:
        raise MessagingApprovalObservationStale(
            "approval observer lease is unavailable"
        )
    try:
        room = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, disbanded_at
                 FROM hosted_rooms WHERE room_id=?""",
            (str(observation.get("room_id") or ""),),
        ).fetchone()
        row = conn.execute(
            """SELECT gateway_id, authority_epoch, process_generation,
                      lease_generation, expires_at, released_at
                 FROM hosted_room_driver_leases WHERE room_id=?""",
            (str(observation.get("room_id") or ""),),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            raise MessagingApprovalObservationStale(
                "approval observer lease is unavailable"
            ) from exc
        raise
    if (
        room is None
        or room["disbanded_at"] is not None
        or str(room["authority_gateway_id"])
        != str(observation.get("authority_gateway_id") or "")
        or int(room["authority_epoch"]) != int(observation.get("authority_epoch") or 0)
        or row is None
        or str(row["gateway_id"]) != str(observation.get("authority_gateway_id") or "")
        or int(row["authority_epoch"]) != int(observation.get("authority_epoch") or 0)
        or str(row["process_generation"]) != observer
        or int(row["lease_generation"]) != lease_generation
        or row["released_at"] is not None
        or float(row["expires_at"]) <= now
    ):
        raise MessagingApprovalObservationStale("approval observer lease changed")


def normalize_pending_approval(
    room_id: Any,
    member_id: Any,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    if action.get("kind") != "approval":
        raise MessagingApprovalError("pending action is not an approval")
    approval = action.get("approval")
    if not isinstance(approval, Mapping):
        raise MessagingApprovalError("pending approval details are unavailable")
    choices = {str(choice or "").casefold() for choice in approval.get("choices") or ()}
    if not {"once", "deny"} <= choices:
        raise MessagingApprovalError("pending approval choices are unsafe")
    generation = int(action.get("execution_generation") or 0)
    if generation < 1:
        raise MessagingApprovalError("pending approval generation is invalid")
    authority_epoch = int(action.get("authority_epoch") or 0)
    if authority_epoch < 1:
        raise MessagingApprovalError("pending approval authority epoch is invalid")
    return {
        "kind": "approval",
        "room_id": _identifier(room_id, label="room_id"),
        "authority_gateway_id": _identifier(
            action.get("authority_gateway_id"),
            label="authority_gateway_id",
        ),
        "authority_epoch": authority_epoch,
        "member_id": _identifier(member_id, label="member_id"),
        "task_id": _identifier(action.get("task_id"), label="task_id"),
        "execution_generation": generation,
        "request_id": _identifier(action.get("request_id"), label="request_id"),
        "profile": (
            _identifier(action.get("profile"), label="profile")
            if action.get("profile")
            else ""
        ),
        "session_id": (
            _identifier(action.get("session_id"), label="session_id")
            if action.get("session_id")
            else ""
        ),
        "observer_generation": _identifier(
            action.get("observer_generation") or "legacy",
            label="observer_generation",
        ),
        "observer_lease_generation": int(action.get("observer_lease_generation") or 0),
        "approval": {
            "description": _text(approval.get("description")),
            "command": _text(approval.get("command")),
            "choices": ["once", "deny"],
        },
    }


def persist_pending_approval(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    pending = normalize_pending_approval(room_id, member_id, action)
    approval = pending["approval"]
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        now = time.time()
        _prune_locked(conn, now=now)
        _require_observer_lease(conn, pending, now=now)
        conn.execute(
            """INSERT INTO hosted_room_pending_approvals(
                   room_id, authority_gateway_id, authority_epoch,
                   member_id, task_id, execution_generation,
                   request_id, profile, session_id, description,
                   observer_generation, observer_lease_generation,
                   command_text, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id, member_id) DO UPDATE SET
                   authority_gateway_id=excluded.authority_gateway_id,
                   authority_epoch=excluded.authority_epoch,
                   task_id=excluded.task_id,
                   execution_generation=excluded.execution_generation,
                   request_id=excluded.request_id,
                   profile=excluded.profile,
                   session_id=excluded.session_id,
                   observer_generation=excluded.observer_generation,
                   observer_lease_generation=excluded.observer_lease_generation,
                   description=excluded.description,
                   command_text=excluded.command_text,
                   updated_at=excluded.updated_at""",
            (
                pending["room_id"],
                pending["authority_gateway_id"],
                pending["authority_epoch"],
                pending["member_id"],
                pending["task_id"],
                pending["execution_generation"],
                pending["request_id"],
                pending["profile"],
                pending["session_id"],
                approval["description"],
                pending["observer_generation"],
                pending["observer_lease_generation"],
                approval["command"],
                now,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return pending


def clear_pending_approval(
    db_path: Path | str,
    *,
    room_id: Any,
    member_id: Any,
    request_id: Any | None = None,
    authority_gateway_id: Any | None = None,
    authority_epoch: Any | None = None,
    observer_generation: Any | None = None,
    observer_lease_generation: Any | None = None,
) -> int:
    room = _identifier(room_id, label="room_id")
    member = _identifier(member_id, label="member_id")
    request = (
        _identifier(request_id, label="request_id") if request_id is not None else ""
    )
    authority_gateway = (
        _identifier(authority_gateway_id, label="authority_gateway_id")
        if authority_gateway_id is not None
        else ""
    )
    epoch = int(authority_epoch or 0)
    if authority_epoch is not None and epoch < 1:
        raise MessagingApprovalError("authority_epoch must be positive")
    observer = str(observer_generation or "")
    observer_lease = int(observer_lease_generation or 0)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if observer:
            _require_observer_lease(
                conn,
                {
                    "room_id": room,
                    "authority_gateway_id": authority_gateway,
                    "authority_epoch": epoch,
                    "observer_generation": observer,
                    "observer_lease_generation": observer_lease,
                },
                now=time.time(),
            )
        changed = conn.execute(
            """DELETE FROM hosted_room_pending_approvals
                WHERE room_id=? AND member_id=?
                  AND (?='' OR request_id=?)
                  AND (?='' OR authority_gateway_id=?)
                  AND (?=0 OR authority_epoch=?)""",
            (
                room,
                member,
                request,
                request,
                authority_gateway,
                authority_gateway,
                epoch,
                epoch,
            ),
        )
        conn.commit()
        return int(changed.rowcount)
    finally:
        conn.close()


def _pending_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "kind": "approval",
        "room_id": str(row["room_id"]),
        "authority_gateway_id": str(row["authority_gateway_id"]),
        "authority_epoch": int(row["authority_epoch"]),
        "member_id": str(row["member_id"]),
        "task_id": str(row["task_id"]),
        "execution_generation": int(row["execution_generation"]),
        "request_id": str(row["request_id"]),
        "profile": str(row["profile"]),
        "session_id": str(row["session_id"]),
        "observer_generation": str(row["observer_generation"]),
        "observer_lease_generation": int(row["observer_lease_generation"]),
        "approval": {
            "description": str(row["description"]),
            "command": str(row["command_text"]),
            "choices": ["once", "deny"],
        },
    }


def list_pending_approvals(
    db_path: Path | str,
    *,
    room_id: Any,
) -> list[dict[str, Any]]:
    room = _identifier(room_id, label="room_id")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM hosted_room_pending_approvals
                WHERE room_id=? AND updated_at>=?
                ORDER BY updated_at, member_id LIMIT ?""",
            (room, time.time() - PENDING_APPROVAL_TTL_SECONDS, MAX_PENDING_APPROVALS),
        ).fetchall()
    finally:
        conn.close()
    return [_pending_from_row(row) for row in rows]


def list_all_pending_approvals(
    db_path: Path | str,
    *,
    limit: int = 512,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM hosted_room_pending_approvals
                WHERE updated_at>=?
                ORDER BY updated_at, room_id, member_id LIMIT ?""",
            (
                time.time() - PENDING_APPROVAL_TTL_SECONDS,
                max(1, min(512, int(limit))),
            ),
        ).fetchall()
    finally:
        conn.close()
    return [_pending_from_row(row) for row in rows]


def begin_approval_command(
    db_path: Path | str,
    *,
    command_id: Any,
    pending: Mapping[str, Any],
    choice: str,
) -> dict[str, Any]:
    command = _identifier(command_id, label="command_id")
    normalized_choice = str(choice or "").casefold()
    if normalized_choice not in {"once", "deny"}:
        raise MessagingApprovalError("approval choice must be once or deny")
    authority_epoch = int(pending.get("authority_epoch") or 0)
    execution_generation = int(pending.get("execution_generation") or 0)
    coordinates = (
        _identifier(pending.get("room_id"), label="room_id"),
        _identifier(
            pending.get("authority_gateway_id"),
            label="authority_gateway_id",
        ),
        authority_epoch,
        _identifier(pending.get("member_id"), label="member_id"),
        _identifier(pending.get("task_id"), label="task_id"),
        execution_generation,
        _identifier(pending.get("request_id"), label="request_id"),
    )
    if authority_epoch < 1:
        raise MessagingApprovalError("approval authority epoch is invalid")
    if execution_generation < 1:
        raise MessagingApprovalError("approval generation is invalid")
    now = time.time()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _prune_locked(conn, now=now)
        existing = conn.execute(
            "SELECT * FROM hosted_room_messaging_approval_commands WHERE command_id=?",
            (command,),
        ).fetchone()
        if existing is not None:
            stored = (
                str(existing["room_id"]),
                str(existing["authority_gateway_id"]),
                int(existing["authority_epoch"]),
                str(existing["member_id"]),
                str(existing["task_id"]),
                int(existing["execution_generation"]),
                str(existing["request_id"]),
                str(existing["choice"]),
            )
            if stored != (*coordinates, normalized_choice):
                raise MessagingApprovalError(
                    "approval command ID was reused with different content"
                )
            conn.commit()
            return {
                "command_id": command,
                "state": str(existing["state"]),
                "result": existing["result_text"],
                "idempotent": True,
                **dict(zip(_APPROVAL_SCOPE_FIELDS, coordinates)),
                "choice": normalized_choice,
            }
        existing_scope = conn.execute(
            """SELECT * FROM hosted_room_messaging_approval_commands
                WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=?
                  AND member_id=? AND task_id=?
                  AND execution_generation=? AND request_id=?""",
            coordinates,
        ).fetchone()
        if existing_scope is not None:
            if str(existing_scope["choice"]) != normalized_choice:
                raise MessagingApprovalError(
                    "A different decision is already queued for this approval."
                )
            conn.commit()
            return {
                "command_id": str(existing_scope["command_id"]),
                "state": str(existing_scope["state"]),
                "result": existing_scope["result_text"],
                "idempotent": True,
                **dict(zip(_APPROVAL_SCOPE_FIELDS, coordinates)),
                "choice": normalized_choice,
            }
        pending_room_count = conn.execute(
            """SELECT COUNT(*) FROM hosted_room_messaging_approval_commands
                WHERE room_id=? AND state='pending'""",
            (coordinates[0],),
        ).fetchone()[0]
        pending_total_count = conn.execute(
            """SELECT COUNT(*) FROM hosted_room_messaging_approval_commands
                WHERE state='pending'""",
        ).fetchone()[0]
        if (
            int(pending_room_count) >= MAX_PENDING_COMMANDS_PER_ROOM
            or int(pending_total_count) >= MAX_PENDING_COMMANDS_TOTAL
        ):
            raise MessagingApprovalError(
                "Too many approval decisions are waiting. Try again later."
            )
        conn.execute(
            """INSERT INTO hosted_room_messaging_approval_commands(
                   command_id, room_id, authority_gateway_id, authority_epoch,
                   member_id, task_id,
                   execution_generation, request_id, choice, state,
                   result_text, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)""",
            (command, *coordinates, normalized_choice, now, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "command_id": command,
        "state": "pending",
        "result": None,
        "idempotent": False,
        **dict(zip(_APPROVAL_SCOPE_FIELDS, coordinates)),
        "choice": normalized_choice,
    }


def list_pending_approval_commands(
    db_path: Path | str,
    *,
    room_id: Any,
) -> list[dict[str, Any]]:
    room = _identifier(room_id, label="room_id")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM hosted_room_messaging_approval_commands
                WHERE room_id=? AND state='pending' AND updated_at>=?
                ORDER BY created_at, command_id LIMIT ?""",
            (
                room,
                time.time() - PENDING_APPROVAL_TTL_SECONDS,
                MAX_PENDING_APPROVALS,
            ),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def list_all_pending_approval_commands(
    db_path: Path | str,
    *,
    limit: int = MAX_PENDING_COMMANDS_TOTAL,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT * FROM hosted_room_messaging_approval_commands
                WHERE state='pending' AND updated_at>=?
                ORDER BY created_at, command_id LIMIT ?""",
            (
                time.time() - PENDING_APPROVAL_TTL_SECONDS,
                max(1, min(MAX_PENDING_COMMANDS_TOTAL, int(limit))),
            ),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def terminalize_unowned_approval_commands(
    db_path: Path | str,
    *,
    local_gateway_id: str,
) -> int:
    """Close decisions that no live local room binding can ever consume."""

    from gateway import hosted_rooms

    completed = 0
    for command in list_all_pending_approval_commands(db_path):
        reason = ""
        try:
            room = hosted_rooms.room_state(db_path, room_id=command["room_id"])
        except (hosted_rooms.RoomNotFoundError, hosted_rooms.RoomQuarantinedError):
            reason = "Approval expired because the Group Chat is no longer available."
        else:
            if (
                str(room["authority_gateway_id"]) != local_gateway_id
                or str(command["authority_gateway_id"])
                != str(room["authority_gateway_id"])
                or int(command["authority_epoch"]) != int(room["authority_epoch"])
            ):
                reason = "Approval expired because Group Chat authority changed."
        if not reason:
            continue
        clear_pending_approval(
            db_path,
            room_id=command["room_id"],
            member_id=command["member_id"],
            request_id=command["request_id"],
            authority_gateway_id=command["authority_gateway_id"],
            authority_epoch=command["authority_epoch"],
        )
        complete_approval_command(
            db_path,
            command_id=command["command_id"],
            result=reason,
        )
        completed += 1
    return completed


def approval_command(
    db_path: Path | str,
    *,
    command_id: Any,
) -> dict[str, Any] | None:
    command = _identifier(command_id, label="command_id")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """SELECT * FROM hosted_room_messaging_approval_commands
                WHERE command_id=?
                  AND ((state='pending' AND updated_at>=?)
                    OR (state='completed' AND updated_at>=?))""",
            (
                command,
                time.time() - PENDING_APPROVAL_TTL_SECONDS,
                time.time() - COMMAND_RETENTION_SECONDS,
            ),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def complete_approval_command(
    db_path: Path | str,
    *,
    command_id: Any,
    result: str,
) -> None:
    command = _identifier(command_id, label="command_id")
    safe_result = _text(result)
    conn = _connect(db_path)
    try:
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        _prune_locked(conn, now=now)
        conn.execute(
            """UPDATE hosted_room_messaging_approval_commands
                  SET state='completed', result_text=?, updated_at=?
                WHERE command_id=? AND state='pending'""",
            (safe_result, now, command),
        )
        conn.commit()
    finally:
        conn.close()


def submit_approval(
    db_path: Path | str,
    *,
    service: Any,
    command_id: Any,
    pending: Mapping[str, Any],
    choice: str,
) -> dict[str, Any]:
    """Freeze a decision, resolving immediately only when this process owns it."""

    plan = begin_approval_command(
        db_path,
        command_id=command_id,
        pending=pending,
        choice=choice,
    )
    if plan["state"] == "completed":
        return {
            **plan,
            "queued": False,
            "applied": plan.get("result") in {"Approved once.", "Denied."},
        }
    if service is not None:
        try:
            service.approve_room_task(
                plan["room_id"],
                member_id=plan["member_id"],
                task_id=plan["task_id"],
                execution_generation=plan["execution_generation"],
                choice=plan["choice"],
                request_id=plan["request_id"],
            )
        except MessagingApprovalTerminalError as exc:
            result = _text(exc) or "Approval is no longer available."
            complete_approval_command(
                db_path,
                command_id=plan["command_id"],
                result=result,
            )
            return {
                **plan,
                "state": "completed",
                "result": result,
                "queued": False,
                "applied": False,
            }
        except Exception:
            return {**plan, "queued": True}
        else:
            result = "Approved once." if plan["choice"] == "once" else "Denied."
            try:
                complete_approval_command(
                    db_path,
                    command_id=plan["command_id"],
                    result=result,
                )
            except Exception:
                return {**plan, "queued": True, "applied": True}
            return {
                **plan,
                "state": "completed",
                "result": result,
                "queued": False,
                "applied": True,
            }
    return {**plan, "queued": True}


def pending_approvals_for_room(
    service: Any,
    room: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if room.get("_room_mode") == "desktop":
        return []
    if room.get("_room_mode") == "remote":
        return []
    else:
        raw_actions = service.status(str(room["room_id"])).get("pending_actions", [])
    actions = raw_actions if isinstance(raw_actions, list) else []
    normalized = [
        normalize_pending_approval(
            room["room_id"],
            action.get("member_id"),
            action,
        )
        for action in actions
        if isinstance(action, Mapping) and action.get("kind") == "approval"
    ]
    authority_gateway_id = str(room.get("authority_gateway_id") or "")
    authority_epoch = int(room.get("authority_epoch") or 0)
    if authority_gateway_id and authority_epoch:
        normalized = [
            action
            for action in normalized
            if action["authority_gateway_id"] == authority_gateway_id
            and action["authority_epoch"] == authority_epoch
        ]
    return normalized[:MAX_PENDING_APPROVALS]


def approval_reference(pending: Mapping[str, Any]) -> str:
    coordinates = "\0".join(str(pending[field]) for field in _APPROVAL_SCOPE_FIELDS)
    return hashlib.sha256(coordinates.encode()).hexdigest()[:8].upper()


def select_pending_approval(
    pending: list[dict[str, Any]],
    selection: str = "",
) -> tuple[int, dict[str, Any]]:
    raw = str(selection or "").strip()
    matches = [
        (index, action)
        for index, action in enumerate(pending, start=1)
        if approval_reference(action).casefold() == raw.casefold()
    ]
    if len(matches) != 1:
        raise MessagingApprovalError(
            "Choose the approval code shown in the Group Chat."
        )
    return matches[0]


def submit_room_approval(
    service: Any,
    room: Mapping[str, Any],
    *,
    command_id: str,
    choice: str,
    installation_owner_authorized: bool,
    selection: str = "",
    expected_request_id: str = "",
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    if installation_owner_authorized is not True:
        raise MessagingApprovalError(
            "Approve or deny Bot commands from the installation owner’s "
            "authorized one-to-one Hermes chat."
        )
    if room.get("_room_mode") == "desktop":
        raise MessagingApprovalError(
            "This Group Chat runs in Desktop. Approve or deny the command there."
        )
    if room.get("_room_mode") == "remote":
        raise MessagingApprovalError(
            "Approvals are available only in an owner chat connected to the "
            "device running this Group Chat."
        )
    existing = approval_command(service.db_path, command_id=command_id)
    if existing is not None:
        result = submit_approval(
            service.db_path,
            service=service.service,
            command_id=command_id,
            pending=existing,
            choice=choice,
        )
        return 0, existing, result
    pending = pending_approvals_for_room(service, room)
    if not pending:
        raise MessagingApprovalError("This Group Chat has no pending approvals.")
    if expected_request_id:
        matches = [
            (candidate_index, action)
            for candidate_index, action in enumerate(pending, start=1)
            if action["request_id"] == expected_request_id
        ]
        if len(matches) != 1:
            raise MessagingApprovalError(
                "That approval changed. Check the Group Chat again."
            )
        index, selected = matches[0]
    else:
        index, selected = select_pending_approval(pending, selection)
    result = submit_approval(
        service.db_path,
        service=service.service,
        command_id=command_id,
        pending=selected,
        choice=choice,
    )
    return index, selected, result


def approval_member_label(room: Mapping[str, Any], member_id: str) -> str:
    for member in room.get("members") or []:
        if not isinstance(member, Mapping):
            continue
        candidate = str(member.get("member_id") or member.get("profile") or "")
        if candidate != member_id:
            continue
        return _display_text(
            member.get("display_name")
            or member.get("handle")
            or member.get("profile")
            or member_id,
            limit=48,
        )
    return _display_text(member_id, limit=48) or "Bot"


def _approval_member_picker_label(room: Mapping[str, Any], member_id: str) -> str:
    label = approval_member_label(room, member_id)
    for member in room.get("members") or []:
        if not isinstance(member, Mapping):
            continue
        candidate = str(member.get("member_id") or member.get("profile") or "")
        if candidate != member_id:
            continue
        handle = _display_text(
            str(member.get("handle") or "").lstrip("@"),
            limit=20,
        )
        if handle:
            suffix = f" · ＠{handle}"
            return f"{label[: max(1, 40 - len(suffix))]}{suffix}"
        break
    suffix = f" · {member_id[:10]}"
    return f"{label[: max(1, 40 - len(suffix))]}{suffix}"


def _approval_display_parts(approval: Mapping[str, Any]) -> tuple[str, str]:
    command = _display_text(
        approval.get("command"),
        limit=MAX_APPROVAL_TEXT_CHARS,
    )
    description = _display_text(
        approval.get("description"),
        limit=160 if command else MAX_APPROVAL_TEXT_CHARS,
    )
    if command == description:
        command = ""
    return description, command


def format_pending_approvals(
    service: Any,
    room: Mapping[str, Any],
    *,
    room_reference: str,
    room_command: str = "/group",
) -> str:
    pending = pending_approvals_for_room(service, room)
    if not pending:
        return ""
    lines = ["⚠️ **Approval needed**"]
    for index, action in enumerate(pending, start=1):
        approval = action["approval"]
        label = approval_member_label(room, str(action["member_id"]))
        description, command = _approval_display_parts(approval)
        detail = description or command or "Command"
        reference = approval_reference(action)
        lines.append(f"{index}. **{label}** · {detail} · `{reference}`")
        if command and command != detail:
            lines.append(f"   Command: {command}")
    lines.extend([
        f"Actions: `{room_command} {room_reference} approvals`",
        f"Approve once: `{room_command} {room_reference} approve <approval code>`",
        f"Deny: `{room_command} {room_reference} deny <approval code>`",
    ])
    return "\n".join(lines)


def format_approval_picker_title(
    room: Mapping[str, Any],
    pending: list[dict[str, Any]],
) -> str:
    lines = ["⚠️ **Approval needed**"]
    for index, action in enumerate(pending, start=1):
        bot = approval_member_label(room, str(action["member_id"]))
        approval = action["approval"]
        description, command = _approval_display_parts(approval)
        lines.append(f"{index}. **{bot}**: {description or command or 'Command'}")
        if command and command != description:
            lines.append(f"   Command: {command}")
    lines.append("Choose **Approve once** or **Deny** below.")
    return "\n".join(lines)


def approval_picker_choices(
    room: Mapping[str, Any],
    pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not 1 <= len(pending) <= 4:
        return []
    choices: list[dict[str, Any]] = []
    for index, action in enumerate(pending, start=1):
        picker_bot = _approval_member_picker_label(
            room,
            str(action["member_id"]),
        )
        coordinates = "\0".join(str(action[field]) for field in _APPROVAL_SCOPE_FIELDS)
        once_token = hashlib.sha256(
            f"{index}\0once\0{coordinates}".encode()
        ).hexdigest()[:20]
        deny_token = hashlib.sha256(
            f"{index}\0deny\0{coordinates}".encode()
        ).hexdigest()[:20]
        choices.extend([
            {
                "value": f"a={index}.o.{once_token}",
                "label": f"✓ {index}. Approve once · {picker_bot}",
                "description": "Approve this command one time",
                "full_width": True,
                "is_current": False,
            },
            {
                "value": f"a={index}.d.{deny_token}",
                "label": f"✕ {index}. Deny · {picker_bot}",
                "description": "Do not run this command",
                "full_width": True,
                "is_current": False,
            },
        ])
    return choices


def resolve_approval_picker_choice(
    room: Mapping[str, Any],
    pending: list[dict[str, Any]],
    value: str,
) -> tuple[int, str, str]:
    choices = approval_picker_choices(room, pending)
    matched = next(
        (choice for choice in choices if choice["value"] == str(value or "")),
        None,
    )
    if matched is None:
        raise MessagingApprovalError(
            "That approval changed. Check the Group Chat again."
        )
    _prefix, encoded = str(matched["value"]).split("=", 1)
    index_text, choice_code, _digest = encoded.split(".", 2)
    index = int(index_text)
    choice = "once" if choice_code == "o" else "deny"
    return index, choice, str(pending[index - 1]["request_id"])
