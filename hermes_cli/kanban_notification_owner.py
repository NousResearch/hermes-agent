"""Explicit, board-local notification ownership, separate from task provenance.

Owners use the existing subscription/delivery pipeline; no adapter fallback or
second notifier is introduced. Ownership is exclusive and explicitly opt-in.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from typing import Any


_TABLE = "kanban_notification_owner"
_ROUTE_FIELDS = {"platform", "chat_id", "thread_id", "user_id", "user_id_alt",
                 "chat_type", "notifier_profile", "delivery_mode", "delivery_metadata"}


def _validate_target(target: Mapping[str, Any]) -> dict[str, Any]:
    from gateway.config import Platform
    from hermes_cli.kanban_db_notify import _NOTIFY_DELIVERY_MODES

    if not isinstance(target, Mapping) or set(target) - _ROUTE_FIELDS:
        raise ValueError("notification owner must be a route with known delivery fields")
    route = dict(target)
    for field in ("platform", "chat_id", "notifier_profile"):
        value = route.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"notification owner requires {field}")
    if route["platform"] not in {p.value for p in Platform} | {"tui"}:
        raise ValueError("unknown notification owner platform")
    route.setdefault("delivery_mode", "notify+wake")
    if route["delivery_mode"] not in _NOTIFY_DELIVERY_MODES:
        raise ValueError("unknown notification owner delivery_mode")
    if route.get("chat_type", "dm") not in ("dm", "group", "channel"):
        raise ValueError("invalid notification owner chat_type")
    route.setdefault("chat_type", "dm")
    for field in ("thread_id", "user_id", "user_id_alt"):
        if route.get(field) is not None and not isinstance(route[field], str):
            raise ValueError(f"notification owner {field} must be a string")
    metadata = route.get("delivery_metadata")
    if metadata is not None and (not isinstance(metadata, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, (str, int, float, bool))
            for k, v in metadata.items())):
        raise ValueError("notification owner metadata must contain scalar delivery fields")
    return route


def get_notification_owner(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Read the explicit owner of this physical board DB; never infer one."""
    try:
        row = conn.execute(f"SELECT target FROM {_TABLE} WHERE id = 1").fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None  # read-only legacy DB before its first migration
        raise
    return _validate_target(json.loads(row[0])) if row else None


def owner_result_fields(conn: sqlite3.Connection) -> dict[str, Any]:
    owner = get_notification_owner(conn)
    return {"notification_owner": owner} if owner is not None else {}


def _cursor(conn: sqlite3.Connection, task_id: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MAX(COALESCE(last_delivered_event_id, last_event_id)), MAX(last_ping_event_id) "
        "FROM kanban_notify_subs WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row[0] is not None:
        return int(row[0]), int(row[1] or 0)
    high = conn.execute("SELECT COALESCE(MAX(id), 0) FROM task_events WHERE task_id = ?",
                        (task_id,)).fetchone()[0]
    return int(high), int(high)


def _bind(conn: sqlite3.Connection, task_id: str, target: dict[str, Any], cursor: tuple[int, int]) -> None:
    from hermes_cli.kanban_db_notify import _encode_notify_delivery_metadata

    conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
    conn.execute(
        "INSERT INTO kanban_notify_subs "
        "(task_id, platform, chat_id, thread_id, user_id, user_id_alt, chat_type, "
        "notifier_profile, delivery_mode, delivery_metadata, created_at, last_event_id, last_ping_event_id, last_delivered_event_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, target["platform"], target["chat_id"], target.get("thread_id") or "",
         target.get("user_id"), target.get("user_id_alt"), target["chat_type"],
         target["notifier_profile"], target["delivery_mode"],
         _encode_notify_delivery_metadata(target.get("delivery_metadata")), int(time.time()), *cursor, cursor[0]),
    )


def sync_task_notification_owner(conn: sqlite3.Connection, task_id: str) -> bool:
    """Inside the caller's write transaction, override inherited/creator routes."""
    target = get_notification_owner(conn)
    if target is None:
        return False
    existing = conn.execute("SELECT * FROM kanban_notify_subs WHERE task_id = ?", (task_id,)).fetchall()
    if len(existing) == 1 and subscription_matches_owner(dict(existing[0]), target):
        return True
    _bind(conn, task_id, target, _cursor(conn, task_id))
    return True


def subscription_matches_owner(sub: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    from hermes_cli.kanban_db_notify import _decode_notify_delivery_metadata

    for field in ("platform", "chat_id", "notifier_profile", "delivery_mode", "chat_type"):
        if sub.get(field) != target.get(field):
            return False
    for field in ("thread_id", "user_id", "user_id_alt"):
        if (sub.get(field) or "") != (target.get(field) or ""):
            return False
    return _decode_notify_delivery_metadata(sub.get("delivery_metadata")) == _decode_notify_delivery_metadata(target.get("delivery_metadata"))


def set_notification_owner(conn: sqlite3.Connection, target: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Atomically transfer all subscriptions and future task routing.

    Existing cursors are retained (the greatest acknowledged cursor when several
    subscribers existed); tasks without subscribers start at their high-water
    mark. No task provenance, history or global board pointer is rewritten.
    Clearing ownership removes its routes rather than guessing a new recipient.
    """
    from hermes_cli import kanban_db as kb

    route = _validate_target(target) if target is not None else None
    with kb.write_txn(conn):
        current = get_notification_owner(conn)
        if route is not None and current is not None and subscription_matches_owner(current, route):
            return current
        if route is None:
            if get_notification_owner(conn) is not None:
                conn.execute("DELETE FROM kanban_notify_subs")
            conn.execute(f"DELETE FROM {_TABLE}")
        else:
            conn.execute(f"INSERT INTO {_TABLE} (id, target) VALUES (1, ?) "
                         "ON CONFLICT(id) DO UPDATE SET target = excluded.target", (json.dumps(route),))
            for row in conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall():
                _bind(conn, row[0], route, _cursor(conn, row[0]))
    return route


def _validate_local_owner_session(target: dict[str, Any]) -> None:
    """The CLI must not silently accept a typo or another profile's session."""
    if target["platform"] not in ("local", "api_server", "tui"):
        return
    from pathlib import Path
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    from gateway.session_context import get_session_env
    from hermes_state import SessionDB

    profile = target["notifier_profile"]
    canonical = get_profile_dir(profile)
    active = get_session_env("HERMES_SESSION_PROFILE", "")
    home = get_hermes_home() if profile == active else canonical
    path = Path(home) / "state.db"
    state = SessionDB(path, read_only=True)
    try:
        session = state.get_session(target["chat_id"])
    finally:
        state.close()
    if not session or (session.get("profile_name") or profile) != profile:
        raise ValueError("notification owner session does not exist in the selected profile")


def subscription_is_current(conn: sqlite3.Connection, sub: Mapping[str, Any]) -> bool:
    """Fail closed when a collected delivery was superseded by an owner transfer."""
    row = conn.execute(
        "SELECT * FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?",
        (sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or ""),
    ).fetchone()
    return row is not None and subscription_matches_owner(dict(row), sub)


def board_subscription_is_current(board: str | None, sub: Mapping[str, Any]) -> bool:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kb.pin_first_board_resolution(), kbc.connect_closing(board=board) as conn:
        if get_notification_owner(conn) is None:
            # Preserve mutable legacy route fields, but not a removed subscription.
            return conn.execute(
                "SELECT 1 FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?",
                (sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or ""),
            ).fetchone() is not None
        return subscription_is_current(conn, sub)


def command_board_owner(args) -> int:
    """``boards owner SLUG`` reads, sets, or explicitly clears responsibility."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_output import _err, _json_out

    try:
        slug = kb._normalize_board_slug(args.slug)
        if not kb.board_exists(slug):
            raise ValueError(f"board {slug!r} does not exist")
        fields = (args.platform, args.chat_id, args.profile)
        setting = any(v is not None for v in fields)
        extras = (args.thread_id, args.user_id, args.user_id_alt, args.delivery_metadata)
        if any(v is not None for v in extras) and not setting:
            raise ValueError("routing fields require --platform, --chat-id and --profile")
        if args.clear and setting:
            raise ValueError("--clear cannot be combined with a new owner")
        if setting and not all(fields):
            raise ValueError("setting an owner requires --platform, --chat-id and --profile")
        with kbc.connect_closing(board=slug) as conn:
            if setting:
                target = _validate_target(dict(platform=args.platform, chat_id=args.chat_id,
                    notifier_profile=args.profile, thread_id=args.thread_id, chat_type=args.chat_type,
                    user_id=args.user_id, user_id_alt=args.user_id_alt, delivery_mode=args.delivery_mode,
                    delivery_metadata=json.loads(args.delivery_metadata) if args.delivery_metadata else None))
                _validate_local_owner_session(target)
                owner = set_notification_owner(conn, target)
            elif args.clear:
                owner = set_notification_owner(conn, None)
            else:
                owner = get_notification_owner(conn)
        result = {"board": slug, "notification_owner": owner}
        if not _json_out(args, result):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        return _err(f"kanban boards owner: {exc}", 2)
