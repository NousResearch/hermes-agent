"""Canonical, profile-scoped routing for cross-surface sessions.

Routes and exact session bindings live beside Projects in ``projects.db``.  The
module never derives a session identity from a project: Telegram, cron and
webhook runs remain distinct rows and merely share project metadata.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli import projects_db
from hermes_cli.sqlite_util import write_txn

_SUPPORTED_ORIGINS = frozenset({"telegram", "cron", "webhook"})
logger = logging.getLogger(__name__)


class InvalidProjectRouteError(RuntimeError):
    """A matching configured route is invalid and must fail closed."""


@dataclass(frozen=True)
class ProjectRoute:
    origin_kind: str
    origin_key: str
    project_id: str
    cwd: str
    telegram_chat_id: Optional[str] = None
    telegram_thread_id: Optional[str] = None


@dataclass(frozen=True)
class SessionRouteBinding:
    session_id: str
    origin_kind: str
    origin_key: str
    project_id: Optional[str]
    cwd: Optional[str]
    telegram_chat_id: Optional[str]
    telegram_thread_id: Optional[str]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_session_routes (
    origin_kind TEXT NOT NULL,
    origin_key TEXT NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    cwd TEXT NOT NULL,
    telegram_chat_id TEXT,
    telegram_thread_id TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (origin_kind, origin_key)
);
CREATE TABLE IF NOT EXISTS project_session_bindings (
    session_id TEXT PRIMARY KEY,
    origin_kind TEXT NOT NULL,
    origin_key TEXT NOT NULL,
    project_id TEXT,
    cwd TEXT,
    telegram_chat_id TEXT,
    telegram_thread_id TEXT,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_session_bindings_origin
    ON project_session_bindings(origin_kind, origin_key);
CREATE TABLE IF NOT EXISTS project_session_mirror_deliveries (
    session_id TEXT NOT NULL,
    message_key TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('pending', 'sent')),
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (session_id, message_key)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _clean_origin(origin_kind: str, origin_key: str) -> tuple[str, str]:
    kind = str(origin_kind or "").strip().lower()
    key = str(origin_key or "").strip()
    if kind not in _SUPPORTED_ORIGINS:
        raise ValueError(f"unsupported route origin: {origin_kind!r}")
    if not key:
        raise ValueError("route origin key must not be empty")
    return kind, key


def telegram_origin_key(chat_id: object, thread_id: object = None) -> str:
    chat = str(chat_id or "").strip()
    thread = str(thread_id or "").strip()
    if not chat:
        raise ValueError("Telegram chat_id must not be empty")
    return f"{chat}:{thread}"


def _normalize_existing_dir(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise InvalidProjectRouteError("project route has no working directory")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise InvalidProjectRouteError(
            f"project route working directory does not exist: {raw}"
        )
    return str(path.resolve())


def _validated_project_cwd(
    conn: sqlite3.Connection, project_id: str, cwd: object = None
) -> str:
    project = projects_db.get_project(conn, str(project_id or ""))
    if project is None or project.archived:
        raise InvalidProjectRouteError(
            f"project does not exist or is archived: {project_id}"
        )
    candidate = cwd or project.primary_path or (
        project.folders[0].path if project.folders else None
    )
    resolved = _normalize_existing_dir(candidate)
    owner = projects_db.project_for_path(conn, resolved)
    if owner is None or owner.id != project.id:
        raise InvalidProjectRouteError(
            f"working directory does not belong to project {project.id}: {resolved}"
        )
    return resolved


def set_project_route(
    conn: sqlite3.Connection,
    *,
    origin_kind: str,
    origin_key: str,
    project_id: str,
    cwd: object = None,
    telegram_chat_id: object = None,
    telegram_thread_id: object = None,
) -> ProjectRoute:
    """Validate and upsert one exact event route."""
    _ensure_schema(conn)
    kind, key = _clean_origin(origin_kind, origin_key)
    resolved = _validated_project_cwd(conn, project_id, cwd)
    chat = str(telegram_chat_id).strip() if telegram_chat_id not in (None, "") else None
    thread = (
        str(telegram_thread_id).strip()
        if telegram_thread_id not in (None, "")
        else None
    )
    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            """INSERT INTO project_session_routes
               (origin_kind, origin_key, project_id, cwd, telegram_chat_id,
                telegram_thread_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(origin_kind, origin_key) DO UPDATE SET
                 project_id=excluded.project_id, cwd=excluded.cwd,
                 telegram_chat_id=excluded.telegram_chat_id,
                 telegram_thread_id=excluded.telegram_thread_id,
                 updated_at=excluded.updated_at""",
            (kind, key, project_id, resolved, chat, thread, now),
        )
    return ProjectRoute(kind, key, project_id, resolved, chat, thread)


def _telegram_delivery_fields(
    kind: str, key: str, value: dict
) -> tuple[Optional[str], Optional[str]]:
    chat = value.get("telegram_chat_id")
    thread = value.get("telegram_thread_id")
    if kind == "telegram":
        parts = key.split(":", 1)
        if len(parts) != 2 or not parts[0].strip():
            raise InvalidProjectRouteError(
                f"invalid Telegram route key (expected chat_id:thread_id): {key!r}"
            )
        chat = chat or parts[0]
        thread = thread if thread not in (None, "") else (parts[1] or None)
    deliver = str(value.get("deliver") or "").strip()
    if deliver.lower().startswith("telegram:"):
        parts = deliver.split(":", 2)
        if len(parts) < 2 or not parts[1]:
            raise InvalidProjectRouteError(f"invalid Telegram delivery target: {deliver!r}")
        chat = parts[1]
        thread = parts[2] if len(parts) == 3 and parts[2] else None
    extra = value.get("deliver_extra")
    if extra is not None:
        if not isinstance(extra, dict):
            raise InvalidProjectRouteError("deliver_extra must be an object")
        chat = extra.get("chat_id", chat)
        thread = extra.get("thread_id", thread)
    clean_chat = str(chat).strip() if chat not in (None, "") else None
    clean_thread = str(thread).strip() if thread not in (None, "") else None
    return clean_chat, clean_thread


def sync_route_table(conn: sqlite3.Connection, profile_home: Path) -> int:
    """Atomically synchronize a profile-local v2 JSON route table."""
    path = Path(profile_home) / "session_project_routes.json"
    if not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidProjectRouteError(f"cannot read v2 project route table: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 2:
        raise InvalidProjectRouteError("session_project_routes.json must use version 2")

    validated: list[ProjectRoute] = []
    seen: set[tuple[str, str]] = set()
    raw_routes: list[tuple[str, object, dict]] = []
    if "routes" in payload:
        routes = payload.get("routes")
        if not isinstance(routes, list):
            raise InvalidProjectRouteError("v2 routes must be a list")
        for value in routes:
            if not isinstance(value, dict):
                raise InvalidProjectRouteError("each v2 route must be an object")
            origin = value.get("origin")
            if not isinstance(origin, dict):
                raise InvalidProjectRouteError("each v2 route must have an origin object")
            kind = str(origin.get("source") or "").strip().lower()
            if kind == "telegram":
                key = telegram_origin_key(origin.get("chat_id"), origin.get("thread_id"))
            elif kind == "cron":
                key = origin.get("job_id")
            elif kind == "webhook":
                key = origin.get("subscription")
            else:
                raise InvalidProjectRouteError(f"unsupported route origin: {kind!r}")
            raw_routes.append((kind, key, value))
    else:
        # Transitional map shape accepted for early v2 generators.
        for kind in ("telegram", "cron", "webhook"):
            entries = payload.get(kind, {})
            if entries is None:
                entries = {}
            if not isinstance(entries, dict):
                raise InvalidProjectRouteError(f"{kind} routes must be an object")
            raw_routes.extend((kind, key, value) for key, value in entries.items())

    for kind, raw_key, value in raw_routes:
        if not isinstance(value, dict):
            raise InvalidProjectRouteError(f"{kind} route {raw_key!r} must be an object")
        clean_kind, key = _clean_origin(kind, str(raw_key or ""))
        identity = (clean_kind, key)
        if identity in seen:
            raise InvalidProjectRouteError(f"duplicate project route: {kind}:{key}")
        seen.add(identity)
        project_id = str(value.get("project_id") or "").strip()
        if not project_id:
            raise InvalidProjectRouteError(f"{kind} route {key!r} has no project_id")
        cwd = _validated_project_cwd(conn, project_id, value.get("cwd"))
        chat, thread = _telegram_delivery_fields(kind, key, value)
        validated.append(ProjectRoute(kind, key, project_id, cwd, chat, thread))

    _ensure_schema(conn)
    now = int(time.time())
    with write_txn(conn):
        conn.execute("DELETE FROM project_session_routes")
        conn.executemany(
            """INSERT INTO project_session_routes
               (origin_kind, origin_key, project_id, cwd, telegram_chat_id,
                telegram_thread_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    route.origin_kind,
                    route.origin_key,
                    route.project_id,
                    route.cwd,
                    route.telegram_chat_id,
                    route.telegram_thread_id,
                    now,
                )
                for route in validated
            ],
        )
    return len(validated)


def resolve_event_route(
    conn: sqlite3.Connection, origin_kind: str, origin_key: str
) -> Optional[ProjectRoute]:
    """Resolve one exact route; absence is normal, a matching invalid row is not."""
    _ensure_schema(conn)
    kind, key = _clean_origin(origin_kind, origin_key)
    row = conn.execute(
        "SELECT * FROM project_session_routes WHERE origin_kind=? AND origin_key=?",
        (kind, key),
    ).fetchone()
    if row is None:
        return None
    project_id = str(row["project_id"])
    cwd = _validated_project_cwd(conn, project_id, row["cwd"])
    return ProjectRoute(
        kind,
        key,
        project_id,
        cwd,
        row["telegram_chat_id"],
        row["telegram_thread_id"],
    )


def bind_inbound_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin_kind: str,
    origin_key: str,
    telegram_chat_id: object = None,
    telegram_thread_id: object = None,
) -> SessionRouteBinding:
    """Bind an exact session to its route without coalescing by project."""
    _ensure_schema(conn)
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id must not be empty")
    kind, key = _clean_origin(origin_kind, origin_key)
    route = resolve_event_route(conn, kind, key)

    # Telegram-origin sessions always mirror back to the exact inbound peer.
    if kind == "telegram":
        chat = str(telegram_chat_id or "").strip()
        if not chat:
            raise InvalidProjectRouteError(
                "Telegram session cannot be bound without an exact chat_id"
            )
        thread = str(telegram_thread_id or "").strip() or None
    else:
        chat = route.telegram_chat_id if route else None
        thread = route.telegram_thread_id if route else None

    binding = SessionRouteBinding(
        session_id=sid,
        origin_kind=kind,
        origin_key=key,
        project_id=route.project_id if route else None,
        cwd=route.cwd if route else None,
        telegram_chat_id=chat,
        telegram_thread_id=thread,
    )
    with write_txn(conn):
        conn.execute(
            """INSERT INTO project_session_bindings
               (session_id, origin_kind, origin_key, project_id, cwd,
                telegram_chat_id, telegram_thread_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 origin_kind=excluded.origin_kind, origin_key=excluded.origin_key,
                 project_id=excluded.project_id, cwd=excluded.cwd,
                 telegram_chat_id=excluded.telegram_chat_id,
                 telegram_thread_id=excluded.telegram_thread_id,
                 updated_at=excluded.updated_at""",
            (
                binding.session_id,
                binding.origin_kind,
                binding.origin_key,
                binding.project_id,
                binding.cwd,
                binding.telegram_chat_id,
                binding.telegram_thread_id,
                int(time.time()),
            ),
        )
    return binding


def bind_gateway_session(
    conn: sqlite3.Connection, session_id: str, source: object
) -> Optional[SessionRouteBinding]:
    """Resolve and bind one gateway source using its canonical event identity."""
    platform = getattr(getattr(source, "platform", None), "value", None)
    platform = str(platform or getattr(source, "platform", "") or "").lower()
    if platform == "telegram":
        chat_id = getattr(source, "chat_id", None)
        thread_id = getattr(source, "thread_id", None)
        return bind_inbound_session(
            conn,
            session_id=session_id,
            origin_kind="telegram",
            origin_key=telegram_origin_key(chat_id, thread_id),
            telegram_chat_id=chat_id,
            telegram_thread_id=thread_id,
        )
    if platform == "webhook":
        route_name = str(getattr(source, "user_name", "") or "").strip()
        if not route_name:
            raise InvalidProjectRouteError(
                "Webhook session cannot be routed without a subscription name"
            )
        return bind_inbound_session(
            conn,
            session_id=session_id,
            origin_kind="webhook",
            origin_key=route_name,
        )
    return None


def apply_gateway_session_route(
    conn: sqlite3.Connection, session_db: object, context: object
) -> Optional[SessionRouteBinding]:
    """Bind a gateway session and stamp its classification cwd before the turn."""
    source = getattr(context, "source", None)
    binding = bind_gateway_session(conn, getattr(context, "session_id", ""), source)
    if binding is None:
        return None
    if binding.cwd:
        setattr(context, "cwd", binding.cwd)
        updater = getattr(session_db, "update_session_cwd", None)
        if not callable(updater):
            raise InvalidProjectRouteError(
                f"cannot persist project route cwd for session {binding.session_id}: no durable updater"
            )
        generation = updater(binding.session_id, binding.cwd)
        if generation is None:
            raise InvalidProjectRouteError(
                f"cannot persist project route cwd for missing session {binding.session_id}"
            )
    return binding


def apply_cron_session_route(
    conn: sqlite3.Connection,
    session_db: object,
    session_id: str,
    job_id: str,
) -> Optional[SessionRouteBinding]:
    """Classify one cron session without changing the job execution cwd."""
    binding = bind_inbound_session(
        conn,
        session_id=session_id,
        origin_kind="cron",
        origin_key=str(job_id),
    )
    if binding.cwd:
        updater = getattr(session_db, "update_session_cwd", None)
        if not callable(updater):
            raise InvalidProjectRouteError(
                f"cannot persist project route cwd for session {binding.session_id}: no durable updater"
            )
        generation = updater(binding.session_id, binding.cwd)
        if generation is None:
            raise InvalidProjectRouteError(
                f"cannot persist project route cwd for missing session {binding.session_id}"
            )
    return binding


def get_session_binding(
    conn: sqlite3.Connection, session_id: str
) -> Optional[SessionRouteBinding]:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM project_session_bindings WHERE session_id=?",
        (str(session_id),),
    ).fetchone()
    if row is None:
        return None
    return SessionRouteBinding(
        session_id=row["session_id"],
        origin_kind=row["origin_kind"],
        origin_key=row["origin_key"],
        project_id=row["project_id"],
        cwd=row["cwd"],
        telegram_chat_id=row["telegram_chat_id"],
        telegram_thread_id=row["telegram_thread_id"],
    )


def copy_session_binding(
    conn: sqlite3.Connection, source_session_id: str, target_session_id: str
) -> Optional[SessionRouteBinding]:
    source = get_session_binding(conn, source_session_id)
    if source is None:
        return None
    _ensure_schema(conn)
    with write_txn(conn):
        conn.execute(
            """INSERT INTO project_session_bindings
               (session_id, origin_kind, origin_key, project_id, cwd,
                telegram_chat_id, telegram_thread_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO NOTHING""",
            (
                str(target_session_id),
                source.origin_kind,
                source.origin_key,
                source.project_id,
                source.cwd,
                source.telegram_chat_id,
                source.telegram_thread_id,
                int(time.time()),
            ),
        )
    return get_session_binding(conn, target_session_id)


def claim_mirror_delivery(
    conn: sqlite3.Connection, session_id: str, message_key: str
) -> bool:
    """Atomically claim one stable session/message identity."""
    _ensure_schema(conn)
    now = int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            """INSERT OR IGNORE INTO project_session_mirror_deliveries
               (session_id, message_key, state, updated_at)
               VALUES (?, ?, 'pending', ?)""",
            (str(session_id), str(message_key), now),
        )
    return cur.rowcount == 1


def complete_mirror_delivery(
    conn: sqlite3.Connection, session_id: str, message_key: str
) -> None:
    _ensure_schema(conn)
    with write_txn(conn):
        conn.execute(
            """UPDATE project_session_mirror_deliveries
               SET state='sent', updated_at=?
               WHERE session_id=? AND message_key=?""",
            (int(time.time()), str(session_id), str(message_key)),
        )


def release_mirror_delivery(
    conn: sqlite3.Connection, session_id: str, message_key: str
) -> None:
    """Release only a failed pending claim; sent receipts are immutable."""
    _ensure_schema(conn)
    with write_txn(conn):
        conn.execute(
            """DELETE FROM project_session_mirror_deliveries
               WHERE session_id=? AND message_key=? AND state='pending'""",
            (str(session_id), str(message_key)),
        )


def _send_exact_telegram(
    profile_home: Path,
    chat_id: str,
    thread_id: Optional[str],
    text: str,
) -> None:
    """Use Hermes' in-process standalone Telegram delivery implementation."""
    from gateway.run import _profile_runtime_scope
    from gateway.config import Platform, load_gateway_config
    from model_tools import _run_async
    from tools.send_message_tool import _send_to_platform

    with _profile_runtime_scope(Path(profile_home)):
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.TELEGRAM)
        if pconfig is None or not pconfig.enabled:
            raise RuntimeError("Telegram is not configured for this profile")
        result = _run_async(
            _send_to_platform(
                Platform.TELEGRAM,
                pconfig,
                str(chat_id),
                str(text),
                thread_id=str(thread_id) if thread_id else None,
                media_files=[],
                force_document=False,
            )
        )
    if not isinstance(result, dict) or not result.get("success"):
        error = result.get("error") if isinstance(result, dict) else result
        raise RuntimeError(str(error or "Telegram delivery failed"))


def mirror_desktop_turn(
    profile_home: Path,
    session_id: str,
    *,
    user_text: str,
    assistant_text: str,
    user_message_key: str,
    assistant_message_key: str,
    send: Optional[Callable[[str, Optional[str], str], None]] = None,
) -> dict[str, bool]:
    """Mirror a persisted Desktop turn for a Telegram-origin session only."""
    result = {"user": False, "assistant": False}
    conn = projects_db.connect(Path(profile_home) / "projects.db")
    try:
        binding = get_session_binding(conn, session_id)
        if (
            binding is None
            or binding.origin_kind != "telegram"
            or not binding.telegram_chat_id
        ):
            return result
        sender = send or (
            lambda chat_id, thread_id, text: _send_exact_telegram(
                Path(profile_home), chat_id, thread_id, text
            )
        )
        messages = (
            ("user", user_message_key, str(user_text or "")),
            ("assistant", assistant_message_key, str(assistant_text or "")),
        )
        for role, key, text in messages:
            if not text.strip() or not str(key or "").strip():
                continue
            if not claim_mirror_delivery(conn, session_id, key):
                continue
            try:
                sender(binding.telegram_chat_id, binding.telegram_thread_id, text)
            except Exception as exc:
                release_mirror_delivery(conn, session_id, key)
                logger.warning(
                    "Desktop Telegram mirror failed for session=%s role=%s: %s",
                    session_id,
                    role,
                    exc,
                )
                if role == "user":
                    break
            else:
                complete_mirror_delivery(conn, session_id, key)
                result[role] = True
        return result
    finally:
        conn.close()
