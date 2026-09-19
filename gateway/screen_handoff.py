"""Durable, profile-scoped browser screen handoffs.

The Bot Desktop lease remains the only input lock.  This module only stores the
short-lived invitation and the explicit return-to-agent intent.  Tokens are
returned to the delivery adapter once and are never written to disk or logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_constants import get_hermes_home, secure_parent_dir

logger = logging.getLogger(__name__)

INVITE_TTL_SECONDS = 10 * 60
CONFIRM_TTL_SECONDS = 2 * 60
WEB_SESSION_TTL_SECONDS = 30 * 60
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_CALLBACKS: dict[str, Callable[[dict[str, Any]], Any]] = {}
_CALLBACK_LOCK = threading.RLock()


def _now() -> float:
    return time.time()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _short_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))


def _db_path(profile_home: Optional[str | Path] = None) -> Path:
    home = Path(profile_home) if profile_home else get_hermes_home()
    return home / "state" / "screen-handoffs.db"


def _private_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path.parent)


@dataclass(frozen=True)
class Handoff:
    request_id: str
    invite_token: str
    confirmation_code: str
    state: str
    session_id: str
    profile_home: str
    source_json: str
    reason: str
    created_at: float
    invite_expires_at: float
    confirmation_expires_at: float
    web_expires_at: Optional[float]
    viewer_id: Optional[str]
    # One-time response value; never persisted and never included by public().
    web_session_token: str = ""

    def public(self) -> dict[str, Any]:
        """Model/gateway-safe representation; excludes the invite and confirmation secrets."""
        return {
            "request_id": self.request_id,
            "state": self.state,
            "session_id": self.session_id,
            "profile_home": self.profile_home,
            "reason": self.reason,
            "created_at": self.created_at,
            "invite_expires_at": self.invite_expires_at,
            "confirmation_expires_at": self.confirmation_expires_at,
            "web_expires_at": self.web_expires_at,
            "viewer_id": self.viewer_id,
        }


class ScreenHandoffStore:
    """SQLite-backed state machine for one profile's screen invitations."""

    def __init__(self, profile_home: Optional[str | Path] = None, *, db_path: Optional[Path] = None):
        self.profile_home = str(Path(profile_home) if profile_home else get_hermes_home())
        self.path = Path(db_path) if db_path else _db_path(self.profile_home)
        _private_db(self.path)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS screen_handoffs (
                    request_id TEXT PRIMARY KEY,
                    invite_digest TEXT NOT NULL UNIQUE,
                    confirmation_digest TEXT NOT NULL,
                    web_session_digest TEXT,
                    state TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    profile_home TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    invite_expires_at REAL NOT NULL,
                    confirmation_expires_at REAL NOT NULL,
                    web_expires_at REAL,
                    viewer_id TEXT,
                    opened_at REAL,
                    authorized_at REAL,
                    human_at REAL,
                    returned_at REAL,
                    resumed_at REAL,
                    revoked_at REAL,
                    last_error TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS screen_handoffs_active_idx "
                "ON screen_handoffs(profile_home, session_id, state)"
            )

    @staticmethod
    def _row(row: Optional[sqlite3.Row], *, invite_token: str = "", confirmation_code: str = "",
             web_session_token: str = "") -> Optional[Handoff]:
        if row is None:
            return None
        return Handoff(
            request_id=row["request_id"], invite_token=invite_token, confirmation_code=confirmation_code,
            state=row["state"], session_id=row["session_id"], profile_home=row["profile_home"],
            source_json=row["source_json"], reason=row["reason"], created_at=float(row["created_at"]),
            invite_expires_at=float(row["invite_expires_at"]),
            confirmation_expires_at=float(row["confirmation_expires_at"]),
            web_expires_at=float(row["web_expires_at"]) if row["web_expires_at"] is not None else None,
            viewer_id=row["viewer_id"], web_session_token=web_session_token,
        )

    def _expire(self, conn: sqlite3.Connection, now: float) -> None:
        conn.execute(
            "UPDATE screen_handoffs SET state='expired' "
            "WHERE state IN ('pending','opened') AND invite_expires_at <= ?",
            (now,),
        )
        conn.execute(
            "UPDATE screen_handoffs SET state='expired' "
            "WHERE state IN ('authorized','human') AND web_expires_at IS NOT NULL AND web_expires_at <= ?",
            (now,),
        )

    def create_or_get(self, *, session_id: str, source_json: str, reason: str) -> tuple[Handoff, bool]:
        if not session_id:
            raise ValueError("screen handoff requires a session id")
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute(
                "SELECT * FROM screen_handoffs WHERE profile_home=? AND session_id=? "
                "AND state IN ('pending','opened','authorized','human') "
                "ORDER BY created_at DESC LIMIT 1",
                (self.profile_home, str(session_id)),
            ).fetchone()
            if row is not None:
                conn.commit()
                return self._row(row), False
            request_id = secrets.token_urlsafe(18)
            invite_token = secrets.token_urlsafe(32)
            code = _short_code()
            invite_expires = now + INVITE_TTL_SECONDS
            confirmation_expires = now + CONFIRM_TTL_SECONDS
            conn.execute(
                "INSERT INTO screen_handoffs(request_id, invite_digest, confirmation_digest, state, session_id, "
                "profile_home, source_json, reason, created_at, invite_expires_at, confirmation_expires_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, _digest(invite_token), _digest(code), "pending", str(session_id), self.profile_home,
                 source_json, str(reason or "").strip()[:500], now, invite_expires, confirmation_expires),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (request_id,)).fetchone()
        return self._row(row, invite_token=invite_token, confirmation_code=code), True

    def reissue(self, *, session_id: str, source_json: str, reason: str) -> Optional[Handoff]:
        """Rotate an invitation for an explicit ``/screen`` recovery command."""
        now = _now()
        invite_token, code = secrets.token_urlsafe(32), _short_code()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute(
                "SELECT * FROM screen_handoffs WHERE profile_home=? AND session_id=? "
                "AND state NOT IN ('expired','revoked','resumed') ORDER BY created_at DESC LIMIT 1",
                (self.profile_home, str(session_id)),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE screen_handoffs SET invite_digest=?, confirmation_digest=?, web_session_digest=NULL, "
                "state='pending', source_json=?, reason=?, invite_expires_at=?, confirmation_expires_at=?, "
                "web_expires_at=NULL, viewer_id=NULL, opened_at=NULL, authorized_at=NULL, human_at=NULL, "
                "returned_at=NULL, last_error=NULL WHERE request_id=?",
                (_digest(invite_token), _digest(code), source_json, str(reason or "")[:500],
                 now + INVITE_TTL_SECONDS, now + CONFIRM_TTL_SECONDS, row["request_id"]),
            )
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row, invite_token=invite_token, confirmation_code=code)

    def by_token(self, token: str, *, mark_opened: bool = False) -> Optional[Handoff]:
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE") if mark_opened else None
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE invite_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] in {"expired", "revoked", "resumed"}:
                if mark_opened:
                    conn.commit()
                return None
            if mark_opened and row["state"] == "pending":
                conn.execute("UPDATE screen_handoffs SET state='opened', opened_at=? WHERE request_id=?", (now, row["request_id"]))
                row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            if mark_opened:
                conn.commit()
            return self._row(row, invite_token=token)

    def authorize(self, token: str, code: str) -> Optional[Handoff]:
        if not token or not code:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE invite_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] not in {"opened", "authorized", "pending"}:
                conn.commit()
                return None
            if float(row["confirmation_expires_at"]) <= now:
                conn.execute("UPDATE screen_handoffs SET state='expired' WHERE request_id=?", (row["request_id"],))
                conn.commit()
                return None
            if not secrets.compare_digest(row["confirmation_digest"], _digest(str(code).strip().upper())):
                conn.commit()
                return None
            web_token = secrets.token_urlsafe(32)
            web_expiry = now + WEB_SESSION_TTL_SECONDS
            conn.execute(
                "UPDATE screen_handoffs SET state='authorized', authorized_at=?, web_expires_at=?, web_session_digest=? WHERE request_id=?",
                (now, web_expiry, _digest(web_token), row["request_id"]),
            )
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row, invite_token=token, confirmation_code=str(code).strip().upper(),
                         web_session_token=web_token)

    def web_session(self, token: str) -> Optional[Handoff]:
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
        if row is None or row["state"] not in {"authorized", "human"} or row["web_expires_at"] is None or float(row["web_expires_at"]) <= now:
            return None
        return self._row(row)

    def any_web_session(self, token: str) -> Optional[Handoff]:
        """Lookup a still-valid cookie, including a returned session for idempotent UI actions."""
        if not token:
            return None
        now = _now()
        with self._connect() as conn:
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
        if row is None or row["state"] in {"expired", "revoked"} or row["web_expires_at"] is None or float(row["web_expires_at"]) <= now:
            return None
        return self._row(row)

    def take_over(self, token: str, viewer_id: str) -> Optional[Handoff]:
        if not token or not viewer_id:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._expire(conn, now)
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] not in {"authorized", "human"}:
                conn.commit()
                return None
            conn.execute("UPDATE screen_handoffs SET state='human', human_at=?, viewer_id=? WHERE request_id=?", (now, viewer_id, row["request_id"]))
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row)

    def return_to_agent(self, token: str, viewer_id: str) -> Optional[Handoff]:
        if not token or not viewer_id:
            return None
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM screen_handoffs WHERE web_session_digest=?", (_digest(token),)).fetchone()
            if row is None or row["state"] not in {"human", "authorized"}:
                conn.commit()
                return None
            if row["state"] == "human" and row["viewer_id"] != viewer_id:
                conn.commit()
                return None
            conn.execute("UPDATE screen_handoffs SET state='returned', returned_at=? WHERE request_id=?", (now, row["request_id"]))
            row = conn.execute("SELECT * FROM screen_handoffs WHERE request_id=?", (row["request_id"],)).fetchone()
            conn.commit()
        return self._row(row)

    def revoke(self, request_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE screen_handoffs SET state='revoked', revoked_at=? WHERE request_id=? "
                "AND state NOT IN ('resumed','revoked','expired')", (_now(), request_id),
            )
            return cur.rowcount == 1

    def refuse(self, token: str) -> bool:
        """Refuse a still-pending invitation using its locator, without authorizing it."""
        handoff = self.by_token(token)
        if handoff is None or handoff.state not in {"pending", "opened"}:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE screen_handoffs SET state='revoked', revoked_at=? WHERE request_id=? AND state IN ('pending','opened')",
                (_now(), handoff.request_id),
            )
            return cur.rowcount == 1

    def claim_returned(self, limit: int = 10) -> list[Handoff]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM screen_handoffs WHERE profile_home=? AND state='returned' ORDER BY returned_at LIMIT ?",
                (self.profile_home, int(limit)),
            ).fetchall()
            out = []
            for row in rows:
                conn.execute("UPDATE screen_handoffs SET state='resuming' WHERE request_id=? AND state='returned'", (row["request_id"],))
                out.append(self._row(row))
            conn.commit()
        return out

    def finish_resume(self, request_id: str, *, error: str = "") -> None:
        state = "returned" if error else "resumed"
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state=?, last_error=? WHERE request_id=? AND state='resuming'", (state, error[:500], request_id))

    def abandon_resume(self, request_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE screen_handoffs SET state='revoked', last_error=? WHERE request_id=? AND state='resuming'", (str(error)[:500], request_id))

    def source(self, handoff: Handoff) -> Any:
        from gateway.session import SessionSource
        return SessionSource.from_dict(json.loads(handoff.source_json))


def register_screen_handoff_notify(session_key: str, callback: Callable[[dict[str, Any]], Any]) -> None:
    if session_key:
        with _CALLBACK_LOCK:
            _CALLBACKS[session_key] = callback


def unregister_screen_handoff_notify(session_key: str) -> None:
    with _CALLBACK_LOCK:
        _CALLBACKS.pop(session_key, None)


def notify_screen_handoff(session_key: str, record: dict[str, Any]) -> bool:
    with _CALLBACK_LOCK:
        callback = _CALLBACKS.get(session_key)
    if callback is None:
        return False
    try:
        callback(record)
        return True
    except Exception:
        logger.exception("screen handoff delivery callback failed")
        return False


def has_screen_handoff_notify(session_key: str) -> bool:
    with _CALLBACK_LOCK:
        return bool(session_key and session_key in _CALLBACKS)


def _reset_for_tests() -> None:
    with _CALLBACK_LOCK:
        _CALLBACKS.clear()
