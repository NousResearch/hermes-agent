"""Safety primitives for linking a local Hermes session to Telegram.

This module intentionally does not inject text into a live CLI/PTY.  It is the
fail-closed state layer a future CLI↔Telegram bridge must pass through before
any remote input or approval is allowed to reach an executor.
"""

from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class BridgeVerdict(Enum):
    """Deterministic policy decision for a bridge event."""

    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class BridgeDecision:
    verdict: BridgeVerdict
    reason: str
    bridge_id: Optional[str] = None
    hermes_session_id: Optional[str] = None


@dataclass(frozen=True)
class BridgeApproval:
    approval_id: int
    bridge_id: str
    nonce: str
    expires_at: datetime


@dataclass(frozen=True)
class BridgeBindingToken:
    token_id: int
    bridge_id: str
    hermes_session_id: str
    token: str
    expires_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_text(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _dt_from_text(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _thread_key(thread_id: Optional[str]) -> str:
    return "" if thread_id is None else str(thread_id)


class BridgeStateStore:
    """Durable fail-closed state for a CLI↔Telegram bridge.

    The store provides the invariants needed before any live bridge can safely
    exist:

    - durable Telegram update dedupe
    - explicit local-session ↔ Telegram-chat binding
    - reply-to correlation only for registered, non-expired, input-enabled bot
      messages
    - single-use, nonce-bound approvals
    - operator pause and filesystem kill switch

    It is intentionally conservative: ambiguous, stale, mismatched, or missing
    state always returns ``BridgeVerdict.REJECT``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        now_fn: Callable[[], datetime] = _utc_now,
        kill_switch_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now_fn = now_fn
        self.kill_switch_path = Path(kill_switch_path) if kill_switch_path else None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS processed_updates (
                    platform TEXT NOT NULL,
                    update_id TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    PRIMARY KEY (platform, update_id)
                );

                CREATE TABLE IF NOT EXISTS bindings (
                    bridge_id TEXT PRIMARY KEY,
                    hermes_session_id TEXT NOT NULL,
                    telegram_chat_id TEXT NOT NULL,
                    telegram_user_id TEXT NOT NULL,
                    telegram_thread_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    pause_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outbound_messages (
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL DEFAULT '',
                    message_id TEXT NOT NULL,
                    bridge_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    input_expected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    PRIMARY KEY (chat_id, thread_id, message_id),
                    FOREIGN KEY (bridge_id) REFERENCES bindings(bridge_id)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bridge_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args_hash TEXT NOT NULL,
                    nonce TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    approved_at TEXT,
                    consumed_at TEXT,
                    FOREIGN KEY (bridge_id) REFERENCES bindings(bridge_id)
                );

                CREATE TABLE IF NOT EXISTS binding_tokens (
                    token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bridge_id TEXT NOT NULL UNIQUE,
                    hermes_session_id TEXT NOT NULL,
                    token TEXT NOT NULL UNIQUE,
                    telegram_chat_id TEXT,
                    telegram_user_id TEXT,
                    telegram_thread_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                );
                """
            )
            approval_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(approvals)").fetchall()
            }
            if "approved_at" not in approval_columns:
                conn.execute("ALTER TABLE approvals ADD COLUMN approved_at TEXT")

    def _now(self) -> datetime:
        now = self._now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def _kill_switch_active(self) -> bool:
        return bool(self.kill_switch_path and self.kill_switch_path.exists())

    def disable_bridge(self) -> None:
        """Activate the filesystem kill switch for all bridge input."""
        if self.kill_switch_path is None:
            raise ValueError("kill_switch_path is not configured")
        self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        self.kill_switch_path.write_text(f"disabled_at={_dt_to_text(self._now())}\n", encoding="utf-8")

    def enable_bridge(self) -> None:
        """Remove the filesystem kill switch if present."""
        if self.kill_switch_path and self.kill_switch_path.exists():
            self.kill_switch_path.unlink()

    def binding_for_telegram(
        self,
        *,
        chat_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        """Return the active binding for a Telegram identity, if any."""
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM bindings
                WHERE telegram_chat_id=? AND telegram_user_id=? AND telegram_thread_id=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (str(chat_id), str(user_id), _thread_key(thread_id)),
            ).fetchone()

    def _format_binding_row(self, row: sqlite3.Row) -> str:
        thread = row["telegram_thread_id"] or "-"
        reason = f", reason={row['pause_reason']}" if row["pause_reason"] else ""
        return (
            f"bridge={row['bridge_id']}, session={row['hermes_session_id']}, "
            f"status={row['status']}{reason}, chat={row['telegram_chat_id']}, "
            f"user={row['telegram_user_id']}, thread={thread}, updated={row['updated_at']}"
        )

    def list_bindings_for_session(self, hermes_session_id: str) -> list[sqlite3.Row]:
        """Return Telegram bridge bindings for one Hermes session."""
        with self._connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT * FROM bindings
                    WHERE hermes_session_id=?
                    ORDER BY updated_at DESC, bridge_id ASC
                    """,
                    (str(hermes_session_id),),
                ).fetchall()
            )

    def describe_session_status(self, hermes_session_id: str) -> str:
        """Human-readable local CLI status for a Hermes session's bridge bindings."""
        kill = "active" if self._kill_switch_active() else "inactive"
        rows = self.list_bindings_for_session(hermes_session_id)
        if not rows:
            return (
                f"Local bridge status for Hermes session `{hermes_session_id}`: no Telegram bindings.\n"
                f"Kill switch: {kill}. Use /bridge bind telegram to create an opt-in token."
            )
        details = "\n".join(f"- {self._format_binding_row(row)}" for row in rows)
        return f"Local bridge status for Hermes session `{hermes_session_id}`. Kill switch: {kill}.\n{details}"

    def _delete_binding_rows(self, conn: sqlite3.Connection, bridge_ids: list[str]) -> None:
        for bridge_id in bridge_ids:
            conn.execute("DELETE FROM outbound_messages WHERE bridge_id=?", (bridge_id,))
            conn.execute("DELETE FROM approvals WHERE bridge_id=?", (bridge_id,))
            conn.execute("DELETE FROM bindings WHERE bridge_id=?", (bridge_id,))

    def delete_binding(self, bridge_id: str) -> Optional[sqlite3.Row]:
        """Delete one bridge binding by id and return the deleted row, if any."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM bindings WHERE bridge_id=?", (str(bridge_id),)).fetchone()
            if row is None:
                return None
            self._delete_binding_rows(conn, [str(bridge_id)])
            return row

    def delete_bindings_for_session(self, hermes_session_id: str) -> list[sqlite3.Row]:
        """Delete all Telegram bridge bindings for one Hermes session."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = list(
                conn.execute(
                    "SELECT * FROM bindings WHERE hermes_session_id=? ORDER BY updated_at DESC, bridge_id ASC",
                    (str(hermes_session_id),),
                ).fetchall()
            )
            self._delete_binding_rows(conn, [str(row["bridge_id"]) for row in rows])
            return rows

    def delete_binding_for_telegram(
        self,
        *,
        chat_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        """Delete this Telegram identity's bridge binding and return it, if present."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM bindings
                WHERE telegram_chat_id=? AND telegram_user_id=? AND telegram_thread_id=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (str(chat_id), str(user_id), _thread_key(thread_id)),
            ).fetchone()
            if row is None:
                return None
            self._delete_binding_rows(conn, [str(row["bridge_id"])])
            return row

    def describe_telegram_status(
        self,
        *,
        chat_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> str:
        """Human-readable status for safe Telegram bridge commands."""
        if self._kill_switch_active():
            return "Bridge kill switch is active. Remote input is disabled."
        row = self.binding_for_telegram(chat_id=chat_id, user_id=user_id, thread_id=thread_id)
        if row is None:
            return "No bridge binding is active for this Telegram chat/user."
        return f"Bridge binding: {self._format_binding_row(row)}"

    def validate_telegram_direct_input(
        self,
        *,
        chat_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> BridgeDecision:
        """Validate a bound Telegram DM before routing plain text to a CLI session.

        This is intentionally stricter than normal gateway authorization. A
        Telegram message may only continue a CLI-originated session after the
        local side minted a token, Telegram consumed it, the binding is still
        active, and the global kill switch is not set.
        """
        if self._kill_switch_active():
            return BridgeDecision(BridgeVerdict.REJECT, "bridge kill switch is active")
        row = self.binding_for_telegram(chat_id=chat_id, user_id=user_id, thread_id=thread_id)
        if row is None:
            return BridgeDecision(BridgeVerdict.REJECT, "no bridge binding for telegram identity")
        if row["status"] != "active":
            reason = row["pause_reason"] or row["status"]
            return BridgeDecision(BridgeVerdict.REJECT, f"binding is paused: {reason}")
        return BridgeDecision(
            BridgeVerdict.ACCEPT,
            "accepted",
            bridge_id=row["bridge_id"],
            hermes_session_id=row["hermes_session_id"],
        )

    def accept_update(self, *, platform: str, update_id: str) -> bool:
        """Return True once per platform/update_id, False for duplicates."""
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO processed_updates(platform, update_id, received_at) VALUES (?, ?, ?)",
                    (platform, str(update_id), _dt_to_text(self._now())),
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def create_binding(
        self,
        *,
        bridge_id: str,
        hermes_session_id: str,
        telegram_chat_id: str,
        telegram_user_id: str,
        telegram_thread_id: Optional[str] = None,
    ) -> None:
        """Create or replace an explicit local-session ↔ Telegram binding."""
        now = _dt_to_text(self._now())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bindings(
                    bridge_id, hermes_session_id, telegram_chat_id,
                    telegram_user_id, telegram_thread_id, status,
                    pause_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?)
                ON CONFLICT(bridge_id) DO UPDATE SET
                    hermes_session_id=excluded.hermes_session_id,
                    telegram_chat_id=excluded.telegram_chat_id,
                    telegram_user_id=excluded.telegram_user_id,
                    telegram_thread_id=excluded.telegram_thread_id,
                    status='active',
                    pause_reason=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    bridge_id,
                    hermes_session_id,
                    str(telegram_chat_id),
                    str(telegram_user_id),
                    _thread_key(telegram_thread_id),
                    now,
                    now,
                ),
            )

    def create_binding_token(
        self,
        *,
        bridge_id: str,
        hermes_session_id: str,
        ttl_seconds: int,
        token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        telegram_user_id: Optional[str] = None,
        telegram_thread_id: Optional[str] = None,
    ) -> BridgeBindingToken:
        """Create a local opt-in token for binding Telegram to a Hermes session.

        A live bridge should mint this token only from the local CLI/TUI side.
        Telegram can consume it once to create the explicit chat/user binding.
        """
        token_value = token or secrets.token_urlsafe(18)
        now = self._now()
        expires = datetime.fromtimestamp(now.timestamp() + max(0, int(ttl_seconds)), timezone.utc)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO binding_tokens(
                    bridge_id, hermes_session_id, token,
                    telegram_chat_id, telegram_user_id, telegram_thread_id,
                    created_at, expires_at, consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(bridge_id) DO UPDATE SET
                    hermes_session_id=excluded.hermes_session_id,
                    token=excluded.token,
                    telegram_chat_id=excluded.telegram_chat_id,
                    telegram_user_id=excluded.telegram_user_id,
                    telegram_thread_id=excluded.telegram_thread_id,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at,
                    consumed_at=NULL
                """,
                (
                    bridge_id,
                    hermes_session_id,
                    token_value,
                    str(telegram_chat_id) if telegram_chat_id is not None else None,
                    str(telegram_user_id) if telegram_user_id is not None else None,
                    _thread_key(telegram_thread_id) if telegram_thread_id is not None else None,
                    _dt_to_text(now),
                    _dt_to_text(expires),
                ),
            )
            token_id_raw = cur.lastrowid
            if token_id_raw is None:
                row = conn.execute("SELECT token_id FROM binding_tokens WHERE bridge_id=?", (bridge_id,)).fetchone()
                if row is None:
                    raise RuntimeError("sqlite did not return a binding token row id")
                token_id_raw = row["token_id"]
            token_id = int(token_id_raw)
        return BridgeBindingToken(
            token_id=token_id,
            bridge_id=bridge_id,
            hermes_session_id=hermes_session_id,
            token=token_value,
            expires_at=expires,
        )

    def consume_binding_token(
        self,
        *,
        token: str,
        telegram_chat_id: str,
        telegram_user_id: str,
        telegram_thread_id: Optional[str] = None,
    ) -> BridgeDecision:
        """Consume a local opt-in token and create the Telegram binding."""
        if self._kill_switch_active():
            return BridgeDecision(BridgeVerdict.REJECT, "bridge kill switch is active")

        now = self._now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM binding_tokens WHERE token=?",
                (token,),
            ).fetchone()
            if row is None:
                return BridgeDecision(BridgeVerdict.REJECT, "binding token not found")
            if row["consumed_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "binding token already consumed")
            if _dt_from_text(row["expires_at"]) <= now:
                return BridgeDecision(BridgeVerdict.REJECT, "binding token expired")
            if row["telegram_chat_id"] is not None and str(telegram_chat_id) != str(row["telegram_chat_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram chat mismatch")
            if row["telegram_user_id"] is not None and str(telegram_user_id) != str(row["telegram_user_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram user mismatch")
            expected_thread = row["telegram_thread_id"]
            if expected_thread is not None and _thread_key(telegram_thread_id) != str(expected_thread):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram thread mismatch")

            conn.execute(
                "UPDATE binding_tokens SET consumed_at=? WHERE token_id=? AND consumed_at IS NULL",
                (_dt_to_text(now), row["token_id"]),
            )
            conn.execute(
                """
                INSERT INTO bindings(
                    bridge_id, hermes_session_id, telegram_chat_id,
                    telegram_user_id, telegram_thread_id, status,
                    pause_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', NULL, ?, ?)
                ON CONFLICT(bridge_id) DO UPDATE SET
                    hermes_session_id=excluded.hermes_session_id,
                    telegram_chat_id=excluded.telegram_chat_id,
                    telegram_user_id=excluded.telegram_user_id,
                    telegram_thread_id=excluded.telegram_thread_id,
                    status='active',
                    pause_reason=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    row["bridge_id"],
                    row["hermes_session_id"],
                    str(telegram_chat_id),
                    str(telegram_user_id),
                    _thread_key(telegram_thread_id),
                    _dt_to_text(now),
                    _dt_to_text(now),
                ),
            )
            return BridgeDecision(
                BridgeVerdict.ACCEPT,
                "accepted",
                bridge_id=row["bridge_id"],
                hermes_session_id=row["hermes_session_id"],
            )

    def pause_binding(self, bridge_id: str, *, reason: str = "paused") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE bindings SET status='paused', pause_reason=?, updated_at=? WHERE bridge_id=?",
                (reason, _dt_to_text(self._now()), bridge_id),
            )

    def resume_binding(self, bridge_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE bindings SET status='active', pause_reason=NULL, updated_at=? WHERE bridge_id=?",
                (_dt_to_text(self._now()), bridge_id),
            )

    def record_outbound_message(
        self,
        *,
        bridge_id: str,
        chat_id: str,
        thread_id: Optional[str],
        message_id: str,
        purpose: str,
        input_expected: bool,
        ttl_seconds: int,
    ) -> None:
        """Register a bot-sent Telegram message for later reply correlation."""
        now = self._now()
        expires_at = now.timestamp() + max(0, int(ttl_seconds))
        expires = datetime.fromtimestamp(expires_at, timezone.utc)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO outbound_messages(
                    chat_id, thread_id, message_id, bridge_id, purpose,
                    input_expected, created_at, expires_at, consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    str(chat_id),
                    _thread_key(thread_id),
                    str(message_id),
                    bridge_id,
                    purpose,
                    1 if input_expected else 0,
                    _dt_to_text(now),
                    _dt_to_text(expires),
                ),
            )

    def validate_reply_input(
        self,
        *,
        chat_id: str,
        thread_id: Optional[str],
        user_id: str,
        reply_to_message_id: str,
        inbound_message_id: str,
    ) -> BridgeDecision:
        """Validate a Telegram reply before it can become remote input.

        Successful validation consumes the outbound prompt message, preventing
        accidental re-use of the same reply anchor for multiple executions.
        """
        if self._kill_switch_active():
            return BridgeDecision(BridgeVerdict.REJECT, "bridge kill switch is active")

        now = self._now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT om.*, b.hermes_session_id, b.telegram_user_id, b.status, b.pause_reason
                FROM outbound_messages om
                JOIN bindings b ON b.bridge_id = om.bridge_id
                WHERE om.chat_id=? AND om.thread_id=? AND om.message_id=?
                """,
                (str(chat_id), _thread_key(thread_id), str(reply_to_message_id)),
            ).fetchone()
            if row is None:
                return BridgeDecision(BridgeVerdict.REJECT, "reply target is not registered")
            if row["consumed_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "reply target already consumed")
            if not row["input_expected"]:
                return BridgeDecision(BridgeVerdict.REJECT, "reply target does not accept input")
            if row["status"] != "active":
                reason = row["pause_reason"] or row["status"]
                return BridgeDecision(BridgeVerdict.REJECT, f"binding is paused: {reason}")
            if str(user_id) != str(row["telegram_user_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram user mismatch")
            if _dt_from_text(row["expires_at"]) <= now:
                return BridgeDecision(BridgeVerdict.REJECT, "reply target expired")

            conn.execute(
                """
                UPDATE outbound_messages
                SET consumed_at=?
                WHERE chat_id=? AND thread_id=? AND message_id=? AND consumed_at IS NULL
                """,
                (
                    _dt_to_text(now),
                    str(chat_id),
                    _thread_key(thread_id),
                    str(reply_to_message_id),
                ),
            )
            return BridgeDecision(
                BridgeVerdict.ACCEPT,
                "accepted",
                bridge_id=row["bridge_id"],
                hermes_session_id=row["hermes_session_id"],
            )

    def create_approval(
        self,
        *,
        bridge_id: str,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args_hash: str,
        ttl_seconds: int,
        nonce: Optional[str] = None,
    ) -> BridgeApproval:
        """Create a single-use approval nonce bound to a tool call."""
        nonce_value = nonce or secrets.token_urlsafe(18)
        now = self._now()
        expires = datetime.fromtimestamp(now.timestamp() + max(0, int(ttl_seconds)), timezone.utc)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO approvals(
                    bridge_id, turn_id, tool_call_id, tool_name, tool_args_hash,
                    nonce, created_at, expires_at, consumed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    bridge_id,
                    turn_id,
                    tool_call_id,
                    tool_name,
                    tool_args_hash,
                    nonce_value,
                    _dt_to_text(now),
                    _dt_to_text(expires),
                ),
            )
            approval_id_raw = cur.lastrowid
            if approval_id_raw is None:
                raise RuntimeError("sqlite did not return an approval row id")
            approval_id = int(approval_id_raw)
        return BridgeApproval(approval_id=approval_id, bridge_id=bridge_id, nonce=nonce_value, expires_at=expires)

    def record_approval_response(
        self,
        *,
        nonce: str,
        chat_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> BridgeDecision:
        """Record a Telegram user's approval response without consuming the executor nonce."""
        if self._kill_switch_active():
            return BridgeDecision(BridgeVerdict.REJECT, "bridge kill switch is active")

        now = self._now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT a.*, b.hermes_session_id, b.telegram_chat_id, b.telegram_user_id,
                       b.telegram_thread_id, b.status, b.pause_reason
                FROM approvals a
                JOIN bindings b ON b.bridge_id = a.bridge_id
                WHERE a.nonce=?
                """,
                (nonce,),
            ).fetchone()
            if row is None:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce not found")
            if row["consumed_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce already consumed")
            if row["approved_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce already approved")
            if row["status"] != "active":
                reason = row["pause_reason"] or row["status"]
                return BridgeDecision(BridgeVerdict.REJECT, f"binding is paused: {reason}")
            if str(chat_id) != str(row["telegram_chat_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram chat mismatch")
            if str(user_id) != str(row["telegram_user_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram user mismatch")
            if _thread_key(thread_id) != str(row["telegram_thread_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram thread mismatch")
            if _dt_from_text(row["expires_at"]) <= now:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce expired")

            conn.execute(
                "UPDATE approvals SET approved_at=? WHERE approval_id=? AND approved_at IS NULL AND consumed_at IS NULL",
                (_dt_to_text(now), row["approval_id"]),
            )
            return BridgeDecision(
                BridgeVerdict.ACCEPT,
                "approved",
                bridge_id=row["bridge_id"],
                hermes_session_id=row["hermes_session_id"],
            )

    def consume_approval(
        self,
        *,
        nonce: str,
        chat_id: str,
        user_id: str,
        hermes_session_id: str,
        tool_args_hash: str,
        turn_id: str | None = None,
        tool_call_id: str | None = None,
        thread_id: Optional[str] = None,
        require_user_approval: bool = False,
    ) -> BridgeDecision:
        """Consume an approval nonce if every binding attribute still matches."""
        if self._kill_switch_active():
            return BridgeDecision(BridgeVerdict.REJECT, "bridge kill switch is active")

        now = self._now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT a.*, b.hermes_session_id, b.telegram_chat_id, b.telegram_user_id,
                       b.telegram_thread_id, b.status, b.pause_reason
                FROM approvals a
                JOIN bindings b ON b.bridge_id = a.bridge_id
                WHERE a.nonce=?
                """,
                (nonce,),
            ).fetchone()
            if row is None:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce not found")
            if row["consumed_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce already consumed")
            if require_user_approval and turn_id is None:
                return BridgeDecision(BridgeVerdict.REJECT, "turn id is required for approved bridge nonce consumption")
            if require_user_approval and tool_call_id is None:
                return BridgeDecision(BridgeVerdict.REJECT, "tool call id is required for approved bridge nonce consumption")
            if require_user_approval and not row["approved_at"]:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce has not been approved")
            if row["status"] != "active":
                reason = row["pause_reason"] or row["status"]
                return BridgeDecision(BridgeVerdict.REJECT, f"binding is paused: {reason}")
            if str(chat_id) != str(row["telegram_chat_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram chat mismatch")
            if str(user_id) != str(row["telegram_user_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram user mismatch")
            if require_user_approval and _thread_key(thread_id) != str(row["telegram_thread_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "telegram thread mismatch")
            if str(hermes_session_id) != str(row["hermes_session_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "hermes session mismatch")
            if turn_id is not None and str(turn_id) != str(row["turn_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "turn id mismatch")
            if tool_call_id is not None and str(tool_call_id) != str(row["tool_call_id"]):
                return BridgeDecision(BridgeVerdict.REJECT, "tool call id mismatch")
            if str(tool_args_hash) != str(row["tool_args_hash"]):
                return BridgeDecision(BridgeVerdict.REJECT, "tool arguments changed")
            if _dt_from_text(row["expires_at"]) <= now:
                return BridgeDecision(BridgeVerdict.REJECT, "approval nonce expired")

            conn.execute(
                "UPDATE approvals SET consumed_at=? WHERE approval_id=? AND consumed_at IS NULL",
                (_dt_to_text(now), row["approval_id"]),
            )
            return BridgeDecision(
                BridgeVerdict.ACCEPT,
                "accepted",
                bridge_id=row["bridge_id"],
                hermes_session_id=row["hermes_session_id"],
            )
