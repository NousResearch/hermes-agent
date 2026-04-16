"""SQLite persistence for brokerage intents and audit events."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from brokerage.models import TradeEvent, TradeIntent


class SQLiteBrokerageStore:
    """Persist brokerage intents and audit events in SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else get_hermes_home() / "brokerage" / "brokerage.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_intents (
                    intent_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    account_mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_type TEXT NOT NULL,
                    limit_price REAL,
                    asset_class TEXT NOT NULL,
                    confirmation_code TEXT,
                    confirmation_expires_at TEXT,
                    session_id TEXT,
                    telegram_chat_id TEXT,
                    ibkr_order_id TEXT,
                    raw_request_text TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_intent(
        self,
        intent: TradeIntent,
        *,
        confirmation_code: str,
        confirmation_expires_at: datetime | None = None,
        raw_request_text: str | None = None,
        session_id: str | None = None,
        telegram_chat_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_intents (
                    intent_id, created_at, status, account_mode, symbol, side,
                    quantity, order_type, limit_price, asset_class,
                    confirmation_code, confirmation_expires_at,
                    session_id, telegram_chat_id, ibkr_order_id, raw_request_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.request_id,
                    datetime.now(timezone.utc).isoformat(),
                    intent.status,
                    intent.account_mode,
                    intent.symbol,
                    intent.side,
                    intent.quantity,
                    intent.order_type,
                    intent.limit_price,
                    intent.asset_class,
                    confirmation_code,
                    confirmation_expires_at.isoformat() if confirmation_expires_at else None,
                    session_id,
                    telegram_chat_id,
                    None,
                    raw_request_text,
                ),
            )
            conn.commit()

    def get_intent(self, intent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trade_intents WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_status(self, intent_id: str, status: str, *, ibkr_order_id: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trade_intents SET status = ?, ibkr_order_id = COALESCE(?, ibkr_order_id) WHERE intent_id = ?",
                (status, ibkr_order_id, intent_id),
            )
            conn.commit()

    def transition_status(self, intent_id: str, from_status: str, to_status: str, *, ibkr_order_id: str | None = None) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE trade_intents
                SET status = ?, ibkr_order_id = COALESCE(?, ibkr_order_id)
                WHERE intent_id = ? AND status = ?
                """,
                (to_status, ibkr_order_id, intent_id, from_status),
            )
            conn.commit()
            return result.rowcount > 0

    def append_event(self, event: TradeEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trade_events (intent_id, event_type, detail, created_at) VALUES (?, ?, ?, ?)",
                (event.intent_id, event.event_type, event.detail, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def list_events(self, intent_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT intent_id, event_type, detail, created_at FROM trade_events WHERE intent_id = ? ORDER BY id ASC",
                (intent_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def expire_pending_confirmations(self, now: datetime) -> int:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE trade_intents
                SET status = 'expired'
                WHERE status = 'pending_confirmation'
                  AND confirmation_expires_at IS NOT NULL
                  AND confirmation_expires_at < ?
                """,
                (now.isoformat(),),
            )
            conn.commit()
            return result.rowcount
