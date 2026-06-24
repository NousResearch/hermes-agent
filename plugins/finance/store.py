"""Local SQLite store for the finance plugin.

Everything Hermes knows about a user's finances lives here, on disk, under the
profile-scoped Hermes home — never in a remote service and never alongside the
provider access tokens (those live in a 0600 file; see ``plaid/auth.py``).

Design notes:

* **Append-only transactions.** Plaid's ``/transactions/sync`` can mark a
  transaction ``removed``; we never hard-delete it. Removed rows are
  tombstoned (``removed=1`` + ``removed_at``) so history and prior agent
  answers stay reproducible. Queries exclude tombstones by default.
* **Profile-safe path.** The DB path derives from ``get_hermes_home()`` so each
  profile gets its own ``finance/finance.db``.
* **Money is stored as Plaid reports it.** Amounts follow Plaid's sign
  convention: a *positive* transaction amount is money leaving the account
  (a debit/outflow); a *negative* amount is money coming in (a credit/inflow).
  Aggregations document where they flip the sign for human-friendly output.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home, secure_parent_dir

# Account types treated as assets vs liabilities for net-worth math.
_LIABILITY_TYPES = {"credit", "loan"}


def default_db_path() -> Path:
    """Return the profile-scoped path to ``finance.db``."""
    return get_hermes_home() / "finance" / "finance.db"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    institution_id TEXT,
    institution_name TEXT,
    environment TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    transactions_cursor TEXT,
    last_sync_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    name TEXT,
    official_name TEXT,
    mask TEXT,
    type TEXT,
    subtype TEXT,
    currency TEXT,
    current_balance REAL,
    available_balance REAL,
    credit_limit REAL,
    removed INTEGER NOT NULL DEFAULT 0,
    last_synced_at TEXT,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_accounts_item ON accounts(item_id);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    date TEXT,
    datetime TEXT,
    name TEXT,
    merchant_name TEXT,
    category_primary TEXT,
    category_detailed TEXT,
    pending INTEGER NOT NULL DEFAULT 0,
    payment_channel TEXT,
    removed INTEGER NOT NULL DEFAULT 0,
    removed_at TEXT,
    raw TEXT,
    first_seen_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_item ON transactions(item_id);
CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);

CREATE TABLE IF NOT EXISTS securities (
    security_id TEXT PRIMARY KEY,
    name TEXT,
    ticker_symbol TEXT,
    type TEXT,
    close_price REAL,
    currency TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS holdings (
    account_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    quantity REAL,
    cost_basis REAL,
    institution_price REAL,
    institution_value REAL,
    currency TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, security_id),
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_holdings_item ON holdings(item_id);

CREATE TABLE IF NOT EXISTS liabilities (
    account_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    item_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    last_payment_amount REAL,
    last_payment_date TEXT,
    next_payment_due_date TEXT,
    minimum_payment_amount REAL,
    outstanding_balance REAL,
    apr REAL,
    detail TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (account_id, kind),
    FOREIGN KEY (item_id) REFERENCES items(item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_liabilities_item ON liabilities(item_id);

CREATE TABLE IF NOT EXISTS balance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    current_balance REAL,
    available_balance REAL,
    currency TEXT,
    captured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_balsnap_account ON balance_snapshots(account_id);
CREATE INDEX IF NOT EXISTS idx_balsnap_captured ON balance_snapshots(captured_at);

CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT,
    provider TEXT,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    added INTEGER NOT NULL DEFAULT 0,
    modified INTEGER NOT NULL DEFAULT 0,
    removed INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS merchant_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_type TEXT NOT NULL DEFAULT 'substring',
    pattern TEXT NOT NULL,
    normalized_name TEXT,
    category TEXT,
    priority INTEGER NOT NULL DEFAULT 100,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS category_overrides (
    scope TEXT NOT NULL,
    key TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);
"""


class FinanceStore:
    """Thread-safe handle to the local finance SQLite database."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.path = Path(db_path) if db_path else default_db_path()
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Lock down the finance/ directory (also holds the token file).
                secure_parent_dir(self.path)
                conn = sqlite3.connect(str(self.path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._apply_pragmas(conn)
                conn.executescript(SCHEMA_SQL)
                self._conn = conn
            return self._conn

    @staticmethod
    def _apply_pragmas(conn: sqlite3.Connection) -> None:
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="finance.db")
        except Exception:
            # Network filesystems and minimal test envs may reject WAL; the
            # default rollback journal is correct, just slower.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def __enter__(self) -> "FinanceStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def upsert_item(
        self,
        item_id: str,
        *,
        provider: str,
        institution_id: Optional[str] = None,
        institution_name: Optional[str] = None,
        environment: Optional[str] = None,
        status: str = "active",
    ) -> None:
        now = _utcnow()
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO items (item_id, provider, institution_id, institution_name,
                                   environment, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    provider=excluded.provider,
                    institution_id=COALESCE(excluded.institution_id, items.institution_id),
                    institution_name=COALESCE(excluded.institution_name, items.institution_name),
                    environment=COALESCE(excluded.environment, items.environment),
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (item_id, provider, institution_id, institution_name,
                 environment, status, now, now),
            )

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            row = conn.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
        return dict(row) if row else None

    def list_items(self, provider: Optional[str] = None, *, include_removed: bool = False) -> List[Dict[str, Any]]:
        conn = self._connection()
        clauses, params = [], []
        if provider:
            clauses.append("provider=?")
            params.append(provider)
        if not include_removed:
            clauses.append("status != 'disconnected'")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = conn.execute(f"SELECT * FROM items{where} ORDER BY created_at", params).fetchall()
        return [dict(r) for r in rows]

    def set_item_cursor(self, item_id: str, cursor: Optional[str]) -> None:
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                "UPDATE items SET transactions_cursor=?, updated_at=? WHERE item_id=?",
                (cursor, _utcnow(), item_id),
            )

    def get_item_cursor(self, item_id: str) -> Optional[str]:
        item = self.get_item(item_id)
        return item.get("transactions_cursor") if item else None

    def mark_item_synced(self, item_id: str, *, error: Optional[str] = None) -> None:
        conn = self._connection()
        now = _utcnow()
        with self._lock, conn:
            conn.execute(
                "UPDATE items SET last_sync_at=?, last_error=?, updated_at=? WHERE item_id=?",
                (now, error, now, item_id),
            )

    def set_item_status(self, item_id: str, status: str) -> None:
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                "UPDATE items SET status=?, updated_at=? WHERE item_id=?",
                (status, _utcnow(), item_id),
            )

    def delete_item(self, item_id: str) -> None:
        """Hard-delete an item and all of its data (used by ``--purge``)."""
        conn = self._connection()
        with self._lock, conn:
            conn.execute("DELETE FROM items WHERE item_id=?", (item_id,))

    # ------------------------------------------------------------------
    # Accounts + balances
    # ------------------------------------------------------------------

    def upsert_accounts(self, item_id: str, provider: str, accounts: Iterable[Dict[str, Any]]) -> int:
        now = _utcnow()
        conn = self._connection()
        count = 0
        with self._lock, conn:
            for acct in accounts:
                account_id = acct.get("account_id")
                if not account_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO accounts (account_id, item_id, provider, name, official_name,
                                          mask, type, subtype, currency, current_balance,
                                          available_balance, credit_limit, removed, last_synced_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(account_id) DO UPDATE SET
                        item_id=excluded.item_id,
                        provider=excluded.provider,
                        name=excluded.name,
                        official_name=excluded.official_name,
                        mask=excluded.mask,
                        type=excluded.type,
                        subtype=excluded.subtype,
                        currency=excluded.currency,
                        current_balance=excluded.current_balance,
                        available_balance=excluded.available_balance,
                        credit_limit=excluded.credit_limit,
                        removed=0,
                        last_synced_at=excluded.last_synced_at
                    """,
                    (
                        account_id, item_id, provider, acct.get("name"),
                        acct.get("official_name"), acct.get("mask"), acct.get("type"),
                        acct.get("subtype"), acct.get("currency"),
                        _to_float(acct.get("current_balance")),
                        _to_float(acct.get("available_balance")),
                        _to_float(acct.get("credit_limit")), now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO balance_snapshots (account_id, item_id, provider,
                                                   current_balance, available_balance,
                                                   currency, captured_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        account_id, item_id, provider,
                        _to_float(acct.get("current_balance")),
                        _to_float(acct.get("available_balance")),
                        acct.get("currency"), now,
                    ),
                )
                count += 1
        return count

    def get_accounts(self, *, provider: Optional[str] = None, include_removed: bool = False) -> List[Dict[str, Any]]:
        conn = self._connection()
        clauses, params = [], []
        if provider:
            clauses.append("provider=?")
            params.append(provider)
        if not include_removed:
            clauses.append("removed=0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = conn.execute(f"SELECT * FROM accounts{where} ORDER BY name", params).fetchall()
        return [dict(r) for r in rows]

    def net_worth(self) -> Dict[str, Any]:
        """Return assets, liabilities, and net worth grouped by currency.

        Liability-type accounts (credit, loan) carry a positive ``current``
        balance representing what is owed, so they subtract from net worth.
        """
        accounts = self.get_accounts()
        by_currency: Dict[str, Dict[str, float]] = {}
        for acct in accounts:
            currency = (acct.get("currency") or "USD").upper()
            bucket = by_currency.setdefault(currency, {"assets": 0.0, "liabilities": 0.0})
            balance = acct.get("current_balance")
            if balance is None:
                continue
            if (acct.get("type") or "").lower() in _LIABILITY_TYPES:
                bucket["liabilities"] += abs(balance)
            else:
                bucket["assets"] += balance
        result = {}
        for currency, bucket in by_currency.items():
            net = round(bucket["assets"] - bucket["liabilities"], 2)
            result[currency] = {
                "assets": round(bucket["assets"], 2),
                "liabilities": round(bucket["liabilities"], 2),
                "net_worth": net,
            }
        return result

    def net_worth_history(self, *, limit_days: int = 90) -> List[Dict[str, Any]]:
        """Return per-day net-worth points from balance snapshots.

        Uses the last snapshot per account per day so a day with multiple
        syncs counts each account once.
        """
        conn = self._connection()
        with self._lock:
            rows = conn.execute(
                """
                SELECT a.type AS type, a.currency AS currency,
                       substr(b.captured_at, 1, 10) AS day,
                       b.current_balance AS current_balance,
                       b.captured_at AS captured_at,
                       b.account_id AS account_id
                FROM balance_snapshots b
                JOIN accounts a ON a.account_id = b.account_id
                ORDER BY b.captured_at
                """
            ).fetchall()
        # Keep the latest snapshot per (day, account).
        latest: Dict[tuple, sqlite3.Row] = {}
        for row in rows:
            latest[(row["day"], row["account_id"])] = row
        per_day: Dict[str, float] = {}
        for (day, _account), row in latest.items():
            balance = row["current_balance"]
            if balance is None:
                continue
            signed = -abs(balance) if (row["type"] or "").lower() in _LIABILITY_TYPES else balance
            per_day[day] = per_day.get(day, 0.0) + signed
        points = [{"date": day, "net_worth": round(val, 2)} for day, val in sorted(per_day.items())]
        return points[-limit_days:] if limit_days else points

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def apply_transactions(
        self,
        item_id: str,
        provider: str,
        *,
        added: Iterable[Dict[str, Any]] = (),
        modified: Iterable[Dict[str, Any]] = (),
        removed: Iterable[str] = (),
    ) -> Dict[str, int]:
        now = _utcnow()
        conn = self._connection()
        counts = {"added": 0, "modified": 0, "removed": 0}
        with self._lock, conn:
            for txn in added:
                self._upsert_transaction(conn, item_id, provider, txn, now)
                counts["added"] += 1
            for txn in modified:
                self._upsert_transaction(conn, item_id, provider, txn, now)
                counts["modified"] += 1
            for txn_id in removed:
                if not txn_id:
                    continue
                conn.execute(
                    "UPDATE transactions SET removed=1, removed_at=?, updated_at=? WHERE transaction_id=?",
                    (now, now, txn_id),
                )
                counts["removed"] += 1
        return counts

    @staticmethod
    def _upsert_transaction(
        conn: sqlite3.Connection, item_id: str, provider: str, txn: Dict[str, Any], now: str
    ) -> None:
        transaction_id = txn.get("transaction_id")
        if not transaction_id:
            return
        raw = txn.get("raw")
        raw_json = json.dumps(raw, ensure_ascii=False) if raw is not None else None
        conn.execute(
            """
            INSERT INTO transactions (transaction_id, account_id, item_id, provider, amount,
                                      currency, date, datetime, name, merchant_name,
                                      category_primary, category_detailed, pending,
                                      payment_channel, removed, removed_at, raw,
                                      first_seen_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
            ON CONFLICT(transaction_id) DO UPDATE SET
                account_id=excluded.account_id,
                amount=excluded.amount,
                currency=excluded.currency,
                date=excluded.date,
                datetime=excluded.datetime,
                name=excluded.name,
                merchant_name=excluded.merchant_name,
                category_primary=excluded.category_primary,
                category_detailed=excluded.category_detailed,
                pending=excluded.pending,
                payment_channel=excluded.payment_channel,
                removed=0,
                removed_at=NULL,
                raw=excluded.raw,
                updated_at=excluded.updated_at
            """,
            (
                transaction_id, txn.get("account_id"), item_id, provider,
                _to_float(txn.get("amount")), txn.get("currency"), txn.get("date"),
                txn.get("datetime"), txn.get("name"), txn.get("merchant_name"),
                txn.get("category_primary"), txn.get("category_detailed"),
                1 if txn.get("pending") else 0, txn.get("payment_channel"),
                raw_json, now, now,
            ),
        )

    def iter_transactions(
        self,
        *,
        account_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search: Optional[str] = None,
        include_pending: bool = True,
        include_removed: bool = False,
        limit: Optional[int] = None,
        order: str = "date_desc",
    ) -> List[Dict[str, Any]]:
        conn = self._connection()
        clauses, params = [], []
        if not include_removed:
            clauses.append("removed=0")
        if account_id:
            clauses.append("account_id=?")
            params.append(account_id)
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)
        if not include_pending:
            clauses.append("pending=0")
        if search:
            clauses.append("(LOWER(name) LIKE ? OR LOWER(merchant_name) LIKE ?)")
            needle = f"%{search.lower()}%"
            params.extend([needle, needle])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order_sql = {
            "date_desc": "date DESC, datetime DESC",
            "date_asc": "date ASC, datetime ASC",
            "amount_desc": "ABS(amount) DESC",
        }.get(order, "date DESC")
        sql = f"SELECT * FROM transactions{where} ORDER BY {order_sql}"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Investments + liabilities
    # ------------------------------------------------------------------

    def upsert_securities(self, securities: Iterable[Dict[str, Any]]) -> int:
        now = _utcnow()
        conn = self._connection()
        count = 0
        with self._lock, conn:
            for sec in securities:
                security_id = sec.get("security_id")
                if not security_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO securities (security_id, name, ticker_symbol, type,
                                            close_price, currency, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(security_id) DO UPDATE SET
                        name=excluded.name,
                        ticker_symbol=excluded.ticker_symbol,
                        type=excluded.type,
                        close_price=excluded.close_price,
                        currency=excluded.currency,
                        updated_at=excluded.updated_at
                    """,
                    (
                        security_id, sec.get("name"), sec.get("ticker_symbol"),
                        sec.get("type"), _to_float(sec.get("close_price")),
                        sec.get("currency"), now,
                    ),
                )
                count += 1
        return count

    def upsert_holdings(self, item_id: str, provider: str, holdings: Iterable[Dict[str, Any]]) -> int:
        now = _utcnow()
        conn = self._connection()
        count = 0
        with self._lock, conn:
            for holding in holdings:
                account_id = holding.get("account_id")
                security_id = holding.get("security_id")
                if not account_id or not security_id:
                    continue
                conn.execute(
                    """
                    INSERT INTO holdings (account_id, security_id, item_id, provider, quantity,
                                          cost_basis, institution_price, institution_value,
                                          currency, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id, security_id) DO UPDATE SET
                        item_id=excluded.item_id,
                        provider=excluded.provider,
                        quantity=excluded.quantity,
                        cost_basis=excluded.cost_basis,
                        institution_price=excluded.institution_price,
                        institution_value=excluded.institution_value,
                        currency=excluded.currency,
                        updated_at=excluded.updated_at
                    """,
                    (
                        account_id, security_id, item_id, provider,
                        _to_float(holding.get("quantity")), _to_float(holding.get("cost_basis")),
                        _to_float(holding.get("institution_price")),
                        _to_float(holding.get("institution_value")),
                        holding.get("currency"), now,
                    ),
                )
                count += 1
        return count

    def get_holdings(self) -> List[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            rows = conn.execute(
                """
                SELECT h.*, s.name AS security_name, s.ticker_symbol AS ticker_symbol,
                       s.type AS security_type, a.name AS account_name
                FROM holdings h
                LEFT JOIN securities s ON s.security_id = h.security_id
                LEFT JOIN accounts a ON a.account_id = h.account_id
                ORDER BY h.institution_value DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_liabilities(self, item_id: str, provider: str, liabilities: Iterable[Dict[str, Any]]) -> int:
        now = _utcnow()
        conn = self._connection()
        count = 0
        with self._lock, conn:
            for liability in liabilities:
                account_id = liability.get("account_id")
                kind = liability.get("kind")
                if not account_id or not kind:
                    continue
                detail = liability.get("detail")
                detail_json = json.dumps(detail, ensure_ascii=False) if detail is not None else None
                conn.execute(
                    """
                    INSERT INTO liabilities (account_id, kind, item_id, provider,
                                             last_payment_amount, last_payment_date,
                                             next_payment_due_date, minimum_payment_amount,
                                             outstanding_balance, apr, detail, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id, kind) DO UPDATE SET
                        item_id=excluded.item_id,
                        provider=excluded.provider,
                        last_payment_amount=excluded.last_payment_amount,
                        last_payment_date=excluded.last_payment_date,
                        next_payment_due_date=excluded.next_payment_due_date,
                        minimum_payment_amount=excluded.minimum_payment_amount,
                        outstanding_balance=excluded.outstanding_balance,
                        apr=excluded.apr,
                        detail=excluded.detail,
                        updated_at=excluded.updated_at
                    """,
                    (
                        account_id, kind, item_id, provider,
                        _to_float(liability.get("last_payment_amount")),
                        liability.get("last_payment_date"),
                        liability.get("next_payment_due_date"),
                        _to_float(liability.get("minimum_payment_amount")),
                        _to_float(liability.get("outstanding_balance")),
                        _to_float(liability.get("apr")), detail_json, now,
                    ),
                )
                count += 1
        return count

    def get_liabilities(self) -> List[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            rows = conn.execute(
                """
                SELECT l.*, a.name AS account_name
                FROM liabilities l
                LEFT JOIN accounts a ON a.account_id = l.account_id
                ORDER BY l.next_payment_due_date
                """
            ).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            if record.get("detail"):
                try:
                    record["detail"] = json.loads(record["detail"])
                except (TypeError, ValueError):
                    pass
            result.append(record)
        return result

    # ------------------------------------------------------------------
    # Sync history
    # ------------------------------------------------------------------

    def record_sync_run(
        self,
        *,
        item_id: Optional[str],
        provider: str,
        started_at: str,
        finished_at: str,
        status: str,
        added: int = 0,
        modified: int = 0,
        removed: int = 0,
        error: Optional[str] = None,
    ) -> None:
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO sync_runs (item_id, provider, started_at, finished_at, status,
                                       added, modified, removed, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (item_id, provider, started_at, finished_at, status, added, modified, removed, error),
            )

    def last_sync_run(self) -> Optional[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM sync_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Merchant rules + category overrides
    # ------------------------------------------------------------------

    def add_merchant_rule(
        self,
        *,
        pattern: str,
        match_type: str = "substring",
        normalized_name: Optional[str] = None,
        category: Optional[str] = None,
        priority: int = 100,
    ) -> int:
        conn = self._connection()
        with self._lock, conn:
            cur = conn.execute(
                """
                INSERT INTO merchant_rules (match_type, pattern, normalized_name, category,
                                            priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (match_type, pattern, normalized_name, category, priority, _utcnow()),
            )
            return int(cur.lastrowid)

    def list_merchant_rules(self) -> List[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM merchant_rules ORDER BY priority, id"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_merchant_rule(self, rule_id: int) -> bool:
        conn = self._connection()
        with self._lock, conn:
            cur = conn.execute("DELETE FROM merchant_rules WHERE id=?", (rule_id,))
            return cur.rowcount > 0

    def set_category_override(self, *, scope: str, key: str, category: str) -> None:
        conn = self._connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO category_overrides (scope, key, category, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scope, key) DO UPDATE SET category=excluded.category
                """,
                (scope, key.lower(), category, _utcnow()),
            )

    def list_category_overrides(self) -> List[Dict[str, Any]]:
        conn = self._connection()
        with self._lock:
            rows = conn.execute("SELECT * FROM category_overrides").fetchall()
        return [dict(r) for r in rows]
