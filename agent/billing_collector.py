"""Server-side billing collector for Hermes API calls.

Captures factual per-call ``usage`` + ``cost`` on every API response (success or
failure) into a SEPARATE SQLite database (``~/.hermes/billing/billing.db``), never
into the core ``state.db``. The collector is gated behind a feature flag
(``telemetry.billing_collector.enabled``, default OFF) plus an env override
(``HERMES_BILLING_COLLECTOR``); when disabled, ``record_*`` is a no-op and never
touches the database, the agent turn, or any secrets.

No API keys / PII are stored. The only identity-bearing columns are the Hermes
profile (``agent``), the kanban task id (``task_id``) and the request id — none of
which are secret.

Usage is pulled from the normalized ``usage`` object; the factual dollar cost
(``actual_cost_usd``) is read from the provider's raw ``usage.cost`` field (OpenRouter
emits it on every response, including streaming). When a cost cannot be read, the row
is still recorded with ``has_usage`` / cost NULL so absence is a separate, queryable
fact rather than silence.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("agent.billing_collector")

# ---------------------------------------------------------------------------
# Configuration / feature flag
# ---------------------------------------------------------------------------

_TTL_DAYS_DEFAULT = 90
_ENV_FLAG = "HERMES_BILLING_COLLECTOR"
_CONFIG_PATH = ("telemetry", "billing_collector", "enabled")


def is_collector_enabled() -> bool:
    """Return True iff billing collection is active.

    Precedence: env override ``HERMES_BILLING_COLLECTOR`` (1/true/on | 0/false/off)
    wins; otherwise the ``telemetry.billing_collector.enabled`` config key; default OFF.
    """
    env = (os.getenv(_ENV_FLAG) or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off", "disabled"):
        return False
    try:
        # Lazy import: avoids a module-level dependency on the config subsystem.
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        node = cfg
        for key in _CONFIG_PATH:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                return False
        return bool(node)
    except Exception:
        logger.debug("billing_collector: config read failed; collector disabled", exc_info=True)
        return False


def _current_profile() -> str:
    """Best-effort Hermes profile name (never secret)."""
    return os.environ.get("HERMES_PROFILE", "") or "default"


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


@dataclass
class BillingRecord:
    """One captured API call (success or failure)."""

    request_id: Optional[str]
    timestamp: str  # ISO-8601 UTC
    model: str
    provider: str
    agent: str
    task_id: Optional[str] = None
    turn_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    actual_cost_usd: Optional[Decimal] = None
    currency: str = "USD"
    billing_source: str = "server_response"
    has_usage: bool = False
    is_error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Collector (SQLite, separate file, WAL)
# ---------------------------------------------------------------------------


def _default_db_path() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    directory = Path(home) / "billing"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "billing.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS billing_records (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id        TEXT,
    timestamp         TEXT NOT NULL,
    model             TEXT NOT NULL DEFAULT '',
    provider          TEXT NOT NULL DEFAULT '',
    agent             TEXT NOT NULL DEFAULT '',
    task_id           TEXT,
    turn_id           TEXT,
    prompt_tokens     INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens  INTEGER,
    actual_cost_usd   REAL,
    currency          TEXT NOT NULL DEFAULT 'USD',
    billing_source    TEXT NOT NULL DEFAULT 'server_response',
    has_usage         INTEGER NOT NULL DEFAULT 0,
    is_error          INTEGER NOT NULL DEFAULT 0,
    error_type        TEXT,
    error_message     TEXT
);
CREATE INDEX IF NOT EXISTS idx_billing_ts        ON billing_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_billing_model     ON billing_records(model);
CREATE INDEX IF NOT EXISTS idx_billing_agent     ON billing_records(agent);
CREATE INDEX IF NOT EXISTS idx_billing_task      ON billing_records(task_id);
CREATE INDEX IF NOT EXISTS idx_billing_day       ON billing_records(substr(timestamp, 1, 10));

CREATE TABLE IF NOT EXISTS billing_cleanup_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at      TEXT NOT NULL,
    deleted_rows INTEGER NOT NULL,
    cutoff_day  TEXT NOT NULL
);
"""


class BillingCollector:
    """Append-only store of per-call billing facts in its own SQLite file."""

    def __init__(self, db_path: Optional[os.PathLike | str] = None, *, ttl_days: int = _TTL_DAYS_DEFAULT):
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._ttl_days = int(ttl_days)
        self._lock = threading.RLock()
        try:
            self._init_db()
        except Exception:
            logger.warning("billing_collector: failed to init db at %s", self._db_path, exc_info=True)

    # -- lifecycle ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # -- writes -------------------------------------------------------------
    def record_call(self, rec: BillingRecord) -> None:
        """Persist a single billing record. Never raises into the caller."""
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO billing_records (
                        request_id, timestamp, model, provider, agent, task_id, turn_id,
                        prompt_tokens, cache_read_tokens, cache_write_tokens, completion_tokens,
                        reasoning_tokens, actual_cost_usd, currency, billing_source,
                        has_usage, is_error, error_type, error_message
                    ) VALUES (?,?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?)
                    """,
                    (
                        rec.request_id,
                        rec.timestamp,
                        rec.model,
                        rec.provider,
                        rec.agent,
                        rec.task_id,
                        rec.turn_id,
                        rec.prompt_tokens,
                        rec.cache_read_tokens,
                        rec.cache_write_tokens,
                        rec.completion_tokens,
                        rec.reasoning_tokens,
                        None if rec.actual_cost_usd is None else str(rec.actual_cost_usd),
                        rec.currency,
                        rec.billing_source,
                        1 if rec.has_usage else 0,
                        1 if rec.is_error else 0,
                        rec.error_type,
                        rec.error_message,
                    ),
                )
                conn.commit()
        except Exception:
            logger.warning("billing_collector: record_call failed", exc_info=True)

    def record_error(
        self,
        *,
        model: str,
        provider: str,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a failed API call (no usage available)."""
        self.record_call(
            BillingRecord(
                request_id=request_id,
                timestamp=timestamp or _now_iso(),
                model=model,
                provider=provider,
                agent=agent or _current_profile(),
                task_id=task_id,
                turn_id=turn_id,
                has_usage=False,
                is_error=True,
                error_type=error_type,
                error_message=_truncate(error_message, 2000),
            )
        )

    # -- reads --------------------------------------------------------------
    def query(
        self,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        day: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        has_usage: Optional[bool] = None,
        is_error: Optional[bool] = None,
        limit: int = 1000,
        order: str = "ASC",
    ) -> List[Dict[str, Any]]:
        """Return rows matching the given filters (read-only)."""
        clauses: List[str] = []
        params: List[Any] = []
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if day is not None:
            clauses.append("substr(timestamp, 1, 10) = ?")
            params.append(day)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if has_usage is not None:
            clauses.append("has_usage = ?")
            params.append(1 if has_usage else 0)
        if is_error is not None:
            clauses.append("is_error = ?")
            params.append(1 if is_error else 0)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order_sql = "DESC" if order.strip().upper().startswith("DESC") else "ASC"
        limit_sql = max(1, min(int(limit), 100000))
        sql = f"SELECT * FROM billing_records{where} ORDER BY id {order_sql} LIMIT {limit_sql}"
        rows: List[Dict[str, Any]] = []
        try:
            with self._lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.execute(sql, params):
                    rows.append(self._row_to_dict(r))
        except Exception:
            logger.warning("billing_collector: query failed", exc_info=True)
        return rows

    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        cost = d.get("actual_cost_usd")
        if cost is not None:
            try:
                d["actual_cost_usd"] = Decimal(str(cost))
            except (InvalidOperation, ValueError):
                d["actual_cost_usd"] = None
        d["has_usage"] = bool(d.get("has_usage"))
        d["is_error"] = bool(d.get("is_error"))
        return d

    def stats(
        self,
        *,
        group_by: str = "day",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        agent: Optional[str] = None,
        task_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        include_no_usage: bool = True,
    ) -> List[Dict[str, Any]]:
        """Aggregate cost + tokens over a grouping key.

        ``group_by`` is one of: ``day``, ``model``, ``agent``, ``task_id``,
        ``provider``. Returns one row per group with sums and call counts.
        """
        valid_groups = {
            "day": "substr(timestamp, 1, 10)",
            "model": "model",
            "agent": "agent",
            "task_id": "task_id",
            "provider": "provider",
        }
        key_expr = valid_groups.get(group_by, "substr(timestamp, 1, 10)")
        clauses: List[str] = []
        params: List[Any] = []
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if provider is not None:
            clauses.append("provider = ?")
            params.append(provider)
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if not include_no_usage:
            clauses.append("has_usage = 1")
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT {key_expr} AS bucket, "
            "COUNT(*) AS calls, "
            "SUM(has_usage) AS calls_with_usage, "
            "SUM(is_error) AS calls_error, "
            "SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens, "
            "SUM(COALESCE(cache_read_tokens, 0)) AS cache_read_tokens, "
            "SUM(COALESCE(cache_write_tokens, 0)) AS cache_write_tokens, "
            "SUM(COALESCE(completion_tokens, 0)) AS completion_tokens, "
            "SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN actual_cost_usd ELSE 0 END) AS total_cost_usd "
            f"FROM billing_records{where} GROUP BY bucket ORDER BY bucket ASC"
        )
        out: List[Dict[str, Any]] = []
        try:
            with self._lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.execute(sql, params):
                    row = dict(r)
                    tc = row.get("total_cost_usd")
                    try:
                        row["total_cost_usd"] = Decimal(str(tc)) if tc is not None else Decimal("0")
                    except (InvalidOperation, ValueError):
                        row["total_cost_usd"] = Decimal("0")
                    row["calls"] = int(row.get("calls") or 0)
                    row["calls_with_usage"] = int(row.get("calls_with_usage") or 0)
                    row["calls_error"] = int(row.get("calls_error") or 0)
                    out.append(row)
        except Exception:
            logger.warning("billing_collector: stats failed", exc_info=True)
        return out

    # -- retention ----------------------------------------------------------
    def cleanup(self, *, ttl_days: Optional[int] = None) -> int:
        """Delete rows older than ``ttl_days`` (default self._ttl_days). Returns deleted count."""
        ttl = int(ttl_days) if ttl_days is not None else self._ttl_days
        cutoff = _day_n_ago_iso(ttl)
        deleted = 0
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM billing_records WHERE timestamp < ?", (cutoff,)
                )
                deleted = cur.rowcount
                conn.execute(
                    "INSERT INTO billing_cleanup_log (ran_at, deleted_rows, cutoff_day) VALUES (?,?,?)",
                    (_now_iso(), deleted, cutoff),
                )
                conn.commit()
        except Exception:
            logger.warning("billing_collector: cleanup failed", exc_info=True)
        return deleted

    def total_cost_usd(self) -> Decimal:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN actual_cost_usd ELSE 0 END) "
                    "FROM billing_records"
                ).fetchone()
                val = row[0] if row else None
                return Decimal(str(val)) if val is not None else Decimal("0")
        except Exception:
            return Decimal("0")


# ---------------------------------------------------------------------------
# Module-level singleton + response ingestion
# ---------------------------------------------------------------------------

_singleton: Optional[BillingCollector] = None
_singleton_lock = threading.Lock()


def get_collector() -> BillingCollector:
    """Lazily construct the process-wide collector (never raises)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = BillingCollector()
    return _singleton


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day_n_ago_iso(n: int) -> str:
    target = datetime.now(timezone.utc).date().toordinal() - n
    return datetime.fromordinal(target).strftime("%Y-%m-%dT00:00:00Z")


def _truncate(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _extract_raw_cost(usage: Any) -> Optional[Decimal]:
    """Pull the factual provider cost from the raw usage object.

    OpenRouter emits ``usage.cost`` (float USD) on every response. Some providers
    nest it under ``usage.cost_details.total``. Returns None when absent.
    """
    if usage is None:
        return None
    # dict shape
    if isinstance(usage, dict):
        raw = usage.get("cost")
        if raw is None:
            details = usage.get("cost_details")
            if isinstance(details, dict):
                raw = details.get("total")
        if raw is None:
            raw = usage.get("total_cost")
    else:
        raw = getattr(usage, "cost", None)
        if raw is None:
            details = getattr(usage, "cost_details", None)
            if isinstance(details, dict):
                raw = details.get("total")
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return None


def record_response(
    agent: Any,
    response: Any,
    *,
    api_request_id: Optional[str] = None,
    task_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    api_duration: Optional[float] = None,
    api_start_time: Optional[float] = None,
) -> None:
    """Capture billing facts from a successful API response. No-op when disabled.

    Safe to call from the agent hot path: every failure path is swallowed and only
    logged at debug level. Never raises.
    """
    if not is_collector_enabled():
        return
    try:
        if response is None:
            return
        model = (getattr(response, "model", None) or getattr(agent, "model", "")) or ""
        provider = getattr(agent, "provider", "") or ""
        # Normalize tokens through the existing pricing normalizer.
        usage = getattr(response, "usage", None)
        prompt = cache_read = cache_write = completion = reasoning = None
        actual_cost: Optional[Decimal] = None
        has_usage = bool(usage)
        if usage is not None:
            try:
                from agent.usage_pricing import normalize_usage

                cu = normalize_usage(usage, provider=provider, api_mode=getattr(agent, "api_mode", None))
                prompt = cu.prompt_tokens
                cache_read = cu.cache_read_tokens or None
                cache_write = cu.cache_write_tokens or None
                completion = cu.output_tokens
                reasoning = cu.reasoning_tokens or None
            except Exception:
                logger.debug("billing_collector: usage normalization failed", exc_info=True)
            actual_cost = _extract_raw_cost(usage)
        currency = "USD"
        cost_obj = getattr(usage, "cost", None) if not isinstance(usage, dict) else (usage or {}).get("cost")
        if isinstance(cost_obj, dict):
            currency = cost_obj.get("currency", "USD") or "USD"

        rec = BillingRecord(
            request_id=api_request_id,
            timestamp=_now_iso(),
            model=str(model),
            provider=str(provider),
            agent=_current_profile(),
            task_id=task_id,
            turn_id=turn_id,
            prompt_tokens=prompt,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            actual_cost_usd=actual_cost,
            currency=currency,
            billing_source="server_response",
            has_usage=has_usage,
        )
        get_collector().record_call(rec)
    except Exception:
        logger.debug("billing_collector: record_response failed", exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point: `python -m agent.billing_collector [--group-by day|model|agent|task_id] [--model ...] ...`
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="billing_collector", description="Hermes billing collector report")
    p.add_argument("--db", default=None, help="Path to billing.db (default: ~/.hermes/billing/billing.db)")
    p.add_argument("--group-by", choices=["day", "model", "agent", "task_id", "provider"], default="day")
    p.add_argument("--model")
    p.add_argument("--provider")
    p.add_argument("--agent")
    p.add_argument("--task-id", dest="task_id")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--no-usage", action="store_true", help="Exclude rows without usage")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args(argv)

    if not is_collector_enabled():
        print("billing collector disabled (set telemetry.billing_collector.enabled: true or HERMES_BILLING_COLLECTOR=1)")
        return 0

    col = BillingCollector(args.db) if args.db else get_collector()
    rows = col.stats(
        group_by=args.group_by,
        model=args.model,
        provider=args.provider,
        agent=args.agent,
        task_id=args.task_id,
        since=args.since,
        until=args.until,
        include_no_usage=not args.no_usage,
    )
    if args.json:
        import json

        print(json.dumps(
            [dict(r, total_cost_usd=str(r["total_cost_usd"])) for r in rows],
            indent=2,
        ))
        return 0

    # Human table
    print(f"Billing stats by {args.group_by}  (total cost: ${col.total_cost_usd():.4f})")
    header = f"{args.group_by:<16} {'calls':>7} {'w/usage':>8} {'err':>5} {'cost_usd':>12} {'prompt_tok':>11}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cost = r.get("total_cost_usd") or Decimal("0")
        print(
            f"{str(r.get('bucket')):<16} {r.get('calls',0):>7} {r.get('calls_with_usage',0):>8} "
            f"{r.get('calls_error',0):>5} {float(cost):>12.4f} {int(r.get('prompt_tokens') or 0):>11}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
