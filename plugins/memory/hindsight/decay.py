"""Optional, profile-scoped soft decay for Hindsight recall results.

Hindsight owns the durable memory graph and its PostgreSQL store.  This module
keeps only a small local ledger of Hindsight result IDs, so Hermes can apply a
LivingMemory-style importance/access policy without reaching into Hindsight's
database or deleting source memories.

The ledger is deliberately a soft layer: memories below the recall threshold
are omitted from Hermes' recall context, but remain intact in Hindsight and can
be recovered by disabling decay or marking them as exempt.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


UTC = timezone.utc


@dataclass(frozen=True)
class DecayPolicy:
    """LivingMemory-inspired decay parameters."""

    rate_per_day: float = 0.01
    access_decay_window_days: int = 30
    initial_importance: float = 0.5
    min_importance: float = 0.2
    cleanup_age_days: int = 60
    exempt_tags: tuple[str, ...] = (
        "permanent",
        "memory:permanent",
        "hindsight:permanent",
    )


def policy_from_config(config: Mapping[str, Any]) -> DecayPolicy:
    """Build a bounded decay policy from provider configuration."""

    def _float(name: str, default: float) -> float:
        try:
            value = float(config.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and value >= 0 else default

    def _int(name: str, default: int) -> int:
        try:
            value = int(config.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    raw_tags = config.get(
        "decay_exempt_tags", "permanent,memory:permanent,hindsight:permanent"
    )
    if isinstance(raw_tags, str):
        tags = tuple(tag.strip() for tag in raw_tags.split(",") if tag.strip())
    else:
        tags = tuple(str(tag).strip() for tag in (raw_tags or []) if str(tag).strip())

    return DecayPolicy(
        rate_per_day=min(_float("decay_rate_per_day", 0.01), 1.0),
        access_decay_window_days=_int("decay_access_window_days", 30),
        initial_importance=min(_float("decay_initial_importance", 0.5), 1.0),
        min_importance=min(_float("decay_min_importance", 0.2), 1.0),
        cleanup_age_days=_int("decay_cleanup_age_days", 60),
        exempt_tags=tags,
    )


def _as_utc(value: Any, fallback: datetime | None = None) -> datetime:
    """Parse a Hindsight timestamp, falling back to *fallback* or now."""

    if isinstance(value, datetime):
        result = value
    elif value:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            result = fallback or datetime.now(UTC)
    else:
        result = fallback or datetime.now(UTC)
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def result_identity(result: Any) -> str:
    """Return a stable ID for a Hindsight result.

    Hindsight 0.6.x exposes ``RecallResult.id``.  The content hash fallback
    keeps the optional layer compatible with older clients and lightweight
    test doubles that only provide ``text``.  Results with the same fallback
    ``document_id`` and text intentionally share one ledger row, so recalling
    either also refreshes the other's preservation window.
    """

    value = _field(result, "id")
    if value:
        return str(value)
    document_id = str(_field(result, "document_id", "") or "")
    text = str(_field(result, "text", "") or "")
    digest = hashlib.sha256(f"{document_id}\0{text}".encode("utf-8")).hexdigest()
    return f"content:{digest}"


def _result_created_at(result: Any, now: datetime) -> datetime:
    metadata = _field(result, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return _as_utc(
        _field(result, "mentioned_at")
        or _field(result, "occurred_start")
        or metadata.get("retained_at"),
        fallback=now,
    )


def _result_is_exempt(result: Any, policy: DecayPolicy) -> bool:
    tags = _field(result, "tags", []) or []
    metadata = _field(result, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    normalized_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    exempt_tags = {tag.strip().lower() for tag in policy.exempt_tags if tag.strip()}
    source = str(metadata.get("source", "")).strip().lower()
    return bool(normalized_tags & exempt_tags) or source == "permanent"


class HindsightDecayStore:
    """SQLite ledger used to soft-filter stale Hindsight recall results."""

    def __init__(
        self, path: str | Path, bank_id: str, policy: DecayPolicy | None = None
    ):
        self.path = Path(path)
        self.bank_id = str(bank_id)
        self.policy = policy or DecayPolicy()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_decay (
                    bank_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT,
                    permanent INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (bank_id, memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_decay_importance
                    ON memory_decay (bank_id, importance);
                CREATE TABLE IF NOT EXISTS memory_decay_meta (
                    bank_id TEXT PRIMARY KEY,
                    last_decay_at TEXT NOT NULL
                );
                """
            )

    def _decay_conn(self, conn: sqlite3.Connection, now: datetime) -> int:
        # Serialize the meta read, importance updates, and timestamp write.
        # A deferred transaction would let two gateways sharing HERMES_HOME
        # read the same last_decay_at before either claims the writer slot,
        # risking duplicate elapsed-time decay (or SQLITE_BUSY_SNAPSHOT).
        conn.execute("BEGIN IMMEDIATE")
        meta = conn.execute(
            "SELECT last_decay_at FROM memory_decay_meta WHERE bank_id = ?",
            (self.bank_id,),
        ).fetchone()
        if meta is None:
            conn.execute(
                "INSERT INTO memory_decay_meta (bank_id, last_decay_at) VALUES (?, ?)",
                (self.bank_id, now.isoformat()),
            )
            return 0

        last_decay = _as_utc(meta["last_decay_at"], fallback=now)
        elapsed_days = max((now - last_decay).total_seconds() / 86400.0, 0.0)
        if elapsed_days <= 0:
            return 0

        affected = 0
        rows = conn.execute(
            """SELECT memory_id, importance, last_accessed_at, permanent
               FROM memory_decay WHERE bank_id = ?""",
            (self.bank_id,),
        ).fetchall()
        for row in rows:
            if row["permanent"]:
                continue
            effective_rate = self.policy.rate_per_day * elapsed_days
            if row["last_accessed_at"]:
                last_accessed = _as_utc(row["last_accessed_at"], fallback=now)
                if (
                    now - last_accessed
                ).total_seconds() < self.policy.access_decay_window_days * 86400:
                    effective_rate *= 0.5
            importance = max(0.0, float(row["importance"]) - effective_rate)
            if abs(importance - float(row["importance"])) > 1e-12:
                conn.execute(
                    "UPDATE memory_decay SET importance = ? WHERE bank_id = ? AND memory_id = ?",
                    (importance, self.bank_id, row["memory_id"]),
                )
                affected += 1

        conn.execute(
            "UPDATE memory_decay_meta SET last_decay_at = ? WHERE bank_id = ?",
            (now.isoformat(), self.bank_id),
        )
        return affected

    def apply_decay(self, now: datetime | None = None) -> int:
        """Apply elapsed-time decay once and return the number of changed rows."""

        current = _as_utc(now)
        with self._connect() as conn:
            return self._decay_conn(conn, current)

    def _ensure_conn(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        created_at: datetime,
        permanent: bool,
        now: datetime,
    ) -> sqlite3.Row:
        row = conn.execute(
            """SELECT memory_id, created_at, importance, access_count,
                      last_accessed_at, permanent
               FROM memory_decay WHERE bank_id = ? AND memory_id = ?""",
            (self.bank_id, memory_id),
        ).fetchone()
        if row is not None:
            if permanent and not row["permanent"]:
                conn.execute(
                    "UPDATE memory_decay SET permanent = 1 WHERE bank_id = ? AND memory_id = ?",
                    (self.bank_id, memory_id),
                )
                row = conn.execute(
                    """SELECT memory_id, created_at, importance, access_count,
                              last_accessed_at, permanent
                       FROM memory_decay WHERE bank_id = ? AND memory_id = ?""",
                    (self.bank_id, memory_id),
                ).fetchone()
            return row

        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)
        importance = (
            self.policy.initial_importance
            if permanent
            else max(
                0.0,
                self.policy.initial_importance - self.policy.rate_per_day * age_days,
            )
        )
        conn.execute(
            """INSERT INTO memory_decay
               (bank_id, memory_id, created_at, importance, permanent)
               VALUES (?, ?, ?, ?, ?)""",
            (
                self.bank_id,
                memory_id,
                created_at.isoformat(),
                importance,
                int(permanent),
            ),
        )
        return conn.execute(
            """SELECT memory_id, created_at, importance, access_count,
                      last_accessed_at, permanent
               FROM memory_decay WHERE bank_id = ? AND memory_id = ?""",
            (self.bank_id, memory_id),
        ).fetchone()

    def filter_results(
        self, results: Iterable[Any], now: datetime | None = None
    ) -> list[Any]:
        """Return recall results that have not softly decayed away.

        Access is recorded only for results that survive the filter.  This
        makes a frequently recalled memory self-preserving while a stale
        result that is already below the threshold remains decayed.
        """

        current = _as_utc(now)
        result_list = list(results or [])
        if not result_list:
            return []

        visible: list[Any] = []
        with self._connect() as conn:
            self._decay_conn(conn, current)
            to_touch: list[str] = []
            for result in result_list:
                memory_id = result_identity(result)
                created_at = _result_created_at(result, current)
                permanent = _result_is_exempt(result, self.policy)
                row = self._ensure_conn(conn, memory_id, created_at, permanent, current)
                age_days = max(
                    (current - _as_utc(row["created_at"], current)).total_seconds()
                    / 86400.0,
                    0.0,
                )
                stale = (
                    not row["permanent"]
                    and float(row["importance"]) < self.policy.min_importance
                    and age_days >= self.policy.cleanup_age_days
                )
                if stale:
                    continue
                visible.append(result)
                to_touch.append(memory_id)

            for memory_id in to_touch:
                conn.execute(
                    """UPDATE memory_decay
                       SET access_count = access_count + 1, last_accessed_at = ?
                       WHERE bank_id = ? AND memory_id = ?""",
                    (current.isoformat(), self.bank_id, memory_id),
                )
        return visible
