"""SQLite persistence for trade episodes."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from hermes_trader.loop.scheduler import CycleResult

from hermes_trader.config import TRADER_HOME_SUBDIR
from hermes_trader.memory.summary import build_market_summary, compute_embedding_id

SCHEMA_VERSION = 2
EPISODES_DB_NAME = "trade_episodes.db"


@dataclass
class TradeEpisode:
    episode_id: str
    timestamp: str
    chain: str
    strategy_tag: Optional[str]
    gate_decision: str
    gate_reason: Optional[str]
    intent: dict[str, Any]
    decision: Optional[dict[str, Any]]
    execution: Optional[dict[str, Any]]
    market_summary: dict[str, Any]
    liquidity_usd: Optional[float]
    token_address: str
    tx_hash: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_usd: Optional[float] = None
    holding_hours: Optional[float] = None
    embedding_id: Optional[str] = None
    created_at: Optional[str] = None
    reflection: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "chain": self.chain,
            "strategy_tag": self.strategy_tag,
            "gate_decision": self.gate_decision,
            "gate_reason": self.gate_reason,
            "intent": self.intent,
            "decision": self.decision,
            "execution": self.execution,
            "market_summary": self.market_summary,
            "liquidity_usd": self.liquidity_usd,
            "token_address": self.token_address,
            "tx_hash": self.tx_hash,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_usd": self.pnl_usd,
            "holding_hours": self.holding_hours,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at,
            "reflection": self.reflection,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TradeEpisode":
        return cls(
            episode_id=row["episode_id"],
            timestamp=row["timestamp"],
            chain=row["chain"] or "",
            strategy_tag=row["strategy_tag"],
            gate_decision=row["gate_decision"],
            gate_reason=row["gate_reason"],
            intent=json.loads(row["intent_json"]),
            decision=json.loads(row["decision_json"]) if row["decision_json"] else None,
            execution=json.loads(row["execution_json"]) if row["execution_json"] else None,
            market_summary=json.loads(row["market_summary_json"] or "{}"),
            liquidity_usd=row["liquidity_usd"],
            token_address=row["token_address"] or "",
            tx_hash=row["tx_hash"],
            entry_price=row["entry_price"],
            exit_price=row["exit_price"],
            pnl_usd=row["pnl_usd"],
            holding_hours=row["holding_hours"],
            embedding_id=row["embedding_id"],
            created_at=row["created_at"],
            reflection=_parse_reflection_row(row),
        )


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_episodes_db_path() -> Path:
    import os

    override = os.environ.get("HERMES_TRADER_EPISODES_DB", "").strip()
    if override:
        return Path(override)
    return _hermes_home() / TRADER_HOME_SUBDIR / EPISODES_DB_NAME


def _schema_sql_path() -> Path:
    return Path(__file__).resolve().parent / "schema.sql"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_reflection_row(row: sqlite3.Row) -> Optional[dict[str, Any]]:
    if "reflection_json" not in row.keys():
        return None
    raw = row["reflection_json"]
    if not raw:
        return None
    data = json.loads(raw)
    return data if isinstance(data, dict) else None


def _extract_tx_hash(execution: Any) -> Optional[str]:
    if execution is None or execution.payload is None:
        return None
    payload = execution.payload
    if isinstance(payload, dict):
        for key in ("tx_hash", "transaction_hash", "hash"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
        result = payload.get("result")
        if isinstance(result, dict):
            for key in ("tx_hash", "transaction_hash", "hash"):
                val = result.get(key)
                if isinstance(val, str) and val:
                    return val
    return None


class EpisodeStore:
    """SQLite-backed trade episode ledger."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path is not None else default_episodes_db_path()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def migrate(self) -> int:
        """Apply schema migrations. Returns schema version."""
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = conn.execute(
                "SELECT MAX(version) AS v FROM schema_migrations"
            ).fetchone()
            current = int(row["v"] or 0)
            if current < 1:
                schema_sql = _schema_sql_path().read_text(encoding="utf-8")
                conn.executescript(schema_sql)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (1, _utc_now_iso()),
                )
                current = 1

            if current < 2:
                v2_path = Path(__file__).resolve().parent / "schema_v2.sql"
                if v2_path.is_file():
                    for statement in v2_path.read_text(encoding="utf-8").split(";"):
                        stmt = statement.strip()
                        if stmt:
                            try:
                                conn.execute(stmt)
                            except sqlite3.OperationalError as exc:
                                if "duplicate column" not in str(exc).lower():
                                    raise
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (2, _utc_now_iso()),
                )
                current = 2

            conn.commit()
            return current

    def record_cycle(self, result: "CycleResult") -> TradeEpisode:
        self.migrate()
        episode = episode_from_cycle(result)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_episodes (
                    episode_id, timestamp, chain, strategy_tag, gate_decision,
                    gate_reason, intent_json, decision_json, execution_json,
                    market_summary_json, liquidity_usd, token_address, tx_hash,
                    entry_price, exit_price, pnl_usd, holding_hours, embedding_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.timestamp,
                    episode.chain,
                    episode.strategy_tag,
                    episode.gate_decision,
                    episode.gate_reason,
                    _json_dumps(episode.intent),
                    _json_dumps(episode.decision) if episode.decision else None,
                    _json_dumps(episode.execution) if episode.execution else None,
                    _json_dumps(episode.market_summary),
                    episode.liquidity_usd,
                    episode.token_address,
                    episode.tx_hash,
                    episode.entry_price,
                    episode.exit_price,
                    episode.pnl_usd,
                    episode.holding_hours,
                    episode.embedding_id,
                    episode.created_at or _utc_now_iso(),
                ),
            )
            conn.commit()
        return episode

    def list_episodes(self, *, limit: int = 100) -> List[TradeEpisode]:
        self.migrate()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trade_episodes
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [TradeEpisode.from_row(row) for row in rows]

    def get_episode(self, episode_id: str) -> Optional[TradeEpisode]:
        self.migrate()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM trade_episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
        return TradeEpisode.from_row(row) if row else None

    def count(self) -> int:
        self.migrate()
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM trade_episodes").fetchone()
        return int(row["c"] if row else 0)

    def update_outcome(
        self,
        episode_id: str,
        *,
        pnl_usd: float,
        exit_price: Optional[float] = None,
        entry_price: Optional[float] = None,
        holding_hours: Optional[float] = None,
    ) -> None:
        self.migrate()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE trade_episodes
                SET pnl_usd = ?, exit_price = COALESCE(?, exit_price),
                    entry_price = COALESCE(?, entry_price),
                    holding_hours = COALESCE(?, holding_hours)
                WHERE episode_id = ?
                """,
                (pnl_usd, exit_price, entry_price, holding_hours, episode_id),
            )
            conn.commit()

    def save_reflection(self, episode_id: str, reflection: dict[str, Any]) -> None:
        self.migrate()
        with self.connect() as conn:
            conn.execute(
                "UPDATE trade_episodes SET reflection_json = ? WHERE episode_id = ?",
                (_json_dumps(reflection), episode_id),
            )
            conn.commit()

    def list_closed_episodes(self, *, limit: int = 200) -> List[TradeEpisode]:
        self.migrate()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trade_episodes
                WHERE pnl_usd IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [TradeEpisode.from_row(row) for row in rows]

    def list_pending_reflection(self, *, limit: int = 50) -> List[TradeEpisode]:
        self.migrate()
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trade_episodes
                WHERE pnl_usd IS NOT NULL
                  AND (reflection_json IS NULL OR reflection_json = '')
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [TradeEpisode.from_row(row) for row in rows]


def episode_from_cycle(result: "CycleResult") -> TradeEpisode:
    """Build a TradeEpisode record from a completed trading cycle."""
    intent = result.intent
    decision = result.decision
    execution = result.execution
    summary = build_market_summary(result.market_state)
    embedding_id = compute_embedding_id(summary)

    gate_decision = "APPROVE" if decision.approved else "REJECT"
    gate_reason = decision.reason_code.value if decision.reason_code else None

    return TradeEpisode(
        episode_id=uuid.uuid4().hex,
        timestamp=result.market_state.captured_at or _utc_now_iso(),
        chain=intent.chain or result.market_state.chain,
        strategy_tag=intent.strategy_tag,
        gate_decision=gate_decision,
        gate_reason=gate_reason,
        intent={
            "action": intent.action,
            "chain": intent.chain,
            "token_address": intent.token_address,
            "size_usd": intent.size_usd,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
            "strategy_tag": intent.strategy_tag,
            "pool_liquidity_usd": intent.pool_liquidity_usd,
            "slippage_bps": intent.slippage_bps,
        },
        decision=decision.to_dict(),
        execution=execution.to_dict() if execution else None,
        market_summary=summary,
        liquidity_usd=intent.pool_liquidity_usd,
        token_address=intent.token_address,
        tx_hash=_extract_tx_hash(execution),
        embedding_id=embedding_id,
        created_at=_utc_now_iso(),
    )