"""SQLite store for confidence memory items."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .schemas import (
    Confidence,
    Layer,
    MemoryItem,
    MemorySource,
    Scope,
    SourceKind,
    Status,
    now_utc,
    parse_ttl,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    layer TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_reinforced_at TEXT,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    ttl TEXT NOT NULL,
    status TEXT NOT NULL,
    superseded_by TEXT,
    scope TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_confmem_active ON memory_items(status, scope, confidence);
CREATE INDEX IF NOT EXISTS idx_confmem_layer ON memory_items(layer);
"""


class ConfidenceMemoryStore:
    """Local SQLite store for layered, confidence-scored memory."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            from hermes_state import apply_wal_with_fallback
            apply_wal_with_fallback(self.conn, db_label="confidence_memory.db")
        except Exception:
            pass
        self.conn.executescript(SCHEMA)
        self._secure_database_files()

    def _secure_database_files(self) -> None:
        """Keep memory data and SQLite sidecars readable only by the owner."""
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        ):
            if path.exists():
                path.chmod(0o600)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ConfidenceMemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _to_row(self, item: MemoryItem) -> tuple:
        return (
            item.id,
            item.layer.value,
            item.statement,
            item.confidence.value,
            json.dumps([s.to_json() for s in item.sources], ensure_ascii=False),
            item.created_at.isoformat(),
            item.last_reinforced_at.isoformat() if item.last_reinforced_at else None,
            item.reinforcement_count,
            item.ttl,
            item.status.value,
            item.superseded_by,
            item.scope.value,
        )

    def _from_row(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            layer=Layer(row["layer"]),
            statement=row["statement"],
            confidence=Confidence(row["confidence"]),
            sources=[MemorySource.from_json(x) for x in json.loads(row["sources_json"])],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_reinforced_at=(
                datetime.fromisoformat(row["last_reinforced_at"])
                if row["last_reinforced_at"]
                else None
            ),
            reinforcement_count=int(row["reinforcement_count"]),
            ttl=row["ttl"],
            status=Status(row["status"]),
            superseded_by=row["superseded_by"],
            scope=Scope(row["scope"]),
        )

    def add_item(self, item: MemoryItem) -> str:
        self.conn.execute(
            """
            INSERT INTO memory_items
              (id, layer, statement, confidence, sources_json, created_at,
               last_reinforced_at, reinforcement_count, ttl, status,
               superseded_by, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._to_row(item),
        )
        self.conn.commit()
        return item.id

    def add(
        self,
        *,
        statement: str,
        layer: Layer | str,
        confidence: Confidence | str,
        sources: list[MemorySource],
        created_at: Optional[datetime] = None,
        ttl: str = "",
        scope: Scope | str = Scope.INJECTION,
    ) -> str:
        item = MemoryItem(
            statement=statement,
            layer=Layer(layer),
            confidence=Confidence(confidence),
            sources=sources,
            created_at=created_at or now_utc(),
            ttl=ttl,
            scope=Scope(scope),
        )
        return self.add_item(item)

    def get(self, item_id: str) -> MemoryItem:
        row = self.conn.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise KeyError(item_id)
        return self._from_row(row)

    def list_items(self, include_inactive: bool = False) -> list[MemoryItem]:
        sql = "SELECT * FROM memory_items"
        if not include_inactive:
            sql += " WHERE status IN ('active','stale')"
        sql += " ORDER BY created_at ASC"
        return [self._from_row(row) for row in self.conn.execute(sql)]

    def search(self, query: str, *, include_inactive: bool = False, limit: int = 10) -> list[MemoryItem]:
        query = (query or "").strip().lower()
        if not query:
            return self.list_items(include_inactive=include_inactive)[:limit]
        candidates = self.list_items(include_inactive=include_inactive)
        terms = [t for t in query.split() if t]
        scored: list[tuple[int, MemoryItem]] = []
        for item in candidates:
            haystack = item.statement.lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].created_at))
        return [item for _, item in scored[:limit]]

    def confirm(self, item_id: str, source: MemorySource) -> None:
        item = self.get(item_id)
        sources = item.sources + [source]
        self.conn.execute(
            """
            UPDATE memory_items
            SET confidence = 'confirmed', status = 'active', scope = 'injection',
                sources_json = ?, last_reinforced_at = ?, reinforcement_count = reinforcement_count + 1
            WHERE id = ?
            """,
            (
                json.dumps([s.to_json() for s in sources], ensure_ascii=False),
                source.observed_at.isoformat(),
                item_id,
            ),
        )
        self.conn.commit()

    def delete(self, item_id: str) -> None:
        self.conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self.conn.commit()

    def refresh_statuses(self, as_of: Optional[datetime] = None) -> None:
        as_of = as_of or now_utc()
        for item in self.list_items(include_inactive=True):
            if item.status not in {Status.ACTIVE, Status.STALE}:
                continue
            age_anchor = item.last_reinforced_at or item.created_at
            ttl = parse_ttl(item.ttl)
            age = as_of - age_anchor
            new_status = item.status
            if age >= ttl:
                new_status = Status.EXPIRED
            elif age >= ttl / 2:
                new_status = Status.STALE
            if new_status != item.status:
                self.conn.execute(
                    "UPDATE memory_items SET status = ? WHERE id = ?",
                    (new_status.value, item.id),
                )
        self.conn.commit()

    def resolve_user_stated_conflict(
        self,
        old_item_id: str,
        new_statement: str,
        source: MemorySource,
        *,
        layer: Optional[Layer] = None,
    ) -> str:
        old = self.get(old_item_id)
        if source.kind not in {SourceKind.USER_STATED, SourceKind.USER_CONFIRMED}:
            raise ValueError("only user-stated/user-confirmed observations can auto-supersede")
        new_id = self.add(
            statement=new_statement,
            layer=layer or old.layer,
            confidence=Confidence.CONFIRMED,
            sources=[source],
            scope=Scope.INJECTION,
        )
        self.conn.execute(
            "UPDATE memory_items SET status = 'superseded', superseded_by = ? WHERE id = ?",
            (new_id, old_item_id),
        )
        self.conn.commit()
        return new_id

    def select_for_injection(self, query: str = "", *, profile_limit: int = 15) -> list[MemoryItem]:
        self.refresh_statuses()
        selected: list[MemoryItem] = []
        profile_count = 0
        for item in self.list_items(include_inactive=False):
            if item.status != Status.ACTIVE:
                continue
            if item.scope != Scope.INJECTION:
                continue
            if item.confidence == Confidence.TENTATIVE:
                continue
            if item.layer == Layer.PROFILE:
                if profile_count >= profile_limit:
                    continue
                profile_count += 1
                selected.append(item)
            elif item.layer == Layer.ONGOING_THEME and query:
                selected.append(item)
            elif item.layer == Layer.UNRESOLVED_QUESTION:
                selected.append(item)
        return selected

    @staticmethod
    def format_for_prompt(items: Iterable[MemoryItem]) -> str:
        return "\n".join(
            f"- [{item.confidence.value}] ({item.layer.value}) {item.statement}"
            for item in items
        )
