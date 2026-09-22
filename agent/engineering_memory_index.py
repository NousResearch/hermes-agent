"""Rebuildable SQLite librarian for approved engineering-memory records."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from agent.engineering_memory_schema import EngineeringMemoryRecord


@dataclass(frozen=True)
class EngineeringMemoryResult:
    record_id: str
    title: str
    summary: str
    repository: str
    component: str
    task_type: str
    status: str
    verified_head: str | None
    source_label: str
    related_record_ids: tuple[str, ...]
    conflict_set: tuple[str, ...]
    score: float
    reason_code: str

    def to_mapping(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class IndexHealth:
    ok: bool
    fts5: bool
    record_count: int
    issues: tuple[str, ...] = ()


class EngineeringMemoryIndex:
    def __init__(self, db_path: Path, *, fts_supported: bool | None = None):
        self.db_path = Path(db_path).expanduser()
        self._fts_override = fts_supported
        self.fts_supported = False
        self.initialize()

    def _connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS engineering_memory_records ("
                "record_id TEXT PRIMARY KEY, status TEXT NOT NULL, repository TEXT NOT NULL, "
                "component TEXT NOT NULL, task_type TEXT NOT NULL, verified_head TEXT, "
                "payload TEXT NOT NULL)"
            )
            requested = self._fts_override is not False
            if requested:
                try:
                    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS engineering_memory_fts USING fts5(record_id UNINDEXED, content)")
                    self.fts_supported = True
                except sqlite3.OperationalError:
                    self.fts_supported = False

    def rebuild(self, records: Iterable[EngineeringMemoryRecord]) -> None:
        materialized = [record for record in records if record.status == "approved"]
        with self._connection() as connection:
            connection.execute("DELETE FROM engineering_memory_records")
            if self.fts_supported:
                connection.execute("DELETE FROM engineering_memory_fts")
            for record in materialized:
                payload = json.dumps(record.to_mapping(), sort_keys=True, ensure_ascii=False)
                connection.execute(
                    "INSERT INTO engineering_memory_records(record_id,status,repository,component,task_type,verified_head,payload) VALUES(?,?,?,?,?,?,?)",
                    (record.record_id, record.status, record.repository, record.component, record.task_type, record.verified_head, payload),
                )
                if self.fts_supported:
                    content = " ".join((record.title, record.summary, record.body, record.component, record.task_type, *record.tags, *record.symptoms))
                    connection.execute("INSERT INTO engineering_memory_fts(record_id,content) VALUES(?,?)", (record.record_id, content))

    def _candidate_ids(self, query: str) -> set[str] | None:
        if not self.fts_supported or not query.strip():
            return None
        terms = [re.sub(r"[^\w-]", "", item) for item in query.split()]
        terms = [item for item in terms if item]
        if not terms:
            return None
        match = " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        try:
            with self._connection() as connection:
                rows = connection.execute("SELECT record_id FROM engineering_memory_fts WHERE content MATCH ?", (match,)).fetchall()
            return {row[0] for row in rows} or None
        except sqlite3.OperationalError:
            return None

    def search(
        self,
        query: str,
        *,
        repository: str | None = None,
        workspace: str | None = None,
        component: str | None = None,
        task_type: str | None = None,
        tags: Sequence[str] = (),
        verified_only: bool = True,
        as_of_head: str | None = None,
        limit: int = 8,
        char_budget: int = 6_000,
    ) -> list[EngineeringMemoryResult]:
        if limit <= 0 or char_budget <= 0:
            return []
        candidate_ids = self._candidate_ids(query)
        terms = {term.lower() for term in re.findall(r"[\w-]+", query)}
        wanted_tags = {tag.lower() for tag in tags}
        with self._connection() as connection:
            rows = connection.execute("SELECT payload FROM engineering_memory_records").fetchall()
        scored: list[EngineeringMemoryResult] = []
        for (payload,) in rows:
            record = EngineeringMemoryRecord.from_mapping(json.loads(payload))
            if candidate_ids is not None and record.record_id not in candidate_ids:
                continue
            if repository and record.repository != repository:
                continue
            if workspace and record.workspace != workspace:
                continue
            if component and record.component != component:
                continue
            if task_type and record.task_type != task_type:
                continue
            if wanted_tags and not wanted_tags.issubset({tag.lower() for tag in record.tags}):
                continue
            if verified_only and (record.status != "approved" or not record.verified_head):
                continue
            if as_of_head and record.verified_head != as_of_head:
                continue
            searchable = " ".join((record.title, record.summary, record.body, record.component, record.task_type, *record.tags, *record.symptoms)).lower()
            matches = sum(1 for term in terms if term in searchable)
            if query.strip() and matches == 0:
                continue
            score = matches / max(len(terms), 1)
            scored.append(
                EngineeringMemoryResult(
                    record_id=record.record_id,
                    title=record.title,
                    summary=record.summary,
                    repository=record.repository,
                    component=record.component,
                    task_type=record.task_type,
                    status=record.status,
                    verified_head=record.verified_head,
                    source_label=record.source_label,
                    related_record_ids=record.related_record_ids,
                    conflict_set=record.conflict_set,
                    score=score,
                    reason_code="fts_match" if self.fts_supported else "token_overlap",
                )
            )
        scored.sort(key=lambda item: (-item.score, item.record_id))
        output: list[EngineeringMemoryResult] = []
        used = 0
        for result in scored[:limit]:
            size = len(json.dumps(result.to_mapping(), ensure_ascii=False))
            if output and used + size > char_budget:
                break
            if not output and size > char_budget:
                # Keep the envelope bounded even for an unusually long title/summary.
                result = EngineeringMemoryResult(**{**result.__dict__, "title": result.title[: max(0, char_budget // 4)], "summary": result.summary[: max(0, char_budget // 4)]})
                size = len(json.dumps(result.to_mapping(), ensure_ascii=False))
                if size > char_budget:
                    continue
            output.append(result)
            used += size
        return output

    def verify(self) -> IndexHealth:
        issues: list[str] = []
        try:
            with self._connection() as connection:
                count = connection.execute("SELECT COUNT(*) FROM engineering_memory_records").fetchone()[0]
                integrity = connection.execute("PRAGMA integrity_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    issues.append("integrity_check_failed")
        except sqlite3.DatabaseError as exc:
            return IndexHealth(False, self.fts_supported, 0, (f"database_error:{type(exc).__name__}",))
        return IndexHealth(not issues, self.fts_supported, int(count), tuple(issues))
