"""SQLite persistence for the Recall memory provider."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Hermes plugin package import
    from .audit import hash_event, verify_audit_chain
    from .redaction import redact_text
    from .schema import SCHEMA_SQL, SCHEMA_VERSION
except ImportError:  # Standalone CLI/test import from repository root
    from audit import hash_event, verify_audit_chain
    from redaction import redact_text
    from schema import SCHEMA_SQL, SCHEMA_VERSION


_QUERY_STOPWORDS = {
    "a",
    "an",
    "are",
    "is",
    "if",
    "needed",
    "only",
    "reply",
    "the",
    "using",
    "was",
    "what",
    "your",
}


def _query_terms(query: str) -> list[str]:
    """Extract high-signal, FTS-safe query terms from user text."""
    terms = re.findall(r"[\w.-]+", query.lower(), flags=re.UNICODE)
    return [term for term in terms if term and term not in _QUERY_STOPWORDS]


def _subject_key(content: str) -> str:
    """Return a conservative semantic key for durable-memory mirror dedupe.

    Built-in memory replacements usually keep a stable leading label
    (``Recall Memory:``, ``Paperclip debugging:``). Use that label when present;
    otherwise fall back to the first few significant terms. This intentionally
    does not power general archive dedupe — only trusted built-in mirrors.
    """
    text = redact_text(content).strip().lower()
    label = text.split(":", 1)[0].strip()
    if 3 <= len(label) <= 80 and re.search(r"[a-z0-9]", label):
        return "label:" + " ".join(_query_terms(label))
    return "terms:" + " ".join(_query_terms(text)[:6])


def _fts_query(query: str) -> str:
    """Convert arbitrary user text into a safe FTS5 query.

    Raw paths like ``E:\\Projects`` or ``/mnt/e`` contain FTS syntax
    characters. Tokenize and quote terms so search never raises syntax errors.
    """
    quoted = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in _query_terms(query)]
    return " OR ".join(quoted)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _specific_marker_count(text: str) -> int:
    """Count local specificity signals useful for quality ranking.

    This is intentionally deterministic and offline: paths, backticked names,
    hashes, issue-like markers, and digit-bearing tokens usually make a memory
    more actionable than vague prose.
    """
    markers = 0
    markers += len(re.findall(r"`[^`]{2,120}`", text))
    markers += len(re.findall(r"(?:/mnt/[\w./-]+|/[\w./-]{8,}|[A-Za-z]:\\\\[\w.\\\\-]+)", text))
    markers += len(re.findall(r"\b[0-9a-f]{7,40}\b", text.lower()))
    markers += len(re.findall(r"\b[A-Z][A-Z0-9_/-]{5,}\b", text))
    markers += len([term for term in _query_terms(text) if any(ch.isdigit() for ch in term)])
    return markers


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RecallStore:
    """Profile-scoped SQLite store for episodes, observations, and audit events."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        # WAL + synchronous=NORMAL avoids one fsync per tiny archive write while
        # preserving normal SQLite atomicity. Recall is an additive archive, not
        # the authoritative MEMORY.md/USER.md store; FULL made stress/dogfood
        # writes painfully slow on WSL/filesystems with expensive syncs.
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def add_episode(
        self,
        *,
        session_id: str,
        project_path: str,
        user_text: str,
        assistant_text: str,
    ) -> str:
        episode_id = str(uuid.uuid4())
        safe_user_text = redact_text(user_text)[:4000]
        safe_assistant_text = redact_text(assistant_text)[:8000]
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO episodes(id, session_id, project_path, user_text, assistant_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (episode_id, session_id, project_path, safe_user_text, safe_assistant_text, utc_now()),
            )
            self.conn.commit()
        return episode_id

    def add_observation(
        self,
        *,
        content: str,
        type: str = "fact",
        scope: str = "project",
        trust_level: str = "archive",
        confidence: float = 0.5,
        importance: float = 0.5,
        status: str = "active",
        source_session_id: str = "",
        project_path: str = "",
        expires_at: str | None = None,
        supersedes: str | None = None,
    ) -> str:
        observation_id = str(uuid.uuid4())
        safe_content = redact_text(content)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO observations(
                    id, type, scope, trust_level, confidence, importance, status,
                    content, redacted_content, source_session_id, project_path,
                    created_at, expires_at, supersedes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    type,
                    scope,
                    trust_level,
                    float(confidence),
                    float(importance),
                    status,
                    safe_content,
                    safe_content,
                    source_session_id,
                    project_path,
                    utc_now(),
                    expires_at,
                    supersedes,
                ),
            )
            self.conn.commit()
        return observation_id

    def add_builtin_mirror_observation(
        self,
        *,
        content: str,
        type: str,
        scope: str,
        source_session_id: str = "",
        project_path: str = "",
        replace: bool = False,
    ) -> str:
        """Add a trusted built-in memory mirror without accumulating stale duplicates.

        Exact active duplicates are returned as-is. Replacement writes supersede
        the newest active built-in mirror with the same conservative subject key,
        so normal search/current views show the latest durable-memory fact while
        export/audit history still preserves the old row.
        """
        safe_content = redact_text(content)
        key = _subject_key(safe_content)
        with self._lock:
            exact = self.conn.execute(
                """
                SELECT id FROM observations
                WHERE trust_level='builtin-mirror' AND status='active'
                  AND type=? AND scope=? AND project_path=? AND redacted_content=?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (type, scope, project_path, safe_content),
            ).fetchone()

            same_subject_rows: list[sqlite3.Row] = []
            if replace and key:
                rows = self.conn.execute(
                    """
                    SELECT id, redacted_content FROM observations
                    WHERE trust_level='builtin-mirror' AND status='active'
                      AND type=? AND scope=? AND project_path=?
                    ORDER BY created_at DESC, rowid DESC
                    """,
                    (type, scope, project_path),
                ).fetchall()
                same_subject_rows = [
                    row for row in rows if _subject_key(str(row["redacted_content"] or "")) == key
                ]

            if exact:
                exact_id = str(exact["id"])
                if replace:
                    self._reject_redundant_builtin_mirrors(
                        keep_id=exact_id,
                        rows=same_subject_rows,
                        reason="exact built-in mirror already exists for replaced memory subject",
                    )
                return exact_id

            supersedes = str(same_subject_rows[0]["id"]) if same_subject_rows else None

        new_id = self.add_observation(
            content=safe_content,
            type=type,
            scope=scope,
            trust_level="builtin-mirror",
            confidence=0.95,
            importance=0.85,
            status="active",
            source_session_id=source_session_id,
            project_path=project_path,
            supersedes=supersedes,
        )
        if replace:
            self._reject_redundant_builtin_mirrors(
                keep_id=new_id,
                rows=same_subject_rows,
                reason="replaced built-in memory subject with newer mirror",
            )
        return new_id

    def _reject_redundant_builtin_mirrors(
        self,
        *,
        keep_id: str,
        rows: list[sqlite3.Row],
        reason: str,
    ) -> None:
        """Quarantine active same-subject built-in mirrors after a replacement.

        The newest/current row remains active. Older same-subject mirrors stay in
        export/audit history, but cannot reappear in current/search results if a
        middle superseder is later rejected during curation.
        """
        redundant_ids = [str(row["id"]) for row in rows if str(row["id"]) != keep_id]
        if not redundant_ids:
            return
        with self._lock:
            for old_id in redundant_ids:
                self.conn.execute("UPDATE observations SET status='rejected' WHERE id=?", (old_id,))
                self.append_audit_event(
                    "result",
                    "builtin_mirror_superseded",
                    "observation",
                    old_id,
                    {"ok": True, "reason": reason, "superseded_by": keep_id},
                )
            self.conn.commit()

    def get_observation(self, observation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM observations WHERE id=?", (observation_id,)).fetchone()
        return dict(row) if row else None

    def mark_observation_status(self, observation_id: str, status: str) -> bool:
        with self._lock:
            cur = self.conn.execute("UPDATE observations SET status=? WHERE id=?", (status, observation_id))
            self.conn.commit()
            return cur.rowcount > 0

    def list_candidates(
        self,
        *,
        status: str = "candidate",
        type: str | None = None,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses = ["status = ?"]
        params: list[Any] = [status]
        if type:
            clauses.append("type = ?")
            params.append(type)
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        params.append(int(limit))
        where = " AND ".join(clauses)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM observations WHERE {where} ORDER BY importance DESC, created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def _not_expired_clause(self, alias: str = "o") -> str:
        return f"({alias}.expires_at IS NULL OR {alias}.expires_at = '' OR {alias}.expires_at > ?)"

    def _active_superseder_clause(self, alias: str = "s") -> str:
        return (
            f"{alias}.supersedes = o.id AND {alias}.status NOT IN ('rejected', 'deleted') "
            f"AND ({alias}.expires_at IS NULL OR {alias}.expires_at = '' OR {alias}.expires_at > ?)"
        )

    def _redacted_observation_item(self, row: sqlite3.Row | dict[str, Any], *, query_terms: list[str] | None = None) -> dict[str, Any]:
        item = dict(row)
        searchable = " ".join(
            str(item.get(field) or "") for field in ("redacted_content", "content", "type", "scope", "project_path")
        ).lower()
        if query_terms is not None:
            item["matched_query_terms"] = [term for term in query_terms if term in searchable]
        item["content"] = redact_text(item.get("redacted_content") or item.get("content") or "")
        item["redacted_content"] = item["content"]
        if item.get("supersedes_content"):
            item["supersedes_content"] = redact_text(str(item["supersedes_content"]))
        return item

    def search_observations(
        self,
        query: str,
        *,
        limit: int = 5,
        scope: str | None = None,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        fts = _fts_query(query)
        if not fts:
            return []
        now = utc_now()
        clauses = [
            "o.status NOT IN ('rejected', 'deleted')",
            self._not_expired_clause("o"),
            f"NOT EXISTS (SELECT 1 FROM observations s WHERE {self._active_superseder_clause('s')})",
        ]
        params: list[Any] = [fts, now, now]
        if scope:
            clauses.append("o.scope = ?")
            params.append(scope)
        if project_path:
            clauses.append("o.project_path = ?")
            params.append(project_path)
        params.append(int(limit))
        where = " AND ".join(clauses)
        query_terms = _query_terms(query)
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT o.*, bm25(observations_fts) AS score, superseded.redacted_content AS supersedes_content
                FROM observations_fts
                JOIN observations o ON o.rowid = observations_fts.rowid
                LEFT JOIN observations superseded ON superseded.id = o.supersedes
                WHERE observations_fts MATCH ? AND {where}
                ORDER BY score ASC, o.importance DESC, o.confidence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._redacted_observation_item(row, query_terms=query_terms) for row in rows]

    def current_observations(
        self,
        *,
        limit: int = 50,
        scope: str | None = None,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return active, non-expired, non-superseded archive observations."""
        now = utc_now()
        clauses = [
            "o.status = 'active'",
            self._not_expired_clause("o"),
            f"NOT EXISTS (SELECT 1 FROM observations s WHERE {self._active_superseder_clause('s')})",
        ]
        params: list[Any] = [now, now]
        if scope:
            clauses.append("o.scope = ?")
            params.append(scope)
        if project_path:
            clauses.append("o.project_path = ?")
            params.append(project_path)
        params.append(int(limit))
        where = " AND ".join(clauses)
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT o.*, superseded.redacted_content AS supersedes_content
                FROM observations o
                LEFT JOIN observations superseded ON superseded.id = o.supersedes
                WHERE {where}
                ORDER BY o.importance DESC, o.confidence DESC, o.created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._redacted_observation_item(row) for row in rows]

    def _quality_rank_item(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = self._redacted_observation_item(row)
        content = str(item.get("content") or "")
        lower = content.lower()
        reasons: list[str] = []
        score = (float(item.get("confidence") or 0.0) * 0.35) + (float(item.get("importance") or 0.0) * 0.35)

        trust = str(item.get("trust_level") or "")
        if trust == "builtin-mirror":
            score += 0.15
            reasons.append("trusted mirror")
        elif trust == "archive":
            reasons.append("archive evidence")

        obs_type = str(item.get("type") or "")
        if obs_type in {"fact", "preference"}:
            score += 0.1
            reasons.append("durable fact shape")
        elif obs_type == "episode":
            score -= 0.15
            reasons.append("episode trace")
        elif obs_type == "delegation":
            score -= 0.05
            reasons.append("delegation trace")

        status = str(item.get("status") or "")
        if status == "candidate":
            score += 0.05
            reasons.append("candidate for curation")
        elif status == "promoted":
            score += 0.03
            reasons.append("already promoted")
        elif status in {"rejected", "deleted"}:
            score -= 0.4
            reasons.append(f"{status} status")
        if item.get("supersedes"):
            score += 0.02
            reasons.append("supersedes older row")

        length = len(content)
        if 60 <= length <= 800:
            score += 0.05
            reasons.append("concise content")
        elif length < 40:
            score -= 0.08
            reasons.append("too short")
        elif length > 1600:
            score -= 0.08
            reasons.append("too long")

        marker_count = _specific_marker_count(content)
        if marker_count >= 2:
            score += 0.1
            reasons.append("specific markers")
        elif marker_count == 1:
            score += 0.04
            reasons.append("one specific marker")

        if ":" in content[:80]:
            score += 0.08
            reasons.append("stable subject label")
        if "user asked:" in lower and "assistant answered:" in lower:
            score -= 0.12
            reasons.append("transcript summary")
        terms = _query_terms(content)
        if terms:
            most_common = max(terms.count(term) for term in set(terms))
            if most_common >= 4:
                score -= 0.08
                reasons.append("repetitive wording")

        item["quality_score"] = round(_clamp01(score), 3)
        item["quality_reasons"] = reasons
        if item["quality_score"] < 0.45:
            item["recommended_action"] = "reject"
        elif status == "candidate" and item["quality_score"] >= 0.75:
            item["recommended_action"] = "promote"
        elif status in {"active", "promoted"}:
            item["recommended_action"] = "keep"
        else:
            item["recommended_action"] = "review"
        item["subject_key"] = _subject_key(content)
        return item

    def rank_observations(
        self,
        *,
        limit: int = 20,
        include_statuses: list[str] | tuple[str, ...] | None = None,
        scope: str | None = None,
        project_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank observations by deterministic local curation quality."""
        statuses = list(include_statuses or ["candidate", "active"])
        if not statuses:
            return []
        now = utc_now()
        clauses = [self._not_expired_clause("o")]
        params: list[Any] = [now]
        placeholders = ", ".join(["?"] * len(statuses))
        clauses.append(f"o.status IN ({placeholders})")
        params.extend(statuses)
        if scope:
            clauses.append("o.scope = ?")
            params.append(scope)
        if project_path:
            clauses.append("o.project_path = ?")
            params.append(project_path)
        where = " AND ".join(clauses)
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT o.*, superseded.redacted_content AS supersedes_content
                FROM observations o
                LEFT JOIN observations superseded ON superseded.id = o.supersedes
                WHERE {where}
                ORDER BY o.importance DESC, o.confidence DESC, o.created_at DESC
                LIMIT ?
                """,
                [*params, max(int(limit) * 5, int(limit))],
            ).fetchall()
        ranked = [self._quality_rank_item(row) for row in rows]
        ranked.sort(key=lambda item: (float(item.get("quality_score") or 0), float(item.get("importance") or 0)), reverse=True)
        return ranked[: int(limit)]

    def apply_consolidation(
        self,
        *,
        canonical_id: str,
        duplicate_ids: list[str] | tuple[str, ...],
        reason: str = "",
    ) -> dict[str, Any]:
        """Apply a reviewed consolidation by rejecting duplicate rows.

        The archive schema models supersession from a newer row to an older row;
        consolidation suggestions may involve arbitrary existing rows. Rejecting
        reviewed duplicates is the safest mutation: current/search views hide
        them, export/audit history preserves them, and the canonical row remains
        intact.
        """
        canonical = self.get_observation(canonical_id)
        if not canonical:
            raise ValueError(f"Canonical Recall observation not found: {canonical_id}")
        duplicate_ids = [str(item) for item in duplicate_ids if str(item) and str(item) != canonical_id]
        if not duplicate_ids:
            raise ValueError("At least one duplicate_id different from canonical_id is required")
        rejected = 0
        with self._lock:
            for duplicate_id in duplicate_ids:
                cur = self.conn.execute(
                    "UPDATE observations SET status='rejected' WHERE id=? AND id<>?",
                    (duplicate_id, canonical_id),
                )
                rejected += cur.rowcount
            self.conn.commit()
        self.append_audit_event(
            "result",
            "consolidation_apply",
            "observation",
            canonical_id,
            {"canonical_id": canonical_id, "duplicate_ids": duplicate_ids, "duplicates_rejected": rejected, "reason": reason},
        )
        return {
            "success": True,
            "canonical_id": canonical_id,
            "duplicate_ids": duplicate_ids,
            "duplicates_rejected": rejected,
        }

    def suggest_consolidations(
        self,
        *,
        limit: int = 20,
        scope: str | None = None,
        project_path: str | None = None,
        include_low_quality: bool = False,
        min_quality_score: float = 0.45,
    ) -> list[dict[str, Any]]:
        """Suggest same-subject groups where weaker rows should be superseded.

        This does not mutate the archive. It gives operators a deterministic
        curation queue; explicit mark/forget/promote tools remain separate.
        Low-quality groups are hidden by default so transcript-summary episode
        traces do not swamp useful fact/preference consolidation queues.
        """
        ranked = self.rank_observations(limit=max(int(limit) * 10, 50), include_statuses=["candidate", "active", "promoted"], scope=scope, project_path=project_path)
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in ranked:
            key = str(item.get("subject_key") or "")
            if key and key not in {"terms:", "label:"}:
                groups.setdefault(key, []).append(item)

        suggestions: list[dict[str, Any]] = []
        for key, items in groups.items():
            if len(items) < 2:
                continue
            ordered = sorted(
                items,
                key=lambda item: (float(item.get("quality_score") or 0), float(item.get("importance") or 0), str(item.get("created_at") or "")),
                reverse=True,
            )
            canonical = ordered[0]
            if not include_low_quality and (
                float(canonical.get("quality_score") or 0.0) < float(min_quality_score)
                or canonical.get("recommended_action") == "reject"
            ):
                continue
            duplicates = [item for item in ordered[1:] if item.get("id") != canonical.get("id")]
            if not duplicates:
                continue
            suggestions.append(
                {
                    "subject_key": key,
                    "canonical_id": canonical["id"],
                    "canonical_quality_score": canonical["quality_score"],
                    "duplicate_ids": [item["id"] for item in duplicates],
                    "duplicate_count": len(duplicates),
                    "recommended_action": "supersede_duplicates",
                    "suggested_content": canonical["content"],
                    "items": ordered,
                }
            )
        suggestions.sort(key=lambda item: (item["duplicate_count"], item["canonical_quality_score"]), reverse=True)
        return suggestions[: int(limit)]

    def append_audit_event(
        self,
        phase: str,
        operation: str,
        target: str,
        content_preview: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            prev = self.conn.execute(
                "SELECT event_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev["event_hash"] if prev else ""
            event_id = str(uuid.uuid4())
            created_at = utc_now()
            metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            preview = redact_text(content_preview)[:500]
            cur = self.conn.execute(
                """
                INSERT INTO audit_events(event_id, phase, operation, target, content_preview, prev_hash, event_hash, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, '', ?, ?)
                """,
                (event_id, phase, operation, target, preview, prev_hash, created_at, metadata_json),
            )
            seq = cur.lastrowid
            row = {
                "seq": seq,
                "event_id": event_id,
                "phase": phase,
                "operation": operation,
                "target": target,
                "content_preview": preview,
                "prev_hash": prev_hash,
                "created_at": created_at,
                "metadata_json": metadata_json,
            }
            event_hash = hash_event(row)
            self.conn.execute("UPDATE audit_events SET event_hash=? WHERE seq=?", (event_hash, seq))
            self.conn.commit()
        return event_id

    def audit_events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM audit_events ORDER BY seq DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    def archive_stats(self) -> dict[str, Any]:
        with self._lock:
            status_rows = self.conn.execute(
                "SELECT status, COUNT(*) AS count FROM observations GROUP BY status ORDER BY status"
            ).fetchall()
            type_rows = self.conn.execute(
                "SELECT type, COUNT(*) AS count FROM observations GROUP BY type ORDER BY type"
            ).fetchall()
            episode_count = self.conn.execute("SELECT COUNT(*) AS count FROM episodes").fetchone()["count"]
            observation_bounds = self.conn.execute(
                "SELECT MIN(created_at) AS oldest, MAX(created_at) AS newest FROM observations"
            ).fetchone()
            expired_count = self.conn.execute(
                "SELECT COUNT(*) AS count FROM observations WHERE expires_at IS NOT NULL AND expires_at != '' AND expires_at <= ?",
                (utc_now(),),
            ).fetchone()["count"]
            audit = verify_audit_chain(self.conn)
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "observations_by_status": {row["status"]: row["count"] for row in status_rows},
            "observations_by_type": {row["type"]: row["count"] for row in type_rows},
            "episode_count": episode_count,
            "expired_observation_count": expired_count,
            "audit": audit,
            "oldest_observation_at": observation_bounds["oldest"],
            "newest_observation_at": observation_bounds["newest"],
            "db_size_bytes": db_size,
        }

    def export_archive(self) -> dict[str, Any]:
        """Export the archive as a portable JSON-serializable backup payload."""
        with self._lock:
            episodes = [dict(row) for row in self.conn.execute("SELECT * FROM episodes ORDER BY created_at, id").fetchall()]
            observations = [
                dict(row) for row in self.conn.execute("SELECT * FROM observations ORDER BY created_at, id").fetchall()
            ]
            audit_events = [dict(row) for row in self.conn.execute("SELECT * FROM audit_events ORDER BY seq").fetchall()]
        return {
            "version": 1,
            "schema_version": SCHEMA_VERSION,
            "exported_at": utc_now(),
            "episodes": episodes,
            "observations": observations,
            "audit_events": audit_events,
        }

    def import_archive(self, payload: dict[str, Any], *, mode: str = "merge") -> dict[str, int | str]:
        """Import a Recall export payload.

        ``merge`` preserves existing rows and upserts by primary ID. It is the
        only supported mode for now because it is the safest default for backups.
        """
        if int(payload.get("version", 0) or 0) != 1:
            raise ValueError("Unsupported Recall archive export version")
        if mode != "merge":
            raise ValueError("Recall import currently supports mode='merge' only")

        episode_fields = ["id", "session_id", "project_path", "user_text", "assistant_text", "created_at"]
        observation_fields = [
            "id",
            "type",
            "scope",
            "trust_level",
            "confidence",
            "importance",
            "status",
            "content",
            "redacted_content",
            "source_session_id",
            "project_path",
            "created_at",
            "expires_at",
            "supersedes",
        ]
        audit_fields = [
            "seq",
            "event_id",
            "phase",
            "operation",
            "target",
            "content_preview",
            "prev_hash",
            "event_hash",
            "created_at",
            "metadata_json",
        ]
        episodes_imported = 0
        observations_imported = 0
        audit_events_imported = 0
        with self._lock:
            for row in payload.get("episodes", []) or []:
                clean = dict(row)
                clean["user_text"] = redact_text(str(clean.get("user_text") or ""))
                clean["assistant_text"] = redact_text(str(clean.get("assistant_text") or ""))
                values = [clean.get(field, "") for field in episode_fields]
                cur = self.conn.execute(
                    f"INSERT OR REPLACE INTO episodes({', '.join(episode_fields)}) VALUES ({', '.join(['?'] * len(episode_fields))})",
                    values,
                )
                episodes_imported += cur.rowcount
            for row in payload.get("observations", []) or []:
                clean = dict(row)
                clean["content"] = redact_text(str(clean.get("content") or ""))
                clean["redacted_content"] = redact_text(str(clean.get("redacted_content") or clean.get("content") or ""))
                values = [clean.get(field) for field in observation_fields]
                cur = self.conn.execute(
                    f"INSERT OR REPLACE INTO observations({', '.join(observation_fields)}) VALUES ({', '.join(['?'] * len(observation_fields))})",
                    values,
                )
                observations_imported += cur.rowcount
            for row in payload.get("audit_events", []) or []:
                clean = dict(row)
                clean["content_preview"] = redact_text(str(clean.get("content_preview") or ""))
                values = [clean.get(field, "") for field in audit_fields]
                cur = self.conn.execute(
                    f"INSERT OR IGNORE INTO audit_events({', '.join(audit_fields)}) VALUES ({', '.join(['?'] * len(audit_fields))})",
                    values,
                )
                audit_events_imported += cur.rowcount
            self.conn.commit()
        return {
            "mode": mode,
            "episodes_imported": episodes_imported,
            "observations_imported": observations_imported,
            "audit_events_imported": audit_events_imported,
        }

    def diagnose(self) -> dict[str, Any]:
        """Run local health checks for operators before trusting Recall output."""
        checks: dict[str, Any] = {}
        try:
            sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)").close()
            checks["fts5_available"] = True
        except Exception as exc:
            checks["fts5_available"] = False
            checks["fts5_error"] = str(exc)
        checks["db_exists"] = self.db_path.exists()
        checks["db_path"] = str(self.db_path)
        checks["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0
        try:
            with self._lock:
                self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS recall_write_check(x TEXT)")
                self.conn.execute("INSERT INTO recall_write_check(x) VALUES ('ok')")
                self.conn.execute("DROP TABLE recall_write_check")
                self.conn.execute("SELECT COUNT(*) FROM observations_fts").fetchone()
                self.conn.commit()
            checks["db_writable"] = True
            checks["fts_index_readable"] = True
        except Exception as exc:
            checks["db_writable"] = False
            checks["fts_index_readable"] = False
            checks["db_error"] = str(exc)
        audit = verify_audit_chain(self.conn)
        checks["audit_chain_ok"] = bool(audit.get("ok"))
        checks["redaction_smoke_ok"] = "secret-value" not in redact_text("API_KEY=secret-value")
        ok = all(
            bool(checks.get(name))
            for name in ("fts5_available", "db_exists", "db_writable", "fts_index_readable", "audit_chain_ok", "redaction_smoke_ok")
        )
        return {"ok": ok, "checks": checks, "audit": audit, "stats": self.archive_stats()}
