"""SQLite store for Ebbinghaus memory with sleep/dream/capacity lifecycle.

Implements the full Ebbinghaus forgetting curve, sleep-cycle consolidation,
dream-based semantic clustering, and policy-driven capacity enforcement.
All state lives in a single SQLite database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .policies import (
    CapacityPolicy,
    DreamPolicy,
    EbbinghausPolicies,
    PolicyConfigError,
    SleepPolicy,
    is_protected,
    resolve_prune_mode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STRENGTH: float = 6.0

_TOKEN_RE = re.compile(r"[\w][\w.+#:/-]{1,}", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")

_STOPWORDS: frozenset[str] = frozenset({
    "about", "after", "also", "and", "are", "because", "been", "but",
    "can", "could", "for", "from", "has", "have", "into", "not", "of",
    "our", "the", "that", "this", "use", "was", "were", "with", "you",
    "your", "です", "ます", "して", "した", "こと", "これ", "それ",
})

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def forgetting_retention(elapsed_days: float, stability_days: float) -> float:
    """Return retention in [0, 1] for the Ebbinghaus exponential curve."""
    if elapsed_days <= 0:
        return 1.0
    stability = max(0.01, float(stability_days))
    return _clamp(math.exp(-float(elapsed_days) / stability), 0.0, 1.0)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return _SPACE_RE.sub(" ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Encode text into retrieval cues (latin terms + CJK n-grams)."""
    normalized = _normalize_text(text).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("._-/#:")
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    for chunk in _CJK_RE.findall(normalized):
        if len(chunk) < 2:
            continue
        tokens.extend(chunk[i : i + 2] for i in range(0, len(chunk) - 1))
        if len(chunk) >= 3:
            tokens.extend(chunk[i : i + 3] for i in range(0, len(chunk) - 2))
    return tokens


def _cue_counts(text: str | Iterable[str]) -> Counter:
    if isinstance(text, str):
        return Counter(_tokenize(text))
    counts: Counter = Counter()
    for item in text:
        counts.update(_tokenize(str(item)))
    return counts


def _top_cues(counts: Counter, limit: int = 64) -> list[str]:
    return [
        token
        for token, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in overlap)
    if numerator <= 0:
        return 0.0
    left = math.sqrt(sum(v * v for v in a.values()))
    right = math.sqrt(sum(v * v for v in b.values()))
    if left == 0 or right == 0:
        return 0.0
    return float(numerator / (left * right))


def _split_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = re.split(r"[,;\n]", tags)
    elif isinstance(tags, Iterable):
        raw = [str(item) for item in tags]
    else:
        raw = [str(tags)]
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in raw:
        value = _normalize_text(tag).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _join_tags(tags: Iterable[str]) -> str:
    return ",".join(_split_tags(list(tags)))


def _timestamp_value(value: Any, *, default: float = 0.0) -> float:
    """Return a Unix timestamp for legacy REAL or ISO-8601 values."""
    if value is None or value == "":
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        pass
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except ValueError:
        logger.debug("Invalid Ebbinghaus timestamp value %r; using default", value)
        return float(default)


def _encode_memory(content: str, tags: Iterable[str]) -> dict:
    counts = _cue_counts([content, *list(tags)])
    cues = _top_cues(counts)
    return {
        "version": 1,
        "kind": "cue_encoding",
        "summary": _normalize_text(content)[:280],
        "cue_vector": {token: counts[token] for token in cues},
        "cues": cues,
        "length": len(content),
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content            TEXT NOT NULL UNIQUE,
    encoded            TEXT NOT NULL,
    cues               TEXT DEFAULT '',
    tags               TEXT DEFAULT '',
    salience           REAL DEFAULT 0.6,
    valence            REAL DEFAULT 0.0,
    strength           REAL DEFAULT 1.0,
    rehearsal_count    INTEGER DEFAULT 0,
    retrieval_count    INTEGER DEFAULT 0,
    source             TEXT DEFAULT '',
    session_id         TEXT DEFAULT '',
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL,
    last_rehearsed_at  REAL,
    last_retrieved_at  REAL,
    state              TEXT NOT NULL DEFAULT 'active',
    last_anchor_at     REAL,
    sleep_rehearsal_count INTEGER NOT NULL DEFAULT 0,
    last_sleep_at      REAL,
    archived_at        REAL,
    archive_reason     TEXT DEFAULT '',
    memory_type        TEXT NOT NULL DEFAULT 'episodic',
    dream_candidate    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ebbinghaus_tags ON memories(tags);
CREATE INDEX IF NOT EXISTS idx_ebbinghaus_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ebbinghaus_salience ON memories(salience DESC);

CREATE TABLE IF NOT EXISTS memory_provenance (
    semantic_memory_id INTEGER NOT NULL,
    source_memory_id   INTEGER NOT NULL,
    relation           TEXT NOT NULL DEFAULT 'dream-derived',
    created_at         REAL NOT NULL,
    PRIMARY KEY (semantic_memory_id, source_memory_id)
);

CREATE TABLE IF NOT EXISTS dream_previews (
    cluster_id TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    ("state", "TEXT NOT NULL DEFAULT 'active'"),
    ("last_anchor_at", "REAL"),
    ("sleep_rehearsal_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_sleep_at", "REAL"),
    ("archived_at", "REAL"),
    ("archive_reason", "TEXT DEFAULT ''"),
    ("memory_type", "TEXT NOT NULL DEFAULT 'episodic'"),
    ("dream_candidate", "INTEGER NOT NULL DEFAULT 0"),
]

# Whitelist for DDL identifiers — never interpolate untrusted strings into SQL.
_MIGRATION_COLUMN_DEFS: dict[str, str] = dict(_MIGRATION_COLUMNS)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_MEMORY_STATES = frozenset({"active", "archived"})
_MAX_IN_CLAUSE = 2048


def _assert_safe_identifier(name: str) -> str:
    if name not in _MIGRATION_COLUMN_DEFS or not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"refusing unsafe SQL identifier: {name!r}")
    return name


def _placeholders(count: int) -> str:
    n = int(count)
    if n < 1 or n > _MAX_IN_CLAUSE:
        raise ValueError(f"IN-clause size out of bounds: {n}")
    return ",".join("?" * n)


def _as_positive_int_ids(raw: Sequence[Any]) -> list[int]:
    ids: list[int] = []
    for item in raw:
        value = int(item)
        if value < 1:
            raise ValueError(f"memory_id must be >= 1, got {value}")
        ids.append(value)
    return ids


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CapacityError(RuntimeError):
    """Raised when capacity is full and no non-protected memory can be archived."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class EbbinghausMemoryStore:
    """SQLite store for encoded memory traces with lifecycle policies."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        base_stability_days: float = 3.0,
        decay_threshold: float = 0.10,
        time_fn: Callable[[], float] | None = None,
        policies: EbbinghausPolicies | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if policies is not None:
            self.policies = policies
            self.base_stability_days = max(0.05, policies.base_stability_days)
            self.decay_threshold = _clamp(policies.decay_threshold, 0.0, 1.0)
        else:
            self.base_stability_days = max(0.05, float(base_stability_days))
            self.decay_threshold = _clamp(float(decay_threshold), 0.0, 1.0)
            self.policies = EbbinghausPolicies(
                base_stability_days=self.base_stability_days,
                decay_threshold=self.decay_threshold,
            )

        self._time_fn = time_fn or time.time
        # In-process cache of dream previews; durable copy lives in dream_previews.
        self._preview_registry: dict[str, dict[str, Any]] = {}
        self._negative_prefetch_suppressed_count = 0
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._hydrate_preview_registry()

    # ------------------------------------------------------------------
    # DB initialisation & migration
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(self._conn, db_label="ebbinghaus_memory.db")
        except Exception:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass

        self._conn.executescript(_BASE_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        self._conn.execute("BEGIN")
        try:
            existing = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            for col_name, col_def in _MIGRATION_COLUMNS:
                safe_name = _assert_safe_identifier(col_name)
                safe_def = _MIGRATION_COLUMN_DEFS[safe_name]
                if safe_def != col_def:
                    raise ValueError(f"migration definition mismatch for {safe_name}")
                if safe_name not in existing:
                    # Identifiers come only from the hardcoded whitelist above.
                    self._conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {safe_name} {safe_def}"
                    )
                    logger.info("Ebbinghaus migration: added column %s", safe_name)

            self._conn.execute(
                """
                UPDATE memories
                SET last_anchor_at = MAX(
                    COALESCE(created_at, 0),
                    COALESCE(last_rehearsed_at, 0),
                    COALESCE(last_retrieved_at, 0)
                )
                WHERE last_anchor_at IS NULL
                """
            )

            for idx_sql in (
                "CREATE INDEX IF NOT EXISTS idx_ebbinghaus_state_anchor "
                "ON memories(state, last_anchor_at ASC)",
                "CREATE INDEX IF NOT EXISTS idx_ebbinghaus_state_salience "
                "ON memories(state, salience ASC)",
            ):
                try:
                    self._conn.execute(idx_sql)
                except sqlite3.OperationalError:
                    pass
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._conn.close()

    def _hydrate_preview_registry(self) -> None:
        """Load durable dream preview payloads after reopen (best-effort)."""
        try:
            rows = self._conn.execute(
                "SELECT cluster_id, payload FROM dream_previews"
            ).fetchall()
        except sqlite3.Error:
            return
        for row in rows:
            cluster_id = str(row["cluster_id"] or "")
            if not cluster_id or "\x00" in cluster_id:
                continue
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                self._preview_registry[cluster_id] = payload

    def _persist_preview(self, cluster_id: str, payload: dict[str, Any], *, now: float) -> None:
        """Write preview ledger to SQLite so apply survives process restart."""
        if not cluster_id or "\x00" in cluster_id or len(cluster_id) > 128:
            raise ValueError(f"invalid dream cluster_id: {cluster_id!r}")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if "\x00" in encoded:
            raise ValueError("dream preview payload must not contain NUL bytes")
        self._conn.execute(
            """
            INSERT INTO dream_previews (cluster_id, payload, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cluster_id) DO UPDATE SET
                payload = excluded.payload,
                created_at = excluded.created_at
            """,
            (cluster_id, encoded, now),
        )
        self._preview_registry[cluster_id] = payload

    def _load_preview(self, cluster_id: str) -> dict[str, Any] | None:
        cached = self._preview_registry.get(cluster_id)
        if isinstance(cached, dict):
            return cached
        row = self._conn.execute(
            "SELECT payload FROM dream_previews WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        self._preview_registry[cluster_id] = payload
        return payload

    def _drop_preview(self, cluster_id: str) -> None:
        self._preview_registry.pop(cluster_id, None)
        self._conn.execute(
            "DELETE FROM dream_previews WHERE cluster_id = ?",
            (cluster_id,),
        )

    def _now(self) -> float:
        return float(self._time_fn())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(value: str) -> dict:
        try:
            decoded = json.loads(value or "{}")
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _stability_days(self, row: sqlite3.Row) -> float:
        salience = float(row["salience"] or 0.0)
        strength = float(row["strength"] or 1.0)
        rehearsals = int(row["rehearsal_count"] or 0)
        retrievals = int(row["retrieval_count"] or 0)
        multiplier = (
            0.45
            + (1.35 * salience)
            + (0.65 * math.log1p(rehearsals))
            + (0.25 * math.log1p(retrievals))
        )
        return max(0.05, self.base_stability_days * strength * multiplier)

    def _retention(self, row: sqlite3.Row) -> float:
        now = self._now()
        anchor = _timestamp_value(row["last_anchor_at"])
        if anchor <= 0:
            anchor = max(
                _timestamp_value(row["created_at"], default=now),
                _timestamp_value(row["last_rehearsed_at"]),
                _timestamp_value(row["last_retrieved_at"]),
            )
        elapsed_days = max(0.0, (now - anchor) / 86400.0)
        return forgetting_retention(elapsed_days, self._stability_days(row))

    def _compute_anchor(self, row: sqlite3.Row) -> float:
        return max(
            _timestamp_value(row["created_at"], default=0.0),
            _timestamp_value(row["last_rehearsed_at"]),
            _timestamp_value(row["last_retrieved_at"]),
        )

    def _update_anchor(self, memory_id: int, anchor: float) -> None:
        self._conn.execute(
            "UPDATE memories SET last_anchor_at = ? WHERE memory_id = ?",
            (anchor, memory_id),
        )

    def _reinforcement_gain(
        self, row: sqlite3.Row, *, kind: str
    ) -> float:
        bases = {"duplicate": 0.15, "retrieval": 0.08, "rehearsal": 0.25}
        base = bases.get(kind, 0.08)
        valence = float(row["valence"] or 0.0)
        threshold = self.policies.sleep.negative_valence_threshold
        multiplier = self.policies.sleep.negative_reinforcement_multiplier
        if valence <= threshold:
            return base * multiplier
        return base

    def _row_to_result(
        self,
        row: sqlite3.Row,
        *,
        query_score: float | None,
        retention: float | None = None,
    ) -> dict:
        retention = self._retention(row) if retention is None else retention
        encoded = self._decode(row["encoded"])
        now = self._now()
        anchor = _timestamp_value(row["last_anchor_at"])
        if anchor <= 0:
            anchor = self._compute_anchor(row)
        created = _timestamp_value(row["created_at"], default=now)

        result: dict[str, Any] = {
            "memory_id": int(row["memory_id"]),
            "content": row["content"],
            "tags": _split_tags(row["tags"]),
            "cues": encoded.get("cues", [])[:12],
            "salience": round(float(row["salience"] or 0.0), 3),
            "valence": round(float(row["valence"] or 0.0), 3),
            "retention": round(float(retention), 4),
            "stability_days": round(self._stability_days(row), 3),
            "age_days": round(max(0.0, (now - created) / 86400.0), 3),
            "days_since_reinforcement": round(
                max(0.0, (now - anchor) / 86400.0), 3
            ),
            "rehearsal_count": int(row["rehearsal_count"] or 0),
            "retrieval_count": int(row["retrieval_count"] or 0),
            "source": row["source"] or "",
            "session_id": row["session_id"] or "",
            "state": row["state"] or "active",
            "memory_type": row["memory_type"] or "episodic",
            "sleep_rehearsal_count": int(row["sleep_rehearsal_count"] or 0),
            "dream_candidate": bool(row["dream_candidate"]),
        }
        if query_score is not None:
            result["score"] = round(float(query_score), 4)
        return result

    def _is_row_protected(self, row: sqlite3.Row) -> bool:
        return is_protected(
            _split_tags(row["tags"]),
            self.policies.capacity.protected_tags,
        )

    def _has_live_semantic_dependents(self, memory_id: int) -> bool:
        """True if memory_id is the sole provenance source for a live semantic."""
        row = self._conn.execute(
            """
            SELECT 1 FROM memory_provenance p
            WHERE p.source_memory_id = ?
              AND NOT EXISTS (
                  SELECT 1 FROM memory_provenance p2
                  WHERE p2.semantic_memory_id = p.semantic_memory_id
                    AND p2.source_memory_id != ?
              )
              AND EXISTS (
                  SELECT 1 FROM memories m
                  WHERE m.memory_id = p.semantic_memory_id
              )
            LIMIT 1
            """,
            (memory_id, memory_id),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # remember
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        tags: Any = None,
        salience: float = 0.65,
        valence: float = 0.0,
        source: str = "",
        session_id: str = "",
        memory_type: str = "episodic",
    ) -> dict:
        content = _normalize_text(content)
        if not content:
            raise ValueError("content must not be empty")
        if "\x00" in content:
            raise ValueError("content must not contain NUL bytes")

        tag_list = _split_tags(tags)
        encoded = _encode_memory(content, tag_list)
        cues = " ".join(encoded["cues"])
        now = self._now()
        salience = _clamp(float(salience), 0.05, 1.0)
        valence = _clamp(float(valence), -1.0, 1.0)
        tag_text = _join_tags(tag_list)

        self._enforce_capacity_headroom()

        try:
            cur = self._conn.execute(
                """
                INSERT INTO memories (
                    content, encoded, cues, tags, salience, valence, strength,
                    source, session_id, created_at, updated_at,
                    last_rehearsed_at, state, last_anchor_at, memory_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    content,
                    json.dumps(encoded, ensure_ascii=False),
                    cues,
                    tag_text,
                    salience,
                    valence,
                    1.0 + salience,
                    source,
                    session_id,
                    now,
                    now,
                    now,
                    now,
                    memory_type,
                ),
            )
            self._conn.commit()
            memory_id = int(cur.lastrowid)
            return {"memory_id": memory_id, "status": "remembered", **self.get(memory_id)}
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE content = ?", (content,)
            ).fetchone()
            if not row:
                raise
            gain = self._reinforcement_gain(row, kind="duplicate")
            merged_tags = sorted(set(_split_tags(row["tags"])) | set(tag_list))
            self._conn.execute(
                """
                UPDATE memories
                SET tags = ?, salience = MAX(salience, ?),
                    rehearsal_count = rehearsal_count + 1,
                    strength = MIN(?, strength + ?),
                    last_rehearsed_at = ?, updated_at = ?,
                    last_anchor_at = ?,
                    state = CASE WHEN state = 'archived' THEN 'active' ELSE state END,
                    archived_at = CASE WHEN state = 'archived' THEN NULL ELSE archived_at END,
                    archive_reason = CASE WHEN state = 'archived' THEN '' ELSE archive_reason END
                WHERE memory_id = ?
                """,
                (
                    _join_tags(merged_tags),
                    salience,
                    _MAX_STRENGTH,
                    gain,
                    now,
                    now,
                    now,
                    row["memory_id"],
                ),
            )
            self._conn.commit()
            memory_id = int(row["memory_id"])
            return {"memory_id": memory_id, "status": "reinforced", **self.get(memory_id)}

    def _protected_category_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        rows = self._conn.execute(
            "SELECT tags FROM memories WHERE state = 'active'"
        ).fetchall()
        protected = {
            str(tag).strip().lower() for tag in self.policies.capacity.protected_tags
        }
        for row in rows:
            tags = {t.lower() for t in _split_tags(row["tags"])}
            for tag in tags & protected:
                breakdown[tag] = breakdown.get(tag, 0) + 1
        return breakdown

    def _eviction_rank(self, row: sqlite3.Row) -> tuple:
        """Lower tuple sorts first = better eviction candidate."""
        retention = self._retention(row)
        return (
            retention,
            float(row["salience"] or 0.0),
            int(row["retrieval_count"] or 0),
            int(row["rehearsal_count"] or 0),
            _timestamp_value(row["last_anchor_at"], default=_timestamp_value(row["created_at"])),
            int(row["memory_id"]),
        )

    def _enforce_capacity_headroom(self) -> None:
        cap = self.policies.capacity
        active_count = self._count_active()
        if active_count < cap.max_active_memories:
            return
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE state = 'active'"
        ).fetchall()
        candidates = [
            r for r in rows if not is_protected(_split_tags(r["tags"]), cap.protected_tags)
        ]
        candidates.sort(key=self._eviction_rank)
        archived = 0
        now = self._now()
        for r in candidates:
            self._archive_memory(r["memory_id"], reason="capacity", now=now)
            archived += 1
            if active_count - archived < cap.max_active_memories:
                self._conn.commit()
                return
        protected_categories = self._protected_category_breakdown()
        raise CapacityError(
            f"Memory capacity full ({cap.max_active_memories} active) "
            "and no non-protected memory can be archived.",
            details={
                "capacity_blocked": True,
                "active_count": active_count,
                "max_active_memories": cap.max_active_memories,
                "protected_count": active_count - archived,
                "protected_categories": protected_categories,
            },
        )

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def get(self, memory_id: int) -> dict:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (int(memory_id),)
        ).fetchone()
        if not row:
            raise KeyError(f"memory_id not found: {memory_id}")
        return self._row_to_result(row, query_score=None)

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.12,
        reinforce: bool = False,
        include_archived: bool = False,
    ) -> list[dict]:
        query = _normalize_text(query)
        if not query:
            return []
        query_counts = _cue_counts(query)
        query_lower = query.lower()

        if include_archived:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE state = 'active'"
            ).fetchall()

        scored: list[dict] = []
        for row in rows:
            encoded = self._decode(row["encoded"])
            memory_counts = Counter(encoded.get("cue_vector") or {})
            tags = _split_tags(row["tags"])
            tag_counts = _cue_counts(tags)
            lexical = _cosine(query_counts, memory_counts + tag_counts)
            substring = 0.35 if query_lower in str(row["content"]).lower() else 0.0
            tag_bonus = 0.12 if set(_split_tags(query)) & set(tags) else 0.0
            if lexical <= 0 and substring <= 0 and tag_bonus <= 0:
                continue

            retention = self._retention(row)
            salience = float(row["salience"] or 0.0)
            rehearsal_bonus = min(
                0.08, math.log1p(int(row["rehearsal_count"] or 0)) * 0.025
            )
            score = (
                max(lexical, substring) * 0.68
                + retention * 0.18
                + salience * 0.08
                + tag_bonus
                + rehearsal_bonus
            )
            if score < min_score:
                continue
            scored.append(
                self._row_to_result(row, query_score=score, retention=retention)
            )

        scored.sort(
            key=lambda item: (item["score"], item["retention"], item["salience"]),
            reverse=True,
        )
        results = scored[: max(1, int(limit))]

        if reinforce:
            for result in results:
                self._reinforce_retrieval(result["memory_id"])
            results = [
                self.get(r["memory_id"]) | {"score": r["score"]}
                for r in results
            ]
        return results

    # ------------------------------------------------------------------
    # rehearse
    # ------------------------------------------------------------------

    def rehearse(
        self,
        *,
        memory_id: int | None = None,
        query: str = "",
        limit: int = 1,
    ) -> list[dict]:
        targets: list[int] = []
        if memory_id is not None:
            targets.append(int(memory_id))
        elif query:
            targets.extend(
                item["memory_id"]
                for item in self.recall(query, limit=limit, reinforce=False)
            )
        else:
            raise ValueError("memory_id or query is required")

        now = self._now()
        for target in targets:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (target,)
            ).fetchone()
            gain = self._reinforcement_gain(row, kind="rehearsal") if row else 0.25
            self._conn.execute(
                """
                UPDATE memories
                SET rehearsal_count = rehearsal_count + 1,
                    strength = MIN(?, strength + ?),
                    last_rehearsed_at = ?, updated_at = ?,
                    last_anchor_at = ?
                WHERE memory_id = ?
                """,
                (_MAX_STRENGTH, gain, now, now, now, target),
            )
        self._conn.commit()
        return [self.get(t) for t in targets]

    # ------------------------------------------------------------------
    # forget (exact delete)
    # ------------------------------------------------------------------

    def forget(self, memory_id: int) -> bool:
        mid = int(memory_id)
        if mid < 1:
            raise ValueError(f"memory_id must be >= 1, got {mid}")
        cur = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (mid,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # decay (legacy)
    # ------------------------------------------------------------------

    def decay(
        self,
        *,
        threshold: float | None = None,
        prune: bool = False,
        limit: int = 50,
    ) -> dict:
        threshold = (
            self.decay_threshold
            if threshold is None
            else _clamp(float(threshold), 0.0, 1.0)
        )
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE state = 'active'"
        ).fetchall()
        decayed = [
            self._row_to_result(row, query_score=None)
            for row in rows
            if self._retention(row) <= threshold
        ]
        decayed.sort(key=lambda item: item["retention"])
        decayed = decayed[: max(1, int(limit))]
        pruned: list[int] = []
        if prune and decayed:
            pruned = [int(item["memory_id"]) for item in decayed]
            self._conn.executemany(
                "DELETE FROM memories WHERE memory_id = ?",
                [(mid,) for mid in pruned],
            )
            self._conn.commit()
        return {"threshold": threshold, "decayed": decayed, "pruned": pruned}

    # ------------------------------------------------------------------
    # sleep_cycle
    # ------------------------------------------------------------------

    def sleep_cycle(
        self,
        *,
        rehearse_threshold: float | None = None,
        forget_threshold: float | None = None,
        salience_keep_threshold: float | None = None,
        prune: bool | None = None,
        limit: int | None = None,
        prune_mode: str | None = None,
        max_sleep_rehearsals: int | None = None,
        max_negative_sleep_rehearsals: int | None = None,
    ) -> dict:
        sp = self.policies.sleep
        cp = self.policies.capacity

        rehearse_threshold = _clamp(
            float(sp.rehearse_threshold if rehearse_threshold is None else rehearse_threshold),
            0.0,
            1.0,
        )
        forget_threshold = _clamp(
            float(sp.forget_threshold if forget_threshold is None else forget_threshold),
            0.0,
            1.0,
        )
        salience_keep_threshold = _clamp(
            float(
                sp.salience_keep_threshold
                if salience_keep_threshold is None
                else salience_keep_threshold
            ),
            0.0,
            1.0,
        )
        limit = max(1, int(sp.limit if limit is None else limit))

        resolved_prune = resolve_prune_mode(
            prune=prune,
            prune_mode=prune_mode,
            default=sp.prune_mode,
        )
        max_sr = (
            sp.max_sleep_rehearsals
            if max_sleep_rehearsals is None
            else int(max_sleep_rehearsals)
        )
        max_neg_sr = (
            sp.max_negative_sleep_rehearsals
            if max_negative_sleep_rehearsals is None
            else int(max_negative_sleep_rehearsals)
        )

        candidate_window = max(limit * 3, limit + 256)
        # Prefer never-slept / oldest-slept first so prune_mode=none still advances.
        candidates = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE state = 'active'
            ORDER BY COALESCE(last_sleep_at, 0) ASC,
                     COALESCE(last_anchor_at, created_at) ASC,
                     memory_id ASC
            LIMIT ?
            """,
            (candidate_window,),
        ).fetchall()

        scored: list[tuple[float, float, sqlite3.Row]] = []
        for row in candidates:
            retention = self._retention(row)
            salience = float(row["salience"] or 0.0)
            scored.append((retention, salience, row))
        scored.sort(key=lambda t: (t[0], -t[1]))

        now = self._now()
        rehearsed: list[int] = []
        forgotten_ids: list[int] = []
        archived_ids: list[int] = []
        dream_candidate_ids: list[int] = []
        negative_rehearsal_suppressed: list[int] = []
        protected_skipped: list[int] = []
        reviewed_ids: list[int] = []
        processed = 0

        for retention, salience, row in scored:
            if processed >= limit:
                break
            processed += 1
            mid = int(row["memory_id"])
            reviewed_ids.append(mid)
            valence = float(row["valence"] or 0.0)
            prot = self._is_row_protected(row)
            src = int(row["sleep_rehearsal_count"] or 0)

            if prot:
                if (
                    src < max_sr
                    and retention <= rehearse_threshold
                    and salience >= salience_keep_threshold
                ):
                    gain = self._reinforcement_gain(row, kind="rehearsal")
                    self._conn.execute(
                        """
                        UPDATE memories
                        SET rehearsal_count = rehearsal_count + 1,
                            strength = MIN(?, strength + ?),
                            last_rehearsed_at = ?, updated_at = ?,
                            last_anchor_at = ?,
                            sleep_rehearsal_count = sleep_rehearsal_count + 1
                        WHERE memory_id = ?
                        """,
                        (_MAX_STRENGTH, gain, now, now, now, mid),
                    )
                    rehearsed.append(mid)
                else:
                    protected_skipped.append(mid)
                continue

            strongly_negative = valence <= sp.negative_valence_threshold

            if strongly_negative:
                # Forget/archive first when retention is exhausted; otherwise
                # allow a small number of attenuated sleep rehearsals.
                if retention <= forget_threshold:
                    forgotten_ids.append(mid)
                    self._conn.execute(
                        "UPDATE memories SET dream_candidate = 1 WHERE memory_id = ?",
                        (mid,),
                    )
                    dream_candidate_ids.append(mid)
                elif src < max_neg_sr and retention <= rehearse_threshold:
                    gain = self._reinforcement_gain(row, kind="rehearsal")
                    self._conn.execute(
                        """
                        UPDATE memories
                        SET rehearsal_count = rehearsal_count + 1,
                            strength = MIN(?, strength + ?),
                            last_rehearsed_at = ?, updated_at = ?,
                            last_anchor_at = ?,
                            sleep_rehearsal_count = sleep_rehearsal_count + 1
                        WHERE memory_id = ?
                        """,
                        (_MAX_STRENGTH, gain, now, now, now, mid),
                    )
                    rehearsed.append(mid)
                else:
                    negative_rehearsal_suppressed.append(mid)
                    self._conn.execute(
                        "UPDATE memories SET dream_candidate = 1 WHERE memory_id = ?",
                        (mid,),
                    )
                    dream_candidate_ids.append(mid)
                continue

            # Instruction §8.6: forget before rehearse so capped high-salience
            # traces can leave the active set once retention collapses.
            if retention <= forget_threshold:
                forgotten_ids.append(mid)
            elif (
                retention <= rehearse_threshold
                and salience >= salience_keep_threshold
                and src < max_sr
            ):
                gain = self._reinforcement_gain(row, kind="rehearsal")
                self._conn.execute(
                    """
                    UPDATE memories
                    SET rehearsal_count = rehearsal_count + 1,
                        strength = MIN(?, strength + ?),
                        last_rehearsed_at = ?, updated_at = ?,
                        last_anchor_at = ?,
                        sleep_rehearsal_count = sleep_rehearsal_count + 1
                    WHERE memory_id = ?
                    """,
                    (_MAX_STRENGTH, gain, now, now, now, mid),
                )
                rehearsed.append(mid)
            elif (
                retention <= rehearse_threshold
                and salience >= salience_keep_threshold
                and src >= max_sr
            ):
                self._conn.execute(
                    "UPDATE memories SET dream_candidate = 1 WHERE memory_id = ?",
                    (mid,),
                )
                dream_candidate_ids.append(mid)

        pruned_ids: list[int] = []
        if forgotten_ids:
            if resolved_prune == "archive":
                for mid in forgotten_ids:
                    self._archive_memory(mid, reason="sleep-forget", now=now)
                archived_ids.extend(forgotten_ids)
            elif resolved_prune == "delete":
                self._conn.executemany(
                    "DELETE FROM memories WHERE memory_id = ?",
                    [(mid,) for mid in forgotten_ids],
                )
                pruned_ids.extend(forgotten_ids)

        capacity_archived = self._enforce_capacity_ceiling(now)

        purged = self._purge_old_archives(now)

        if reviewed_ids:
            self._conn.executemany(
                "UPDATE memories SET last_sleep_at = ? WHERE memory_id = ?",
                [(now, mid) for mid in reviewed_ids],
            )

        self._conn.commit()

        return {
            "mode": "sleep_cycle",
            "reviewed": processed,
            "candidate_window": candidate_window,
            "rehearse_threshold": rehearse_threshold,
            "forget_threshold": forget_threshold,
            "salience_keep_threshold": salience_keep_threshold,
            "prune_mode": resolved_prune,
            "rehearsed": rehearsed,
            "forgotten": forgotten_ids,
            "archived": archived_ids,
            "pruned": pruned_ids,
            "negative_rehearsal_suppressed": negative_rehearsal_suppressed,
            "dream_candidates": dream_candidate_ids,
            "capacity_archived": capacity_archived,
            "purged": purged,
            "protected_skipped": protected_skipped,
            "active_count": self._count_active(),
            "archived_count": self._count_archived(),
        }

    # ------------------------------------------------------------------
    # archive / capacity helpers
    # ------------------------------------------------------------------

    def _archive_memory(self, memory_id: int, *, reason: str, now: float) -> None:
        self._conn.execute(
            """
            UPDATE memories
            SET state = 'archived', archived_at = ?, archive_reason = ?,
                updated_at = ?
            WHERE memory_id = ? AND state = 'active'
            """,
            (now, reason, now, memory_id),
        )

    def _enforce_capacity_ceiling(self, now: float) -> list[int]:
        cp = self.policies.capacity
        active = self._count_active()
        if active <= cp.max_active_memories:
            return []
        overflow = active - cp.max_active_memories
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE state = 'active'"
        ).fetchall()
        candidates = [
            r for r in rows if not is_protected(_split_tags(r["tags"]), cp.protected_tags)
        ]
        candidates.sort(key=self._eviction_rank)
        archived: list[int] = []
        for r in candidates:
            if len(archived) >= overflow:
                break
            self._archive_memory(r["memory_id"], reason="capacity", now=now)
            archived.append(int(r["memory_id"]))
        return archived

    def _purge_old_archives(self, now: float) -> list[int]:
        cp = self.policies.capacity
        purged: list[int] = []
        protected = {str(tag).strip().lower() for tag in cp.protected_tags}

        cutoff = now - (cp.archive_retention_days * 86400.0)
        old_rows = self._conn.execute(
            """
            SELECT memory_id, tags FROM memories
            WHERE state = 'archived' AND archived_at IS NOT NULL AND archived_at < ?
            ORDER BY archived_at ASC
            """,
            (cutoff,),
        ).fetchall()
        for r in old_rows:
            mid = int(r["memory_id"])
            tags = {t.lower() for t in _split_tags(r["tags"])}
            if tags & protected:
                continue
            if not self._has_live_semantic_dependents(mid):
                self._conn.execute(
                    "DELETE FROM memories WHERE memory_id = ?", (mid,)
                )
                purged.append(mid)

        archived_count = self._count_archived()
        if archived_count > cp.max_archived_memories:
            excess = archived_count - cp.max_archived_memories
            excess_rows = self._conn.execute(
                """
                SELECT memory_id, tags FROM memories
                WHERE state = 'archived'
                ORDER BY COALESCE(archived_at, updated_at) ASC
                LIMIT ?
                """,
                (excess + 50,),
            ).fetchall()
            removed = 0
            for r in excess_rows:
                if removed >= excess:
                    break
                mid = int(r["memory_id"])
                if mid in purged:
                    continue
                tags = {t.lower() for t in _split_tags(r["tags"])}
                if tags & protected:
                    continue
                if not self._has_live_semantic_dependents(mid):
                    self._conn.execute(
                        "DELETE FROM memories WHERE memory_id = ?", (mid,)
                    )
                    purged.append(mid)
                    removed += 1

        return purged

    def _count_active(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE state = 'active'"
        ).fetchone()
        return int(row["c"])

    def _count_archived(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE state = 'archived'"
        ).fetchone()
        return int(row["c"])

    # ------------------------------------------------------------------
    # reinforcement
    # ------------------------------------------------------------------

    def _reinforce_retrieval(self, memory_id: int) -> None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (int(memory_id),)
        ).fetchone()
        gain = self._reinforcement_gain(row, kind="retrieval") if row else 0.08
        now = self._now()
        self._conn.execute(
            """
            UPDATE memories
            SET retrieval_count = retrieval_count + 1,
                strength = MIN(?, strength + ?),
                last_retrieved_at = ?, updated_at = ?,
                last_anchor_at = ?
            WHERE memory_id = ?
            """,
            (_MAX_STRENGTH, gain, now, now, now, int(memory_id)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # dream_preview
    # ------------------------------------------------------------------

    def dream_preview(self) -> dict:
        dp = self.policies.dreaming
        if not dp.enabled:
            return {"mode": "dream_preview", "enabled": False, "clusters": []}

        candidates = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE state = 'active' AND dream_candidate = 1
            ORDER BY COALESCE(last_anchor_at, created_at) ASC
            LIMIT ?
            """,
            (dp.candidate_limit,),
        ).fetchall()

        if len(candidates) < dp.min_source_count:
            return {"mode": "dream_preview", "enabled": True, "clusters": []}

        groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for row in candidates:
            tag_list = _split_tags(row["tags"])
            source = row["source"] or ""
            session = row["session_id"] or ""
            key_parts = sorted(set(tag_list)) + [source, session]
            group_key = hashlib.sha256(
                "|".join(key_parts).encode()
            ).hexdigest()[:16]
            groups[group_key].append(row)

        encoded_groups: dict[str, list[sqlite3.Row]] = {}
        for key, rows in groups.items():
            if len(rows) < dp.min_source_count:
                for r in rows:
                    cue_set = frozenset(
                        self._decode(r["encoded"]).get("cues", [])[:8]
                    )
                    placed = False
                    for ekey, erows in encoded_groups.items():
                        sample_cues = frozenset(
                            self._decode(erows[0]["encoded"]).get("cues", [])[:8]
                        )
                        if len(cue_set & sample_cues) >= 2:
                            erows.append(r)
                            placed = True
                            break
                    if not placed:
                        encoded_groups.setdefault(key, []).append(r)
            else:
                encoded_groups[key] = rows

        clusters: list[dict] = []
        now = self._now()
        stamp = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d")
        for _key, rows in sorted(
            encoded_groups.items(), key=lambda t: -len(t[1])
        ):
            if len(rows) < dp.min_source_count:
                continue
            if len(clusters) >= dp.max_clusters:
                break
            # Skip forced merge of strongly conflicting valence polarities.
            valences = [float(r["valence"] or 0.0) for r in rows]
            if max(valences) - min(valences) > 1.2:
                continue
            cluster_id = f"dream_{stamp}_{len(clusters) + 1:03d}"
            source_ids = [int(r["memory_id"]) for r in rows]
            themes = sorted({tag for r in rows for tag in _split_tags(r["tags"])})[:10]
            contains_neg = any(
                float(r["valence"] or 0.0) <= self.policies.sleep.negative_valence_threshold
                for r in rows
            )
            payload = {
                "cluster_id": cluster_id,
                "source_memory_ids": source_ids,
                "source_ids": source_ids,
                "themes": themes,
                "suggested_kind": "semantic_lesson",
                "contains_strong_negative_valence": contains_neg,
                "llm_output_schema": {
                    "cluster_id": cluster_id,
                    "source_memory_ids": source_ids,
                    "summary": "reusable lesson only",
                    "tags": ["dream-summary", "semantic"],
                    "salience": 0.7,
                    "valence": 0.0,
                },
            }
            # Persist preview ledger so apply works after process restart.
            # Memories table itself remains unchanged by preview.
            self._persist_preview(cluster_id, payload, now=now)
            clusters.append(payload)

        if clusters:
            self._conn.commit()
        return {"mode": "dream_preview", "enabled": True, "clusters": clusters}

    def _find_semantic_by_source_set(self, source_ids: Sequence[int]) -> int | None:
        wanted = sorted({int(x) for x in source_ids})
        if not wanted:
            return None
        rows = self._conn.execute(
            """
            SELECT semantic_memory_id, GROUP_CONCAT(source_memory_id) AS sources
            FROM memory_provenance
            GROUP BY semantic_memory_id
            """
        ).fetchall()
        for row in rows:
            raw = str(row["sources"] or "")
            have = sorted(int(part) for part in raw.split(",") if part.strip())
            if have == wanted:
                return int(row["semantic_memory_id"])
        return None

    def dream_apply(self, dreams: list[dict] | None) -> dict:
        dp = self.policies.dreaming
        if not dp.enabled:
            return {"mode": "dream_apply", "enabled": False, "applied": []}
        if not dreams:
            raise ValueError("dreams array is required for apply")

        results: list[dict] = []
        now = self._now()
        self._conn.execute("BEGIN")
        try:
            for dream in dreams:
                cluster_id = str(dream.get("cluster_id", ""))
                if not cluster_id or len(cluster_id) > 128 or "\x00" in cluster_id:
                    raise ValueError("invalid dream cluster_id")
                content = _normalize_text(
                    str(dream.get("summary") or dream.get("content") or "")
                )
                if "\x00" in content:
                    raise ValueError("dream summary must not contain NUL bytes")
                tags = _split_tags(dream.get("tags") or ["dream-summary", "semantic"])
                if "dream-summary" not in tags:
                    tags.append("dream-summary")
                if "semantic" not in tags:
                    tags.append("semantic")
                salience = _clamp(float(dream.get("salience", 0.6)), 0.05, 1.0)
                valence = _clamp(float(dream.get("valence", 0.0)), -1.0, 1.0)

                if not content or len(content) < 8:
                    raise ValueError(f"invalid dream summary for cluster {cluster_id!r}")

                preview = self._load_preview(cluster_id)
                provided_raw = dream.get("source_memory_ids") or dream.get("source_ids") or []

                # Idempotency first: same content OR same source-set already consolidated.
                # This path stays valid even after the durable preview row is dropped.
                early_source_ids = (
                    _as_positive_int_ids(provided_raw) if provided_raw else []
                )
                existing = self._conn.execute(
                    """
                    SELECT memory_id FROM memories
                    WHERE content = ? AND memory_type = 'semantic'
                    """,
                    (content,),
                ).fetchone()
                by_sources = (
                    self._find_semantic_by_source_set(early_source_ids)
                    if early_source_ids
                    else None
                )
                if existing or by_sources is not None:
                    semantic_id = int(
                        existing["memory_id"] if existing else by_sources  # type: ignore[index]
                    )
                    results.append({
                        "cluster_id": cluster_id,
                        "status": "idempotent",
                        "semantic_memory_id": semantic_id,
                        "archived_sources": [],
                    })
                    continue

                if preview is None:
                    raise ValueError(f"unknown dream preview cluster_id: {cluster_id}")

                expected_ids = _as_positive_int_ids(
                    preview.get("source_memory_ids")
                    or preview.get("source_ids")
                    or []
                )
                provided_ids = (
                    _as_positive_int_ids(provided_raw) if provided_raw else list(expected_ids)
                )
                if sorted(provided_ids) != sorted(expected_ids):
                    raise ValueError(f"source_memory_ids do not match preview for {cluster_id}")

                source_ids = expected_ids
                source_rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE memory_id IN ({_placeholders(len(source_ids))})",
                    source_ids,
                ).fetchall()
                if len(source_rows) != len(source_ids):
                    raise ValueError(f"missing source memories for {cluster_id}")

                max_source_salience = max(
                    (float(r["salience"] or 0.0) for r in source_rows),
                    default=0.5,
                )
                if salience > max_source_salience + 0.15:
                    raise ValueError(
                        f"salience {salience} exceeds source max {max_source_salience} + 0.15"
                    )

                if preview.get("contains_strong_negative_valence") and valence < dp.negative_summary_max_valence:
                    valence = dp.negative_summary_max_valence

                encoded = _encode_memory(content, tags)
                cur = self._conn.execute(
                    """
                    INSERT INTO memories (
                        content, encoded, cues, tags, salience, valence,
                        strength, source, session_id, created_at, updated_at,
                        last_rehearsed_at, state, last_anchor_at, memory_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, 'semantic')
                    """,
                    (
                        content,
                        json.dumps(encoded, ensure_ascii=False),
                        " ".join(encoded["cues"]),
                        _join_tags(tags),
                        salience,
                        valence,
                        1.0 + salience,
                        f"dream:{cluster_id}",
                        "",
                        now,
                        now,
                        now,
                        now,
                    ),
                )
                semantic_id = int(cur.lastrowid)
                for sid in source_ids:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_provenance
                            (semantic_memory_id, source_memory_id, relation, created_at)
                        VALUES (?, ?, 'dream-derived', ?)
                        """,
                        (semantic_id, sid, now),
                    )
                archived_sources: list[int] = []
                if dp.archive_sources_after_apply:
                    for sid in source_ids:
                        self._archive_memory(sid, reason="dream-consolidated", now=now)
                        archived_sources.append(sid)
                results.append({
                    "cluster_id": cluster_id,
                    "status": "applied",
                    "semantic_memory_id": semantic_id,
                    "archived_sources": archived_sources,
                })
                # Drop durable preview after successful apply (idempotent re-apply
                # still works via content / source-set provenance checks).
                self._drop_preview(cluster_id)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return {"mode": "dream_apply", "enabled": True, "applied": results}

    def list_memories(
        self,
        *,
        limit: int = 20,
        state: str | None = None,
        include_archived: bool = False,
    ) -> list[dict]:
        if state:
            state_key = str(state).strip().lower()
            if state_key not in _ALLOWED_MEMORY_STATES:
                raise ValueError(
                    f"state must be one of {sorted(_ALLOWED_MEMORY_STATES)}, got {state!r}"
                )
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE state = ? ORDER BY updated_at DESC LIMIT ?",
                (state_key, max(1, int(limit))),
            ).fetchall()
        elif include_archived:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM memories WHERE state = 'active'
                ORDER BY updated_at DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [self._row_to_result(row, query_score=None) for row in rows]

    def stats(self) -> dict:
        active_row = self._conn.execute(
            """
            SELECT COUNT(*) AS count,
                   COALESCE(AVG(salience), 0) AS avg_salience,
                   COALESCE(AVG(valence), 0) AS avg_valence,
                   COALESCE(SUM(rehearsal_count), 0) AS rehearsals,
                   COALESCE(SUM(retrieval_count), 0) AS retrievals,
                   COALESCE(SUM(sleep_rehearsal_count), 0) AS sleep_rehearsals
            FROM memories WHERE state = 'active'
            """
        ).fetchone()

        archived_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE state = 'archived'"
        ).fetchone()
        total_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories"
        ).fetchone()
        episodic_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE memory_type = 'episodic'"
        ).fetchone()
        semantic_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE memory_type = 'semantic'"
        ).fetchone()
        dream_row = self._conn.execute(
            "SELECT COUNT(*) AS count FROM memories WHERE dream_candidate = 1 AND state = 'active'"
        ).fetchone()

        active_rows = self._conn.execute(
            "SELECT * FROM memories WHERE state = 'active'"
        ).fetchall()
        retained = sum(
            1 for row in active_rows if self._retention(row) > self.decay_threshold
        )
        protected_count = sum(1 for row in active_rows if self._is_row_protected(row))
        negative_active = sum(
            1
            for row in active_rows
            if float(row["valence"] or 0.0) <= self.policies.sleep.negative_valence_threshold
        )
        active_count = int(active_row["count"])
        capacity_blocked = (
            active_count >= self.policies.capacity.max_active_memories
            and protected_count >= self.policies.capacity.max_active_memories
        )
        warning = None
        if active_count and protected_count / max(1, active_count) > 0.10:
            warning = "protected memories exceed 10% of active store"

        now = self._now()
        oldest_active_age_days = 0.0
        if active_rows:
            oldest_created = min(
                _timestamp_value(row["created_at"], default=now) for row in active_rows
            )
            oldest_active_age_days = max(0.0, (now - oldest_created) / 86400.0)
        archived_rows = self._conn.execute(
            "SELECT archived_at, created_at FROM memories WHERE state = 'archived'"
        ).fetchall()
        oldest_archived_age_days = 0.0
        if archived_rows:
            oldest_arch = min(
                _timestamp_value(row["archived_at"])
                or _timestamp_value(row["created_at"], default=now)
                for row in archived_rows
            )
            oldest_archived_age_days = max(0.0, (now - oldest_arch) / 86400.0)

        sleep_rehearsals = int(active_row["sleep_rehearsals"])
        negative_rehearsal_ratio = (
            round(negative_active / max(1, active_count), 4) if active_count else 0.0
        )
        return {
            "count": int(total_row["count"]),
            "active_count": active_count,
            "archived_count": int(archived_row["count"]),
            "episodic_count": int(episodic_row["count"]),
            "semantic_count": int(semantic_row["count"]),
            "protected_count": protected_count,
            "negative_active_count": negative_active,
            "dream_candidate_count": int(dream_row["count"]),
            "retained_count": retained,
            "avg_salience": round(float(active_row["avg_salience"]), 3),
            "avg_valence": round(float(active_row["avg_valence"]), 3),
            "rehearsal_count": int(active_row["rehearsals"]),
            "retrieval_count": int(active_row["retrievals"]),
            "sleep_rehearsal_count": sleep_rehearsals,
            "negative_rehearsal_ratio": negative_rehearsal_ratio,
            "negative_prefetch_suppressed_count": int(
                self._negative_prefetch_suppressed_count
            ),
            "oldest_active_age_days": round(oldest_active_age_days, 3),
            "oldest_archived_age_days": round(oldest_archived_age_days, 3),
            "decay_threshold": self.decay_threshold,
            "base_stability_days": self.base_stability_days,
            "max_active_memories": self.policies.capacity.max_active_memories,
            "max_archived_memories": self.policies.capacity.max_archived_memories,
            "capacity_blocked": capacity_blocked,
            "warning": warning,
            "db_path": str(self.db_path),
            "dreaming_enabled": self.policies.dreaming.enabled,
        }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "forgetting_retention",
    "EbbinghausMemoryStore",
    "CapacityError",
    "_assert_safe_identifier",
    "_placeholders",
]
