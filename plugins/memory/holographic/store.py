"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import os
import re
import sqlite3
import threading
from pathlib import Path

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
# Multi-word proper nouns: "John Doe", "New York City"
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# Single-word proper nouns: "Michael", "Anne", "Obsidian"
# Matches any standalone capitalized word (3+ chars to avoid "I", "A",
# "An").  Sentence-initial common words ("The", "When", "This", …) are
# filtered out by the _RE_STOPWORDS set below rather than by position,
# because real proper nouns can also appear at the start of a sentence
# ("Michael prefers dark roast coffee.").
_RE_SINGLE_CAP   = re.compile(r'\b([A-Z][a-z]{2,})\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)
# Common English words that are capitalized but are not proper nouns.
# These are filtered out of single-word entity candidates.
_RE_STOPWORDS = frozenset({
    'The', 'This', 'That', 'These', 'Those', 'There', 'Their', 'They',
    'Then', 'Thus', 'Therefore', 'However', 'Moreover', 'Meanwhile',
    'When', 'Where', 'While', 'What', 'Who', 'Why', 'How', 'Which',
    'After', 'Before', 'Although', 'Because', 'Since', 'Unless',
    'Some', 'Such', 'Same', 'Both', 'Each', 'Every', 'Either', 'Neither',
    'It', 'Its', 'In', 'On', 'At', 'By', 'For', 'From', 'Of', 'To',
    'As', 'An', 'And', 'But', 'Or', 'Nor', 'So', 'Yet',
    'I', 'We', 'You', 'He', 'She', 'They', 'Me', 'Him', 'Her', 'Us',
    'My', 'Your', 'His', 'Our',
    'A', 'If', 'No', 'Not', 'Yes', 'Oh', 'Ah',
    'Also', 'Even', 'Only', 'Just', 'Still', 'Now', 'Here',
    'First', 'Second', 'Third', 'Last', 'Next', 'Once',
    'Yes', 'Maybe', 'Perhaps',
    'Does', 'Did', 'Do', 'Done', 'Doing',
    'Has', 'Had', 'Have', 'Having',
    'Was', 'Were', 'Will', 'Would', 'Could', 'Should', 'Shall',
    'Can', 'May', 'Might', 'Must',
    'Is', 'Are', 'Am', 'Be', 'Been', 'Being',
    'About', 'Above', 'Across', 'After', 'Against', 'Along',
    'Among', 'Around', 'Behind', 'Below', 'Beside', 'Between',
    'Beyond', 'During', 'Except', 'Inside', 'Into', 'Near',
    'Outside', 'Over', 'Through', 'Throughout', 'Toward', 'Towards',
    'Under', 'Until', 'Upon', 'Within', 'Without',
})


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    # --- Process-wide shared connection registry -------------------------
    # SQLite permits only one writer at a time. Each MemoryStore instance used
    # to open its own connection guarded by its own RLock, so the several
    # providers that coexist in one process (the main agent plus every
    # delegate_task subagent) raced as independent WAL writers. Combined with
    # writes that were not rolled back on error, one connection could leave an
    # open write transaction that pinned the write lock and made every other
    # connection's write fail with "database is locked" for the full busy
    # timeout. All instances for the same database now share ONE connection and
    # ONE re-entrant lock, so access is fully serialized and cross-connection
    # contention is impossible. The shared connection is refcounted, so closing
    # one instance never tears the connection out from under a live sibling.
    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY

        # Acquire (or open) the process-wide shared connection for this DB.
        # resolve() (not just expanduser) so symlinked/relative paths to the
        # same file share ONE connection instead of silently reintroducing
        # the multi-writer contention this registry exists to prevent.
        try:
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                conn = sqlite3.connect(
                    self._key,
                    check_same_thread=False,
                    timeout=10.0,
                    # Autocommit: every statement is its own transaction, so a
                    # write that raises mid-method can never leave a dangling
                    # transaction (and its write lock) open. The explicit
                    # commit() calls below become harmless no-ops.
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                entry = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False}
                MemoryStore._shared[self._key] = entry
            entry["refs"] += 1
            self._entry = entry
            self._conn = entry["conn"]
            self._lock = entry["lock"]

        # Initialise the schema once per shared connection.
        with self._lock:
            if not self._entry["ready"]:
                self._init_db()
                self._entry["ready"] = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, category, tags, self.default_trust),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)

            return fact_id

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts using FTS5.

        Returns a list of fact dicts ordered by FTS5 rank, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # FTS5 AND-joins tokens by default, which zeroes out recall on
            # natural-language queries. Reuse the retriever's sanitizer
            # (stopword drop + OR-join content tokens). Imported lazily to
            # avoid a store->retrieval import cycle.
            from plugins.memory.holographic.retrieval import FactRetriever

            match_query = FactRetriever._sanitize_fts_query(query)
            params: list = [match_query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """

            rows = self._conn.execute(sql, params).fetchall()
            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Rebuild bank for relevant category
            cat = category or self._conn.execute(
                "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()["category"]
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using simple regex rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Single capitalized words (non-sentence-initial, non-stopword) e.g. "Michael"
        3. Double-quoted terms             e.g. "Python"
        4. Single-quoted terms             e.g. 'pytest'
        5. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        # Single-word proper nouns — skip common English stopwords that
        # happen to be capitalized (e.g. "The", "When", "This").
        # Words already captured as part of a multi-word match are deduped
        # by the ``seen`` set, so we only add genuinely new single tokens.
        for m in _RE_SINGLE_CAP.finditer(text):
            word = m.group(1)
            if word and word not in _RE_STOPWORDS:
                _add(word)

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``.

        ``close()`` is refcount-driven, so a live holder (e.g. an agent's
        memory provider) keeps a profile's SQLite handle open indefinitely.
        That is exactly what a profile delete must break on Windows: the
        desktop's main ``serve`` process opens ``memory_store.db`` for every
        known profile, and ``rmtree`` of the profile directory fails with
        ``WinError 32`` while any of those handles is open (#88347). This
        closes the matching connections unconditionally — the directory is
        going away, so later use by a stale holder is expected to fail — and
        returns how many were closed. In a process that holds none (e.g. the
        CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            # Snapshot the keys first so the registry stays stable while
            # connections are closed inside their per-database locks (closing
            # can run no user code, but this keeps the invariant obvious).
            doomed = [
                key
                for key in cls._shared
                if os.path.normcase(key).startswith(root)
            ]
            for key in doomed:
                entry = cls._shared.pop(key)
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    # A connection that is already closed or broken must not
                    # abort releasing its siblings.
                    pass
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference to the shared connection.

        The underlying connection is closed only when the last MemoryStore
        referencing the same database is closed, so closing one instance can
        never break sibling instances that still hold it. Idempotent.
        """
        if getattr(self, "_entry", None) is None:
            return
        with MemoryStore._shared_guard:
            entry = self._entry
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry. After release_all_under() force-
                    # closed this entry (profile delete, #88347) a same-path
                    # store may have re-registered a FRESH entry under the
                    # same key; a stale holder's late close() must not evict
                    # it — that would silently reintroduce the multi-writer
                    # contention this registry exists to prevent.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
