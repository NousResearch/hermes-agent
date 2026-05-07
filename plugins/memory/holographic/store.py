"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_CURRENT_SCHEMA_VERSION = 2
_CURRENT_ENCODING_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL UNIQUE,
    category         TEXT DEFAULT 'general',
    tags             TEXT DEFAULT '',
    trust_score      REAL DEFAULT 0.5,
    helpful_count    INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector       BLOB,
    encoding_version INTEGER NOT NULL DEFAULT 1
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

CREATE TABLE IF NOT EXISTS plugin_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2 schema migration.

    Drops `facts.retrieval_count` (ADR-002) via the SQLite table-rebuild
    dance: capture surviving columns, drop the FTS5 virtual table and
    its triggers (they reference `facts`), rebuild `facts` without the
    column, copy data, swap names, recreate indexes/FTS5/triggers, and
    rebuild the FTS5 index from the surviving content. Wrapped in a
    single transaction so a mid-flight crash leaves v1 intact.

    Idempotent: returns early if `retrieval_count` is already absent.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()]
    if "retrieval_count" not in cols:
        return
    keep = [c for c in cols if c != "retrieval_count"]
    keep_csv = ", ".join(keep)

    # Python sqlite3's legacy isolation mode auto-issues BEGIN before DML
    # and treats DDL as auto-commit. Switch to manual mode so the rebuild
    # below runs as one explicit transaction, then restore the caller's
    # mode in finally. Commit any pending implicit transaction first so
    # the caller's prior writes aren't tangled with our atomic block.
    if conn.in_transaction:
        conn.commit()
    prev_isolation_level = conn.isolation_level
    conn.isolation_level = None

    try:
        conn.execute("BEGIN")

        # FTS5 + triggers reference `facts` — drop before rebuilding.
        for stmt in (
            "DROP TRIGGER IF EXISTS facts_ai",
            "DROP TRIGGER IF EXISTS facts_ad",
            "DROP TRIGGER IF EXISTS facts_au",
            "DROP TABLE IF EXISTS facts_fts",
        ):
            conn.execute(stmt)

        conn.execute("""
            CREATE TABLE facts_new (
                fact_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content          TEXT NOT NULL UNIQUE,
                category         TEXT DEFAULT 'general',
                tags             TEXT DEFAULT '',
                trust_score      REAL DEFAULT 0.5,
                helpful_count    INTEGER DEFAULT 0,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hrr_vector       BLOB,
                encoding_version INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute(
            f"INSERT INTO facts_new ({keep_csv}) SELECT {keep_csv} FROM facts"
        )
        conn.execute("DROP TABLE facts")
        conn.execute("ALTER TABLE facts_new RENAME TO facts")

        # Recreate non-FTS indexes.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_trust ON facts(trust_score DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)"
        )

        # Recreate FTS5 + triggers, then rebuild the FTS index from facts.
        conn.execute("""
            CREATE VIRTUAL TABLE facts_fts USING fts5(
                content, tags, content=facts, content_rowid=fact_id
            )
        """)
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        conn.execute("""
            CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, new.content, new.tags);
            END
        """)
        conn.execute("""
            CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
            END
        """)
        conn.execute("""
            CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, new.content, new.tags);
            END
        """)

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # No transaction to roll back (BEGIN itself failed) — nothing
            # to undo, but we still want the original exception.
            pass
        raise
    finally:
        conn.isolation_level = prev_isolation_level


# Registry of schema migrations keyed by target version. Each entry is a
# function ``(conn) -> None`` that mutates the DB in a single transaction
# and is idempotent (safe to re-run if the prior bump didn't land). The
# runner in ``_init_db`` invokes them in version order.
_MIGRATIONS: "dict[int, Callable[[sqlite3.Connection], None]]" = {
    2: _migrate_v1_to_v2,
}


# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
        canonicalizer=None,
        known_entities: "list[str] | None" = None,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._canonicalizer = canonicalizer
        self._known_entities: list[str] = list(known_entities) if known_entities else []
        # Pre-compile lookup pattern for known entities. Word boundaries on each side.
        if self._known_entities:
            sorted_names = sorted(set(self._known_entities), key=len, reverse=True)
            self._known_entities_re = re.compile(
                r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted_names) + r")(?!\w)",
                re.IGNORECASE,
            )
            self._known_canonical_map = {n.lower(): n for n in self._known_entities}
        else:
            self._known_entities_re = None
            self._known_canonical_map = {}
        self._hrr_available = hrr._HAS_NUMPY
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables/indexes/triggers, reconcile columns, run data migrations.

        Schema management mirrors hermes_state.py: ``_SCHEMA`` is the single
        source of truth, ``_reconcile_columns`` declaratively ADDs any column
        present in ``_SCHEMA`` but missing from the live DB, and the
        ``schema_version`` table gates one-shot data migrations that cannot
        be expressed declaratively.
        """
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._reconcile_columns()

        row = self._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        if row is None:
            # No version row yet. Two cases:
            #   1. Truly fresh DB — _SCHEMA just built v2 tables; migrations
            #      are no-ops (idempotency guard inside each).
            #   2. Legacy DB that predates schema_version — needs migrations
            #      to run from v1 forward.
            # Both are handled by stamping current=1 and letting the runner
            # advance us; a truly fresh DB will pass through every migration
            # as a no-op since the post-migration shape already matches.
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (1,)
            )
            current = 1
        else:
            current = row["version"] if isinstance(row, sqlite3.Row) else row[0]

        # Run any registered migrations from current+1 up to the target,
        # in version order. Column additions are still handled declaratively
        # above; _MIGRATIONS is for changes that cannot be expressed that
        # way (column drops, table drops, data rewrites).
        for target in sorted(_MIGRATIONS):
            if current < target <= _CURRENT_SCHEMA_VERSION:
                _MIGRATIONS[target](self._conn)
                self._conn.execute(
                    "UPDATE schema_version SET version = ?", (target,)
                )
                current = target
        if current < _CURRENT_SCHEMA_VERSION:
            # No registered migration for the trailing gap — record the
            # version bump so subsequent boots don't re-attempt the search.
            self._conn.execute(
                "UPDATE schema_version SET version = ?",
                (_CURRENT_SCHEMA_VERSION,),
            )
        self._conn.commit()

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> "dict[str, dict[str, str]]":
        """Use an in-memory SQLite to parse declared columns per table.

        Avoids regex edge cases (DEFAULT expressions with commas, inline
        constraints, etc.) by letting SQLite itself parse the DDL and
        extracting column metadata via PRAGMA table_info. Returns
        ``{table: {column: "TYPE [NOT NULL] [DEFAULT x]"}}`` suitable
        for feeding into ALTER TABLE ADD COLUMN.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            tables: dict[str, dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: dict[str, str] = {}
                for row in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    _, name, ctype, notnull, default, pk = row
                    parts = [ctype] if ctype else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[name] = " ".join(parts)
                tables[tbl] = cols
            return tables
        finally:
            ref.close()

    def _reconcile_columns(self) -> None:
        """Declarative ADD COLUMN: diff live tables against ``_SCHEMA``.

        Idempotent and self-healing: if a deploy skips a step or a future
        column is added to ``_SCHEMA``, the next open of the DB picks it up.
        """
        expected = self._parse_schema_columns(_SCHEMA)
        for table_name, declared in expected.items():
            try:
                rows = self._conn.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue
            live_cols = {r[1] if isinstance(r, (tuple, list)) else r["name"] for r in rows}
            for col_name, col_type in declared.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        self._conn.execute(
                            f'ALTER TABLE "{table_name}" '
                            f'ADD COLUMN "{safe_name}" {col_type}'
                        )
                    except sqlite3.OperationalError:
                        # duplicate column from a race, or a constraint that
                        # ALTER cannot retro-add (e.g. UNIQUE) — surface only
                        # at debug; not load-bearing for correctness.
                        pass

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
            if self._canonicalizer:
                content = self._canonicalizer(content)

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
        descending.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            params: list = [query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.helpful_count,
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
            return [self._row_to_dict(r) for r in rows]

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
                       helpful_count, created_at, updated_at
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
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

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

        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        # Known-entity match (catches names the capitalized regex misses, e.g. "L-Charge",
        # "ATI", and ensures aliases get resolved to the canonical form).
        if self._known_entities_re is not None:
            for m in self._known_entities_re.finditer(text):
                hit = m.group(1)
                canonical = self._known_canonical_map.get(hit.lower(), hit)
                _add(canonical)

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

    def _compute_hrr_vector(self, fact_id: int, content: str, *, commit: bool = True) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable.

        Stamps ``encoding_version = _CURRENT_ENCODING_VERSION`` on every write
        so future algebra changes can be detected and self-healed by the
        memory doctor. Pass ``commit=False`` to participate in a larger
        transaction (e.g. ``rename_entity``).
        """
        with self._lock:
            if not self._hrr_available:
                return

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
                "UPDATE facts SET hrr_vector = ?, encoding_version = ? "
                "WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), _CURRENT_ENCODING_VERSION, fact_id),
            )
            if commit:
                self._conn.commit()

    def _rebuild_bank(self, category: str, *, commit: bool = True) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors.

        Pass ``commit=False`` to participate in a larger transaction.
        """
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
                if commit:
                    self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

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
            if commit:
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

    def resolve_alias_to_canonical(self, name: str) -> str:
        """If *name* matches a canonical entity name or any of its aliases
        (case-insensitive, hyphen/space tolerant), return the canonical name.
        Otherwise return *name* unchanged.
        """
        if not name:
            return name
        normalized = re.sub(r"[-\s_]+", "", name.lower())

        with self._lock:
            # Try exact name match (case-insensitive)
            row = self._conn.execute(
                "SELECT name FROM entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (name,),
            ).fetchone()
            if row is not None:
                return row["name"]

            # Try alias match — entity rows have aliases as comma-separated string.
            # Pull all rows that have aliases at all and scan in Python (fewer than
            # ~1k entities; comparison cost is trivial).
            rows = self._conn.execute(
                "SELECT name, aliases FROM entities WHERE aliases IS NOT NULL AND aliases <> ''"
            ).fetchall()
            for r in rows:
                aliases = (r["aliases"] or "").split(",")
                for a in aliases:
                    if re.sub(r"[-\s_]+", "", a.strip().lower()) == normalized:
                        return r["name"]
        return name

    def seed_canonical_entities(self, allowlist: list) -> int:
        """Pre-populate the entities table with canonical names + aliases so
        probes by any alias resolve to the canonical entity. Returns count
        of entities upserted.

        *allowlist* is a list of {"canonical": str, "aliases": [str,...]} dicts.
        """
        if not allowlist:
            return 0
        upserted = 0
        with self._lock:
            for entry in allowlist:
                canonical = (entry.get("canonical") or "").strip()
                if not canonical:
                    continue
                aliases = entry.get("aliases") or []
                aliases_str = ",".join(a.strip() for a in aliases if a.strip())
                row = self._conn.execute(
                    "SELECT entity_id, aliases FROM entities WHERE LOWER(name) = LOWER(?)",
                    (canonical,),
                ).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO entities (name, aliases) VALUES (?, ?)",
                        (canonical, aliases_str),
                    )
                else:
                    # Merge new aliases into existing
                    existing_aliases = {a.strip().lower() for a in (row["aliases"] or "").split(",") if a.strip()}
                    new_aliases = {a.strip().lower() for a in aliases if a.strip()}
                    merged = existing_aliases | new_aliases
                    if merged != existing_aliases:
                        # Preserve original casing where possible
                        casing_map = {a.strip().lower(): a.strip() for a in (row["aliases"] or "").split(",") if a.strip()}
                        for a in aliases:
                            casing_map[a.strip().lower()] = a.strip()
                        merged_str = ",".join(casing_map[k] for k in merged if k)
                        self._conn.execute(
                            "UPDATE entities SET aliases = ? WHERE entity_id = ?",
                            (merged_str, row["entity_id"]),
                        )
                upserted += 1
            self._conn.commit()
        return upserted

    def rename_entity(
        self,
        entity_id: int,
        new_name: str,
        *,
        add_aliases: "list[str] | None" = None,
    ) -> dict:
        """Rename a canonical entity and re-encode every fact linked to it.

        Atomic: the entity row update, alias merge, and per-fact HRR
        re-encode all happen in a single transaction. If anything fails
        mid-way, the DB is rolled back to its prior state — no half-renamed
        entities, no facts with stale vectors pointing at a renamed atom.

        The old canonical name is automatically merged into the entity's
        aliases so probes by the previous name still resolve. Pass
        ``add_aliases`` to merge additional aliases in the same transaction.

        Returns a summary dict::

            {"entity_id": int, "old_name": str, "new_name": str,
             "facts_re_encoded": int, "categories_rebuilt": [str, ...]}

        Raises ``KeyError`` if ``entity_id`` does not exist, ``ValueError``
        if ``new_name`` is empty.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("new_name must not be empty")

        with self._lock:
            row = self._conn.execute(
                "SELECT entity_id, name, aliases FROM entities WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"entity_id {entity_id} not found")

            self.backup_before("rename_entity")

            old_name: str = row["name"]
            existing_aliases = [
                a.strip() for a in (row["aliases"] or "").split(",") if a.strip()
            ]

            # Merge old_name + add_aliases into aliases, preserving casing.
            casing_map = {a.lower(): a for a in existing_aliases}
            if old_name and old_name.lower() != new_name.lower():
                casing_map.setdefault(old_name.lower(), old_name)
            for alias in add_aliases or []:
                a = (alias or "").strip()
                if a:
                    casing_map.setdefault(a.lower(), a)
            # Drop the new canonical from aliases if it appears (would be redundant).
            casing_map.pop(new_name.lower(), None)
            merged_aliases = ",".join(casing_map.values())

            # Find every fact linked to this entity (snapshot before commit).
            fact_rows = self._conn.execute(
                """
                SELECT f.fact_id, f.content, f.category
                FROM facts f
                JOIN fact_entities fe ON fe.fact_id = f.fact_id
                WHERE fe.entity_id = ?
                """,
                (entity_id,),
            ).fetchall()
            affected_facts = [
                (r["fact_id"], r["content"]) for r in fact_rows
            ]
            categories = sorted({r["category"] for r in fact_rows})

            # Single transaction. Python's sqlite3 starts an implicit
            # transaction before the first DML below; everything runs
            # inside it until the final commit() or rollback().
            try:
                self._conn.execute(
                    "UPDATE entities SET name = ?, aliases = ? WHERE entity_id = ?",
                    (new_name, merged_aliases, entity_id),
                )
                # Re-encode each linked fact (reads the new entity name via JOIN).
                for fid, content in affected_facts:
                    self._compute_hrr_vector(fid, content, commit=False)
                # Rebuild affected category banks from the fresh vectors.
                for cat in categories:
                    self._rebuild_bank(cat, commit=False)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            return {
                "entity_id":         entity_id,
                "old_name":          old_name,
                "new_name":          new_name,
                "facts_re_encoded":  len(affected_facts),
                "categories_rebuilt": categories,
            }

    def merge_entities(self, source_id: int, target_id: int) -> dict:
        """Merge ``source`` entity into ``target``. Atomic.

        Re-points every ``fact_entities`` row from ``source_id`` to
        ``target_id`` (using INSERT OR IGNORE so facts already linked to
        both collapse to one row), unions ``source.name`` and
        ``source.aliases`` into ``target.aliases``, deletes the source
        entity row, re-encodes every formerly-source-linked fact's
        hrr_vector against the new entity set, and rebuilds affected
        category banks. All inside a single transaction; rollback on
        any failure leaves the DB exactly as it was.

        Distinct from ``rename_entity``: rename mutates one row in
        place; merge consolidates two rows into one and reassigns the
        join table. Use rename when canonical name changes; use merge
        when historical ingestion drift produced two rows for the same
        conceptual entity.

        Returns::

            {"source_id": int, "target_id": int,
             "source_name": str, "target_name": str,
             "facts_re_pointed": int, "facts_re_encoded": int,
             "categories_rebuilt": [str, ...]}

        Raises ``KeyError`` if either id is missing, ``ValueError`` if
        ``source_id == target_id``.
        """
        if source_id == target_id:
            raise ValueError("source_id and target_id must differ")

        with self._lock:
            source = self._conn.execute(
                "SELECT entity_id, name, aliases FROM entities WHERE entity_id = ?",
                (source_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"source entity_id {source_id} not found")
            target = self._conn.execute(
                "SELECT entity_id, name, aliases FROM entities WHERE entity_id = ?",
                (target_id,),
            ).fetchone()
            if target is None:
                raise KeyError(f"target entity_id {target_id} not found")

            self.backup_before("merge_entities")

            source_name: str = source["name"]
            target_name: str = target["name"]

            # Union aliases (target ∪ source.aliases ∪ {source.name}); strip
            # target.name itself so the canonical isn't listed redundantly.
            target_aliases = [
                a.strip() for a in (target["aliases"] or "").split(",") if a.strip()
            ]
            source_aliases = [
                a.strip() for a in (source["aliases"] or "").split(",") if a.strip()
            ]
            casing_map = {a.lower(): a for a in target_aliases}
            for a in source_aliases:
                casing_map.setdefault(a.lower(), a)
            if source_name and source_name.lower() != target_name.lower():
                casing_map.setdefault(source_name.lower(), source_name)
            casing_map.pop(target_name.lower(), None)
            merged_aliases = ",".join(casing_map.values())

            # Snapshot facts linked to source — they need re-encoding after
            # the join move, since their entity-name set just changed.
            fact_rows = self._conn.execute(
                """
                SELECT f.fact_id, f.content, f.category
                FROM facts f
                JOIN fact_entities fe ON fe.fact_id = f.fact_id
                WHERE fe.entity_id = ?
                """,
                (source_id,),
            ).fetchall()
            affected_facts = [(r["fact_id"], r["content"]) for r in fact_rows]
            categories = sorted({r["category"] for r in fact_rows})

            # Single transaction. Implicit BEGIN on first DML below.
            try:
                # Re-point: copy (fact_id, target) for every (fact_id, source);
                # INSERT OR IGNORE collapses pre-existing dual links.
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) "
                    "SELECT fact_id, ? FROM fact_entities WHERE entity_id = ?",
                    (target_id, source_id),
                )
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE entity_id = ?",
                    (source_id,),
                )
                # Merge aliases and drop the source entity row.
                self._conn.execute(
                    "UPDATE entities SET aliases = ? WHERE entity_id = ?",
                    (merged_aliases, target_id),
                )
                self._conn.execute(
                    "DELETE FROM entities WHERE entity_id = ?",
                    (source_id,),
                )
                # Re-encode each affected fact (now reads the new entity set
                # — source is gone, target may already have been there).
                for fid, content in affected_facts:
                    self._compute_hrr_vector(fid, content, commit=False)
                for cat in categories:
                    self._rebuild_bank(cat, commit=False)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

            return {
                "source_id":          source_id,
                "target_id":          target_id,
                "source_name":        source_name,
                "target_name":        target_name,
                "facts_re_pointed":   len(affected_facts),
                "facts_re_encoded":   len(affected_facts),
                "categories_rebuilt": categories,
            }

    def get_state(self, key: str) -> "str | None":
        """Read a value from the plugin_state key/value table."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM plugin_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        """Upsert a value into the plugin_state key/value table."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO plugin_state (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def canonicalize_existing_facts(self, canonicalizer, since_days: "int | None" = None) -> dict:
        """Apply *canonicalizer* to every existing fact's content.

        If *since_days* is a positive int, only facts whose updated_at is within
        that window are walked. Older facts are left as-is — trade-off accepted:
        once the corpus stabilizes, new aliases won't retroactively rewrite
        long-settled facts. Run a full pass (since_days=None) when that matters.

        For facts whose canonicalized content collides with another existing
        fact, the loser is removed (its entity links are reattached to the
        survivor). Returns a summary dict with counts.
        """
        if canonicalizer is None:
            return {"changed": 0, "merged": 0, "skipped": 0}

        with self._lock:
            if since_days and since_days > 0:
                rows = self._conn.execute(
                    "SELECT fact_id, content FROM facts "
                    "WHERE updated_at >= datetime('now', ?) "
                    "ORDER BY fact_id",
                    (f"-{int(since_days)} days",),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT fact_id, content FROM facts ORDER BY fact_id"
                ).fetchall()

        changed = 0
        merged = 0
        skipped = 0
        for row in rows:
            fid = int(row["fact_id"])
            old_content = row["content"]
            new_content = canonicalizer(old_content)
            if new_content == old_content:
                skipped += 1
                continue

            with self._lock:
                existing = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ? AND fact_id != ?",
                    (new_content, fid),
                ).fetchone()

            if existing is not None:
                # Merge: rewire entity links from this fact to the survivor, then drop.
                survivor = int(existing["fact_id"])
                with self._lock:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) "
                        "SELECT ?, entity_id FROM fact_entities WHERE fact_id = ?",
                        (survivor, fid),
                    )
                    self._conn.execute(
                        "DELETE FROM fact_entities WHERE fact_id = ?", (fid,)
                    )
                    self._conn.execute(
                        "DELETE FROM facts WHERE fact_id = ?", (fid,)
                    )
                    self._conn.commit()
                merged += 1
            else:
                # Update in place. update_fact handles entity re-extraction + HRR rebuild.
                self.update_fact(fid, content=new_content)
                changed += 1

        # Rebuild banks for all categories, since vectors moved.
        with self._lock:
            cats = [r["category"] for r in self._conn.execute(
                "SELECT DISTINCT category FROM facts"
            ).fetchall()]
        for cat in cats:
            self._rebuild_bank(cat)

        return {"changed": changed, "merged": merged, "skipped": skipped}

    def backup_before(self, operation_name: str, *, keep: int = 30) -> Path:
        """Snapshot the live DB before a destructive op.

        Writes ``<db_parent>/backups/{operation_name}-{timestamp}.db`` using
        SQLite's online backup API (WAL-safe, sees the latest committed
        state). Then prunes all but the *keep* most recent ``.db`` files in
        that directory so the safety net stays bounded.

        Called automatically by ``rename_entity`` and ``merge_entities``;
        also safe to call directly before ad-hoc destructive work. If the
        snapshot can't be written, raises — a destructive op without its
        backup is worse than the op not happening at all.
        """
        op = (operation_name or "").strip()
        if not op:
            raise ValueError("operation_name must not be empty")
        # Strip path separators so callers can't escape the backups dir.
        op = op.replace("/", "_").replace("\\", "_")

        backups_dir = self.db_path.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)

        # Microseconds disambiguate same-second calls (rotation tests, fast loops).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backups_dir / f"{op}-{timestamp}.db"

        with self._lock:
            dest = sqlite3.connect(str(backup_path))
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

        self._prune_backups(backups_dir, keep=keep)
        return backup_path

    def list_backups(self, operation_name: "str | None" = None) -> "list[Path]":
        """List existing backup snapshots, newest first.

        Filename timestamps are formatted ``YYYYMMDD_HHMMSS_micro`` so a
        descending lexicographic sort = reverse-chronological.

        Pass ``operation_name`` to filter (e.g. only ``rename_entity`` snapshots);
        omit to see everything in the backups directory.
        """
        backups_dir = self.db_path.parent / "backups"
        if not backups_dir.exists():
            return []
        if operation_name:
            op = operation_name.replace("/", "_").replace("\\", "_")
            pattern = f"{op}-*.db"
        else:
            pattern = "*.db"
        return sorted(backups_dir.glob(pattern), reverse=True)

    @staticmethod
    def _prune_backups(backups_dir: Path, keep: int) -> None:
        """Delete all but the *keep* most recent ``.db`` files in *backups_dir*.

        Sorts by filename. Timestamps are formatted ``YYYYMMDD_HHMMSS_micro``
        so lexicographic order equals chronological order — more deterministic
        than mtime when many files are created in the same second.
        """
        if keep < 0:
            return
        files = sorted(backups_dir.glob("*.db"), reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
