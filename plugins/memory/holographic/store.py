"""SQLite-backed fact store with entity resolution and trust scoring (single-user Hermes memory plugin)."""

import logging
import os
import re
import sqlite3
import threading
from pathlib import Path

from . import holographic as hrr

logger = logging.getLogger(__name__)

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

# FTS5 content-sync index + triggers, extracted for self-heal rebuilds
# (identical to the FTS section of _SCHEMA above).
_FTS_SCHEMA = """
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
"""

_HELPFUL_DELTA, _UNHELPFUL_DELTA = 0.05, -0.10

# Entity extraction patterns, applied in order: capitalized multi-word phrases ("John Doe"), double-quoted terms,
# single-quoted terms, then "X aka Y" (both sides).
_RE_SINGLE_ENTITY = (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'), re.compile(r'"([^"]+)"'), re.compile(r"'([^']+)'"))
_RE_AKA = re.compile(r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE)
_ENTITY_NAMES_SQL = "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?"
# Entity lookup order: exact name, then aliases (comma-separated; wrapped in commas for whole-alias matching).
_ENTITY_LOOKUPS = ("SELECT entity_id FROM entities WHERE name LIKE ?",
                   "SELECT entity_id FROM entities WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'")


# R4 temporal lifecycle states. ACTIVE is the default; the lifecycle paths
# (stale/superseded/revoked) never delete history — only the explicit,
# user-invoked remove_fact() deletes (with lineage cleanup).
LIFECYCLE_ACTIVE = "active"
LIFECYCLE_SUPERSEDED = "superseded"
LIFECYCLE_STALE = "stale"
LIFECYCLE_REVOKED = "revoked"
LIFECYCLE_CONFLICT = "conflict"
LIFECYCLES = frozenset({LIFECYCLE_ACTIVE, LIFECYCLE_SUPERSEDED, LIFECYCLE_STALE,
                        LIFECYCLE_REVOKED, LIFECYCLE_CONFLICT})

# Retrieval precedence classes (lower = higher). Verified current knowledge
# always outranks stale/superseded history, regardless of trust scores.
LIFECYCLE_RANK = {LIFECYCLE_ACTIVE: 1, LIFECYCLE_CONFLICT: 2, LIFECYCLE_STALE: 3,
                  LIFECYCLE_SUPERSEDED: 4, LIFECYCLE_REVOKED: 5}
_VERIFIED_RANK = 0

_SOURCE_READ_CAP = 1024 * 1024  # hash at most 1MB of source evidence


def _split_slot_local(content: str) -> tuple[str, str, str]:
    """Minimal subject/predicate/value split (twin of retrieval._split_slot,
    kept local to avoid a store<->retrieval import cycle)."""
    import re as _re
    import unicodedata as _ud
    norm = _ud.normalize("NFKC", (content or "").strip().lower())
    parts = _re.split(r"\s*(=|->|→|คือ)\s*", norm, maxsplit=1)
    if len(parts) < 3 or not parts[0].strip() or not parts[2].strip():
        return "", "", ""
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _source_signature(path: str) -> str:
    """Deterministic file signature size:sha8 (first 1MB).

    mtime is deliberately EXCLUDED: a content-identical touch/copy must not
    invalidate evidence (mtime-only invalidation contradicts event-driven
    revalidation). Never raises; unstatable/unreadable paths yield ''
    (unverifiable, not an error). Only READS for hashing — never writes,
    executes, or follows the path beyond the OS's own resolution. NOTE: the
    existence/change signal is visible to API callers, so source_ref paths
    must come from trusted code (a memory-tool caller could otherwise probe
    file existence)."""
    import hashlib as _hl
    import os as _os
    try:
        if not path or not _os.path.isfile(path):
            return ""
        st = _os.stat(path)
        h = _hl.sha256()
        with open(path, "rb") as f:
            h.update(f.read(_SOURCE_READ_CAP))
        return f"{st.st_size}:{h.hexdigest()[:8]}"
    except Exception:
        return ""


def _escape_like_param(value: str) -> str:
    """Escape %, _ and backslash so entity names with wildcards match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _clamp_trust(value: float) -> float:
    return max(0.0, min(1.0, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring.

    Process-wide shared connection registry: SQLite allows one writer at a time and several providers
    coexist per process (main agent + every delegate_task subagent), so all instances for the same database
    share ONE connection and ONE re-entrant lock — writes are fully serialized and "database is locked" is
    impossible. Refcounted: closing one instance never tears the connection out from under a sibling."""

    _shared: dict = {}
    _shared_guard = threading.Lock()

    def __init__(self, db_path: "str | Path | None" = None, default_trust: float = 0.5, hrr_dim: int = 1024) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust, self.hrr_dim, self._hrr_available = _clamp_trust(default_trust), hrr_dim, hrr._HAS_NUMPY
        try:  # resolve() so symlinked/relative paths to the same file share ONE connection
            self._key = str(self.db_path.resolve())
        except OSError:
            self._key = str(self.db_path)
        with MemoryStore._shared_guard:
            entry = MemoryStore._shared.get(self._key)
            if entry is None:
                # Autocommit: a write that raises mid-method can't leave a dangling transaction (and its
                # write lock) open; the explicit commit() calls in _write are then harmless no-ops.
                conn = sqlite3.connect(self._key, check_same_thread=False, timeout=10.0, isolation_level=None)
                conn.row_factory = sqlite3.Row
                entry = MemoryStore._shared[self._key] = {"conn": conn, "lock": threading.RLock(), "refs": 0, "ready": False}
            entry["refs"] += 1
            self._entry, self._conn, self._lock = entry, entry["conn"], entry["lock"]
        with self._lock:  # schema initialised once per shared connection
            if not entry["ready"]:
                self._init_db()
                entry["ready"] = True

    def _init_db(self) -> None:
        """Create schema, enable WAL via the shared fallback helper (NFS/SMB/FUSE degrade gracefully), add hrr_vector to pre-HRR DBs."""
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        # Legacy DBs may predate base columns (category/tags/trust_score/...).
        # CREATE INDEX in _SCHEMA validates columns at creation time, so add
        # any missing base column BEFORE executescript (additive, keeps data).
        # NOTE: SQLite forbids CURRENT_TIMESTAMP as an ADD COLUMN default, so
        # legacy timestamp columns backfill as TEXT DEFAULT '' (temporal code
        # treats missing/unparseable timestamps as no-decay, never a crash).
        _base_cols = {"category": "TEXT DEFAULT 'general'", "tags": "TEXT DEFAULT ''",
                      "trust_score": "REAL DEFAULT 0.5", "retrieval_count": "INTEGER DEFAULT 0",
                      "helpful_count": "INTEGER DEFAULT 0",
                      "created_at": "TEXT DEFAULT ''",
                      "updated_at": "TEXT DEFAULT ''", "hrr_vector": "BLOB"}
        try:
            _tables = {r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
            if "facts" in _tables:
                _have = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
                for _name, _ddl in _base_cols.items():
                    if _name not in _have:
                        self._conn.execute(f"ALTER TABLE facts ADD COLUMN {_name} {_ddl}")
        except Exception:
            pass
        self._conn.executescript(_SCHEMA)
        if "hrr_vector" not in {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        try:  # FTS backfill for rows predating facts_fts (legacy DBs): the
            # UPDATE/DELETE triggers corrupt the FTS image when they touch a
            # row the index never saw. Rebuild only when counts disagree.
            n_facts = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            n_fts = self._conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
            if n_facts != n_fts:
                self._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        except Exception:
            pass
        try:  # self-heal probe: ancient DBs can carry an FTS image whose
            # trigger-driven writes fail on PRE-EXISTING rows (DatabaseError
            # on UPDATE/DELETE) while fresh rows work fine. Probe with a
            # no-op trigger-fired UPDATE on a real legacy row inside a
            # savepoint (zero side effects); on failure the index + triggers
            # are reconstructed from the facts table (derived state).
            probe_row = self._conn.execute(
                "SELECT fact_id FROM facts LIMIT 1").fetchone()
            if probe_row is not None:
                self._conn.execute("SAVEPOINT _fts_probe")
                try:
                    self._conn.execute(
                        "UPDATE facts SET updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                        (probe_row["fact_id"],))
                finally:
                    self._conn.execute("ROLLBACK TO SAVEPOINT _fts_probe")
                    self._conn.execute("RELEASE _fts_probe")
        except Exception:
            for _stmt in ("ROLLBACK TO SAVEPOINT _fts_probe", "RELEASE _fts_probe"):
                try:
                    self._conn.execute(_stmt)
                except Exception:
                    pass
            self._conn.execute("DROP TABLE IF EXISTS facts_fts")
            for _trigger in ("facts_ai", "facts_ad", "facts_au"):
                try:
                    self._conn.execute(f"DROP TRIGGER IF EXISTS {_trigger}")
                except Exception:
                    pass
            self._conn.executescript(_FTS_SCHEMA)
            self._conn.execute("INSERT INTO facts_fts(rowid, content, tags) "
                               "SELECT fact_id, content, COALESCE(tags, '') FROM facts")
        # R3: additive bank-sum column for exact incremental updates (nullable;
        # banks without it fall back to full rebuild — legacy-safe, idempotent).
        try:
            _bank_cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_banks)").fetchall()}
            if "vector_sum" not in _bank_cols:
                self._conn.execute("ALTER TABLE memory_banks ADD COLUMN vector_sum BLOB")
        except Exception:
            pass
        # R4: temporal lifecycle columns (all TEXT/INTEGER defaults; ADD COLUMN
        # defaults must be constants — no CURRENT_TIMESTAMP here) + lineage
        # table. Additive, idempotent, legacy-safe; history is never deleted.
        _lifecycle_cols = {"lifecycle": "TEXT DEFAULT 'active'",
                           "superseded_by": "INTEGER DEFAULT NULL",
                           "verified_at": "TEXT DEFAULT ''",
                           "verified_by": "TEXT DEFAULT ''",
                           "source_ref": "TEXT DEFAULT ''",
                           "source_sig": "TEXT DEFAULT ''",
                           "lifecycle_reason": "TEXT DEFAULT ''"}
        try:
            _have = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
            for _name, _ddl in _lifecycle_cols.items():
                if _name not in _have:
                    self._conn.execute(f"ALTER TABLE facts ADD COLUMN {_name} {_ddl}")
        except Exception:
            pass
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS fact_lineage ("
            " lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " old_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),"
            " new_fact_id INTEGER NOT NULL REFERENCES facts(fact_id),"
            " relation TEXT NOT NULL,"
            " reason TEXT DEFAULT '',"
            " verifier TEXT DEFAULT '',"
            " active INTEGER DEFAULT 1,"
            " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
            "CREATE INDEX IF NOT EXISTS idx_lineage_old ON fact_lineage(old_fact_id);"
            "CREATE INDEX IF NOT EXISTS idx_lineage_new ON fact_lineage(new_fact_id);")
        try:  # backfill for lineage tables created before the active flag
            _lcols = {row[1] for row in self._conn.execute("PRAGMA table_info(fact_lineage)").fetchall()}
            if "active" not in _lcols:
                self._conn.execute("ALTER TABLE fact_lineage ADD COLUMN active INTEGER DEFAULT 1")
        except Exception:
            pass
        self._conn.commit()

    def _one(self, sql: str, params=()):
        return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, params=()) -> sqlite3.Cursor:
        # Commit only outside an explicit transaction: callers inside
        # add_facts_batch()'s BEGIN rely on the single COMMIT at the end
        # (atomic batch). Normal flows never BEGIN, so behavior is unchanged.
        cur = self._conn.execute(sql, params)
        if not self._conn.in_transaction:
            self._conn.commit()
        return cur

    def add_fact(self, content: str, category: str = "general", tags: str = "") -> int:
        """Insert a fact and return its fact_id; on duplicate content (UNIQUE) return the existing fact_id untouched.
        Links extracted entities and rebuilds the category bank."""
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
            with self._atomic():
                try:
                    fact_id: int = self._conn.execute(
                        "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                        (content, category, tags, self.default_trust)).lastrowid  # type: ignore[assignment]
                except sqlite3.IntegrityError:
                    return int(self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))["fact_id"])
                self._link_entities(fact_id, content)
                self._compute_hrr_vector(fact_id, content)
                blob = self._one("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,))["hrr_vector"]
                self._update_bank_incremental(category, [blob] if blob is not None else [])
                return fact_id

    def update_fact(self, fact_id: int, content: str | None = None, trust_delta: float | None = None,
                    tags: str | None = None, category: str | None = None) -> bool:
        """Partially update a fact (trust clamped to [0, 1]). Returns True if the row existed.

        Changing ``content`` voids prior verification (verified_at/by/sig are
        cleared — old evidence must not vouch for new text)."""
        with self._lock:
            row = self._one("SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            changes = {col: val for col, val in {
                "content": content.strip() if content is not None else None, "tags": tags, "category": category,
                "trust_score": _clamp_trust(row["trust_score"] + trust_delta) if trust_delta is not None else None,
            }.items() if val is not None}
            if content is not None:  # new text, new evidence needed
                changes.update({"verified_at": "", "verified_by": "", "source_sig": ""})
            assignments = ", ".join(["updated_at = CURRENT_TIMESTAMP"] + [f"{col} = ?" for col in changes])
            with self._atomic():
                self._write(f"UPDATE facts SET {assignments} WHERE fact_id = ?", [*changes.values(), fact_id])
                if content is not None:  # re-extract entities and recompute the HRR vector
                    self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
                    self._link_entities(fact_id, content)
                    self._compute_hrr_vector(fact_id, content)
                self._rebuild_bank(category or self._one("SELECT category FROM facts WHERE fact_id = ?", (fact_id,))["category"])
            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. This is the explicit,
        user-invoked deletion path (stock API, covered by stock tests) — the
        R4 lifecycle paths (stale/supersede/revoke) never delete. Lineage
        rows touching the fact are removed with it (no orphans); rows it
        superseded demote to stale history (never stranded, never current).
        All-or-nothing nestable transaction."""
        with self._lock:
            row = self._one("SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            with self._atomic():
                self._conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
                try:
                    self._conn.execute("DELETE FROM fact_lineage WHERE old_fact_id = ? OR new_fact_id = ?",
                                       (fact_id, fact_id))
                    self._conn.execute("UPDATE facts SET lifecycle = 'stale', superseded_by = NULL "
                                       "WHERE superseded_by = ?", (fact_id,))
                except Exception as e:
                    logger.debug("Holographic lineage cleanup skipped: %s", e)
                self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._rebuild_bank(row["category"])
            return True

    def list_facts(self, category: str | None = None, min_trust: float = 0.0, limit: int = 50) -> list[dict]:
        """Browse facts ordered by trust_score descending, optionally filtered by category / min trust.
        Rows carry a computed ``stale`` flag (True unless lifecycle is active)."""
        with self._lock:
            category_clause = "AND category = ? " if category is not None else ""
            params = [min_trust] + ([category] if category is not None else []) + [limit]
            sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, "
                   f"created_at, updated_at FROM facts WHERE trust_score >= ? {category_clause}"
                   "ORDER BY trust_score DESC LIMIT ?")
            rows: list[dict] = []
            lifecycles: dict[int, str] = {}
            try:
                rows = [dict(r) for r in self._conn.execute(sql, params).fetchall()]
                lifecycles = {r["fact_id"]: (r["lifecycle"] or "active")
                              for r in self._conn.execute(
                                  "SELECT fact_id, lifecycle FROM facts").fetchall()}
            except Exception:
                pass
            for row in rows:
                row["stale"] = lifecycles.get(row["fact_id"], "active").lower() != "active"
            return rows

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Adjust trust asymmetrically: helpful -> +0.05 and helpful_count += 1; unhelpful -> -0.10.
        Returns {fact_id, old_trust, new_trust, helpful_count}. Raises KeyError if fact_id is unknown."""
        with self._lock:
            row = self._one("SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")
            old_trust: float = row["trust_score"]
            new_trust = _clamp_trust(old_trust + (_HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA))
            increment = 1 if helpful else 0
            self._write("UPDATE facts SET trust_score = ?, helpful_count = helpful_count + ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (new_trust, increment, fact_id))
            return {"fact_id": fact_id, "old_trust": old_trust, "new_trust": new_trust, "helpful_count": row["helpful_count"] + increment}

    # -- R4 temporal lifecycle (explicit transitions; history never deleted) --

    def _atomic(self):
        """Nestable transaction scope (savepoints): atomic batch/lineage
        updates with rollback on error. Safe under the re-entrant RLock and
        when an outer scope already began a transaction."""
        import contextlib as _cl

        @_cl.contextmanager
        def _scope():
            self._conn.execute("SAVEPOINT _r4_atomic")
            try:
                yield
                self._conn.execute("RELEASE _r4_atomic")
            except Exception:
                try:
                    self._conn.execute("ROLLBACK TO SAVEPOINT _r4_atomic")
                except Exception:
                    pass
                try:
                    self._conn.execute("RELEASE _r4_atomic")
                except Exception:
                    pass
                raise
        return _scope()

    def verify_fact(self, fact_id: int, verifier: str = "", source_ref: str = "",
                    lifecycle: str | None = None) -> bool:
        """Mark a fact verified with source evidence. Verification only ever
        confers ACTIVE (a forged ``lifecycle`` value is refused) and only
        from active/stale rows: revoked/conflict/superseded rows cannot be
        resurrected by verification (no silent graph repair). Timestamps are
        server-side; caller input is limited to verifier/source strings,
        which are stored as DATA. Returns False when the row is unknown."""
        if lifecycle is not None and lifecycle != LIFECYCLE_ACTIVE:
            return False
        with self._lock:
            row = self._one("SELECT fact_id, lifecycle FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            if (row["lifecycle"] or LIFECYCLE_ACTIVE) not in (LIFECYCLE_ACTIVE, LIFECYCLE_STALE):
                return False
            sig = _source_signature(source_ref)
            self._write("UPDATE facts SET lifecycle = 'active', verified_at = CURRENT_TIMESTAMP, "
                        "verified_by = ?, source_ref = ?, source_sig = ?, lifecycle_reason = '', "
                        "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                        (verifier or "", source_ref or "", sig, fact_id))
            return True

    def mark_stale(self, fact_id: int, reason: str = "") -> bool:
        """Event-driven invalidation (no blind TTL) from ACTIVE rows only.
        Revoked rows stay terminal (restore via a new fact); superseded rows
        keep their successor link (stale-marking would promote them).
        Keeps the row and all provenance. Returns False when refused/unknown."""
        with self._lock:
            row = self._one("SELECT fact_id, lifecycle FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            if (row["lifecycle"] or LIFECYCLE_ACTIVE) != LIFECYCLE_ACTIVE:
                return False  # only ACTIVE rows stale-mark (terminal/history stay)
            self._write("UPDATE facts SET lifecycle = 'stale', lifecycle_reason = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (reason or "", fact_id))
            return True

    def revoke_fact(self, fact_id: int, reason: str = "") -> bool:
        """Revoke a fact proven wrong. The row is retained (never deleted);
        prior attestation is cleared (a revoked row must not keep verified
        credentials — same rule as content updates)."""
        with self._lock:
            if self._one("SELECT fact_id FROM facts WHERE fact_id = ?", (fact_id,)) is None:
                return False
            self._write("UPDATE facts SET lifecycle = 'revoked', lifecycle_reason = ?, "
                        "verified_at = '', verified_by = '', source_sig = '', "
                        "updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?", (reason or "", fact_id))
            return True

    def supersede_fact(self, old_id: int, new_id: int, reason: str = "",
                       verifier: str = "") -> bool:
        """Explicit lineage: old -> superseded_by new, both rows preserved.
        The successor must be active/stale (never revoked/conflict); the old
        fact must not be revoked (revocation is terminal — restore via a new
        fact instead). The new link becomes the single canonical one (prior
        supersedes-links from the same old fact are retired but kept).
        Retry-safe: identical links are not duplicated. Self-supersession
        and unknown ids are refused."""
        if old_id == new_id:
            return False
        with self._lock:
            old = self._one("SELECT fact_id, lifecycle FROM facts WHERE fact_id = ?", (old_id,))
            new = self._one("SELECT fact_id, lifecycle FROM facts WHERE fact_id = ?", (new_id,))
            if old is None or new is None:
                return False
            if (old["lifecycle"] or LIFECYCLE_ACTIVE) == LIFECYCLE_REVOKED:
                return False
            if (new["lifecycle"] or LIFECYCLE_ACTIVE) not in (LIFECYCLE_ACTIVE, LIFECYCLE_STALE):
                return False
            if self._would_cycle(old_id, new_id):
                return False
            with self._atomic():
                self._write("UPDATE facts SET lifecycle = 'superseded', superseded_by = ?, "
                            "lifecycle_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?",
                            (new_id, reason or "", old_id))
                self._write("UPDATE fact_lineage SET active = 0 WHERE old_fact_id = ? "
                            "AND relation = 'supersedes'", (old_id,))
                dup = self._one("SELECT lineage_id FROM fact_lineage WHERE old_fact_id = ? "
                                "AND new_fact_id = ? AND relation = 'supersedes'", (old_id, new_id))
                if dup is None:
                    self._write("INSERT INTO fact_lineage (old_fact_id, new_fact_id, relation, reason, verifier) "
                                "VALUES (?, ?, 'supersedes', ?, ?)", (old_id, new_id, reason or "", verifier or ""))
                else:
                    self._write("UPDATE fact_lineage SET active = 1 WHERE lineage_id = ?", (dup["lineage_id"],))
            return True

    def _would_cycle(self, old_id: int, new_id: int) -> bool:
        """True if linking old -> new would close a supersession cycle
        (new already descends from old through superseded_by edges)."""
        try:
            seen: set = set()
            cur = new_id
            while cur is not None and cur not in seen:
                if cur == old_id:
                    return True
                seen.add(cur)
                row = self._one("SELECT superseded_by FROM facts WHERE fact_id = ?", (cur,))
                if row is None:
                    return False
                cur = row["superseded_by"]
            return False
        except Exception:
            return True  # fail-closed: refuse on unreadable graph

    def revalidate_fact(self, fact_id: int) -> str:
        """Re-check a fact against its source evidence. Returns UNCHANGED /
        CHANGED / MISSING / CONFLICT and applies event-driven invalidation
        (CHANGED/MISSING -> STALE, CONFLICT -> conflict). UNCHANGED never
        mutates state (no resurrection, no blind refresh). Unknown id ->
        MISSING without writes."""
        with self._lock:
            row = self._one("SELECT fact_id, content, lifecycle, category, source_ref, source_sig FROM facts "
                            "WHERE fact_id = ?", (fact_id,))
            if row is None:
                return "MISSING"
            if self._has_active_conflict(fact_id, row["content"] or "", row["category"]):
                self._write("UPDATE facts SET lifecycle = 'conflict', updated_at = CURRENT_TIMESTAMP "
                            "WHERE fact_id = ?", (fact_id,))
                return "CONFLICT"
            ref = row["source_ref"] or ""
            if not ref:
                return "UNCHANGED"
            import os as _os
            if not _os.path.isfile(ref):
                self._write("UPDATE facts SET lifecycle = 'stale', updated_at = CURRENT_TIMESTAMP "
                            "WHERE fact_id = ?", (fact_id,))
                return "MISSING"
            if (row["source_sig"] or "") and _source_signature(ref) != row["source_sig"]:
                self._write("UPDATE facts SET lifecycle = 'stale', updated_at = CURRENT_TIMESTAMP "
                            "WHERE fact_id = ?", (fact_id,))
                return "CHANGED"
            return "UNCHANGED"

    def _has_active_conflict(self, fact_id: int, content: str, category: str | None = None) -> bool:
        """Same (subject, predicate) with a different value among ACTIVE rows
        of the SAME category (cross-category same-slots are independent
        scopes, not conflicts)."""
        subj, pred, val = _split_slot_local(content)
        if not subj or not pred or not val:
            return False
        try:
            if category is None:
                own = self._one("SELECT category FROM facts WHERE fact_id = ?", (fact_id,))
                category = (own["category"] or "general") if own else "general"
            rows = self._conn.execute(
                "SELECT fact_id, content FROM facts WHERE fact_id != ? AND lifecycle = 'active' "
                "AND category = ?", (fact_id, category)).fetchall()
        except Exception:
            return False
        for other in rows:
            o_subj, o_pred, o_val = _split_slot_local(other["content"] or "")
            if o_subj == subj and o_pred == pred and o_val and o_val != val:
                return True
        return False

    def _extract_entities(self, text: str) -> list[str]:
        """Regex entity candidates (see the pattern table), deduplicated case-insensitively in first-seen order."""
        raw = [m.group(1) for pattern in _RE_SINGLE_ENTITY for m in pattern.finditer(text)]
        for m in _RE_AKA.finditer(text):
            raw += [m.group(1), m.group(2)]
        uniq: dict[str, str] = {}  # lower-cased key -> first-seen spelling, insertion-ordered
        for name in filter(None, (n.strip() for n in raw)):
            uniq.setdefault(name.lower(), name)
        return list(uniq.values())

    def _link_entities(self, fact_id: int, content: str) -> None:
        """Extract entities from content, resolve/create them, and link each to the fact."""
        for name in self._extract_entities(content):
            self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (fact_id, self._resolve_entity(name)))

    def _resolve_entity(self, name: str) -> int:
        """Return the entity_id for a case-insensitive name or alias match, creating the entity if absent."""
        for sql in _ENTITY_LOOKUPS:
            row = self._one(sql + " ESCAPE '\\'", (_escape_like_param(name),))
            if row is not None:
                return int(row["entity_id"])
        return int(self._write("INSERT INTO entities (name) VALUES (?)", (name,)).lastrowid)  # type: ignore[arg-type]

    def add_entity_alias(self, name: str, alias: str) -> bool:
        """Attach an alias (short name, acronym, known variant) to an entity.

        Creates the entity when absent. Returns False on empty input only;
        never raises. Powers query-side alias expansion in FactRetriever."""
        name = (name or "").strip()
        alias = (alias or "").strip()
        if not name or not alias or name.lower() == alias.lower():
            return False
        with self._lock:
            try:
                eid = self._resolve_entity(name)
                row = self._one("SELECT aliases FROM entities WHERE entity_id = ?", (eid,))
                current = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
                if alias.lower() in {a.lower() for a in current}:
                    return True
                current.append(alias)
                self._write("UPDATE entities SET aliases = ? WHERE entity_id = ?",
                            (", ".join(current), eid))
                return True
            except Exception:
                return False

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store the HRR vector for a fact (linked entities as roles). No-op without numpy."""
        if not self._hrr_available:
            return
        entities = [row["name"] for row in self._conn.execute(_ENTITY_NAMES_SQL, (fact_id,)).fetchall()]
        blob = hrr.phases_to_bytes(hrr.encode_fact(content, entities, self.hrr_dim))
        self._write("UPDATE facts SET hrr_vector = ? WHERE fact_id = ?", (blob, fact_id))

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors.

        Also (re)writes the complex aggregate sum used by exact incremental
        updates; banks are derived state and always rebuildable from facts.
        fact_id ordering keeps summation order identical to incremental
        appends (insertion order), so exactness survives VACUUM/deletes."""
        if not self._hrr_available:
            return
        bank_name = f"cat:{category}"
        # Backfill vectors for legacy rows that predate HRR (derived state
        # only; content untouched). Without this, upgraded legacy DBs can
        # never rebuild banks and health reports NEEDS_REPAIR permanently
        # while repair claims success (R9 legacy-upgrade gate).
        for fid, content in self._conn.execute(
                "SELECT fact_id, content FROM facts WHERE category = ? AND hrr_vector IS NULL",
                (category,)).fetchall():
            try:
                self._compute_hrr_vector(fid, content)
            except Exception:
                pass
        rows = self._conn.execute("SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL ORDER BY fact_id", (category,)).fetchall()
        if not rows:
            self._write("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
            return
        vecs = [hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows]
        bank_vector = hrr.bundle(*vecs)
        hrr.snr_estimate(self.hrr_dim, len(rows))  # warns when near capacity
        params = (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, len(rows))
        try:  # keep the exact incremental sum alongside the phases
            total = hrr.phases_to_complex(vecs[0])
            for _v in vecs[1:]:
                total = total + hrr.phases_to_complex(_v)
            self._write("INSERT INTO memory_banks (bank_name, vector, vector_sum, dim, fact_count, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(bank_name) DO UPDATE SET "
                        "vector = excluded.vector, vector_sum = excluded.vector_sum, dim = excluded.dim, "
                        "fact_count = excluded.fact_count, updated_at = excluded.updated_at",
                        (bank_name, hrr.phases_to_bytes(bank_vector), hrr.complex_sum_to_bytes(total),
                         self.hrr_dim, len(rows)))
        except Exception:
            self._write("INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at) "
                        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(bank_name) DO UPDATE SET "
                        "vector = excluded.vector, dim = excluded.dim, fact_count = excluded.fact_count, "
                        "updated_at = excluded.updated_at", params)

    def _update_bank_incremental(self, category: str, blobs: list) -> None:
        """Fold new fact vectors into the category bank in O(dim) via the
        stored complex sum (complex addition is associative, so the result is
        bitwise-identical to a full rebuild — verified by tests). Falls back
        to a full rebuild when no usable sum exists (legacy/empty/dim drift)."""
        live = [b for b in blobs if b is not None]
        if not self._hrr_available or not live:
            self._rebuild_bank(category)
            return
        bank_name = f"cat:{category}"
        try:
            row = self._one("SELECT vector_sum, dim, fact_count FROM memory_banks WHERE bank_name = ?", (bank_name,))
            if row is None or row["vector_sum"] is None or int(row["dim"]) != self.hrr_dim:
                raise ValueError("no usable sum")
            total = hrr.bytes_to_complex_sum(row["vector_sum"], self.hrr_dim)
        except Exception:
            self._rebuild_bank(category)
            return
        try:
            for blob in live:
                total = total + hrr.phases_to_complex(hrr.bytes_to_phases(blob, dim=self.hrr_dim))
            count = int(row["fact_count"] or 0) + len(live)
            phases = hrr.complex_to_phases(total)
            self._write("UPDATE memory_banks SET vector = ?, vector_sum = ?, fact_count = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE bank_name = ?",
                        (hrr.phases_to_bytes(phases), hrr.complex_sum_to_bytes(total), count, bank_name))
        except Exception:
            self._rebuild_bank(category)

    def add_facts_batch(self, items) -> list[int]:
        """Insert many facts atomically: one transaction, one bank update per
        category. Duplicate contents return existing ids (no row growth, no
        duplicate lineage). Any error rolls back the whole batch (no partial
        writes). Each item is (content, category, tags) or a dict."""
        entries = list(items)
        with self._lock:
            with self._atomic():
                ids: list[int] = []
                blobs_by_cat: dict[str, list] = {}
                for entry in entries:
                    if isinstance(entry, dict):
                        content = entry.get("content", "")
                        category = entry.get("category", "general")
                        tags = entry.get("tags", "")
                    else:
                        parts = list(entry)
                        content = parts[0] if len(parts) > 0 else ""
                        category = parts[1] if len(parts) > 1 else "general"
                        tags = parts[2] if len(parts) > 2 else ""
                    content = (content or "").strip()
                    if not content:
                        raise ValueError("content must not be empty")
                    try:
                        fid = self._conn.execute(
                            "INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                            (content, category, tags, self.default_trust)).lastrowid
                    except sqlite3.IntegrityError:
                        fid = int(self._one(
                            "SELECT fact_id FROM facts WHERE content = ?", (content,))["fact_id"])
                        ids.append(fid)
                        continue
                    self._link_entities(fid, content)
                    self._compute_hrr_vector(fid, content)
                    blob = self._one("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,))["hrr_vector"]
                    blobs_by_cat.setdefault(category, []).append(blob)
                    ids.append(int(fid))
                for _cat, _blobs in blobs_by_cat.items():
                    self._update_bank_incremental(_cat, _blobs)
                return ids

    @classmethod
    def release_all_under(cls, directory: "str | Path") -> int:
        """Force-close every shared connection whose database lives under ``directory``; returns the count.
        close() is refcount-driven, so a live holder (e.g. an agent's provider) keeps a profile's SQLite handle
        open, which on Windows makes rmtree of the profile fail. The directory is going away, so later use by a
        stale holder is expected to fail.

        That is exactly what a profile delete must break on Windows: the desktop's main ``serve`` process
        opens ``memory_store.db`` for every known profile, and ``rmtree`` of the profile directory fails
        with ``WinError 32`` while any of those handles is open (#88347). In a process that holds none (e.g.
        the CLI deleting from outside serve) this is a harmless no-op returning 0.
        """
        root = os.path.normcase(str(Path(directory).expanduser().resolve())) + os.sep
        with cls._shared_guard:
            doomed = [cls._shared.pop(key) for key in list(cls._shared) if os.path.normcase(key).startswith(root)]
            for entry in doomed:
                try:
                    with entry["lock"]:
                        entry["conn"].close()
                except Exception:
                    pass  # an already-closed/broken connection must not abort releasing siblings
        return len(doomed)

    def close(self) -> None:
        """Release this instance's reference; the connection closes with the last holder. Idempotent."""
        with MemoryStore._shared_guard:
            entry = getattr(self, "_entry", None)
            if entry is None:
                return
            entry["refs"] -= 1
            if entry["refs"] <= 0:
                try:
                    entry["conn"].close()
                finally:
                    # Pop only OUR entry: after release_all_under() a same-path store may have
                    # registered a FRESH entry under this key; a stale late close() must not evict it.
                    # See #88347.
                    if MemoryStore._shared.get(self._key) is entry:
                        MemoryStore._shared.pop(self._key, None)
            self._entry = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
