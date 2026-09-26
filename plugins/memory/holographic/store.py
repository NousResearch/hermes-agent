"""SQLite-backed fact store with entity resolution and trust scoring (single-user Hermes memory plugin)."""

import os
import re
import sqlite3
import threading
from pathlib import Path

from . import holographic as hrr

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

_HELPFUL_DELTA, _UNHELPFUL_DELTA = 0.05, -0.10

# Entity extraction patterns. A NAME token starts uppercase, contains at least one lowercase letter
# or digit, and ends alphanumerically, so trailing punctuation is not swallowed ("Acme." -> "Acme").
#
# The "lowercase letter or digit" requirement is what keeps bare initials and initialisms out of the
# index. `[A-Z][a-z]+` -- the original -- also rejected them, so the only tokens newly accepted here
# are those containing a digit ("B2B", "NS-606", "V2.0"), which is the actual defect being fixed.
# Accepting all-caps tokens split one person across two entities ("The CEO John Smith approved"
# became "CEO John Smith" instead of "John Smith") and admitted junk ("I Think", "OK Google").
_CAP = r"[A-Z](?=[A-Za-z0-9&.\-]*[a-z0-9])[A-Za-z0-9&.\-]*[A-Za-z0-9]"
# Applied in order: capitalised multi-word phrases ("John Doe", "B2B Scaler") and quoted terms.
#
# NOTE the deliberate limit: a single capitalised word is NOT matched on its own. Requiring two words
# is what keeps ordinary sentence-initial words out of the index, and a loose single-token rule (e.g.
# "any capitalised token containing a digit") buys "NS-606" at the cost of "Q3", "B2B" and every
# other alphanumeric label -- junk entities that then co-occur with real ones and pollute related().
# Single-word names are covered instead by names the store already knows (_known_names_in), and by
# link_entities() for a name seen for the first time.
_RE_SINGLE_ENTITY = (re.compile(rf'\b({_CAP}(?:\s+{_CAP})+)\b'), re.compile(r'"([^"]+)"'), re.compile(r"'([^']+)'"))
# Both sides are restricted to capitalised NAME tokens. With a bare `\w+` the right-hand side ran on
# greedily -- "Alice Cooper aka The Falcon joined" produced an entity called "Falcon joined".
_RE_AKA = re.compile(rf'({_CAP}(?:\s+{_CAP})*)\s+(?i:aka|also known as)\s+({_CAP}(?:\s+{_CAP})*)')
# A leading article is not part of a name. Without this, "The Robosmart pilot renewed" yields an entity
# named "The Robosmart", which co-occurs with the real "Robosmart" and so pollutes related() as well as
# probe(): the junk entity looks like a genuine second party.
_RE_LEADING_ARTICLE = re.compile(r'^(?:The|A|An)\s+(?=[A-Z])')
_ENTITY_NAMES_SQL = "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?"
# Entity lookup order: exact name, then aliases (comma-separated; wrapped in commas for whole-alias matching).
_ENTITY_LOOKUPS = ("SELECT entity_id FROM entities WHERE name LIKE ?",
                   "SELECT entity_id FROM entities WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'")
_ENTITY_GAZETTEER_SQL = "SELECT name, aliases FROM entities"


def _strip_leading_article(name: str) -> str:
    """Drop a leading "The"/"A"/"An" so it is not stored as part of the name."""
    return _RE_LEADING_ARTICLE.sub("", name, count=1).strip()


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
        self._conn.executescript(_SCHEMA)
        if "hrr_vector" not in {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}:
            from hermes_cli.sqlite_util import add_column_if_missing
            add_column_if_missing(self._conn, "facts", "hrr_vector", "hrr_vector BLOB")
        self._conn.commit()

    def _one(self, sql: str, params=()):
        return self._conn.execute(sql, params).fetchone()

    def _write(self, sql: str, params=()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def add_fact(self, content: str, category: str = "general", tags: str = "") -> int:
        """Insert a fact and return its fact_id; on duplicate content (UNIQUE) return the existing fact_id untouched.
        Links extracted entities and rebuilds the category bank."""
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")
            try:
                fact_id: int = self._write("INSERT INTO facts (content, category, tags, trust_score) VALUES (?, ?, ?, ?)",
                                           (content, category, tags, self.default_trust)).lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                return int(self._one("SELECT fact_id FROM facts WHERE content = ?", (content,))["fact_id"])
            self._link_entities(fact_id, content)
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category)
            return fact_id

    def update_fact(self, fact_id: int, content: str | None = None, trust_delta: float | None = None,
                    tags: str | None = None, category: str | None = None) -> bool:
        """Partially update a fact (trust clamped to [0, 1]). Returns True if the row existed."""
        with self._lock:
            row = self._one("SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            changes = {col: val for col, val in {
                "content": content.strip() if content is not None else None, "tags": tags, "category": category,
                "trust_score": _clamp_trust(row["trust_score"] + trust_delta) if trust_delta is not None else None,
            }.items() if val is not None}
            assignments = ", ".join(["updated_at = CURRENT_TIMESTAMP"] + [f"{col} = ?" for col in changes])
            self._write(f"UPDATE facts SET {assignments} WHERE fact_id = ?", [*changes.values(), fact_id])
            if content is not None:  # re-extract entities and recompute the HRR vector
                self._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
                self._link_entities(fact_id, content)
                self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(category or self._one("SELECT category FROM facts WHERE fact_id = ?", (fact_id,))["category"])
            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._one("SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,))
            if row is None:
                return False
            self._conn.execute("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
            self._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._rebuild_bank(row["category"])
            return True

    def list_facts(self, category: str | None = None, min_trust: float = 0.0, limit: int = 50) -> list[dict]:
        """Browse facts ordered by trust_score descending, optionally filtered by category / min trust."""
        with self._lock:
            category_clause = "AND category = ? " if category is not None else ""
            params = [min_trust] + ([category] if category is not None else []) + [limit]
            sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, "
                   f"created_at, updated_at FROM facts WHERE trust_score >= ? {category_clause}"
                   "ORDER BY trust_score DESC LIMIT ?")
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

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

    def _extract_entities(self, text: str) -> list[str]:
        """Entity candidates: the regex table, "X aka Y", plus names already in the store.

        Known names are matched directly because the patterns deliberately require TWO capitalised
        words -- loose enough to catch every capitalised word otherwise. That makes a single-word
        name ("Robosmart", "Miranda") invisible to the patterns, so names the store already knows are
        matched explicitly. A single-word name the store has NEVER seen is still not discoverable
        here, by design; link_entities() is the explicit path for that.
        """
        raw = [m.group(1) for pattern in _RE_SINGLE_ENTITY for m in pattern.finditer(text)]
        for m in _RE_AKA.finditer(text):
            raw += [m.group(1), m.group(2)]
        raw += self._known_names_in(text)
        uniq: dict[str, str] = {}  # lower-cased key -> first-seen spelling, insertion-ordered
        for name in filter(None, (n.strip() for n in raw)):
            stripped = _strip_leading_article(name)
            if stripped:
                uniq.setdefault(stripped.lower(), stripped)
        return list(uniq.values())

    def _known_names_in(self, text: str) -> list[str]:
        """Canonical names of entities already in the store that appear in `text`.

        The gazetteer and its alternation are cached until a new entity is created: rebuilding
        them on every add_fact would scan the whole entities table once per fact."""
        gazetteer = self._entry.get("gazetteer")  # shared per DATABASE, not per instance
        if gazetteer is None:
            lookup: dict[str, str] = {}
            for row in self._conn.execute(_ENTITY_GAZETTEER_SQL).fetchall():
                canonical = row["name"]
                lookup[canonical.lower()] = canonical
                for alias in (row["aliases"] or "").split(","):
                    alias = alias.strip()
                    if alias:
                        lookup.setdefault(alias.lower(), canonical)
            # Longest first, so "B2B Scaler LLC" wins over "B2B Scaler".
            pattern = (re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(lookup, key=len, reverse=True)) + r")(?!\w)",
                                  re.IGNORECASE) if lookup else None)
            gazetteer = (lookup, pattern)
            self._entry["gazetteer"] = gazetteer
        lookup, pattern = gazetteer
        if pattern is None:
            return []
        found: dict[str, str] = {}
        for match in pattern.finditer(text):
            found.setdefault(match.group(1).lower(), lookup[match.group(1).lower()])
        return list(found.values())

    def _link_entities(self, fact_id: int, content: str) -> None:
        """Extract entities from content, resolve/create them, and link each to the fact."""
        for name in self._extract_entities(content):
            self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (fact_id, self._resolve_entity(name)))

    def link_entities(self, fact_id: int, names: "list[str]") -> list[int]:
        """Link `names` to `fact_id`, creating entities as needed; returns their entity ids.

        The public write-side counterpart to the read-only entity lookups. Extraction is
        deliberately high-precision, so this is how a caller attaches an entity it already knows
        about -- notably a single-word name the extractor has never seen."""
        if isinstance(names, str):  # a bare string would iterate CHARACTERS and create one entity per letter
            names = [names]
        with self._lock:
            if self._one("SELECT fact_id FROM facts WHERE fact_id = ?", (fact_id,)) is None:
                raise KeyError(f"fact_id {fact_id} not found")
            ids: list[int] = []
            for name in names or []:
                clean = _strip_leading_article((name or "").strip())
                if not clean:
                    continue
                entity_id = self._resolve_entity(clean)
                self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                            (fact_id, entity_id))
                if entity_id not in ids:
                    ids.append(entity_id)
            return ids

    def reindex_entities(self, fact_id: "int | None" = None, prune: bool = False) -> dict:
        """Re-run entity extraction over stored facts and link whatever it now finds.

        `prune=True` also REMOVES links the current extractor would not produce, which is what
        actually repairs a store polluted before an extraction fix: re-extraction can only ever
        ADD, so a row carrying junk like "The Robosmart" would otherwise keep it forever. Off by
        default because it equally drops entities attached explicitly via link_entities().

        HRR vectors AND category banks are recomputed: entities are encoded into the vectors as
        roles, and probe(category=...) reads the banks.
        """
        with self._lock:
            if fact_id is None:
                rows = self._conn.execute("SELECT fact_id, content FROM facts").fetchall()
            else:
                rows = self._conn.execute("SELECT fact_id, content FROM facts WHERE fact_id = ?", (fact_id,)).fetchall()
            for row in rows:
                row_id = int(row["fact_id"])
                if prune:
                    self._write("DELETE FROM fact_entities WHERE fact_id = ?", (row_id,))
                self._link_entities(row_id, row["content"])
                self._compute_hrr_vector(row_id, row["content"])
            for category_row in self._conn.execute("SELECT DISTINCT category FROM facts").fetchall():
                self._rebuild_bank(category_row["category"])
            return {"facts": len(rows)}

    def _resolve_entity(self, name: str) -> int:
        """Return the entity_id for a case-insensitive name or alias match, creating the entity if absent."""
        for sql in _ENTITY_LOOKUPS:
            row = self._one(sql, (name,))
            if row is not None:
                return int(row["entity_id"])
        self._entry["gazetteer"] = None  # a new entity changes the gazetteer for every instance
        return int(self._write("INSERT INTO entities (name) VALUES (?)", (name,)).lastrowid)  # type: ignore[arg-type]

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store the HRR vector for a fact (linked entities as roles). No-op without numpy."""
        if not self._hrr_available:
            return
        entities = [row["name"] for row in self._conn.execute(_ENTITY_NAMES_SQL, (fact_id,)).fetchall()]
        blob = hrr.phases_to_bytes(hrr.encode_fact(content, entities, self.hrr_dim))
        self._write("UPDATE facts SET hrr_vector = ? WHERE fact_id = ?", (blob, fact_id))

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        if not self._hrr_available:
            return
        bank_name = f"cat:{category}"
        rows = self._conn.execute("SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL", (category,)).fetchall()
        if not rows:
            self._write("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
            return
        bank_vector = hrr.bundle(*[hrr.bytes_to_phases(row["hrr_vector"], dim=self.hrr_dim) for row in rows])
        hrr.snr_estimate(self.hrr_dim, len(rows))  # warns when near capacity
        self._write("INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at) "
                    "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT(bank_name) DO UPDATE SET "
                    "vector = excluded.vector, dim = excluded.dim, fact_count = excluded.fact_count, "
                    "updated_at = excluded.updated_at", (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, len(rows)))

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
