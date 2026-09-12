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

# Entity extraction patterns, applied in order: capitalized multi-word phrases ("John Doe"), hyphenated names
# ("Pi-hole"), single capitalized words ("Alice"), double-quoted terms, single-quoted terms, then "X aka Y"
# (both sides). Words are matched as Unicode letter runs and shape-checked in Python (_is_name_word), because
# re cannot express "initial capital, lowercase rest" for any script. Single words shorter than three
# characters are ignored, as are the function words below. Quoted terms are capped at 48 characters: longer
# than that is a pasted paragraph, not a name (with no bound the pattern stored entire passages as entities -
# 31 such rows in one live store, the worst 426 characters).
_RE_MULTI_CAP = re.compile(r'\b([^\W\d_]+(?:\s+[^\W\d_]+)+)\b')
_RE_HYPHENATED = re.compile(r'\b([^\W\d_]{2,}(?:-[^\W\d_]+)+)\b')
_RE_SINGLE_CAP = re.compile(r'\b([^\W\d_]{3,})\b')
_RE_QUOTED = (re.compile(r'"([^"]{2,48})"'), re.compile(r"'([^']{2,48})'"))
# Function words, pronouns and sentence-starters that are capitalized like a proper noun but carry no
# entity meaning. Conservative by design: a missed name costs recall, a stored non-name adds a noise
# component to every bundle that mentions it. Shared with the local backfill script - keep one list.
_ENTITY_STOPLIST = frozenset("""
    A An The This That These Those It He She They We You I Me Us Him Her Them
    Is Are Was Were Be Been Has Have Had Do Does Did Will Would Can Could Should May Might
    Not No Yes But And Or Nor If In On At To For With From By As Of Into Over
    True False None
    All Any Some Each Both Few Many Most Other Such Only Own Same So Than Too Very Just
    Now Then Also Even Still Yet Like Get Got One Two Three First Last New Old More Less
    After Before During Since Until While About Above Across Again Against Along Among Around
    Here There Where When Why How What Which Who
    However Therefore Because Although Though Unless Whether""".split())
# Words dropped from the FRONT of a multi-word match ("The Thursday deployment" -> "Thursday").
# Deliberately much smaller than the list above: stripping every function word mangles real names that
# merely start with one ("One Page Rules" -> "Page Rules", "Old Fire Station" -> "Fire Station"), and a
# renamed entity is worse than a junk one because probing the real name then finds nothing.
_PHRASE_LEAD_STOPWORDS = frozenset(("A", "An", "The", "This", "That", "These", "Those"))
# Tokenizer for a matched phrase: offsets are taken from the match itself, so arbitrary whitespace
# between the words (double spaces, newlines) cannot shift a claimed span.
_RE_WORD = re.compile(r'[^\W\d_]+')


def _is_name_word(word: str) -> bool:
    """One initial capital then lowercase letters (Alice, Jozsef, Ivan) - excludes acronyms like BDFL."""
    return len(word) > 1 and word[0].isupper() and word[1:].islower()


def _is_hyphenated_name(name: str) -> bool:
    """First part a name word, later parts letter-initial and otherwise lowercase (Pi-hole, Foo-Bar-Baz)."""
    parts = name.split("-")
    return _is_name_word(parts[0]) and all(
        p[:1].isalpha() and (len(p) == 1 or p[1:].islower()) for p in parts[1:])
_RE_AKA = re.compile(r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE)
_ENTITY_NAMES_SQL = "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?"
# Entity lookup order: exact name, then aliases (comma-separated; wrapped in commas for whole-alias matching).
_ENTITY_LOOKUPS = ("SELECT entity_id FROM entities WHERE name LIKE ?",
                   "SELECT entity_id FROM entities WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'")


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
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
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
        """Regex entity candidates (see the pattern table), deduplicated case-insensitively in first-seen order.

        Two de-overlap rules keep one phrase from yielding several entities: a leading determiner is dropped
        from a multi-word match ("The Thursday deployment" -> "Thursday"), and a single capitalized word
        covered by a multi-word or hyphenated match is suppressed ("Acme Corporation" does not also bind
        "Acme" and "Corporation"; "Pi-hole" does not bind "Pi"). Only determiners are trimmed from the front
        of a phrase - see _PHRASE_LEAD_STOPWORDS. Names are matched as Unicode letter runs plus a shape check,
        so accented and non-Latin names extract the way ASCII ones do; scripts with no letter case at all
        (CJK) yield nothing, which no capitalisation rule can fix.
        """
        phrases = [p for m in _RE_MULTI_CAP.finditer(text) for p in self._phrase_entities(m)]
        hyphenated = [(m.group(1), m.span(1)) for m in _RE_HYPHENATED.finditer(text)
                      if _is_hyphenated_name(m.group(1))]
        claimed = [span for _, span in phrases + hyphenated]  # character ranges already bound as one entity

        def _covered(start: int, end: int) -> bool:
            return any(s <= start and end <= e for s, e in claimed)

        raw = [name for name, _ in phrases]
        raw += [name for name, _ in hyphenated]
        raw += [m.group(1) for m in _RE_SINGLE_CAP.finditer(text)
                if _is_name_word(m.group(1)) and m.group(1) not in _ENTITY_STOPLIST and not _covered(*m.span(1))]
        # A quoted term must carry a letter: "1234" or a run of punctuation is not an entity.
        for pattern in _RE_QUOTED:
            raw += [m.group(1).strip() for m in pattern.finditer(text) if any(c.isalpha() for c in m.group(1))]
        for m in _RE_AKA.finditer(text):
            raw += [m.group(1), m.group(2)]
        uniq: dict[str, str] = {}  # lower-cased key -> first-seen spelling, insertion-ordered
        for name in filter(None, (n.strip() for n in raw)):
            uniq.setdefault(name.lower(), name)
        return list(uniq.values())

    @staticmethod
    def _phrase_entities(match: "re.Match[str]") -> list[tuple[str, tuple[int, int]]]:
        """Every run of two or more capitalized words in a word run, minus any determiner prefix, plus spans.

        The pattern matches whole word runs because re cannot say "capitalized" for every script, so the runs
        are located here: "Alice works at Acme Corporation" yields "Acme Corporation" and "One Horse Town is a
        phrase" yields "One Horse Town" - the phrase is not required to start the sentence.
        """
        phrases: list[tuple[str, tuple[int, int]]] = []
        tokens = list(_RE_WORD.finditer(match.group(1)))
        index = 0
        while index < len(tokens):
            if not _is_name_word(tokens[index].group(0)):
                index += 1
                continue
            start = index
            while index < len(tokens) and _is_name_word(tokens[index].group(0)):
                index += 1
            run = tokens[start:index]
            if len(run) < 2:
                continue  # a lone capitalized word belongs to the single-word rule
            first = 0
            while first < len(run) and run[first].group(0) in _PHRASE_LEAD_STOPWORDS:
                first += 1
            if first >= len(run):
                continue
            name = " ".join(t.group(0) for t in run[first:])  # normalises the whitespace between words
            phrases.append((name, (match.start(1) + run[first].start(), match.start(1) + run[-1].end())))
        return phrases

    def _link_entities(self, fact_id: int, content: str) -> None:
        """Extract entities from content, resolve/create them, and link each to the fact."""
        for name in self._extract_entities(content):
            self._write("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (fact_id, self._resolve_entity(name)))

    def _resolve_entity(self, name: str) -> int:
        """Return the entity_id for a case-insensitive name or alias match, creating the entity if absent."""
        for sql in _ENTITY_LOOKUPS:
            row = self._one(sql, (name,))
            if row is not None:
                return int(row["entity_id"])
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
