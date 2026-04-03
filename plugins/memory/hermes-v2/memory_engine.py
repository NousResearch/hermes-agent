"""
Hermes Memory Engine V2 — SQLite-backed memory with FTS5 search and tiered lifecycle.

Cannibalized from:
- HiveMind (memory.rs): schema, hybrid search, tiers, power-law decay, strength model
- Claude Code (memdir/): type taxonomy, consolidation patterns

Design principles:
- Zero new dependencies (sqlite3 is stdlib, FTS5 is built-in)
- Backward-compatible with flat-file MemoryStore
- Frozen snapshot pattern preserved (no mid-session prompt changes)
- WAL mode for concurrent access (CLI + gateway + cron)
"""

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .yake import extract_keywords

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_TYPES = ("general", "preference", "correction", "project", "reference")
MEMORY_TIERS = ("active", "archived", "consolidated", "superseded")
MEMORY_TARGETS = ("memory", "user")

# Trust scoring constants (ported from holographic)
# Asymmetric: bad info sinks fast, good info rises slowly
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0
_TRUST_DEFAULT   =  0.5

# Entity extraction regex patterns (ported from holographic store.py)
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA          = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)

TIER_WEIGHTS = {"active": 1.0, "archived": 0.5, "consolidated": 0.3, "superseded": 0.2}
TYPE_BOOSTS = {
    "preference": 1.2,
    "correction": 1.3,
    "project": 1.0,
    "reference": 0.8,
    "general": 1.0,
}

# Power-law decay exponent (from HiveMind memory.rs)
RECENCY_DECAY_EXPONENT = -0.3

# Minimum BM25 score to include in results
DEFAULT_MIN_RELEVANCE = 0.1

# Stale threshold for archival (days)
ARCHIVE_STALE_DAYS = 90
ARCHIVE_MIN_STRENGTH = 1.1

# Near-duplicate threshold (raw BM25 score — higher = stricter)
# Exact duplicates score 10+. Topically similar but different content scores 1-5.
# Near-exact rephrases score ~7+. Topical overlap with different info scores 3-6.
# Previous threshold of 5.0 caused false supersessions (unrelated entries sharing
# a few keywords like "Discord" or "HiveMind" got incorrectly superseded).
DEDUP_THRESHOLD = 8.0

SCHEMA_VERSION = 3

# Hard budget: max active memories per target before forced compaction.
# memory target: more entries (project facts, references, observations)
# user target: fewer entries (preferences, profile, corrections)
# When exceeded, lowest-strength active entries are archived to make room.
MAX_ACTIVE_MEMORY = 50
MAX_ACTIVE_USER = 25

# Chunking constants (from HiveMind memory.rs)
CHUNK_MAX_CHARS = 1600
CHUNK_OVERLAP_CHARS = 320
CHUNK_MIN_CONTENT_LEN = 500

# Type tag prefixes for prompt rendering
TYPE_TAGS = {
    "preference": "pref",
    "correction": "corr",
    "project": "proj",
    "reference": "ref",
    "general": "gen",
}

# Module-level cache for fastembed local embedder
_LOCAL_EMBEDDER = None

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    target          TEXT NOT NULL DEFAULT 'memory',
    type            TEXT NOT NULL DEFAULT 'general',
    source          TEXT NOT NULL DEFAULT 'agent',
    tags            TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_accessed   TEXT,
    access_count    INTEGER NOT NULL DEFAULT 0,
    strength        REAL NOT NULL DEFAULT 1.0,
    trust_score     REAL NOT NULL DEFAULT 0.5,
    helpful_count   INTEGER NOT NULL DEFAULT 0,
    tier            TEXT NOT NULL DEFAULT 'active',
    superseded_by   TEXT,
    session_id      TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, type,
    content='memories', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, type)
    VALUES (new.rowid, new.content, new.tags, new.type);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
    VALUES ('delete', old.rowid, old.content, old.tags, old.type);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content, tags, type ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
    VALUES ('delete', old.rowid, old.content, old.tags, old.type);
    INSERT INTO memories_fts(rowid, content, tags, type)
    VALUES (new.rowid, new.content, new.tags, new.type);
END;

CREATE TABLE IF NOT EXISTS memory_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- Graph / chunking / embedding tables (V2)

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    hash        TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id    TEXT PRIMARY KEY,
    embedding   BLOB,
    model       TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.5,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, relation)
);

-- FTS5 for chunks
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks', content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_chunks_memory_id ON chunks(memory_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

-- Trust score index (V3)
CREATE INDEX IF NOT EXISTS idx_memories_trust ON memories(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_target_tier ON memories(target, tier);

-- Entity resolution tables (V3, ported from holographic)
CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id   INTEGER NOT NULL REFERENCES entities(entity_id),
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
"""

_META_DEFAULTS = {
    "schema_version": str(SCHEMA_VERSION),
    "migrated_from_flat": "0",
    "last_consolidation": "",
    "consolidation_session_count": "0",
}


# ---------------------------------------------------------------------------
# Module-level utilities (ported from HiveMind memory.rs)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors. Uses numpy when available, pure Python fallback.

    Ported from HiveMind memory.rs lines 539-556.
    Returns 0.0 on empty, mismatched dims, or near-zero norms.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import numpy as np
        a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
        norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
    except ImportError:
        # Pure Python fallback
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            norm_a += a[i] * a[i]
            norm_b += b[i] * b[i]
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def chunk_text(
    content: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list:
    """Split text into overlapping chunks for indexing.

    Ported from HiveMind memory.rs lines 455-504.
    Returns list of (start_line, end_line, text) tuples.
    Line numbers are 1-based.
    """
    lines = content.split("\n")
    if not lines:
        return []

    chunks = []
    current_lines = []  # list of (line_index, line_text)
    current_chars = 0

    def flush():
        if not current_lines:
            return
        start = current_lines[0][0] + 1
        end = current_lines[-1][0] + 1
        text = "\n".join(l for _, l in current_lines)
        chunks.append((start, end, text))

    for i, line in enumerate(lines):
        line_size = len(line) + 1

        if current_chars + line_size > max_chars and current_lines:
            flush()

            if overlap_chars > 0:
                acc = 0
                kept_start = len(current_lines)
                for j in range(len(current_lines) - 1, -1, -1):
                    acc += len(current_lines[j][1]) + 1
                    kept_start = j
                    if acc >= overlap_chars:
                        break
                current_lines = current_lines[kept_start:]
                current_chars = sum(len(l) + 1 for _, l in current_lines)
            else:
                current_lines = []
                current_chars = 0

        current_lines.append((i, line))
        current_chars += line_size

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Topic classification (ported from HiveMind memory.rs lines 1186-1370)
# ---------------------------------------------------------------------------

_TECH_KEYWORDS = frozenset([
    "function", "class", "api", "database", "server", "deploy", "code",
    "debug", "error", "bug", "compile", "build", "test", "config",
    "docker", "git", "rust", "typescript", "python", "javascript",
    "react", "tauri", "sql", "http", "endpoint", "backend", "frontend",
    "algorithm", "struct", "module", "import", "dependency", "package",
    "binary", "runtime", "compiler", "lint", "refactor", "migration",
    "schema", "query", "index", "cache", "thread", "async", "mutex",
    "vector", "embedding", "model", "inference", "gpu", "vram", "cuda",
    "wsl", "linux", "windows", "terminal", "bash", "command",
])

_PROJECT_KEYWORDS = frozenset([
    "plan", "roadmap", "phase", "milestone", "priority", "design",
    "architecture", "approach", "strategy", "goal", "requirement",
    "feature", "sprint", "task", "ticket", "issue",
])

_PERSONAL_KEYWORDS = frozenset([
    "prefer", "like", "hate", "favorite", "hobby", "style",
    "morning", "evening", "feel", "mood", "name", "birthday",
])

_PROJECT_TAGS = frozenset(["decision", "instruction", "correction"])
_PERSONAL_TAGS = frozenset(["preference"])


def classify_topic(content: str, keywords: list = None, tags: list = None) -> str:
    """Keyword-based topic classification. Ported from HiveMind memory.rs.

    Returns one of MEMORY_TYPES: 'general', 'preference', 'project', 'reference'.
    Maps topic:technical -> 'general', topic:personal -> 'preference',
    topic:project -> 'project'.
    """
    if keywords is None:
        keywords = extract_keywords(content)
    if tags is None:
        tags = []

    lower = content.lower()
    kw_set = [k.lower() for k in keywords]
    tag_set = [t.lower() for t in tags]

    # Technical score
    tech_score = sum(1 for k in kw_set if k in _TECH_KEYWORDS)
    has_code = any(marker in lower for marker in ["```", "fn ", "const ", "import ", "class ", "def "])
    tech_total = tech_score + (2 if has_code else 0)

    # Project score
    project_tag_score = sum(1 for t in tag_set if t in _PROJECT_TAGS)
    project_kw = sum(1 for k in kw_set if k in _PROJECT_KEYWORDS)
    project_total = project_tag_score + project_kw

    # Personal score
    personal_tag_score = sum(1 for t in tag_set if t in _PERSONAL_TAGS)
    personal_kw = sum(1 for k in kw_set if k in _PERSONAL_KEYWORDS)
    personal_total = personal_tag_score + personal_kw

    scores = [
        (tech_total, "general"),      # topic:technical -> general type
        (project_total, "project"),
        (personal_total, "preference"),
    ]
    best = max(scores, key=lambda x: x[0])

    if best[0] >= 2:
        return best[1]
    return "general"


# ---------------------------------------------------------------------------
# Embedding stub (content-hash caching pattern from HiveMind)
# ---------------------------------------------------------------------------


def _get_local_embedding(content: str) -> list:
    """Generate embedding locally via fastembed (zero API keys needed)."""
    try:
        from fastembed import TextEmbedding
        global _LOCAL_EMBEDDER
        if _LOCAL_EMBEDDER is None:
            _LOCAL_EMBEDDER = TextEmbedding('BAAI/bge-small-en-v1.5')
        embeddings = list(_LOCAL_EMBEDDER.embed([content]))
        return list(float(x) for x in embeddings[0])
    except Exception:
        return []


def generate_embedding(content: str, model: str = None, config: dict = None) -> list:
    """Generate embedding vector using whatever provider Hermes has configured.

    Provider cascade (adapts to YOUR API keys):
    0. Local fastembed (BAAI/bge-small-en-v1.5) — zero API keys, always available
    1. Configured model from config (memory.embedding_model)
    2. Auto-detect from available keys:
       - OPENAI_API_KEY -> text-embedding-3-small
       - ANTHROPIC_API_KEY -> voyage-3 (via litellm)
       - OPENROUTER_API_KEY -> openrouter/openai/text-embedding-3-small
    3. Graceful degradation: no keys / no litellm -> return []
       (search falls back to BM25-only, which still works)
    """
    config = config or {}

    # Try local embeddings first (zero API key, always available)
    if not model and config.get("embedding_provider", "auto") in ("auto", "local"):
        local = _get_local_embedding(content)
        if local:
            return local

    # Resolve model from config or auto-detect from available API keys
    if not model:
        model = config.get("embedding_model", "")
    if not model:
        model = _detect_embedding_model()
    if not model:
        return []  # No provider available — BM25 fallback

    try:
        from litellm import embedding as litellm_embedding
        response = litellm_embedding(model=model, input=[content])
        return response.data[0]["embedding"]
    except ImportError:
        logger.debug("litellm not installed — embeddings disabled")
        return []
    except Exception as e:
        logger.debug("Embedding generation failed (%s): %s", model, e)
        return []


def _detect_embedding_model() -> str:
    """Auto-detect the best embedding model from available API keys.

    Checks environment for provider keys and returns the appropriate
    embedding model string for litellm. Returns '' if no provider found.
    """
    import os
    if os.environ.get("OPENAI_API_KEY"):
        return "text-embedding-3-small"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter/openai/text-embedding-3-small"
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Anthropic doesn't have native embeddings, but Voyage AI
        # is available via litellm with VOYAGE_API_KEY
        if os.environ.get("VOYAGE_API_KEY"):
            return "voyage/voyage-3-lite"
        return ""  # Anthropic alone can't do embeddings
    if os.environ.get("GLM_API_KEY"):
        return ""  # GLM embeddings not in litellm yet
    return ""


def _clamp_trust(value: float) -> float:
    """Clamp trust score to [0.0, 1.0]."""
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


def _hash_text(text: str) -> str:
    """SHA256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# MemoryEngine
# ---------------------------------------------------------------------------


def _memory_staleness_suffix(mem: dict, now: datetime = None) -> str:
    """Return a staleness warning suffix for memories >7 days old.

    Ported from Claude Code's memoryAge.ts: memoryAgeDays / memoryFreshnessText.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    updated_str = mem.get("updated_at") or mem.get("created_at")
    if not updated_str:
        return ""

    try:
        updated = datetime.fromisoformat(updated_str)
        # Ensure timezone-aware comparison
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        delta_days = max(0, (now - updated).days)
    except (ValueError, TypeError):
        return ""

    if delta_days <= 7:
        return ""

    return f"({delta_days}d old \u2014 verify)"


class MemoryEngine:
    """SQLite-backed memory store with FTS5 search and tiered lifecycle.

    Thread-safe via per-thread connections and WAL mode.
    """

    def __init__(self, db_path: Optional[Path] = None, config: Optional[dict] = None):
        config = config or {}
        if db_path is None:
            raise ValueError(
                "db_path is required — the plugin provider must pass it explicitly"
            )

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._config = config
        self._local = threading.local()
        self._lock = threading.Lock()

        # Snapshot for prompt injection (frozen at session start)
        self._snapshot: Optional[dict] = None

        # Initialize schema
        self._init_db()

    # -- Connection management -----------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            # Use autocommit mode to avoid implicit transactions holding locks
            # during slow operations (e.g., fastembed model loading).
            # All writes use explicit conn.execute("BEGIN") / conn.commit().
            conn.isolation_level = None
            self._local.conn = conn
        return conn

    def _init_db(self):
        """Create tables, seed metadata, and run migrations.

        Migration runs BEFORE full schema application so that ALTER TABLE
        adds columns before CREATE INDEX references them.
        """
        conn = self._get_conn()

        # Check if this is an existing database that needs migration.
        # We detect by checking if memory_meta table exists with a version.
        needs_migration = False
        try:
            row = conn.execute(
                "SELECT value FROM memory_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row:
                current_version = int(row["value"]) if row else 1
                if current_version < SCHEMA_VERSION:
                    needs_migration = True
                    self._migrate_schema(conn, current_version)
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet — fresh database

        # Apply full schema (CREATE IF NOT EXISTS is idempotent)
        conn.executescript(_SCHEMA_SQL)

        # Seed metadata defaults
        for key, default in _META_DEFAULTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO memory_meta (key, value) VALUES (?, ?)",
                (key, default),
            )
        conn.commit()

        # If we migrated, update the version stamp
        if needs_migration:
            conn.execute(
                "INSERT OR REPLACE INTO memory_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection, current_version: int):
        """Apply incremental schema migrations for existing databases.

        V2 -> V3: Add trust_score, helpful_count columns; entities + memory_entities tables.
        Must run BEFORE _SCHEMA_SQL so new indexes can reference new columns.
        """
        if current_version < 3:
            # V2 -> V3: Add trust scoring columns to memories table
            for alter_sql in [
                "ALTER TABLE memories ADD COLUMN trust_score REAL NOT NULL DEFAULT 0.5",
                "ALTER TABLE memories ADD COLUMN helpful_count INTEGER NOT NULL DEFAULT 0",
            ]:
                try:
                    conn.execute(alter_sql)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning("Migration ALTER failed: %s", e)

            conn.commit()
            logger.info("Schema migrated from V%d to V3 (trust scoring + entities)", current_version)

    def close(self):
        """Close the thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # -- Meta ----------------------------------------------------------------

    def _get_meta(self, key: str) -> Optional[str]:
        row = self._get_conn().execute(
            "SELECT value FROM memory_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    # -- Embedding management -------------------------------------------------

    def _get_or_create_embedding(self, chunk_id: str, content: str) -> list:
        """Get cached embedding or generate and store a new one.

        Content-hash caching pattern from HiveMind: avoids redundant API calls.
        Carefully structured to not hold DB locks during slow embedding generation.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row and row["embedding"]:
            try:
                return json.loads(row["embedding"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Commit any implicit read transaction before slow embedding generation
        # to avoid holding DB locks (especially during fastembed cold start).
        conn.commit()

        vec = generate_embedding(content, config=self._config)
        if vec:
            now = datetime.now(timezone.utc).isoformat()
            blob = json.dumps(vec)
            # Detect the model that was actually used
            used_model = self._config.get("embedding_model", "") or _detect_embedding_model() or "local/bge-small-en-v1.5"
            conn.execute(
                """INSERT OR REPLACE INTO embeddings (chunk_id, embedding, model, created_at)
                   VALUES (?, ?, ?, ?)""",
                (chunk_id, blob, used_model, now),
            )
            conn.commit()
        return vec

    def _generate_embeddings_background(self, memory_id: str, content: str, chunk_ids: list = None):
        """Fire-and-forget embedding generation in a background thread."""
        def _worker():
            # Create a dedicated connection for this background thread
            # (thread-local _get_conn would create one anyway, but we need
            # to ensure it gets closed when the thread finishes).
            conn = sqlite3.connect(
                str(self._db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            conn.isolation_level = None
            # Temporarily set as thread-local so _get_or_create_embedding works
            self._local.conn = conn
            try:
                if chunk_ids:
                    for cid in chunk_ids:
                        row = conn.execute(
                            "SELECT content FROM chunks WHERE id = ?", (cid,)
                        ).fetchone()
                        if row:
                            self._get_or_create_embedding(cid, row["content"])
                else:
                    self._get_or_create_embedding(memory_id, content)
            except Exception as e:
                logger.debug("Background embedding generation failed: %s", e)
            finally:
                conn.close()
                self._local.conn = None

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # -- Core CRUD -----------------------------------------------------------

    def add(
        self,
        content: str,
        target: str = "memory",
        type: str = "general",
        tags: str = "",
        source: str = "agent",
        session_id: str = None,
    ) -> dict:
        """Add a new memory. Returns {success, id, ...} or {error}."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        if target not in MEMORY_TARGETS:
            return {"success": False, "error": f"Invalid target: {target}. Use: {MEMORY_TARGETS}"}
        if type not in MEMORY_TYPES:
            type = "general"

        # Auto-classify type if type=='general' using keyword-based classification
        if type == "general":
            keywords = extract_keywords(content)
            classified = classify_topic(content, keywords=keywords)
            if classified in MEMORY_TYPES:
                type = classified
            # Auto-extract keywords and merge with existing tags
            if keywords:
                existing_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
                merged = list(dict.fromkeys(existing_tags + keywords[:5]))  # dedup, keep order
                tags = ",".join(merged)

        # Exact duplicate check (fast, reliable)
        conn = self._get_conn()
        exact = conn.execute(
            "SELECT id, content FROM memories WHERE target = ? AND tier = 'active' AND content = ?",
            (target, content),
        ).fetchone()
        if exact:
            return {
                "success": False,
                "error": f"Exact duplicate exists: {exact['content'][:80]}...",
                "duplicate_id": exact["id"],
            }

        # Release read lock before potentially slow embedding generation
        conn.commit()

        # Near-duplicate detection via embeddings (HiveMind threshold: cosine > 0.92)
        new_embedding = generate_embedding(content, config=self._config)
        if new_embedding:
            # Check against active memories in same target
            active_rows = conn.execute(
                """SELECT e.chunk_id, e.embedding, m.id, m.content
                   FROM embeddings e
                   JOIN memories m ON (e.chunk_id = m.id OR e.chunk_id LIKE m.id || ':chunk_%%')
                   WHERE m.tier = 'active' AND m.target = ?
                   LIMIT 200""",
                (target,),
            ).fetchall()
            for row in active_rows:
                try:
                    stored_emb = json.loads(row["embedding"])
                except (json.JSONDecodeError, TypeError):
                    continue
                sim = cosine_similarity(new_embedding, stored_emb)
                if sim > 0.92:
                    return {
                        "success": False,
                        "error": f"Near-duplicate (cosine={sim:.3f}): {row['content'][:80]}...",
                        "duplicate_id": row["id"],
                    }

        # Commit any implicit read transaction from duplicate checks above
        # to avoid holding DB locks during concurrent background embedding writes.
        conn.commit()

        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO memories (id, content, target, type, source, tags,
               created_at, updated_at, session_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, content, target, type, source, tags, now, now, session_id),
        )

        # Chunk long content and insert into chunks table
        chunks = None
        if len(content) > CHUNK_MIN_CONTENT_LEN:
            chunks = chunk_text(content)
            for idx, (start_line, end_line, chunk_content) in enumerate(chunks):
                chunk_id = f"{memory_id}:chunk_{idx}"
                chunk_hash = _hash_text(chunk_content)
                conn.execute(
                    """INSERT INTO chunks (id, memory_id, chunk_index, content,
                       start_line, end_line, hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, memory_id, idx, chunk_content, start_line, end_line, chunk_hash),
                )

        conn.commit()

        # Generate embeddings in background (non-blocking)
        if chunks:
            chunk_ids = [f"{memory_id}:chunk_{idx}" for idx in range(len(chunks))]
            self._generate_embeddings_background(memory_id, content, chunk_ids=chunk_ids)
        else:
            self._generate_embeddings_background(memory_id, content)

        # Auto-create edges via keyword overlap (port from HiveMind lines 1372-1437)
        self._auto_create_edges(memory_id, content)

        # Extract entities and link to memory (V3, ported from holographic)
        self._extract_and_link_entities(memory_id, content)

        # Check for supersession candidates
        self._check_supersession(memory_id, content, target)

        # Enforce budget — archive weakest if over cap
        self.enforce_budget(target)

        logger.debug("Memory added: %s [%s/%s] %s", memory_id[:8], target, type, content[:60])
        return {"success": True, "id": memory_id, "target": target, "type": type}

    def replace(self, memory_id: str, new_content: str) -> dict:
        """Update a memory's content."""
        new_content = new_content.strip()
        if not new_content:
            return {"success": False, "error": "Content cannot be empty."}

        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
            (new_content, now, memory_id),
        )
        conn.commit()

        if cur.rowcount == 0:
            return {"success": False, "error": f"Memory {memory_id} not found."}

        # Re-extract entities on content change (V3)
        self._unlink_memory_entities(memory_id)
        self._extract_and_link_entities(memory_id, new_content)

        return {"success": True, "id": memory_id}

    def remove(self, memory_id: str) -> dict:
        """Delete a memory."""
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

        if cur.rowcount == 0:
            return {"success": False, "error": f"Memory {memory_id} not found."}
        return {"success": True, "id": memory_id}

    def get(self, memory_id: str) -> Optional[dict]:
        """Get a single memory by ID."""
        row = self._get_conn().execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- Search --------------------------------------------------------------

    def search_fts(self, query: str, target: str = None, limit: int = 20) -> list:
        """BM25 full-text search. Returns memories ranked by relevance."""
        query = query.strip()
        if not query:
            return []

        conn = self._get_conn()
        # Build FTS5 query — tokenize for safety
        # Strip quotes/apostrophes that break FTS5 syntax, keep only alnum words
        words = re.findall(r'[a-zA-Z0-9_]+', query)
        words = [w for w in words if len(w) > 1]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)

        sql = """
            SELECT m.*, bm25(memories_fts) AS bm25_rank
            FROM memories m
            JOIN memories_fts ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ?
        """
        params = [fts_query]

        if target:
            sql += " AND m.target = ?"
            params.append(target)

        sql += " ORDER BY bm25_rank LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed: %s (query=%r)", e, fts_query)
            return []

        results = []
        for row in rows:
            d = dict(row)
            # FTS5 bm25() returns negative values (more negative = better match).
            # We negate to get positive scores. Do NOT normalize to 0-1 across
            # the result set — that makes the best match always 1.0 regardless
            # of actual relevance. Raw magnitude is more useful for thresholding.
            raw = -d.pop("bm25_rank", 0)
            d["bm25_score"] = max(raw, 0.0)
            results.append(d)

        return results

    def search(
        self,
        query: str,
        target: str = None,
        limit: int = 10,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        auxiliary_client=None,
    ) -> list:
        """Hybrid search: FTS5 BM25 + recency + strength + tier weighting.

        Adapted from HiveMind memory.rs score_memory().
        """
        candidates = self.search_fts(query, target=target, limit=limit * 3)
        if not candidates:
            return []

        now = datetime.now(timezone.utc)

        scored = []
        # Hybrid search: if embeddings available, use HiveMind formula
        # Final score = (0.7 * cosine + 0.3 * normalized_bm25) * recency * strength * tier_w * type_w
        query_embedding = generate_embedding(query, config=self._config)
        # Normalize BM25 scores for blending
        bm25_scores = [mem.get("bm25_score", 0) for mem in candidates]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 < 1e-10:
            max_bm25 = 1.0
        has_embeddings = bool(query_embedding)

        # Batch-fetch all embeddings upfront to avoid N+1 queries
        candidate_embeddings = {}  # memory_id -> embedding vector
        if has_embeddings:
            candidate_ids = [mem["id"] for mem in candidates]
            conn = self._get_conn()
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                # Fetch memory-level embeddings
                emb_rows = conn.execute(
                    f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})",
                    candidate_ids,
                ).fetchall()
                for row in emb_rows:
                    try:
                        candidate_embeddings[row["chunk_id"]] = json.loads(row["embedding"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # For candidates without memory-level embeddings, try chunk embeddings
                missing_ids = [mid for mid in candidate_ids if mid not in candidate_embeddings]
                if missing_ids:
                    like_patterns = [mid + ":chunk_%" for mid in missing_ids]
                    # Batch fetch chunk embeddings — one query per missing id
                    # (SQLite doesn't support LIKE with IN, but we can use OR)
                    if like_patterns:
                        like_clauses = " OR ".join("chunk_id LIKE ?" for _ in like_patterns)
                        chunk_rows = conn.execute(
                            f"SELECT chunk_id, embedding FROM embeddings WHERE {like_clauses}",
                            like_patterns,
                        ).fetchall()
                        for row in chunk_rows:
                            # Extract memory_id from chunk_id (format: "memory_id:chunk_N")
                            mem_id = row["chunk_id"].rsplit(":chunk_", 1)[0]
                            if mem_id not in candidate_embeddings:
                                try:
                                    candidate_embeddings[mem_id] = json.loads(row["embedding"])
                                except (json.JSONDecodeError, TypeError):
                                    pass

        for mem in candidates:
            bm25 = mem.get("bm25_score", 0)
            normalized_bm25 = bm25 / max_bm25

            # Recency decay (power-law, from HiveMind)
            try:
                updated = datetime.fromisoformat(mem["updated_at"])
                hours = max((now - updated).total_seconds() / 3600, 0)
            except (ValueError, TypeError):
                hours = 0
            recency = (1 + hours) ** RECENCY_DECAY_EXPONENT

            # Strength (logarithmic reinforcement, from HiveMind)
            access_count = mem.get("access_count", 0)
            strength = 1.0 + 0.1 * math.log(1 + access_count)

            # Tier weight
            tier_w = TIER_WEIGHTS.get(mem.get("tier", "active"), 0.5)

            # Type boost
            type_w = TYPE_BOOSTS.get(mem.get("type", "general"), 1.0)

            # Trust score weight (V3, ported from holographic)
            trust_w = mem.get("trust_score", _TRUST_DEFAULT)

            # Compute relevance base score (always in 0-1 range)
            if has_embeddings:
                mem_embedding = candidate_embeddings.get(mem["id"])
                if mem_embedding:
                    cos_sim = cosine_similarity(query_embedding, mem_embedding)
                    base_score = 0.7 * cos_sim + 0.3 * normalized_bm25
                else:
                    # No embedding for this candidate, fall back to normalized BM25
                    base_score = normalized_bm25
            else:
                # No query embedding available, pure normalized BM25
                # (use normalized score so min_relevance has consistent semantics
                # regardless of whether embeddings are available)
                base_score = normalized_bm25

            score = base_score * recency * strength * tier_w * type_w * trust_w
            if score >= min_relevance:
                mem["relevance_score"] = round(score, 4)
                scored.append(mem)

        scored.sort(key=lambda m: m["relevance_score"], reverse=True)
        top_results = scored[:limit]

        # Graph-augmented expansion: 1-hop traversal with 0.5x weight boost
        if top_results:
            seen_ids = {m["id"] for m in top_results}
            graph_additions = []
            for mem in top_results[:3]:  # expand from top 3 only
                related = self.get_related(mem["id"], hops=1)
                for rel in related:
                    if rel["id"] not in seen_ids:
                        seen_ids.add(rel["id"])
                        rel["relevance_score"] = round(mem["relevance_score"] * 0.5, 4)
                        rel["graph_expanded"] = True
                        graph_additions.append(rel)
            if graph_additions:
                top_results.extend(graph_additions)
                top_results.sort(key=lambda m: m["relevance_score"], reverse=True)
                top_results = top_results[:limit]

        return top_results

    # -- Lifecycle -----------------------------------------------------------

    def reinforce(self, memory_id: str):
        """Increment access count and update strength. Called on search hit."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE memories
               SET access_count = access_count + 1,
                   strength = 1.0 + 0.1 * ln(1 + access_count + 1),
                   last_accessed = ?
               WHERE id = ?""",
            (now, memory_id),
        )
        conn.commit()

    def archive_stale(self, days: int = ARCHIVE_STALE_DAYS, min_strength: float = ARCHIVE_MIN_STRENGTH) -> int:
        """Archive memories that are old and weak. Returns count archived."""
        conn = self._get_conn()
        cutoff = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """UPDATE memories
               SET tier = 'archived', updated_at = ?
               WHERE tier = 'active'
                 AND strength < ?
                 AND julianday(?) - julianday(updated_at) > ?""",
            (cutoff, min_strength, cutoff, days),
        )
        conn.commit()
        count = cur.rowcount
        if count:
            logger.info("Archived %d stale memories (>%d days, strength<%.1f)", count, days, min_strength)
        return count

    def supersede(self, old_id: str, new_id: str):
        """Mark old memory as superseded by new one."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE memories SET tier = 'superseded', superseded_by = ?, updated_at = ? WHERE id = ?",
            (new_id, now, old_id),
        )
        conn.commit()

    def enforce_budget(self, target: str = None) -> int:
        """Archive lowest-strength memories when active count exceeds budget.

        Inspired by Stanford Generative Agents' memory budget + MemoryBank's
        strength-based pruning. Keeps the most reinforced/important memories alive.

        Returns count of memories archived to make room.
        """
        targets = [target] if target else list(MEMORY_TARGETS)
        total_archived = 0
        conn = self._get_conn()

        for t in targets:
            budget = MAX_ACTIVE_USER if t == "user" else MAX_ACTIVE_MEMORY
            count = self.count_active(t)

            if count <= budget:
                continue

            overflow = count - budget
            # Archive the weakest active memories (lowest strength, then oldest)
            # Corrections and preferences are protected — they cost more to lose
            rows = conn.execute(
                """SELECT id FROM memories
                   WHERE target = ? AND tier = 'active'
                   ORDER BY
                       CASE WHEN type IN ('correction', 'preference') THEN 1 ELSE 0 END ASC,
                       strength ASC,
                       updated_at ASC
                   LIMIT ?""",
                (t, overflow),
            ).fetchall()

            now = datetime.now(timezone.utc).isoformat()
            for row in rows:
                conn.execute(
                    "UPDATE memories SET tier = 'archived', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                total_archived += 1

            if rows:
                conn.commit()
                logger.info(
                    "Budget enforcement [%s]: archived %d memories (%d/%d over budget)",
                    t, len(rows), count, budget,
                )

        return total_archived

    def purge_dead(self, max_age_days: int = 30) -> int:
        """Hard-delete superseded and archived memories older than max_age_days.

        This is the only place memories are truly removed from the DB.
        Prevents monotonic DB growth from supersession and archival.
        Also cleans up orphaned chunks, embeddings, and edges.
        """
        conn = self._get_conn()
        cutoff = datetime.now(timezone.utc).isoformat()

        # Find memories to purge
        rows = conn.execute(
            """SELECT id FROM memories
               WHERE tier IN ('superseded', 'archived')
                 AND julianday(?) - julianday(updated_at) > ?""",
            (cutoff, max_age_days),
        ).fetchall()

        if not rows:
            return 0

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)

        # Delete entity links for these memories (V3)
        conn.execute(
            f"DELETE FROM memory_entities WHERE memory_id IN ({placeholders})",
            ids,
        )

        # Delete edges referencing these memories
        conn.execute(
            f"DELETE FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            ids + ids,
        )

        # Delete embeddings for chunks of these memories
        conn.execute(
            f"""DELETE FROM embeddings WHERE chunk_id IN (
                SELECT id FROM chunks WHERE memory_id IN ({placeholders})
            )""",
            ids,
        )

        # Delete chunks
        conn.execute(f"DELETE FROM chunks WHERE memory_id IN ({placeholders})", ids)

        # Delete the memories themselves
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)

        conn.commit()
        logger.info("Purged %d dead memories (superseded/archived >%d days)", len(ids), max_age_days)

        return len(ids)

    def _check_supersession(self, new_id: str, content: str, target: str):
        """Check if this new memory supersedes an existing one.

        From HiveMind: cosine > 0.85 + same topic -> supersede.
        We approximate with FTS5 BM25 since we don't have embeddings yet.
        """
        candidates = self.search_fts(content, target=target, limit=5)
        for c in candidates:
            if c["id"] == new_id:
                continue
            if c.get("bm25_score", 0) > DEDUP_THRESHOLD and c.get("tier") == "active":
                self.supersede(c["id"], new_id)
                logger.debug(
                    "Memory %s superseded by %s (score=%.2f)",
                    c["id"][:8], new_id[:8], c["bm25_score"],
                )

    # -- Graph operations (MAGMA-inspired) ------------------------------------

    def _auto_create_edges(self, memory_id: str, content: str):
        """Auto-create edges between memories sharing keywords.

        Ported from HiveMind memory.rs auto_create_edges().
        """
        keywords = extract_keywords(content)
        if not keywords:
            return

        # Use top 3 keywords to find related memories via FTS5
        query_terms = [f'"{k}"' for k in keywords[:3]]
        fts_query = " OR ".join(query_terms)

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT DISTINCT m.id FROM memories m
                   JOIN memories_fts ON memories_fts.rowid = m.rowid
                   WHERE memories_fts MATCH ? LIMIT 10""",
                (fts_query,),
            ).fetchall()
        except sqlite3.OperationalError:
            return

        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            related_id = row["id"]
            if related_id == memory_id:
                continue

            # Check if edge already exists
            existing = conn.execute(
                "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ? AND relation = ?",
                (memory_id, related_id, "related_to"),
            ).fetchone()
            if existing:
                continue

            conn.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, relation, weight, created_at)
                   VALUES (?, ?, 'related_to', 0.5, ?)""",
                (memory_id, related_id, now),
            )
        conn.commit()

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 0.5) -> dict:
        """Add a typed edge between two memories."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, target_id, relation, weight, now),
            )
            conn.commit()
            return {"success": True, "source_id": source_id, "target_id": target_id, "relation": relation}
        except sqlite3.Error as e:
            return {"success": False, "error": str(e)}

    def get_edges(self, memory_id: str, relation: str = None) -> list:
        """Get all edges for a memory (both outgoing and incoming)."""
        conn = self._get_conn()
        if relation:
            rows = conn.execute(
                """SELECT * FROM edges
                   WHERE (source_id = ? OR target_id = ?) AND relation = ?""",
                (memory_id, memory_id, relation),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edges WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_related(self, memory_id: str, hops: int = 1) -> list:
        """BFS traversal to find related memories up to N hops away."""
        conn = self._get_conn()
        visited = {memory_id}
        frontier = {memory_id}

        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                # Outgoing
                rows = conn.execute(
                    "SELECT target_id FROM edges WHERE source_id = ? LIMIT 20",
                    (nid,),
                ).fetchall()
                for r in rows:
                    tid = r["target_id"]
                    if tid not in visited:
                        next_frontier.add(tid)
                        visited.add(tid)

                # Incoming
                rows = conn.execute(
                    "SELECT source_id FROM edges WHERE target_id = ? LIMIT 20",
                    (nid,),
                ).fetchall()
                for r in rows:
                    sid = r["source_id"]
                    if sid not in visited:
                        next_frontier.add(sid)
                        visited.add(sid)
            frontier = next_frontier

        # Fetch the actual memory records (exclude the start node)
        visited.discard(memory_id)
        results = []
        for mid in visited:
            mem = self.get(mid)
            if mem:
                results.append(mem)
        return results

    # -- Trust Scoring (ported from holographic) --------------------------------

    def record_feedback(self, memory_id: str, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns dict with memory_id, old_trust, new_trust, helpful_count.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, trust_score, helpful_count FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"memory_id {memory_id} not found")

        old_trust = row["trust_score"]
        delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
        new_trust = _clamp_trust(old_trust + delta)
        helpful_increment = 1 if helpful else 0

        conn.execute(
            """
            UPDATE memories
            SET trust_score   = ?,
                helpful_count = helpful_count + ?,
                updated_at    = ?
            WHERE id = ?
            """,
            (new_trust, helpful_increment, datetime.now(timezone.utc).isoformat(), memory_id),
        )
        conn.commit()

        return {
            "memory_id":     memory_id,
            "old_trust":     round(old_trust, 4),
            "new_trust":     round(new_trust, 4),
            "helpful_count": row["helpful_count"] + helpful_increment,
        }

    # -- Entity Extraction & Resolution (ported from holographic) ---------------

    def _extract_entities(self, text: str) -> list:
        """Extract entity candidates from text using regex rules.

        Rules (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Double-quoted terms             e.g. "Python"
        3. Single-quoted terms             e.g. 'pytest'
        4. AKA patterns                    e.g. "Guido aka BDFL" -> two entities
        """
        seen: set = set()
        candidates: list = []

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

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find existing entity by name or alias (case-insensitive), or create one."""
        conn = self._get_conn()

        # Exact name match (case-insensitive via LIKE)
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — stored as comma-separated, use LIKE with boundary markers
        alias_row = conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO entities (name, created_at) VALUES (?, ?)", (name, now)
        )
        conn.commit()
        return int(cur.lastrowid)

    def _link_memory_entity(self, memory_id: str, entity_id: int) -> None:
        """Insert into memory_entities, ignore if link exists."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, entity_id),
        )
        conn.commit()

    def _extract_and_link_entities(self, memory_id: str, content: str) -> None:
        """Extract entities from content and link them to the memory.

        Called after add() and replace() to maintain entity graph.
        """
        entities = self._extract_entities(content)
        for entity_name in entities:
            try:
                entity_id = self._resolve_entity(entity_name)
                self._link_memory_entity(memory_id, entity_id)
            except Exception as e:
                logger.debug("Entity linking failed for '%s': %s", entity_name, e)

    def _unlink_memory_entities(self, memory_id: str) -> None:
        """Remove all entity links for a memory (used before re-extraction on update)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
        conn.commit()

    def find_related_entities(self, memory_id: str) -> list:
        """Return entities linked to a memory.

        Returns list of dicts with entity_id, name, entity_type, aliases.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.entity_id, e.name, e.entity_type, e.aliases
            FROM entities e
            JOIN memory_entities me ON me.entity_id = e.entity_id
            WHERE me.memory_id = ?
            """,
            (memory_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Contradiction Detection (ported from holographic) ----------------------

    def find_contradictions(
        self,
        entity_name: str = None,
        threshold: float = 0.3,
        limit: int = 20,
    ) -> list:
        """Find potentially contradictory memories via entity overlap + content divergence.

        Two memories contradict when they share entities (same subject) but have
        low content similarity (different claims).

        Args:
            entity_name: Optional filter — only consider memories linked to this entity.
            threshold: Minimum contradiction_score to flag (default 0.3).
            limit: Max contradictions to return (default 20).

        Returns list of dicts with memory_a, memory_b, entity_overlap,
        content_similarity, contradiction_score, shared_entities.
        """
        conn = self._get_conn()

        # Get recent active memories (cap at 500 to avoid O(n^2) explosion)
        if entity_name:
            # Filter to memories linked to the specified entity
            rows = conn.execute(
                """
                SELECT m.id, m.content, m.type, m.tags, m.trust_score,
                       m.created_at, m.updated_at
                FROM memories m
                JOIN memory_entities me ON me.memory_id = m.id
                JOIN entities e ON e.entity_id = me.entity_id
                WHERE m.tier = 'active' AND e.name LIKE ?
                ORDER BY m.updated_at DESC
                LIMIT 500
                """,
                (entity_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, content, type, tags, trust_score,
                       created_at, updated_at
                FROM memories
                WHERE tier = 'active'
                ORDER BY updated_at DESC
                LIMIT 500
                """,
            ).fetchall()

        if len(rows) < 2:
            return []

        # Build entity sets per memory
        memory_entities_map: dict = {}  # memory_id -> set of lowercased entity names
        memories_list = []
        for row in rows:
            mem = dict(row)
            mid = mem["id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN memory_entities me ON me.entity_id = e.entity_id
                WHERE me.memory_id = ?
                """,
                (mid,),
            ).fetchall()
            memory_entities_map[mid] = {r["name"].lower() for r in entity_rows}
            memories_list.append(mem)

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        contradictions = []
        for i in range(len(memories_list)):
            for j in range(i + 1, len(memories_list)):
                m1, m2 = memories_list[i], memories_list[j]
                ents1 = memory_entities_map.get(m1["id"], set())
                ents2 = memory_entities_map.get(m2["id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                union = ents1 | ents2
                intersection = ents1 & ents2
                entity_overlap = len(intersection) / len(union) if union else 0
                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap

                # Content similarity — token Jaccard (zero dependencies)
                tokens1 = set(m1["content"].lower().split())
                tokens2 = set(m2["content"].lower().split())
                token_union = tokens1 | tokens2
                content_sim = len(tokens1 & tokens2) / len(token_union) if token_union else 0

                # High entity overlap + low content similarity = contradiction
                contradiction_score = entity_overlap * (1.0 - content_sim)

                if contradiction_score >= threshold:
                    contradictions.append({
                        "memory_a": {"id": m1["id"], "content": m1["content"][:200]},
                        "memory_b": {"id": m2["id"], "content": m2["content"][:200]},
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(intersection),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    # -- Retrieval for prompts -----------------------------------------------

    def get_active_memories(self, target: str, limit: int = None) -> list:
        """Get all active memories for a target, ordered by strength desc."""
        conn = self._get_conn()
        sql = "SELECT * FROM memories WHERE target = ? AND tier = 'active' ORDER BY strength DESC"
        params = [target]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def format_for_prompt(self, target: str, char_budget: int = None) -> Optional[str]:
        """Format memories for system prompt injection.

        Adds type tags, respects budget, returns None if empty.
        Memories older than 7 days get a staleness suffix (from Claude Code memoryAge.ts).
        """
        memories = self.get_active_memories(target)
        if not memories:
            return None

        now = datetime.now(timezone.utc)
        lines = []
        total_chars = 0
        for mem in memories:
            tag = TYPE_TAGS.get(mem.get("type", "general"), "gen")
            line = f"[{tag}] {mem['content']}"

            # Staleness caveat: memories >7 days old get age suffix
            staleness = _memory_staleness_suffix(mem, now)
            if staleness:
                line = f"{line} {staleness}"

            if char_budget and total_chars + len(line) + 3 > char_budget:
                break
            lines.append(line)
            total_chars += len(line) + 3  # +3 for delimiter

        if not lines:
            return None

        content = "\n\u00a7\n".join(lines)
        total = self.count_active(target)
        shown = len(lines)
        budget_str = f"{total_chars}" if not char_budget else f"{total_chars}/{char_budget}"

        header_name = "MEMORY (your personal notes)" if target == "memory" else "USER PROFILE (who the user is)"
        header = f"{'═' * 46}\n{header_name} [{shown}/{total} entries — {budget_str} chars]\n{'═' * 46}"

        return f"{header}\n{content}\n"

    # -- Snapshot (frozen at session start) ----------------------------------

    def snapshot(self) -> dict:
        """Capture current state as frozen snapshot for prompt caching."""
        self._snapshot = {
            "memory": self.format_for_prompt("memory"),
            "user": self.format_for_prompt("user"),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._snapshot

    def get_snapshot(self, target: str) -> Optional[str]:
        """Get frozen snapshot for a target. Returns None if not captured or empty."""
        if self._snapshot is None:
            self.snapshot()
        return self._snapshot.get(target)

    # -- Stats ---------------------------------------------------------------

    def count_active(self, target: str = None) -> int:
        """Count active memories, optionally filtered by target."""
        conn = self._get_conn()
        if target:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE target = ? AND tier = 'active'",
                (target,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE tier = 'active'"
            ).fetchone()
        return row["c"] if row else 0

    def stats(self) -> dict:
        """Memory statistics: counts per target/tier, total embeddings, total edges."""
        conn = self._get_conn()

        # Per-target, per-tier counts
        rows = conn.execute(
            "SELECT target, tier, COUNT(*) as count FROM memories GROUP BY target, tier"
        ).fetchall()

        by_target_tier = {}
        for r in rows:
            target = r["target"]
            if target not in by_target_tier:
                by_target_tier[target] = {}
            by_target_tier[target][r["tier"]] = r["count"]

        total_memories = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        total_embeddings = conn.execute("SELECT COUNT(*) as c FROM embeddings").fetchone()["c"]
        total_edges = conn.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]

        return {
            "total_memories": total_memories,
            "by_target_tier": by_target_tier,
            "total_embeddings": total_embeddings,
            "total_edges": total_edges,
            "schema_version": self._get_meta("schema_version"),
        }

    # -- Migration from flat files -------------------------------------------

    def migrate_from_flat_files(self, memory_dir: Path = None) -> dict:
        """Import entries from MEMORY.md and USER.md into SQLite.

        Preserves flat files as .bak. Idempotent (skips if already migrated).
        """
        if self._get_meta("migrated_from_flat") == "1":
            return {"migrated": False, "reason": "Already migrated."}

        if memory_dir is None:
            memory_dir = self._db_path.parent

        count = 0
        for target, filename in [("memory", "MEMORY.md"), ("user", "USER.md")]:
            filepath = memory_dir / filename
            if not filepath.exists():
                continue

            text = filepath.read_text(encoding="utf-8").strip()
            if not text:
                continue

            entries = [e.strip() for e in text.split("\n\u00a7\n") if e.strip()]
            for entry in entries:
                self.add(
                    content=entry,
                    target=target,
                    type="general",
                    source="migration",
                )
                count += 1

            # Backup flat file
            bak = filepath.with_suffix(".md.bak")
            if not bak.exists():
                filepath.rename(bak)
                logger.info("Backed up %s -> %s", filepath.name, bak.name)

        self._set_meta("migrated_from_flat", "1")
        logger.info("Migrated %d entries from flat files to SQLite", count)
        return {"migrated": True, "count": count}

    # -- Manifest for extraction dedup (from Claude Code pattern) -----------

    def get_manifest(self, target: str = None) -> str:
        """Return a compact manifest of all active memories for dedup checking.

        Used by the auto-extractor to avoid duplicating existing memories.
        """
        memories = []
        for t in ([target] if target else list(MEMORY_TARGETS)):
            for mem in self.get_active_memories(t):
                memories.append(f"[{mem['id'][:8]}|{mem.get('type','gen')}|{mem['target']}] {mem['content'][:120]}")
        return "\n".join(memories) if memories else "(no memories yet)"
