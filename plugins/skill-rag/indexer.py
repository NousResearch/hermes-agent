"""Skill scanning, embeddings, SQLite index.

Reuses parse_frontmatter, EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS
from agent/skill_utils.py Hermes Agent.
"""
import hashlib
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Max SKILL.md file size (1 MB)
_MAX_SKILL_FILE_SIZE = 1 * 1024 * 1024

# Reused from Hermes Agent
from agent.skill_utils import parse_frontmatter as _hermes_parse_frontmatter
from agent.skill_utils import EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS

from .config import (
    SKILLS_ROOT, FALLBACK_INDEX_DIR, DB_FILENAME, FTS_TABLE, SKILL_FILE,
    LOCAL_MODEL, LOCAL_REVISION,
    API_BATCH_SIZE, API_TIMEOUT_CONNECT, API_TIMEOUT_READ,
    EMBEDDING_DIM, PREFIX_PASSAGE, FIELD_MAX_LEN, LOG_PREFIX,
    DEFAULTS,
)

logger = logging.getLogger(__name__)

# API key is a secret — read from environment variable
_API_KEY = os.environ.get("SKILL_RAG_API_KEY", "local-not-needed")


def _model_fingerprint(config: Dict[str, Any]) -> str:
    """Hash of everything that affects the vector."""
    provider = config.get("provider", DEFAULTS["provider"])
    parts = [provider, str(EMBEDDING_DIM), PREFIX_PASSAGE]
    if provider == "local":
        parts += [LOCAL_MODEL, str(LOCAL_REVISION)]
    else:
        parts += [config.get("api_base", DEFAULTS["api_base"]),
                  config.get("api_model", DEFAULTS["api_model"])]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _path_id(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode()).hexdigest()[:16]


def _content_hash(composed_text: str) -> str:
    return hashlib.sha256(composed_text.encode()).hexdigest()


def _stringify(val) -> str:
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    return str(val or "").strip()


def _compose_text(meta: dict, fallback_name: str) -> str:
    """Compose embedding text from frontmatter fields."""
    parts = []
    name = (meta.get("name") or fallback_name or "").strip()
    if name:
        parts.append(name)

    for field in ("description", "when_to_use", "triggers", "category"):
        val = _stringify(meta.get(field))
        if val:
            if len(val) > FIELD_MAX_LEN:
                val = val[:FIELD_MAX_LEN]
            parts.append(val)

    tags = meta.get("tags")
    if isinstance(tags, list) and tags:
        parts.append("tags: " + ", ".join(str(t) for t in tags))
    elif isinstance(tags, str) and tags.strip():
        parts.append("tags: " + tags.strip())

    return ". ".join(parts)


def _is_excluded_dir(path: Path, skills_root: Path) -> bool:
    """Check if path is in an excluded directory.

    Uses EXCLUDED_SKILL_DIRS from Hermes Agent.
    """
    try:
        rel = path.relative_to(skills_root)
    except ValueError:
        return False
    # Check each path part against EXCLUDED_SKILL_DIRS
    for part in rel.parts[:-1]:  # all parts except filename
        if part in EXCLUDED_SKILL_DIRS:
            return True
        if part in SKILL_SUPPORT_DIRS:
            return True
    return False


_session_local = threading.local()


def _get_session():
    session = getattr(_session_local, "session", None)
    if session is None:
        import requests
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=8, max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _session_local.session = session
    return session


def _is_skill_visible(name: str, frontmatter: dict) -> bool:
    """Check if skill is visible for system prompt.

    Uses the same checks as build_skills_system_prompt:
    - disabled from config.yaml
    - platforms/environments/requires_apps from frontmatter
    - session_platforms/requires_toolsets/fallback_for_toolsets
    """
    from agent.skill_utils import (
        get_disabled_skill_names,
        extract_skill_conditions,
        skill_matches_platform,
        skill_matches_environment,
        skill_matches_apps,
    )
    from agent.prompt_builder import _skill_should_show

    # 1. Disabled
    disabled = get_disabled_skill_names()
    if name in disabled:
        return False

    # 2. Platform / environment / apps (same as _parse_skill_file)
    if not skill_matches_platform({"platforms": frontmatter.get("platforms") or []}):
        return False
    if not skill_matches_environment({"environments": frontmatter.get("environments") or []}):
        return False
    if not skill_matches_apps({"requires_apps": frontmatter.get("requires_apps") or []}):
        return False

    # 3. Toolset / tool / session_platform conditions (same as _skill_should_show)
    conditions = extract_skill_conditions(frontmatter)
    if not _skill_should_show(conditions, None, None):
        return False

    return True


class Indexer:

    def __init__(self, skills_root: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.skills_root = Path(skills_root) if skills_root else SKILLS_ROOT
        self.config = dict(DEFAULTS)
        if config:
            self.config.update(config)
        self.db_path = self._resolve_db_path()
        self.model_hash = _model_fingerprint(self.config)
        self._model = None
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available = False
        self._path_cache: dict[str, Path] = {}
        self._skill_count = 0
        self._lock = threading.Lock()

    # --- Initialization ---

    def _resolve_db_path(self) -> Path:
        primary = self.skills_root / DB_FILENAME
        try:
            self.skills_root.mkdir(parents=True, exist_ok=True)
            primary.touch(exist_ok=True)
            return primary
        except OSError:
            FALLBACK_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(str(self.skills_root).encode()).hexdigest()[:12]
            return FALLBACK_INDEX_DIR / f"{key}.db"

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._init_schema()
            return self._conn

    def close(self):
        """Close database connection. Call on shutdown."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _init_schema(self):
        """Initialize SQLite schema: skills table + FTS5.

        Creates skills table with fields for storing metadata and embeddings.
        Creates FTS5 virtual table for full-text search.
        FTS5 may be unavailable — in this case _fts_available = False.
        """
        c = self.conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                path_id      TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                description  TEXT NOT NULL,
                category     TEXT DEFAULT '',
                when_to_use  TEXT DEFAULT '',
                rel_path     TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                model_hash   TEXT NOT NULL,
                embedding    BLOB,
                updated_at   REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")
        try:
            c.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} "
                f"USING fts5(name, description, category, when_to_use, path_id UNINDEXED)"
            )
            self._fts_available = True
        except sqlite3.OperationalError as e:
            logger.warning(f"{LOG_PREFIX} FTS5 unavailable: {e}")
            self._fts_available = False
        c.commit()

    # --- Model ---

    def load_model(self):
        if self.config.get("provider") != "local":
            return None
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(LOCAL_MODEL, revision=LOCAL_REVISION)
        return self._model

    def embed(self, text: str, prefix: str) -> Optional[np.ndarray]:
        provider = self.config.get("provider")
        if provider == "local":
            return self._embed_local(text, prefix)
        if provider == "openai_compatible":
            vecs = self._embed_api([text], prefix, batch=False)
            return vecs[0] if vecs else None
        return None

    def embed_batch(self, texts, prefix: str):
        """Batch-embedding of a list of texts.

        Args:
            texts: List of texts to embed.
            prefix: Prefix for each text (e.g., "passage: ").

        Returns:
            List of numpy vectors (normalized). Empty list on error.
        """
        provider = self.config.get("provider")
        if provider == "local":
            try:
                model = self.load_model()
                arr = model.encode(
                    [prefix + t for t in texts],
                    normalize_embeddings=True,
                    batch_size=API_BATCH_SIZE,
                )
                return [np.asarray(v, dtype=np.float32) for v in arr]
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} local batch embed failed: {e}")
                return []
        if provider == "openai_compatible":
            return self._embed_api(texts, prefix, batch=True)
        return []

    def _embed_local(self, text, prefix):
        """Embed a single text via local model (sentence-transformers).

        Args:
            text: Text to embed.
            prefix: Prefix (e.g., "query: ").

        Returns:
            Normalized numpy vector or None on error.
        """
        try:
            model = self.load_model()
            vec = model.encode([prefix + text], normalize_embeddings=True)[0]
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} local embed failed: {e}")
            return None

    def _embed_api(self, texts, prefix, batch):
        """Embed via OpenAI-compatible API (LM Studio, Ollama, vLLM).

        Args:
            texts: List of texts to embed.
            prefix: Prefix for each text.
            batch: If True — return all vectors; if False — only first.

        Returns:
            List of normalized numpy vectors. Empty list on error.
        """
        try:
            resp = _get_session().post(
                f"{self.config.get('api_base')}/embeddings",
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.get("api_model"),
                    "input": [prefix + t for t in texts],
                },
                timeout=(API_TIMEOUT_CONNECT, API_TIMEOUT_READ),
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            out = []
            for d in data:
                arr = np.asarray(d["embedding"], dtype=np.float32)
                n = np.linalg.norm(arr)
                out.append(arr / n if n > 0 else arr)
            return out if batch else out[:1]
        except Exception as e:
            logger.warning(f"{LOG_PREFIX} API embed failed: {e}")
            return []

    # --- Scanning ---

    def scan_skills(self):
        """Scan skills with Hermes filtering.

        Uses _is_skill_visible() to exclude:
        - disabled skills (config.yaml skills.disabled)
        - skills with incompatible platform/environment/apps
        - skills with unmet requires_toolsets/requires_tools
        """
        if not self.skills_root.exists():
            return []
        found = []
        for path in self.skills_root.rglob(SKILL_FILE):
            # Skip excluded directories
            if _is_excluded_dir(path, self.skills_root):
                continue
            try:
                # M3: File size check before reading
                if path.stat().st_size > _MAX_SKILL_FILE_SIZE:
                    logger.warning("%s skipping %s: file too large (%d bytes)",
                                   LOG_PREFIX, path, path.stat().st_size)
                    continue
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = _hermes_parse_frontmatter(text)
            if not meta:
                continue
            name = (meta.get("name") or path.parent.name).strip()
            description = (meta.get("description") or "").strip()
            if not name or not description:
                continue
            # Filtering: disabled / platform / tools
            if not _is_skill_visible(name, meta):
                logger.debug("%s skipping %s: not visible in system prompt", LOG_PREFIX, name)
                continue
            composed = _compose_text(meta, name)
            rel_path = str(path.relative_to(self.skills_root))
            found.append({
                "name": name,
                "description": description,
                "category": (meta.get("category") or "").strip(),
                "when_to_use": _stringify(
                    meta.get("when_to_use") or meta.get("triggers")
                ),
                "composed": composed,
                "rel_path": rel_path,
                "path_id": _path_id(rel_path),
                "content_hash": _content_hash(composed),
                "auto_created": bool(meta.get("auto_created")),
            })
        return found

    def _find_skill_path(self, name) -> Optional[Path]:
        if name in self._path_cache:
            return self._path_cache[name]
        for path in self.skills_root.rglob(SKILL_FILE):
            if _is_excluded_dir(path, self.skills_root):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, body = _hermes_parse_frontmatter(text)
            if meta:
                n = meta.get("name") or path.parent.name
                self._path_cache[n] = path
                if n == name:
                    return path
        return None

    # --- Synchronization ---

    def sync_startup(self):
        found = self.scan_skills()
        self._skill_count = len(found)
        by_id = {s["path_id"]: s for s in found}

        c = self.conn
        existing = {
            row[0]: (row[1], row[2])
            for row in c.execute(
                "SELECT path_id, content_hash, model_hash FROM skills"
            )
        }

        for path_id in existing:
            if path_id not in by_id:
                c.execute("DELETE FROM skills WHERE path_id = ?", (path_id,))
                if self._fts_available:
                    c.execute(f"DELETE FROM {FTS_TABLE} WHERE path_id = ?", (path_id,))

        to_reindex = [
            s for pid, s in by_id.items()
            if existing.get(pid) is None
            or existing[pid][0] != s["content_hash"]
            or existing[pid][1] != self.model_hash
        ]

        for i in range(0, len(to_reindex), API_BATCH_SIZE):
            batch = to_reindex[i:i + API_BATCH_SIZE]
            texts = [s["composed"] for s in batch]
            vecs = self.embed_batch(texts, PREFIX_PASSAGE)
            for skill, vec in zip(batch, vecs):
                self.upsert_with_vec(skill, vec)

        c.commit()
        logger.info(
            f"{LOG_PREFIX} sync: {len(found)} skills, "
            f"{len(to_reindex)} reindexed"
        )

    def upsert_with_vec(self, skill, vec):
        """If vec is None — do not update hashes.

        This keeps the skill "stale" until the next successful attempt,
        so BM25 fallback activates on subsequent turns.
        """
        c = self.conn
        if vec is None:
            c.execute("""
                INSERT INTO skills
                    (path_id, name, description, category, when_to_use,
                     rel_path, content_hash, model_hash, embedding, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(path_id) DO UPDATE SET
                    name        = excluded.name,
                    description = excluded.description,
                    category    = excluded.category,
                    when_to_use = excluded.when_to_use,
                    rel_path    = excluded.rel_path,
                    updated_at  = excluded.updated_at
            """, (
                skill["path_id"], skill["name"], skill["description"],
                skill["category"], skill["when_to_use"],
                skill["rel_path"], "STALE", "STALE", time.time(),
            ))
            return

        blob = vec.tobytes()
        c.execute("""
            INSERT INTO skills
                (path_id, name, description, category, when_to_use,
                 rel_path, content_hash, model_hash, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path_id) DO UPDATE SET
                name         = excluded.name,
                description  = excluded.description,
                category     = excluded.category,
                when_to_use  = excluded.when_to_use,
                rel_path     = excluded.rel_path,
                content_hash = excluded.content_hash,
                model_hash   = excluded.model_hash,
                embedding    = excluded.embedding,
                updated_at   = excluded.updated_at
        """, (
            skill["path_id"], skill["name"], skill["description"],
            skill["category"], skill["when_to_use"],
            skill["rel_path"], skill["content_hash"], self.model_hash,
            blob, time.time(),
        ))
        if self._fts_available:
            c.execute(f"DELETE FROM {FTS_TABLE} WHERE path_id = ?", (skill["path_id"],))
            c.execute(
                f"INSERT INTO {FTS_TABLE} "
                f"(name, description, category, when_to_use, path_id) "
                f"VALUES (?, ?, ?, ?, ?)",
                (skill["name"], skill["description"], skill["category"],
                 skill["when_to_use"], skill["path_id"]),
            )

    def reindex(self, names):
        """Targeted re-indexing by names."""
        if not names:
            return
        names_set = set(names)
        to_reindex = []
        for skill in self.scan_skills():
            if skill["name"] in names_set:
                to_reindex.append(skill)
                self._path_cache.pop(skill["name"], None)

        for i in range(0, len(to_reindex), API_BATCH_SIZE):
            batch = to_reindex[i:i + API_BATCH_SIZE]
            texts = [s["composed"] for s in batch]
            vecs = self.embed_batch(texts, PREFIX_PASSAGE)
            if len(vecs) < len(batch):
                logger.warning(
                    "%s reindex: got %d vectors for %d skills (partial failure)",
                    LOG_PREFIX, len(vecs), len(batch),
                )
            for skill, vec in zip(batch, vecs):
                self.upsert_with_vec(skill, vec)
        self.conn.commit()

    def check_fresh(self, names):
        """Check freshness of only the specified names."""
        if not names:
            return []
        stale = []
        for name in set(names):
            path = self._find_skill_path(name)
            if path is None:
                stale.append(name)
                continue
            try:
                # M3: File size check
                if path.stat().st_size > _MAX_SKILL_FILE_SIZE:
                    stale.append(name)
                    continue
                text = path.read_text(encoding="utf-8")
            except OSError:
                stale.append(name)
                continue
            meta, body = _hermes_parse_frontmatter(text)
            if not meta:
                stale.append(name)
                continue
            n = (meta.get("name") or path.parent.name).strip()
            composed = _compose_text(meta, n)
            h = _content_hash(composed)
            row = self.conn.execute(
                "SELECT content_hash, model_hash FROM skills WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None or row[0] != h or row[1] != self.model_hash:
                stale.append(name)
        return stale
