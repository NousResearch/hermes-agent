"""Layer 5 indexer: deterministic SQLite + FTS5 over existing markdown.

Design guarantees (per memory-architecture.md §7, §14):

* **Markdown is truth; SQLite is a cache.** ``index.db`` is fully derivable
  from the markdown sources and is gitignored.
* **Deterministic rebuild.** Same inputs -> identical database. Files are
  processed in sorted order; chunks within a file keep their source position;
  timestamps come from file mtime (stable for identical inputs).
* **Graceful degradation.** If FTS5 is unavailable at create time, the indexer
  builds the base tables only and falls back to a ``LIKE`` scan at search time
  (``retrieval_method == "sqlite-like"``). It never raises on missing FTS5.
* **No interpretation.** The index stores *raw content chunks*, never
  summaries or LLM output.

Out of scope (explicitly NOT done): auto-extraction, summarization,
embeddings, Graphiti, Holographic.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

from hermes_cli.memory_router.provenance import SearchResult

# Relative source paths (to HERMES_HOME) the indexer knows about.
_L1_FILES = (
    ("SOUL.md", "L1-identity"),
    ("memories/IDENTITY.md", "L1-identity"),
    ("memories/USER.md", "L1-identity"),
    ("memories/MEMORY.md", "L1-identity"),
)
_SPECIAL_FILES = (
    ("HERMES_PROJECTS.md", "L5-projects"),
    ("HERMES_SESSION.md", "L5-session"),
)

_FTS5_TABLE = "notes_fts"
_BASE_TABLE = "notes"

# Stopwords dropped from FTS5 token queries (cosmetic; improves recall focus).
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "be", "by", "with", "at", "from", "as",
        "it", "this", "that", "i", "you", "we", "they", "what", "who",
    }
)

# Master switch for FTS5 usage. Tests force this to False to exercise the
# sqlite-like fallback path. In normal operation it stays True and only the
# per-connection OperationalError detection downgrades to the fallback.
_FTS5_ENABLED: bool = True

# Max chunk size in characters (paragraphs longer than this get split).
_MAX_CHUNK = 500


class MemoryIndex:
    """Deterministic FTS5 index over existing Hermes markdown."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        hermes_home: Optional[Path] = None,
    ) -> None:
        self.hermes_home = Path(hermes_home) if hermes_home else get_hermes_home()
        if db_path is None:
            db_path = self.hermes_home / "memory" / "index.db"
        self.db_path = Path(db_path)
        # Detection of FTS5 support at the db level. Overridable for tests.
        self._fts5_enabled: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Public lifecycle
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True if the index database exists and can be opened."""
        if not self.db_path.exists():
            return False
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(f"SELECT count(*) FROM {_BASE_TABLE}")
            return True
        except (sqlite3.Error, OSError):
            return False

    def fts5_available(self) -> bool:
        """True if FTS5 is usable on the current index (graceful-degradation aware)."""
        if not self.available():
            return False
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                return self._detect_fts5(conn)
        except (sqlite3.Error, OSError):
            return False

    def document_count(self) -> int:
        """Number of indexed source documents (distinct source_file rows)."""
        if not self.available():
            return 0
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute(
                    f"SELECT count(DISTINCT source_file) FROM {_BASE_TABLE}"
                )
                return int(cur.fetchone()[0])
        except (sqlite3.Error, OSError):
            return 0

    def index_health(self) -> dict:
        """Structured health snapshot of the Layer 5 index cache."""
        db_exists = self.db_path.exists()
        opens = readable = fts = False
        if db_exists:
            try:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    conn.execute(f"SELECT count(*) FROM {_BASE_TABLE}")
                    opens = readable = True
                    fts = self._detect_fts5(conn)
                finally:
                    conn.close()
            except (sqlite3.Error, OSError):
                pass
        return {
            "db_path": str(self.db_path),
            "db_exists": db_exists,
            "opens": opens,
            "base_table_readable": readable,
            "fts5": fts,
            "status": "ok" if (opens and readable) else ("empty" if not db_exists else "corrupt"),
        }

    def build(self, hermes_home: Optional[Path] = None) -> int:
        """Build (or rebuild in place) the index. Returns number of chunks.

        Deterministic: same inputs -> identical database contents.
        """
        home = Path(hermes_home) if hermes_home else self.hermes_home
        self.hermes_home = home
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        sources = self._discover_sources(home)
        rows: list[dict[str, Any]] = []
        for abs_path, rel_path, layer in sources:
            rows.extend(self._file_rows(abs_path, rel_path, layer))

        conn = sqlite3.connect(str(self.db_path))
        try:
            self._init_schema(conn)
            # Deterministic clear of the base + fts tables, then insert.
            self._clear(conn)
            for row in rows:
                self._insert(conn, row)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def rebuild(self, hermes_home: Optional[Path] = None) -> int:
        """Drop all tables and rebuild from scratch. Deterministic."""
        if self.db_path.exists():
            self.db_path.unlink()
        return self.build(hermes_home)

    # ------------------------------------------------------------------ #
    # Phase 3 — archive lifecycle (incremental, lag-tolerant, non-blocking)
    # ------------------------------------------------------------------ #
    def enqueue(self, source_file: str) -> None:
        """Mark a raw transcript pending (re)indexing. UPSERT: resets status.

        Idempotent. The actual indexing happens in ``refresh_pending()``
        (lazy on next search/status, or via the async flush in the listener).
        Never raises on read error — the file is checked for existence here so
        a missing transcript simply isn't enqueued.
        """
        rel = source_file
        if Path(source_file).is_absolute():
            try:
                rel = str(Path(source_file).relative_to(self.hermes_home))
            except ValueError:
                rel = str(Path(source_file))
        if not (self.hermes_home / rel).is_file():
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            self._init_schema(conn)
            conn.execute(
                "INSERT INTO index_pending (source_file, enqueued_at, attempts, "
                "last_error, last_attempt, status) VALUES (?, ?, 0, NULL, NULL, 'pending') "
                "ON CONFLICT(source_file) DO UPDATE SET "
                "enqueued_at=excluded.enqueued_at, attempts=0, "
                "last_error=NULL, last_attempt=NULL, status='pending'",
                (rel, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def index_session(self, source_file: str) -> int:
        """Idempotently (re)index ONE raw transcript. DELETE+INSERT by source.

        Returns number of chunks indexed. Boring on purpose: no diffing, no
        partial updates, no migration. The raw file is READ ONLY — never
        mutated (ownership rule in docs/memory-archive-contract.md §0).
        """
        abs_path = Path(source_file)
        if not abs_path.is_absolute():
            abs_path = self.hermes_home / source_file
        if not abs_path.is_file():
            return 0
        rel = source_file
        if Path(source_file).is_absolute():
            try:
                rel = str(abs_path.relative_to(self.hermes_home))
            except ValueError:
                rel = str(abs_path)
        rows = self._file_rows(abs_path, rel, "L3-archive")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            self._init_schema(conn)
            # Idempotent: drop this file's existing rows, then re-insert.
            conn.execute(
                f"DELETE FROM {_BASE_TABLE} WHERE source_file = ? AND memory_layer = 'L3-archive'",
                (rel,),
            )
            for row in rows:
                self._insert(conn, row)
            conn.commit()
        finally:
            conn.close()
        return len(rows)

    def refresh_pending(self, *, force: bool = False) -> dict:
        """Drain pending/failed rows into the index. Per-row fault isolation.

        Returns a stats dict: {drained, ok, failed, errors}.

        Called lazily by ``search()``/``archive_stats()`` (so a closed session
        is searchable by the next query even if the async flush never ran), and
        explicitly by the listener's fire-and-forget thread. Never raises.
        """
        if not self.db_path.exists():
            return {"drained": 0, "ok": 0, "failed": 0, "errors": []}
        conn = sqlite3.connect(str(self.db_path))
        try:
            try:
                rows = conn.execute(
                    "SELECT source_file FROM index_pending "
                    "WHERE status IN ('pending','failed')"
                ).fetchall()
            except sqlite3.OperationalError:
                return {"drained": 0, "ok": 0, "failed": 0, "errors": []}
            stats = {"drained": len(rows), "ok": 0, "failed": 0, "errors": []}
            for (sf,) in rows:
                last_attempt = datetime.now(timezone.utc).isoformat()
                try:
                    n = self.index_session(sf)
                    conn.execute(
                        "UPDATE index_pending SET attempts=attempts+1, "
                        "last_attempt=?, last_error=NULL, status='done' WHERE source_file=?",
                        (last_attempt, sf),
                    )
                    stats["ok"] += 1 if n >= 0 else 0
                except Exception as e:  # noqa: BLE001 — one bad file must not block others
                    err = f"{type(e).__name__}: {e}"[:500]
                    conn.execute(
                        "UPDATE index_pending SET attempts=attempts+1, "
                        "last_attempt=?, last_error=?, status='failed' WHERE source_file=?",
                        (last_attempt, err, sf),
                    )
                    stats["failed"] += 1
                    stats["errors"].append((sf, err))
            conn.commit()
            return stats
        finally:
            conn.close()

    def archive_stats(self) -> dict:
        """L3 archive lifecycle counters for `hermes memory status`.

        Triggers a lazy ``refresh_pending()`` first so the numbers reflect
        reality by the time the user looks. Returns:
          indexed_sessions, pending, failed, last_refresh, last_error
        """
        self.refresh_pending()
        if not self.db_path.exists():
            return {
                "indexed_sessions": 0,
                "pending": 0,
                "failed": 0,
                "last_refresh": None,
                "last_error": None,
            }
        conn = sqlite3.connect(str(self.db_path))
        try:
            try:
                indexed = conn.execute(
                    "SELECT count(DISTINCT source_file) FROM notes WHERE memory_layer='L3-archive'"
                ).fetchone()[0]
                pending = conn.execute(
                    "SELECT count(*) FROM index_pending WHERE status='pending'"
                ).fetchone()[0]
                failed = conn.execute(
                    "SELECT count(*) FROM index_pending WHERE status='failed'"
                ).fetchone()[0]
                last = conn.execute(
                    "SELECT MAX(last_attempt) FROM index_pending WHERE status='done'"
                ).fetchone()[0]
                last_err = conn.execute(
                    "SELECT last_error FROM index_pending WHERE status='failed' "
                    "ORDER BY last_attempt DESC LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError:
                return {
                    "indexed_sessions": 0,
                    "pending": 0,
                    "failed": 0,
                    "last_refresh": None,
                    "last_error": None,
                }
            return {
                "indexed_sessions": int(indexed or 0),
                "pending": int(pending or 0),
                "failed": int(failed or 0),
                "last_refresh": last,
                "last_error": last_err[0] if last_err else None,
            }
        finally:
            conn.close()

    def archive(
        self,
        query: str = "",
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Retrieve conversation-archive chunks (L3-archive), read-only.

        Scoped to ``memory_layer='L3-archive'``. Optional ``session_id`` narrows
        to one closed session. With an empty ``query`` and a ``session_id``,
        returns that session's chunks ordered by position. A lazy
        ``refresh_pending()`` runs first so a just-closed session is retrievable
        by the next query even if the async flush never ran.
        """
        if not self.db_path.exists():
            return []
        self.refresh_pending()
        conn = sqlite3.connect(str(self.db_path))
        try:
            clauses = ["memory_layer = 'L3-archive'"]
            params: list[Any] = []
            if session_id:
                clauses.append("json_extract(extra, '$.session_id') = ?")
                params.append(session_id)
            if query:
                toks = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower())
                        if t not in _STOPWORDS] or [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower())]
                sub = " OR ".join(["(content LIKE ? OR tags LIKE ?)" for _ in toks]) or "1=0"
                clauses.append(f"({sub})")
                for t in toks:
                    params.extend([f"%{t}%", f"%{t}%"])
            sql = (
                "SELECT source_file, memory_layer, content, created_at, extra "
                f"FROM {_BASE_TABLE} WHERE {' AND '.join(clauses)} "
                "ORDER BY id LIMIT ?"
            )
            params.append(limit)
            try:
                cur = conn.execute(sql, params)
            except sqlite3.OperationalError:
                return []
            return [self._row_to_result(r, "sqlite-like", None) for r in cur.fetchall()]
        finally:
            conn.close()

    def recent(self, limit: int = 10) -> list[SearchResult]:
        """Most-recently-updated chunks across L3-archive + L1/L5-notes.

        Read-only. Recent-ness is by source ``updated_at`` (file mtime). Used by
        the ``recent()`` API operation. Returns provenance-bearing results only.
        """
        if not self.db_path.exists():
            return []
        self.refresh_pending()
        conn = sqlite3.connect(str(self.db_path))
        try:
            try:
                cur = conn.execute(
                    f"SELECT source_file, memory_layer, content, created_at, extra "
                    f"FROM {_BASE_TABLE} "
                    "WHERE memory_layer IN ('L3-archive','L1-identity','L5-notes') "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?",
                    (limit,),
                )
            except sqlite3.OperationalError:
                return []
            return [self._row_to_result(r, "sqlite-like", None) for r in cur.fetchall()]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        limit: int = 10,
        scope: Optional[str] = None,
    ) -> list[SearchResult]:
        """Full-text search over indexed content.

        Uses FTS5 when available, otherwise falls back to a ``LIKE`` scan.
        Every returned :class:`SearchResult` carries full provenance. The
        index never interprets content — it returns raw matched chunks.

        A lazy ``refresh_pending()`` runs first so a just-closed session
        (enqueued by the archive-lifecycle listener) is searchable by the next
        query even if the async flush never ran. Boring and safe.
        """
        if not self.db_path.exists():
            return []
        self.refresh_pending()
        conn = sqlite3.connect(str(self.db_path))
        try:
            fts5 = self._detect_fts5(conn)
            if fts5:
                return self._search_fts5(conn, query, limit)
            return self._search_like(conn, query, limit)
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _discover_sources(
        self, home: Path
    ) -> list[tuple[Path, str, str]]:
        """Return (abs_path, rel_path, layer) tuples, sorted deterministically."""
        found: list[tuple[Path, str, str]] = []
        for rel, layer in _L1_FILES + _SPECIAL_FILES:
            p = home / rel
            if p.is_file():
                found.append((p, rel, layer))
        # Any other *.md under memories/ (covers notes beyond the 3 identity
        # files, which were already picked up above via _L1_FILES).
        mem_dir = home / "memories"
        if mem_dir.is_dir():
            for p in sorted(mem_dir.rglob("*.md")):
                rel = p.relative_to(home).as_posix()
                if any(rel == r for r, _ in _L1_FILES):
                    continue  # already counted as L1
                found.append((p, rel, "L5-notes"))
        # L3 conversation archive: index existing session/archive JSONL in
        # place (Phase 2, Q1). No migration, no new writers. The raw files are
        # the source of truth; this is a read-only cache extension.
        for pat in ("sessions", "archive"):
            base = home / pat
            if base.is_dir():
                for p in sorted(base.rglob("*.jsonl")):
                    rel = p.relative_to(home).as_posix()
                    found.append((p, rel, "L3-archive"))
        # Deterministic ordering by (layer, rel path).
        found.sort(key=lambda t: (t[2], t[1]))
        return found

    def _file_rows(
        self, abs_path: Path, rel_path: str, layer: str
    ) -> list[dict[str, Any]]:
        try:
            raw = abs_path.read_text(encoding="utf-8", errors="replace")
            mtime = abs_path.stat().st_mtime
        except OSError:
            return []
        ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        # L3 archive rows are derived from JSONL events, not markdown chunks.
        if layer == "L3-archive":
            return self._archive_rows(abs_path, rel_path, ts)

        body = self._strip_frontmatter(raw)
        chunks = self._chunk(body)
        rows: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            rows.append(
                {
                    "source_file": rel_path,
                    "memory_layer": layer,
                    "content": chunk,
                    "tags": "",
                    "created_at": ts,
                    "updated_at": ts,
                    "extra": json.dumps({"chunk_index": idx, "chunk_count": len(chunks)}),
                }
            )
        return rows

    def _archive_rows(
        self, abs_path: Path, rel_path: str, file_ts: str
    ) -> list[dict[str, Any]]:
        """Index a session/archive .jsonl: one row per chat event line.

        Raw file is never mutated. Only lines with a non-empty ``content``
        string and a chat ``role`` (user/assistant/system/tool) become chunks.
        The leading ``session_meta`` line is skipped. Provenance is enriched
        with session_id/role/event_ts/chunk_index and the optional
        project_context / hermes_version / git_commit / working_directory
        fields — all best-effort; absence never blocks indexing.
        """
        session_id = Path(rel_path).stem
        env = self._archive_enrichment()
        rows: list[dict[str, Any]] = []
        for idx, line in enumerate(raw := abs_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON line: index verbatim as a raw note (rare; preserves
                # fidelity for malformed-but-present archive lines).
                rows.append(self._archive_row(rel_path, file_ts, idx, "raw", line, session_id, env))
                continue
            role = ev.get("role")
            if role in (None, "session_meta"):
                continue  # session_meta is not a retrievable chat event
            content = ev.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            rows.append(self._archive_row(rel_path, file_ts, idx, role, content, session_id, env, event_ts=ev.get("ts")))
        return rows

    @staticmethod
    def _archive_row(rel_path, file_ts, idx, role, content, session_id, env, event_ts=None):
        extra = {
            "session_id": session_id,
            "role": role,
            "chunk_index": idx,
            "event_ts": event_ts or file_ts,
        }
        # Optional enrichment — never block indexing if any are missing.
        for k in ("project_context", "hermes_version", "git_commit", "working_directory"):
            if env.get(k):
                extra[k] = env[k]
        return {
            "source_file": rel_path,
            "memory_layer": "L3-archive",
            "content": content,
            "tags": "",
            "created_at": event_ts or file_ts,
            "updated_at": file_ts,
            "extra": json.dumps(extra),
        }

    @staticmethod
    def _archive_enrichment() -> dict[str, str]:
        """Best-effort optional provenance enrichment for archive rows.

        Each field is gathered without raising; missing fields are simply
        absent from the row's ``extra`` (see _archive_row). No external calls.
        """
        env: dict[str, str] = {}
        # hermes_version: package version if importable.
        try:
            from hermes_cli import __version__

            env["hermes_version"] = str(__version__)
        except Exception:
            pass
        # git_commit + working_directory: only if a git repo is discoverable
        # from the current working directory (best-effort, non-fatal).
        import subprocess

        try:
            cwd = Path.cwd()
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(cwd), capture_output=True, text=True, timeout=2,
            )
            if commit.returncode == 0 and commit.stdout.strip():
                env["git_commit"] = commit.stdout.strip()
                env["working_directory"] = str(cwd)
        except Exception:
            pass
        return env

    @staticmethod
    def _strip_frontmatter(raw: str) -> str:
        """Return body with a leading YAML frontmatter block removed.

        Parses frontmatter if ``yaml`` is importable; otherwise treats the
        first ``---``...``---`` block as a non-indexed header and drops it.
        """
        if not raw.startswith("---"):
            return raw
        # Find the closing '---' on its own line.
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.DOTALL)
        if not m:
            return raw
        fm, body = m.group(1), m.group(2)
        # If yaml is available, we could parse fm for tags; keep it simple and
        # just drop the frontmatter from indexed body text.
        return body

    @staticmethod
    def _chunk(text: str) -> list[str]:
        """Split text into <=_MAX_CHUNK char chunks, preserving order.

        Paragraphs (blank-line separated) are the primary unit; oversized
        paragraphs are further split on whitespace boundaries.
        """
        text = text.strip()
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= _MAX_CHUNK:
                chunks.append(para)
                continue
            # Split large paragraph into word-bounded pieces.
            words = para.split()
            cur = ""
            for w in words:
                if cur and len(cur) + len(w) + 1 > _MAX_CHUNK:
                    chunks.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                chunks.append(cur)
        return chunks

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text(encoding="utf-8")
        if not _FTS5_ENABLED:
            # Forced fallback (e.g. tests). Build base tables only.
            self._fts5_enabled = False
            conn.executescript(self._base_schema_only())
            return
        try:
            conn.executescript(sql)
            self._fts5_enabled = True
        except sqlite3.OperationalError:
            # FTS5 not compiled into this sqlite3. Build base tables only.
            self._fts5_enabled = False
            conn.executescript(self._base_schema_only())

    def _base_schema_only(self) -> str:
        """Base-table DDL without any FTS5 virtual tables (fallback)."""
        return """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, memory_layer TEXT NOT NULL,
            content TEXT, tags TEXT, created_at TEXT, updated_at TEXT, extra TEXT);
        CREATE INDEX IF NOT EXISTS idx_notes_source ON notes(source_file);
        CREATE INDEX IF NOT EXISTS idx_notes_layer ON notes(memory_layer);
        CREATE TABLE IF NOT EXISTS index_pending (
            source_file  TEXT PRIMARY KEY,
            enqueued_at  TEXT NOT NULL,
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_error   TEXT,
            last_attempt TEXT,
            status       TEXT NOT NULL DEFAULT 'pending'
        );
        """

    def _clear(self, conn: sqlite3.Connection) -> None:
        if self._fts5_enabled:
            try:
                conn.execute(f"DELETE FROM {_FTS5_TABLE}")
            except sqlite3.OperationalError:
                pass
        conn.execute(f"DELETE FROM {_BASE_TABLE}")

    def _insert(self, conn: sqlite3.Connection, row: dict[str, Any]) -> None:
        conn.execute(
            f"INSERT INTO {_BASE_TABLE} "
            "(source_file, memory_layer, content, tags, created_at, updated_at, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["source_file"],
                row["memory_layer"],
                row["content"],
                row["tags"],
                row["created_at"],
                row["updated_at"],
                row["extra"],
            ),
        )
        if self._fts5_enabled:
            try:
                conn.execute(
                    f"INSERT INTO {_FTS5_TABLE}(rowid, content, tags) "
                    "VALUES ((SELECT last_insert_rowid()), ?, ?)",
                    (row["content"], row["tags"]),
                )
            except sqlite3.OperationalError:
                # FTS5 row insert failed; search will fall back to LIKE.
                self._fts5_enabled = False

    def _detect_fts5(self, conn: sqlite3.Connection) -> bool:
        if not _FTS5_ENABLED:
            return False
        if self._fts5_enabled is False:
            return False
        try:
            conn.execute(f"SELECT count(*) FROM {_FTS5_TABLE}")
            return True
        except sqlite3.OperationalError:
            return False

    def _build_fts_query(self, query: str) -> str:
        toks = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if t not in _STOPWORDS]
        if not toks:
            toks = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower())]
        # OR-join so partial recall still surfaces hits. Quote tokens that
        # FTS5 would otherwise treat as operators.
        safe = [f'"{t}"' for t in toks] or ['""']
        return " OR ".join(safe)

    def _search_fts5(
        self, conn: sqlite3.Connection, query: str, limit: int
    ) -> list[SearchResult]:
        q = self._build_fts_query(query)
        sql = (
            f"SELECT n.source_file, n.memory_layer, n.content, n.created_at, n.extra "
            f"FROM {_BASE_TABLE} n JOIN {_FTS5_TABLE} f "
            "ON n.id = f.rowid "
            f"WHERE {_FTS5_TABLE} MATCH ? ORDER BY rank LIMIT ?"
        )
        try:
            cur = conn.execute(sql, (q, limit))
        except sqlite3.OperationalError:
            return self._search_like(conn, query, limit)
        return [self._row_to_result(r, "fts5", None) for r in cur.fetchall()]

    def _search_like(
        self, conn: sqlite3.Connection, query: str, limit: int
    ) -> list[SearchResult]:
        toks = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if t not in _STOPWORDS]
        if not toks:
            toks = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower())]
        clauses = " OR ".join(["(content LIKE ? OR tags LIKE ?)" for _ in toks]) or "1=0"
        params: list[Any] = []
        for t in toks:
            params.extend([f"%{t}%", f"%{t}%"])
        params.append(limit)
        sql = (
            f"SELECT source_file, memory_layer, content, created_at, extra "
            f"FROM {_BASE_TABLE} WHERE {clauses} ORDER BY id LIMIT ?"
        )
        cur = conn.execute(sql, params)
        return [self._row_to_result(r, "sqlite-like", None) for r in cur.fetchall()]

    @staticmethod
    def _row_to_result(
        row: tuple[Any, ...], method: str, score: Optional[float]
    ) -> SearchResult:
        # Columns: source_file, memory_layer, content, created_at, extra
        source_file, layer, content, created_at, extra = row
        # The indexer stores a JSON object in `extra`; surface it as a real
        # dict on the result so provenance enrichment (session_id, role,
        # event_ts, optional project_context/hermes_version/git_commit/
        # working_directory) is visible, not buried in a string.
        extra_dict: dict[str, Any] = {}
        if extra:
            try:
                parsed = json.loads(extra)
                if isinstance(parsed, dict):
                    extra_dict = parsed
            except (json.JSONDecodeError, TypeError):
                extra_dict = {"index_extra": extra}
        return SearchResult(
            source_file=source_file,
            memory_layer=layer,
            retrieval_method=method,
            content=content or "",
            timestamp=created_at,
            snippet=(content or "")[:200],
            score=score,
            intent="historical",
            capability="L5-index",
            extra=extra_dict,
        )
