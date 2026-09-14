"""The corpus index: SQLite FTS5, in a database NOVA owns outright.

Not the runtime's database. The runtime keeps sessions, memories and task state in its own
store, on its own schema, which upstream is free to change in any release. Writing NOVA's
corpus in there would convert every upstream merge into a data-migration question and make
the patch budget a fiction. A separate file costs nothing and keeps the boundary honest.

FTS5 with the ``porter`` stemmer, ranked by ``bm25``, and no embeddings anywhere. That is a
deliberate stopping point, not an unfinished one: BM25 over a few thousand well-chunked
enterprise documents is genuinely good, it is exact and explainable when a customer asks why
a document ranked where it did, and it adds no model provider, no vector store, no
dimension-compatibility problem and no re-embedding cost on every ingest. Semantic retrieval
is a real improvement for some corpora and belongs behind the same interface later; shipping
it first would have meant shipping a dependency stack before shipping retrieval.

The write side is transactional per document, so an ingest interrupted halfway leaves a
consistent index of the documents it had finished rather than a half-written one.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from nova.errors import NovaError
from nova.knowledge.chunk import Chunk
from nova.knowledge.query import format_citation, search as _search

#: Bumped when the schema changes in a way an existing index cannot be read under. The
#: response is always to rebuild: the index is derived data, so recreating it is cheap and
#: migrating it would be code nobody can test against a customer's real corpus.
SCHEMA_VERSION = 1

#: Column 0 must be ``text``: ``snippet()`` and ``bm25()`` address columns positionally, and
#: the query module — which is copied into the agent plugin and cannot see this file —
#: hard-codes index 0 as the searchable one.
_CREATE = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id      TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL,
        doc_path    TEXT NOT NULL,
        doc_title   TEXT NOT NULL DEFAULT '',
        digest      TEXT NOT NULL,
        byte_size   INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        method      TEXT NOT NULL DEFAULT '',
        indexed_at  TEXT NOT NULL,
        UNIQUE (source_id, doc_path)
    )
    """,
    "CREATE INDEX IF NOT EXISTS documents_by_source ON documents (source_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        text,
        chunk_id      UNINDEXED,
        doc_id        UNINDEXED,
        source_id     UNINDEXED,
        doc_path      UNINDEXED,
        doc_title     UNINDEXED,
        ordinal       UNINDEXED,
        start_line    UNINDEXED,
        end_line      UNINDEXED,
        heading_trail UNINDEXED,
        tokenize = 'porter unicode61'
    )
    """,
)


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk, with everything needed to cite it."""

    chunk_id: str
    source_id: str
    doc_path: str
    doc_title: str
    start_line: int
    end_line: int
    heading_trail: tuple[str, ...]
    text: str
    snippet: str
    score: float

    @property
    def citation(self) -> str:
        return format_citation(self.doc_path, self.start_line, self.end_line, self.heading_trail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "doc_path": self.doc_path,
            "doc_title": self.doc_title,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "heading_trail": list(self.heading_trail),
            "citation": self.citation,
            "snippet": self.snippet,
            "score": self.score,
            "text": self.text,
        }


@dataclass(frozen=True)
class DocumentRecord:
    """What the index remembers about one ingested document."""

    source_id: str
    doc_path: str
    digest: str
    doc_title: str = ""
    byte_size: int = 0
    method: str = ""


def fts5_available(connection: Optional[sqlite3.Connection] = None) -> bool:
    """Whether this Python's SQLite was built with FTS5.

    Worth probing rather than assuming: FTS5 is a compile-time option, and the failure on a
    build without it is an ``OperationalError`` deep inside a CREATE statement rather than
    anything a customer could act on.
    """
    own = connection is None
    connection = connection or sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        connection.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        if own:
            connection.close()


class KnowledgeIndex:
    """A corpus index, opened for reading or for writing."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self._connection = connection

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str, *, create: bool = True) -> "KnowledgeIndex":
        path = Path(path).expanduser()
        if not path.exists() and not create:
            raise NovaError(f"no knowledge index at {path}")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path))
        try:
            if not fts5_available(connection):
                raise NovaError(
                    "this Python's SQLite was built without FTS5, which NOVA Knowledge "
                    "requires. Install a python3 built against a full SQLite (the "
                    "python.org and Debian/Ubuntu builds both include FTS5)."
                )
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            index = cls(path, connection)
            if create:
                index._create_schema()
            index._check_schema_version()
            return index
        except Exception:
            connection.close()
            raise

    def _create_schema(self) -> None:
        with self._connection:
            for statement in _CREATE:
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _check_schema_version(self) -> None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        found = int(row[0]) if row and str(row[0]).isdigit() else 0
        if found != SCHEMA_VERSION:
            raise NovaError(
                f"knowledge index at {self.path} is schema version {found}, this NOVA "
                f"expects {SCHEMA_VERSION}. The index is derived data — delete it and "
                "re-run `nova knowledge ingest` to rebuild."
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "KnowledgeIndex":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def known_digest(self, source_id: str, doc_path: str) -> Optional[str]:
        """The digest recorded for this document, or None if it has never been ingested."""
        row = self._connection.execute(
            "SELECT digest FROM documents WHERE source_id = ? AND doc_path = ?",
            (source_id, doc_path),
        ).fetchone()
        return row[0] if row else None

    def replace_document(self, record: DocumentRecord, chunks: Sequence[Chunk]) -> int:
        """Write one document's chunks, replacing any earlier version of it. Atomic.

        Delete-then-insert rather than a diff. The chunk ids of an edited document shift
        wherever text moved, so computing a minimal update would cost more than rewriting a
        few dozen rows, and would be a new class of bug for no measurable gain.
        """
        doc_id = document_id(record.source_id, record.doc_path)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with self._connection:
            self._connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._connection.executemany(
                """
                INSERT INTO chunks (
                    text, chunk_id, doc_id, source_id, doc_path, doc_title,
                    ordinal, start_line, end_line, heading_trail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.text, chunk.chunk_id, doc_id, record.source_id,
                        record.doc_path, record.doc_title, chunk.ordinal,
                        chunk.start_line, chunk.end_line,
                        json.dumps(list(chunk.heading_trail)),
                    )
                    for chunk in chunks
                ],
            )
            self._connection.execute(
                """
                INSERT INTO documents (
                    doc_id, source_id, doc_path, doc_title, digest,
                    byte_size, chunk_count, method, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (doc_id) DO UPDATE SET
                    doc_title = excluded.doc_title,
                    digest = excluded.digest,
                    byte_size = excluded.byte_size,
                    chunk_count = excluded.chunk_count,
                    method = excluded.method,
                    indexed_at = excluded.indexed_at
                """,
                (
                    doc_id, record.source_id, record.doc_path, record.doc_title,
                    record.digest, record.byte_size, len(chunks), record.method, now,
                ),
            )
        return len(chunks)

    def touch_document(self, record: DocumentRecord) -> None:
        """Record that an unchanged document was seen, without rewriting its chunks."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with self._connection:
            self._connection.execute(
                "UPDATE documents SET indexed_at = ? WHERE doc_id = ?",
                (now, document_id(record.source_id, record.doc_path)),
            )

    def prune(self, source_id: str, keep_paths: Iterable[str]) -> list[str]:
        """Drop documents of *source_id* that are no longer on disk. Returns what went.

        A deleted file that stays searchable is a live disclosure: the customer took the
        document down and their agents kept quoting it. So pruning is part of every ingest,
        not a separate maintenance command someone has to remember to run.
        """
        keep = set(keep_paths)
        present = [
            row[0]
            for row in self._connection.execute(
                "SELECT doc_path FROM documents WHERE source_id = ?", (source_id,)
            )
        ]
        gone = sorted(set(present) - keep)
        if not gone:
            return []
        with self._connection:
            for doc_path in gone:
                doc_id = document_id(source_id, doc_path)
                self._connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                self._connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return gone

    def drop_source(self, source_id: str) -> int:
        """Remove a whole corpus — used when a source is undeclared."""
        with self._connection:
            removed = self._connection.execute(
                "SELECT COUNT(*) FROM documents WHERE source_id = ?", (source_id,)
            ).fetchone()[0]
            self._connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            self._connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
        return int(removed)

    def optimize(self) -> None:
        """Merge the FTS index's b-trees. Worth one call at the end of an ingest."""
        with self._connection:
            self._connection.execute("INSERT INTO chunks(chunks) VALUES ('optimize')")

    # -- reading -----------------------------------------------------------

    def search(
        self, question: str, *, source_ids: Sequence[str], limit: int = 5
    ) -> list[SearchHit]:
        """Search, scoped to *source_ids*. See :func:`nova.knowledge.query.search`."""
        return [
            SearchHit(
                chunk_id=hit["chunk_id"],
                source_id=hit["source_id"],
                doc_path=hit["doc_path"],
                doc_title=hit["doc_title"],
                start_line=int(hit["start_line"] or 0),
                end_line=int(hit["end_line"] or 0),
                heading_trail=tuple(hit["heading_trail"]),
                text=hit["text"],
                snippet=hit["snippet"],
                score=hit["score"],
            )
            for hit in _search(self._connection, question, source_ids=source_ids, limit=limit)
        ]

    def stats(self) -> dict[str, dict[str, int]]:
        """Documents and chunks per source, for the dashboard and the ingest report."""
        rows = self._connection.execute(
            """
            SELECT source_id, COUNT(*), COALESCE(SUM(chunk_count), 0), COALESCE(SUM(byte_size), 0)
              FROM documents GROUP BY source_id ORDER BY source_id
            """
        ).fetchall()
        return {
            str(source_id): {
                "documents": int(documents),
                "chunks": int(chunks),
                "bytes": int(byte_size),
            }
            for source_id, documents, chunks, byte_size in rows
        }

    def documents(self, source_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Every indexed document, newest ingest first. For the dashboard's corpus view."""
        sql = (
            "SELECT source_id, doc_path, doc_title, digest, byte_size, chunk_count, "
            "method, indexed_at FROM documents"
        )
        parameters: tuple[Any, ...] = ()
        if source_id:
            sql += " WHERE source_id = ?"
            parameters = (source_id,)
        sql += " ORDER BY source_id, doc_path"
        keys = (
            "source_id", "doc_path", "doc_title", "digest",
            "byte_size", "chunk_count", "method", "indexed_at",
        )
        return [dict(zip(keys, row)) for row in self._connection.execute(sql, parameters)]


def document_id(source_id: str, doc_path: str) -> str:
    """A stable id for one document within one corpus."""
    import hashlib

    return hashlib.sha256(f"{source_id}\0{doc_path}".encode("utf-8")).hexdigest()[:32]
