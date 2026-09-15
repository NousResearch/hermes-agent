"""Durable storage for the LCM context engine.

Three properties of this module are load-bearing for the B04 acceptance
criteria, so they are stated once here rather than re-litigated per method:

1. **Raw messages are append-only.**  ``sync_transcript`` only ever appends;
   there is no UPDATE or DELETE path for a recorded message.  Compaction
   summarizes *by reference*, so the originals survive it.
2. **Every summary node keeps lineage.**  ``create_node`` writes one
   ``node_sources`` row per source (message id or parent node id), and the
   ``(kind, id)`` pair is what ``lcm_expand`` walks.
3. **Paging is enforced here, not by the caller.**  ``page_window`` and
   ``page_node_sources`` clamp their page size to the configured ceiling and
   report ``has_more`` / ``next_cursor``, so no caller can ask for "everything".
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

_MESSAGES = """
CREATE TABLE IF NOT EXISTS messages (
    session_id  TEXT    NOT NULL,
    seq         INTEGER NOT NULL,
    message_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    compacted   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    PRIMARY KEY (session_id, seq)
)
"""

_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id     TEXT    PRIMARY KEY,
    session_id  TEXT    NOT NULL,
    level       INTEGER NOT NULL DEFAULT 1,
    summary     TEXT    NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL
)
"""

_NODE_SOURCES = """
CREATE TABLE IF NOT EXISTS node_sources (
    node_id     TEXT    NOT NULL,
    source_kind TEXT    NOT NULL,
    source_id   TEXT    NOT NULL,
    ordinal     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (node_id, source_kind, source_id)
)
"""

_MESSAGES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(session_id UNINDEXED, message_id UNINDEXED, body)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS messages_msgid ON messages (session_id, message_id)",
    "CREATE INDEX IF NOT EXISTS nodes_session ON nodes (session_id, level)",
    "CREATE INDEX IF NOT EXISTS node_sources_node ON node_sources (node_id, ordinal)",
)


def _migrate_to_1(conn: sqlite3.Connection) -> None:
    """v1: the raw-message / summary-node / lineage core."""
    conn.execute(_SCHEMA_META)
    conn.execute(_MESSAGES)
    conn.execute(_NODES)
    conn.execute(_NODE_SOURCES)


def _migrate_to_2(conn: sqlite3.Connection) -> None:
    """v2: the FTS index and the supporting secondary indexes."""
    conn.execute(_MESSAGES_FTS)
    for statement in _INDEXES:
        conn.execute(statement)


_MIGRATIONS = {1: _migrate_to_1, 2: _migrate_to_2}


def schema_version(conn: sqlite3.Connection) -> int:
    """Return the recorded schema version, or 0 for a pre-v1 database."""
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    if not row:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def migrate(conn: sqlite3.Connection) -> int:
    """Bring *conn* up to ``SCHEMA_VERSION``; return the resulting version.

    Migrations are additive and idempotent, so an already-current database is
    a no-op and a partially-migrated one resumes where it stopped.  Existing
    rows are never touched — a v1 database keeps every message and node.
    """
    conn.execute(_SCHEMA_META)
    current = schema_version(conn)
    while current < SCHEMA_VERSION:
        _MIGRATIONS[current + 1](conn)
        current += 1
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(current),),
    )
    return current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def render_body(message: Dict[str, Any]) -> str:
    """Render a message to searchable text (bounded by the caller)."""
    content = message.get("content")
    parts: List[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
    elif content is not None:
        parts.append(str(content))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                fn = call.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    parts.append(f"[tool_call {name}]")

    name = message.get("name")
    if name:
        parts.append(f"[name {name}]")
    return "\n".join(part for part in parts if part).strip()


def _canonical(message: Dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, default=str)


def _message_id(session_id: str, seq: int, payload: str) -> str:
    digest = hashlib.sha256(
        f"{session_id}\x1f{seq}\x1f{payload}".encode("utf-8", "replace")
    ).hexdigest()
    return f"m{digest[:20]}"


def _node_id(
    session_id: str, level: int, ordinal: int, sources: Iterable[Tuple[str, str]]
) -> str:
    """Content-derived node id.

    Seeded by the *sources* rather than the rendered summary text, so the id is
    stable and knowable before the summary is written — the summary can then
    cite its own node id without a second node having to be created.
    """
    seed = "\x1f".join(
        [session_id, str(level), str(ordinal)]
        + [f"{kind}:{sid}" for kind, sid in sources]
    )
    return "n" + hashlib.sha256(seed.encode("utf-8", "replace")).hexdigest()[:20]


def _fts_query(query: str) -> str:
    """Quote every token so FTS5 MATCH cannot be driven by user syntax.

    FTS5 treats ``"``, ``-``, ``*``, ``:``, ``(``, ``)`` and bare ``AND``/``OR``
    as operators; a raw passthrough therefore both errors and lets a query
    change its own semantics.  Quoting each token as a string literal turns the
    whole thing into a plain conjunction of phrases.
    """
    tokens = [tok for tok in str(query).replace('"', " ").split() if tok]
    if not tokens:
        return ""
    return " AND ".join(f'"{tok}"' for tok in tokens[:32])


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class LCMStore:
    """SQLite-backed raw-message store with DAG summary nodes and lineage."""

    def __init__(
        self,
        db_path: Path,
        *,
        max_page_size: int = 50,
        page_size: int = 10,
        body_chars: int = 2_000,
        max_search_results: int = 8,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_page_size = max(1, int(max_page_size))
        self.page_size = max(1, min(int(page_size), self.max_page_size))
        self.body_chars = max(200, int(body_chars))
        self.max_search_results = max(1, int(max_search_results))

        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available: Optional[bool] = None

    # -- connection --------------------------------------------------------

    def open(self) -> "LCMStore":
        """Open (and migrate) the database.  Idempotent and thread-safe."""
        with self._lock:
            if self._conn is not None:
                return self
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: compression runs on a pooled daemon
            # thread (context_timeout_seconds), so the connection must be
            # shareable; every statement is serialized by ``_lock``.
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            migrate(conn)
            conn.commit()
            self._conn = conn
            return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                except sqlite3.Error:
                    pass
                self._conn.close()
                self._conn = None

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Connection]:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        with self._lock:
            yield self._conn

    def version(self) -> int:
        with self._cursor() as conn:
            return schema_version(conn)

    def fts_available(self) -> bool:
        """Whether the FTS5 index exists; ``search`` falls back to LIKE if not."""
        if self._fts_available is None:
            with self._cursor() as conn:
                try:
                    conn.execute(_MESSAGES_FTS)
                    self._fts_available = True
                except sqlite3.Error:
                    self._fts_available = False
        return bool(self._fts_available)

    # -- raw messages ------------------------------------------------------

    def max_seq(self, session_id: str) -> int:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT MAX(seq) AS m FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["m"]) if row and row["m"] is not None else 0

    def _tail_rows(self, session_id: str, limit: int) -> List[Tuple[str, str]]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT message_id, payload FROM messages WHERE session_id = ? "
                "ORDER BY seq DESC LIMIT ?",
                (session_id, max(0, int(limit))),
            ).fetchall()
        return [(row["message_id"], row["payload"]) for row in reversed(rows)]

    def sync_transcript(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> List[str]:
        """Append any *new* tail of *messages*; return ids aligned to *messages*.

        The transcript is re-read in full on every hook, so this diffs by
        longest-suffix overlap: find the largest ``k`` such that the last ``k``
        stored payloads equal the first ``k`` incoming payloads, then append
        everything after ``k``.  The returned list is positional — ``result[i]``
        is the stored id for ``messages[i]``, whether that row pre-existed or
        was appended by this call — which is what lets compaction cite exact
        lineage for the range it absorbs.

        A compacted transcript (whose middle was replaced by a summary message
        this engine wrote so long before that no overlap remains) contributes
        nothing: the pre-compaction originals were already recorded, so
        returning "nothing new" is the safe direction.  Unresolvable positions
        come back as an empty string.
        """
        if not messages:
            return []
        if not all(isinstance(m, dict) for m in messages):
            return ["" for _ in messages]

        payloads = [_canonical(m) for m in messages]
        with self._cursor() as conn:
            stored = self._tail_rows(session_id, limit=len(payloads))

            overlap = 0
            if stored:
                upper = min(len(stored), len(payloads))
                for k in range(upper, 0, -1):
                    if [p for _mid, p in stored[len(stored) - k:]] == payloads[:k]:
                        overlap = k
                        break
                if overlap == 0:
                    # No suffix overlap.  Two very different situations:
                    #
                    #  * the caller re-read a *rewritten* transcript (our own
                    #    compaction marker replaced the middle) — its surviving
                    #    head/tail rows are already stored, so appending would
                    #    duplicate history; or
                    #  * the caller handed us only freshly-produced messages
                    #    (a tail, not a transcript) — nothing overlaps because
                    #    none of it is stored yet.
                    #
                    # A bounded membership check tells them apart.  Losing a
                    # genuinely new message is the worse error for an engine
                    # whose whole promise is retention, so the tie goes to
                    # appending.
                    recent = {
                        payload
                        for _mid, payload in self._tail_rows(
                            session_id, limit=max(200, len(payloads))
                        )
                    }
                    if any(payload in recent for payload in payloads):
                        return ["" for _ in messages]

            ids: List[str] = [""] * len(messages)
            for index in range(overlap):
                ids[index] = stored[len(stored) - overlap + index][0]

            seq = self.max_seq(session_id)
            now = time.time()
            for index in range(overlap, len(messages)):
                seq += 1
                message = messages[index]
                payload = payloads[index]
                message_id = _message_id(session_id, seq, payload)
                body = render_body(message)[: self.body_chars * 4]
                conn.execute(
                    "INSERT OR IGNORE INTO messages "
                    "(session_id, seq, message_id, role, body, payload, "
                    " compacted, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        session_id,
                        seq,
                        message_id,
                        str(message.get("role") or "unknown"),
                        body,
                        payload,
                        now,
                    ),
                )
                if self.fts_available():
                    conn.execute(
                        "INSERT INTO messages_fts (session_id, message_id, body) "
                        "VALUES (?, ?, ?)",
                        (session_id, message_id, body),
                    )
                ids[index] = message_id
            conn.commit()
            return ids

    def message(self, session_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            ).fetchone()
        return _message_record(row) if row else None

    def mark_compacted(self, session_id: str, message_ids: Iterable[str]) -> int:
        """Flag messages as absorbed into a summary node (they remain readable)."""
        ids = [mid for mid in message_ids if mid]
        if not ids:
            return 0
        with self._cursor() as conn:
            updated = 0
            for chunk_start in range(0, len(ids), 200):
                chunk = ids[chunk_start:chunk_start + 200]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"UPDATE messages SET compacted = 1 WHERE session_id = ? "
                    f"AND message_id IN ({placeholders})",
                    [session_id, *chunk],
                )
                updated += cur.rowcount or 0
            conn.commit()
            return updated

    def page_window(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return one bounded page of raw messages ordered by ``seq``."""
        size = self._clamp_page(page_size)
        with self._cursor() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND seq > ? "
                "ORDER BY seq LIMIT ?",
                (session_id, int(after_seq), size + 1),
            ).fetchall()
        has_more = len(rows) > size
        rows = rows[:size]
        records = [_message_record(row) for row in rows]
        next_cursor = records[-1]["seq"] if (records and has_more) else None
        return {
            "items": records,
            "returned": len(records),
            "total": total,
            "page_size": size,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }

    def page_node_sources(
        self, node_id: str, *, page: int = 0, page_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Return one bounded page of a node's lineage edges."""
        size = self._clamp_page(page_size)
        offset = max(0, int(page)) * size
        with self._cursor() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM node_sources WHERE node_id = ?",
                    (node_id,),
                ).fetchone()["c"]
            )
            rows = conn.execute(
                "SELECT source_kind, source_id, ordinal FROM node_sources "
                "WHERE node_id = ? ORDER BY ordinal LIMIT ? OFFSET ?",
                (node_id, size, offset),
            ).fetchall()
        items = [
            {"kind": row["source_kind"], "id": row["source_id"], "ordinal": row["ordinal"]}
            for row in rows
        ]
        consumed = offset + len(items)
        return {
            "items": items,
            "returned": len(items),
            "total": total,
            "page": max(0, int(page)),
            "page_size": size,
            "has_more": consumed < total,
        }

    def _clamp_page(self, page_size: Optional[int]) -> int:
        if page_size is None:
            return self.page_size
        try:
            parsed = int(page_size)
        except (TypeError, ValueError):
            return self.page_size
        return max(1, min(parsed, self.max_page_size))

    # -- summary nodes -----------------------------------------------------

    def node_count(self, session_id: str) -> int:
        with self._cursor() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM nodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )

    def resolve_node_id(
        self, session_id: str, level: int, sources: List[Tuple[str, str]]
    ) -> str:
        """Derive the id ``create_node`` will assign for these inputs."""
        return _node_id(session_id, int(level), self.node_count(session_id), sources)

    def create_node(
        self,
        session_id: str,
        *,
        level: int,
        summary: str,
        sources: List[Tuple[str, str]],
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a summary node plus its lineage edges; return its record.

        Pass ``node_id`` (from ``resolve_node_id``) when the summary text needs
        to cite its own node id — the id is derived from the sources, so it is
        knowable before the text is rendered.
        """
        with self._cursor() as conn:
            ordinal = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM nodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )
            if node_id is None:
                node_id = _node_id(session_id, int(level), ordinal, sources)
            now = time.time()
            conn.execute(
                "INSERT OR REPLACE INTO nodes "
                "(node_id, session_id, level, summary, source_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, session_id, int(level), summary, len(sources), now),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO node_sources "
                "(node_id, source_kind, source_id, ordinal) VALUES (?, ?, ?, ?)",
                [
                    (node_id, kind, source_id, position)
                    for position, (kind, source_id) in enumerate(sources)
                ],
            )
            conn.commit()
        return {
            "node_id": node_id,
            "session_id": session_id,
            "level": int(level),
            "summary": summary,
            "source_count": len(sources),
            "created_at": now,
        }

    def node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "node_id": row["node_id"],
            "session_id": row["session_id"],
            "level": int(row["level"]),
            "summary": row["summary"],
            "source_count": int(row["source_count"]),
            "created_at": float(row["created_at"]),
        }

    def node_sources(self, node_id: str) -> List[Tuple[str, str]]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT source_kind, source_id FROM node_sources "
                "WHERE node_id = ? ORDER BY ordinal",
                (node_id,),
            ).fetchall()
        return [(row["source_kind"], row["source_id"]) for row in rows]

    # -- search ------------------------------------------------------------

    def search(
        self, session_id: str, query: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Bounded full-text search over this session's raw messages."""
        size = self._clamp_limit(limit)
        text = _fts_query(query)
        if not text:
            return []
        if self.fts_available():
            with self._cursor() as conn:
                try:
                    rows = conn.execute(
                        "SELECT message_id, body FROM messages_fts "
                        "WHERE messages_fts MATCH ? AND session_id = ? "
                        "ORDER BY rank LIMIT ?",
                        (text, session_id, size),
                    ).fetchall()
                    return [
                        {
                            "message_id": row["message_id"],
                            "body": row["body"][: self.body_chars],
                            "truncated": len(row["body"]) > self.body_chars,
                            "match": "fts",
                        }
                        for row in rows
                    ]
                except sqlite3.Error:
                    pass  # malformed MATCH -> fall through to LIKE
        return self._like_search(session_id, query, size)

    def _like_search(
        self, session_id: str, query: str, size: int
    ) -> List[Dict[str, Any]]:
        needle = f"%{str(query).strip()}%"
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT message_id, body FROM messages "
                "WHERE session_id = ? AND body LIKE ? ORDER BY seq LIMIT ?",
                (session_id, needle, size),
            ).fetchall()
        return [
            {
                "message_id": row["message_id"],
                "body": row["body"][: self.body_chars],
                "truncated": len(row["body"]) > self.body_chars,
                "match": "like",
            }
            for row in rows
        ]

    def _clamp_limit(self, limit: Optional[int]) -> int:
        if limit is None:
            return self.max_search_results
        try:
            parsed = int(limit)
        except (TypeError, ValueError):
            return self.max_search_results
        return max(1, min(parsed, self.max_search_results))

    # -- stats -------------------------------------------------------------

    def stats(self, session_id: str) -> Dict[str, Any]:
        with self._cursor() as conn:
            messages = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )
            compacted = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE session_id = ? "
                    "AND compacted = 1",
                    (session_id,),
                ).fetchone()["c"]
            )
            nodes = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM nodes WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )
            edges = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM node_sources ns JOIN nodes n "
                    "ON n.node_id = ns.node_id WHERE n.session_id = ?",
                    (session_id,),
                ).fetchone()["c"]
            )
        return {
            "messages": messages,
            "compacted_messages": compacted,
            "nodes": nodes,
            "lineage_edges": edges,
            "schema_version": self.version(),
            "fts": self.fts_available(),
        }


def _message_record(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "seq": int(row["seq"]),
        "role": row["role"],
        "body": row["body"],
        "compacted": bool(row["compacted"]),
        "created_at": float(row["created_at"]),
    }


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


def default_db_path(hermes_home: Path) -> Path:
    return Path(hermes_home) / "lcm" / "lcm.db"


def backup_db(db_path: Path, dest: Optional[Path] = None) -> Path:
    """Write a consistent snapshot of *db_path*; return the snapshot path.

    Uses the SQLite online backup API so a live WAL database is copied
    consistently rather than by file-level ``cp`` (which can capture a torn
    ``-wal``).  The backup is opened read-only and verified to carry a schema
    version before being accepted.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"no LCM database at {db_path}")
    if dest is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dest = db_path.with_name(f"{db_path.stem}.backup-{stamp}.db")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    target = sqlite3.connect(str(dest))
    try:
        with target:
            source.backup(target)
    finally:
        target.close()
        source.close()

    check = sqlite3.connect(str(dest))
    try:
        if schema_version(check) != SCHEMA_VERSION:
            dest.unlink(missing_ok=True)
            raise RuntimeError("backup verification failed: schema version mismatch")
    finally:
        check.close()
    return dest
