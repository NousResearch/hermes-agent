"""Bounded evidence from exact host-approved sessions on a read-only SessionDB."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

from hermes_state import SessionDB

from .runner import Evidence, ToolBudgetExceeded, _positive_int


class ScopedSessionEvidenceReader:
    def __init__(self, *, db_path: Path, owner_key: str, allowed_session_ids: frozenset[str],
                 max_sessions: int = 20, max_hits: int = 10, page_size: int = 8,
                 max_rows: int = 64, max_bytes: int = 65536, max_field_bytes: int = 2048):
        if not isinstance(db_path, Path) or not db_path.is_absolute() or db_path.name != "state.db":
            raise ValueError("host must bind an absolute state.db path")
        if not owner_key or not isinstance(allowed_session_ids, frozenset) or not allowed_session_ids:
            raise ValueError("host-vetted owner and session scope required")
        if not all(_positive_int(v) for v in (max_sessions, max_hits, page_size, max_rows,
                                              max_bytes, max_field_bytes)):
            raise ValueError("finite positive read bounds required")
        if len(allowed_session_ids) > max_sessions or page_size > max_rows or max_field_bytes > max_bytes:
            raise ValueError("session scope exceeds bound")
        self.owner_key, self.allowed_session_ids = owner_key, allowed_session_ids
        self.max_hits, self.page_size = max_hits, page_size
        self.max_rows, self.max_bytes, self.max_field_bytes = max_rows, max_bytes, max_field_bytes
        self._db = SessionDB(db_path=db_path, read_only=True)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _envelope(ref: str, payload: dict, status: str, coverage: str, reads: int,
                  source_digest: str | None = None) -> Evidence:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return Evidence(
            source_kind="session", exact_reference=ref,
            digest="sha256:" + (source_digest or hashlib.sha256(body.encode("utf-8")).hexdigest()),
            location=ref, read_at=datetime.now(timezone.utc).isoformat(),
            status=status, coverage=coverage, content=payload, underlying_reads=max(1, reads),
        )

    def _scan(self, session_id: str, reserve_read: Callable[[], None],
              remaining_rows: int, remaining_bytes: int):
        """SQL slices content before Python materializes each bounded page."""
        reads = 0

        def query(sql, params):
            nonlocal reads
            reserve_read()
            reads += 1
            return self._db._read_all(sql, params)

        meta = query("SELECT id, substr(source, 1, 128) AS source, started_at "
                     "FROM sessions WHERE id = ? LIMIT 1", (session_id,))
        if not meta:
            return [], None, "missing", reads, remaining_rows, remaining_bytes
        rows, after_id, clipped = [], 0, False
        while remaining_rows > 0 and remaining_bytes > 0:
            page_limit = min(self.page_size, remaining_rows,
                             max(1, remaining_bytes // self.max_field_bytes))
            field_limit = min(self.max_field_bytes, remaining_bytes // page_limit)
            if field_limit < 1:
                break
            page = query(
                "SELECT id, substr(role, 1, 64) AS role, "
                "substr(CAST(content AS BLOB), 1, ?) AS content_prefix, "
                "length(CAST(content AS BLOB)) AS content_bytes, timestamp "
                "FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) "
                "AND id > ? ORDER BY id LIMIT ?",
                (field_limit, session_id, after_id, page_limit),
            )
            if not page:
                return rows, dict(meta[0]), "truncated" if clipped else "complete", reads, remaining_rows, remaining_bytes
            for row in page:
                prefix = row["content_prefix"] or b""
                if isinstance(prefix, str):
                    prefix = prefix.encode("utf-8")
                # A SQL byte slice can end inside a code point. Drop that tail so
                # exposed UTF-8 never grows past the requested byte allowance.
                content = prefix.decode("utf-8", errors="ignore")
                returned_bytes = len(content.encode("utf-8"))
                content_clipped = (row["content_bytes"] or 0) > returned_bytes
                clipped |= content_clipped
                shaped = {"id": row["id"], "role": row["role"], "content": content,
                          "timestamp": row["timestamp"]}
                if content_clipped:
                    shaped.update(content_truncated=True, original_content_bytes=row["content_bytes"])
                rows.append(shaped)
                after_id = row["id"]
                remaining_rows -= 1
                remaining_bytes -= returned_bytes
            if len(page) < page_limit:
                return rows, dict(meta[0]), "truncated" if clipped else "complete", reads, remaining_rows, remaining_bytes
        more = query("SELECT id FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1) "
                     "AND id > ? ORDER BY id LIMIT 1", (session_id, after_id))
        status = "truncated" if clipped or more else "complete"
        return rows, dict(meta[0]), status, reads, remaining_rows, remaining_bytes

    def read(self, reference: str, reserve_read: Callable[[], None] = lambda: None) -> Evidence:
        if not isinstance(reference, str) or not reference.startswith("session:"):
            raise ValueError("session reference required")
        session_id = reference.removeprefix("session:")
        if session_id not in self.allowed_session_ids:
            raise PermissionError("session outside host-bound scope")
        measured_reads = 0

        def metered():
            nonlocal measured_reads
            reserve_read()
            measured_reads += 1

        try:
            rows, meta, status, reads, _, _ = self._scan(
                session_id, metered, self.max_rows, self.max_bytes)
        except ToolBudgetExceeded:
            raise
        except Exception as exc:
            return self._envelope(reference, {"error_type": type(exc).__name__}, "error",
                                  "approved session read failed", measured_reads)
        payload = {"session_id": session_id, "message_count": len(rows), "messages": rows,
                   "session_meta": meta, "truncated": status == "truncated"}
        digest = hashlib.sha256(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False,
                                           sort_keys=True).encode("utf-8")).hexdigest()
        return self._envelope(reference, payload, status,
                              "approved session only; active and compacted rows by id; excludes rewind; "
                              f"{len(rows)} rows and bounded content prefixes covered",
                              reads, digest)

    def search(self, query: str, reserve_read: Callable[[], None] = lambda: None) -> Evidence:
        if not isinstance(query, str) or not query.strip() or len(query) > 1024:
            raise ValueError("bounded search query required")
        hits, statuses = [], []
        source_hasher = hashlib.sha256()
        reads, scanned = 0, 0
        measured_reads = 0

        def metered():
            nonlocal measured_reads
            reserve_read()
            measured_reads += 1

        remaining_rows, remaining_bytes = self.max_rows, self.max_bytes
        for session_id in sorted(self.allowed_session_ids):
            try:
                rows, meta, status, count, remaining_rows, remaining_bytes = self._scan(
                    session_id, metered, remaining_rows, remaining_bytes)
            except ToolBudgetExceeded:
                raise
            except Exception as exc:
                return self._envelope("session-query:" + query,
                                      {"hits": hits, "error_type": type(exc).__name__}, "error",
                                      f"failed after {scanned} approved sessions", measured_reads)
            reads += count
            scanned += 1
            statuses.append(status)
            source_hasher.update(json.dumps({"session_id": session_id, "meta": meta, "rows": rows},
                                            ensure_ascii=False, sort_keys=True).encode("utf-8"))
            for row in rows:
                if query.casefold() in row["content"].casefold():
                    hits.append({"session_id": session_id, "message_id": row["id"],
                                 "role": row["role"], "snippet": row["content"][:1000]})
                    if len(hits) >= self.max_hits:
                        statuses.append("truncated")
                        break
            if len(hits) >= self.max_hits or remaining_rows == 0 or remaining_bytes == 0:
                break
        status = "missing" if "missing" in statuses else (
            "truncated" if "truncated" in statuses or scanned < len(self.allowed_session_ids) else "complete")
        return self._envelope("session-query:" + query, {"hits": hits, "sessions_scanned": scanned},
                              status, f"{scanned} approved sessions; bounded active and compacted rows; "
                              "excludes rewind; content prefixes only when truncated",
                              reads, source_hasher.hexdigest())
