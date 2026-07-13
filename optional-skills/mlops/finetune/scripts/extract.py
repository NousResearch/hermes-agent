#!/usr/bin/env python3
"""
Session extractor for the finetune pipeline.

Reads from ~/.hermes/state.db, normalizes sessions to JSONL, and supports
incremental extraction and external imports.

Each session is extracted standalone (a compression child's leading summary
makes it a coherent conversation in its own right). Lineage is kept only as
a `root_session_id` field so downstream train/eval splitting can keep a
session and its continuations on the same side of the split.

Usage:
    python extract.py [--min-turns N] [--exclude-sources cron,gateway] [--since YYYY-MM-DD] [--full]
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    EXTRACTED_DIR, EXTRACT_STATE_PATH, IMPORTED_DIR, STATE_DB_PATH,
    ensure_dirs, load_config, load_json, save_json, read_jsonl, append_jsonl,
    load_records_dedup, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Mirrors SessionDB._CONTENT_JSON_PREFIX (hermes_state.py). Structured
# (list/dict) message content is stored as this sentinel + JSON; these
# scripts run standalone and can't import core, so the decode is replicated
# here — same pattern as common._default_hermes_home.
_CONTENT_JSON_PREFIX = "\x00json:"

# Mirrors hermes_state._delegate_from_json: delegate subagent sessions carry
# a `_delegate_from` marker in model_config (set at creation; backfilled for
# old rows by the core v16 migration).
_DELEGATE_MARKER_SQL = (
    "json_extract(COALESCE(s.model_config, '{}'), '$._delegate_from')"
)


def _decode_content(content: Any) -> Any:
    """Reverse SessionDB._encode_content; returns scalars unchanged.

    Decoding multipart content back to a parts list lets downstream
    flattening (common.content_to_text) drop non-text parts — stored as the
    raw sentinel string, base64 image blobs would flow into training data.
    """
    if isinstance(content, str) and content.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(content[len(_CONTENT_JSON_PREFIX):])
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to decode JSON-encoded message content; keeping raw string"
            )
    return content


class SessionExtractor:
    """Extract and normalize sessions from the Hermes state DB."""

    def __init__(self, db_path: Path = None, config: dict = None):
        self.db_path = db_path or STATE_DB_PATH
        self.config = config or load_config().get("extract", {})
        self.min_turns = self.config.get("min_turns", 2)
        self.exclude_sources = self.config.get("exclude_sources", [])
        # Delegate subagent sessions are agent-to-agent traffic, not user
        # conversations — excluded from training data by default.
        self.include_delegates = bool(self.config.get("include_delegates", False))
        self._column_cache: Dict[str, set] = {}

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only connection to state.db."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"State DB not found: {self.db_path}")
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        """Whether `table` has `column`.

        The `messages.active` and `sessions.archived` columns are added by
        core migrations; a DB that predates them cannot contain soft-deleted
        or archived rows, so skipping those filters on such a DB is exact,
        not lossy.
        """
        cols = self._column_cache.get(table)
        if cols is None:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            self._column_cache[table] = cols
        return column in cols

    def _get_session_message_counts(self) -> Dict[str, int]:
        """Per-session message counts recorded at last extraction.

        Used instead of a pure started_at watermark so sessions that gain
        messages after being extracted (still in progress at extraction time)
        are re-extracted.
        """
        state = load_json(EXTRACT_STATE_PATH, {})
        return dict(state.get("session_message_counts", {}))

    def _save_extraction_state(
        self, timestamp: float, count: int,
        session_message_counts: Dict[str, int] = None,
    ):
        """Persist extraction state for incremental runs."""
        state = load_json(EXTRACT_STATE_PATH, {})
        state["last_extraction_time"] = timestamp
        state["last_extraction_count"] = count
        state["last_extraction_date"] = datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).isoformat()
        state["total_extracted"] = state.get("total_extracted", 0) + count
        if session_message_counts:
            counts = state.get("session_message_counts", {})
            counts.update(session_message_counts)
            state["session_message_counts"] = counts
        save_json(EXTRACT_STATE_PATH, state)

    def _query_sessions(
        self, conn: sqlite3.Connection,
        date_after: str = None,
    ) -> List[Dict[str, Any]]:
        """Query sessions matching filters.

        Archived sessions and (by default) delegate subagent sessions are
        excluded. min_turns is applied to the normalized record in extract(),
        not here, because message_count can include rows that the active
        filter drops.
        """
        where = ["s.message_count >= 1"]
        params: list = []

        if not self.include_delegates:
            where.append(f"{_DELEGATE_MARKER_SQL} IS NULL")

        if self._has_column(conn, "sessions", "archived"):
            where.append("s.archived = 0")

        if self.exclude_sources:
            placeholders = ",".join("?" for _ in self.exclude_sources)
            where.append(f"s.source NOT IN ({placeholders})")
            params.extend(self.exclude_sources)

        if date_after:
            ts = datetime.fromisoformat(date_after).timestamp()
            where.append("s.started_at > ?")
            params.append(ts)

        where_sql = " AND ".join(where)
        query = f"""
            SELECT s.id, s.source, s.model, s.parent_session_id,
                   s.started_at, s.ended_at, s.message_count,
                   s.tool_call_count, s.input_tokens, s.output_tokens
            FROM sessions s
            WHERE {where_sql}
            ORDER BY s.started_at ASC
        """
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def _get_messages(self, conn: sqlite3.Connection, session_id: str) -> List[Dict[str, Any]]:
        """Get a session's active messages in insertion order.

        Mirrors SessionDB.get_messages: filters `active = 1` (rewound /
        retracted turns and pre-compaction originals are soft-deleted with
        active=0) and orders by AUTOINCREMENT id, not timestamp — core
        deliberately orders by id because wall-clock regressions (e.g. WSL2)
        can reorder timestamps.
        """
        active_clause = (
            " AND active = 1"
            if self._has_column(conn, "messages", "active") else ""
        )
        cursor = conn.execute(
            f"""SELECT role, content, tool_call_id, tool_calls, tool_name
               FROM messages
               WHERE session_id = ?{active_clause}
               ORDER BY id""",
            (session_id,),
        )
        messages = []
        for row in cursor.fetchall():
            msg = {"role": row["role"], "content": _decode_content(row["content"])}
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            messages.append(msg)
        return messages

    _SESSION_COLUMNS = (
        "id, source, model, parent_session_id, started_at, ended_at, "
        "message_count, tool_call_count, input_tokens, output_tokens"
    )

    def _fetch_session_row(self, conn: sqlite3.Connection, session_id: str) -> Optional[Dict]:
        """Fetch a single session row by ID (no filters)."""
        cursor = conn.execute(
            f"SELECT {self._SESSION_COLUMNS} FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def _find_root_id(self, conn: sqlite3.Connection, session: Dict) -> str:
        """Walk parent_session_id links up to the root ancestor's id.

        Used only to stamp `root_session_id` on the record (so format.py can
        key the train/eval split on the whole lineage). If a parent row is
        missing from the DB, the highest reachable ancestor is the root.
        """
        current = session
        seen = {session["id"]}
        while current.get("parent_session_id"):
            parent = self._fetch_session_row(conn, current["parent_session_id"])
            if parent is None or parent["id"] in seen:
                break
            seen.add(parent["id"])
            current = parent
        return current["id"]

    def _normalize_session(
        self, conn: sqlite3.Connection, session: Dict,
        root_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a single session into the extraction format.

        Sessions are extracted standalone — never concatenated with parents
        or children. A compression child re-flushes its summary plus verbatim
        copies of retained parent turns at creation, so concatenating a
        lineage would duplicate content; each session row is already a
        coherent conversation on its own.
        """
        messages = self._get_messages(conn, session["id"])
        if not messages:
            return None

        # Build turns list
        turns = []
        for msg in messages:
            turn = {"role": msg["role"], "content": msg.get("content", "")}
            if msg.get("tool_calls"):
                turn["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                turn["tool_call_id"] = msg["tool_call_id"]
            if msg.get("tool_name"):
                turn["tool_name"] = msg["tool_name"]
            turns.append(turn)

        # Timezone-aware UTC ISO timestamp (retro assumes UTC downstream)
        started_at = datetime.fromtimestamp(
            session["started_at"], tz=timezone.utc
        ).isoformat()

        metadata = {
            "source": session.get("source", "unknown"),
            "model": session.get("model", ""),
            "parent_session_id": session.get("parent_session_id"),
            "tool_call_count": session.get("tool_call_count", 0) or 0,
            "total_tokens": (
                (session.get("input_tokens", 0) or 0)
                + (session.get("output_tokens", 0) or 0)
            ),
        }

        return {
            "session_id": session["id"],
            "root_session_id": root_id or session["id"],
            "started_at": started_at,
            "turns": turns,
            "metadata": metadata,
        }

    def extract(
        self, full: bool = False, since: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract sessions from state.db.

        Args:
            full: If True, extract all sessions (ignore incremental state).
            since: Only extract sessions after this ISO date.

        Returns:
            List of normalized session dicts.
        """
        ensure_dirs()
        conn = self._connect()

        try:
            prev_counts = {} if full else self._get_session_message_counts()
            sessions = self._query_sessions(conn, date_after=since)

            if not sessions:
                logger.info("No new sessions to extract.")
                return []

            # A session needs (re-)extraction when it's new OR its message
            # count changed since the last run (session still in progress at
            # extraction time).
            changed = [
                s for s in sessions
                if prev_counts.get(s["id"]) != s.get("message_count")
            ]
            if not changed:
                logger.info("No new sessions to extract.")
                return []

            extracted = []
            # Record counts for every visible session (not just changed ones)
            # so unchanged sessions don't re-trigger next run.
            new_counts: Dict[str, int] = {
                s["id"]: s.get("message_count", 0) for s in sessions
            }
            for session in changed:
                root_id = self._find_root_id(conn, session)
                normalized = self._normalize_session(conn, session, root_id=root_id)
                if normalized and len(normalized["turns"]) >= self.min_turns:
                    extracted.append(normalized)

            if extracted:
                # Write to timestamped file
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = EXTRACTED_DIR / f"extract_{ts}.jsonl"
                append_jsonl(output_path, extracted)

                logger.info(
                    "Extracted %d sessions to %s", len(extracted), output_path
                )

            # Update extraction state (even when nothing met min_turns, so
            # trivial sessions aren't rescanned as "changed" forever).
            max_time = max(s["started_at"] for s in sessions)
            self._save_extraction_state(max_time, len(extracted), new_counts)
            return extracted

        finally:
            conn.close()

    def import_external(self, import_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Import sessions from external sources (Claude exports, other OpenAI-format logs).

        External files should be JSONL in the normalized session format, placed in
        ~/.hermes/finetune/data/imported/
        """
        import_dir = import_dir or IMPORTED_DIR
        if not import_dir.exists():
            return []

        ensure_dirs()
        imported = []
        done_dir = import_dir / "processed"
        for path in sorted(import_dir.glob("*.jsonl")):
            records = read_jsonl(path)
            if not records:
                # Nothing parsed — leave the file in place so the failure is
                # visible and the data isn't silently lost.
                logger.error(
                    "Import failed for %s: no records parsed — leaving file "
                    "in place (fix the format and re-run).", path.name,
                )
                continue
            imported.extend(records)
            done_dir.mkdir(exist_ok=True)
            path.rename(done_dir / path.name)
            logger.info("Imported %d sessions from %s", len(records), path.name)

        if imported:
            # Persist imported records as an extraction batch so downstream
            # scoring/clustering picks them up like normal extractions.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = EXTRACTED_DIR / f"extract_{ts}_imported.jsonl"
            append_jsonl(output_path, imported)
            logger.info(
                "Wrote %d imported sessions to %s", len(imported), output_path
            )

        return imported

    def get_all_extracted(self) -> List[Dict[str, Any]]:
        """Load all previously extracted sessions, deduped by session_id
        (re-extracted sessions keep the newest copy)."""
        return load_records_dedup(EXTRACTED_DIR, "extract_*.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Extract sessions from Hermes state DB")
    parser.add_argument("--min-turns", type=int, default=None,
                        help="Minimum turn count (default: from config)")
    parser.add_argument("--exclude-sources", type=str, default=None,
                        help="Comma-separated sources to exclude")
    parser.add_argument("--since", type=str, default=None,
                        help="Only extract sessions after this date (YYYY-MM-DD)")
    parser.add_argument("--full", action="store_true",
                        help="Full extraction (ignore incremental state)")
    parser.add_argument("--import-external", action="store_true",
                        help="Also import from ~/.hermes/finetune/data/imported/")
    args = parser.parse_args()

    if args.since:
        try:
            datetime.fromisoformat(args.since)
        except ValueError:
            parser.error(
                f"--since must be an ISO date like 2026-01-31, got {args.since!r}"
            )

    config = load_config().get("extract", {})
    if args.min_turns is not None:
        config["min_turns"] = args.min_turns
    if args.exclude_sources:
        config["exclude_sources"] = args.exclude_sources.split(",")

    extractor = SessionExtractor(config=config)
    sessions = extractor.extract(full=args.full, since=args.since)

    if args.import_external:
        imported = extractor.import_external()
        sessions.extend(imported)

    print(f"Total extracted: {len(sessions)} sessions")


if __name__ == "__main__":
    main()
