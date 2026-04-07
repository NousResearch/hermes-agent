#!/usr/bin/env python3
"""
Session extractor for the finetune pipeline.

Reads from ~/.hermes/state.db, normalizes sessions to JSONL, and supports
incremental extraction and external imports.

Usage:
    python extract.py [--min-turns N] [--exclude-sources cron,gateway] [--since YYYY-MM-DD] [--full]
"""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    EXTRACTED_DIR, EXTRACT_STATE_PATH, IMPORTED_DIR, STATE_DB_PATH,
    ensure_dirs, load_config, load_json, save_json, read_jsonl, append_jsonl,
    logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SessionExtractor:
    """Extract and normalize sessions from the Hermes state DB."""

    def __init__(self, db_path: Path = None, config: dict = None):
        self.db_path = db_path or STATE_DB_PATH
        self.config = config or load_config().get("extract", {})
        self.min_turns = self.config.get("min_turns", 2)
        self.exclude_sources = self.config.get("exclude_sources", [])

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only connection to state.db."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"State DB not found: {self.db_path}")
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_last_extraction_time(self) -> float:
        """Get timestamp of last extraction for incremental mode."""
        state = load_json(EXTRACT_STATE_PATH, {})
        return state.get("last_extraction_time", 0.0)

    def _save_extraction_state(self, timestamp: float, count: int):
        """Persist extraction state for incremental runs."""
        state = load_json(EXTRACT_STATE_PATH, {})
        state["last_extraction_time"] = timestamp
        state["last_extraction_count"] = count
        state["last_extraction_date"] = datetime.fromtimestamp(timestamp).isoformat()
        state["total_extracted"] = state.get("total_extracted", 0) + count
        save_json(EXTRACT_STATE_PATH, state)

    def _query_sessions(
        self, conn: sqlite3.Connection,
        since: float = 0.0,
        date_after: str = None,
    ) -> List[Dict[str, Any]]:
        """Query sessions matching filters."""
        where = ["s.message_count >= ?"]
        params: list = [self.min_turns]

        if self.exclude_sources:
            placeholders = ",".join("?" for _ in self.exclude_sources)
            where.append(f"s.source NOT IN ({placeholders})")
            params.extend(self.exclude_sources)

        if since > 0:
            where.append("s.started_at > ?")
            params.append(since)

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
        """Get all messages for a session in conversation order."""
        cursor = conn.execute(
            """SELECT role, content, tool_call_id, tool_calls, tool_name
               FROM messages
               WHERE session_id = ?
               ORDER BY timestamp, id""",
            (session_id,),
        )
        messages = []
        for row in cursor.fetchall():
            msg = {"role": row["role"], "content": row["content"]}
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

    def _reconstruct_lineage(
        self, conn: sqlite3.Connection, sessions: List[Dict],
    ) -> Dict[str, List[str]]:
        """Map parent sessions to their children for lineage reconstruction."""
        lineage: Dict[str, List[str]] = {}
        for s in sessions:
            parent = s.get("parent_session_id")
            if parent:
                lineage.setdefault(parent, []).append(s["id"])
        return lineage

    def _normalize_session(
        self, conn: sqlite3.Connection, session: Dict,
        child_ids: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a session into the extraction format."""
        messages = self._get_messages(conn, session["id"])

        # Reconstruct compression-split conversations
        if child_ids:
            for child_id in sorted(child_ids):
                child_msgs = self._get_messages(conn, child_id)
                messages.extend(child_msgs)

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

        started_at = datetime.fromtimestamp(session["started_at"]).isoformat()

        return {
            "session_id": session["id"],
            "started_at": started_at,
            "turns": turns,
            "metadata": {
                "source": session.get("source", "unknown"),
                "model": session.get("model", ""),
                "parent_session_id": session.get("parent_session_id"),
                "tool_call_count": session.get("tool_call_count", 0),
                "total_tokens": (session.get("input_tokens", 0) or 0)
                    + (session.get("output_tokens", 0) or 0),
            },
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
            last_time = 0.0 if full else self._get_last_extraction_time()
            sessions = self._query_sessions(conn, since=last_time, date_after=since)

            if not sessions:
                logger.info("No new sessions to extract.")
                return []

            # Build lineage map for compression-split reconstruction
            lineage = self._reconstruct_lineage(conn, sessions)

            # Skip child sessions (they're merged into parents)
            child_set = set()
            for children in lineage.values():
                child_set.update(children)

            extracted = []
            for session in sessions:
                if session["id"] in child_set:
                    continue
                children = lineage.get(session["id"], [])
                normalized = self._normalize_session(conn, session, children)
                if normalized:
                    extracted.append(normalized)

            if extracted:
                # Write to timestamped file
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = EXTRACTED_DIR / f"extract_{ts}.jsonl"
                append_jsonl(output_path, extracted)

                # Update extraction state
                max_time = max(s["started_at"] for s in sessions)
                self._save_extraction_state(max_time, len(extracted))

                logger.info(
                    "Extracted %d sessions to %s", len(extracted), output_path
                )
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

        imported = []
        for path in sorted(import_dir.glob("*.jsonl")):
            records = read_jsonl(path)
            imported.extend(records)
            # Move processed file
            done_dir = import_dir / "processed"
            done_dir.mkdir(exist_ok=True)
            path.rename(done_dir / path.name)
            logger.info("Imported %d sessions from %s", len(records), path.name)

        return imported

    def get_all_extracted(self) -> List[Dict[str, Any]]:
        """Load all previously extracted sessions."""
        all_sessions = []
        for path in sorted(EXTRACTED_DIR.glob("extract_*.jsonl")):
            all_sessions.extend(read_jsonl(path))
        return all_sessions


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
