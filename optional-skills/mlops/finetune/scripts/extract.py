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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    EXTRACTED_DIR, EXTRACT_STATE_PATH, IMPORTED_DIR, STATE_DB_PATH,
    ensure_dirs, load_config, load_json, save_json, read_jsonl, append_jsonl,
    load_records_dedup, logger,
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

    def _get_session_message_counts(self) -> Dict[str, int]:
        """Per-session message counts recorded at last extraction.

        Used instead of a pure started_at watermark so sessions that gain
        messages after being extracted (still in progress at extraction time,
        or later compression-split children) are re-extracted.
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

        Note: min_turns is applied to the *merged* (lineage-reconstructed)
        record in extract(), not here, so small child sessions are still
        visible and can trigger re-extraction of their root.
        """
        where = ["s.message_count >= 1"]
        params: list = []

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

    def _find_root(self, conn: sqlite3.Connection, session: Dict) -> Dict:
        """Walk parent_session_id links up to the root ancestor.

        If a parent row is missing from the DB, the highest reachable
        ancestor is treated as the root (nothing is dropped).
        """
        current = session
        seen = {session["id"]}
        while current.get("parent_session_id"):
            parent = self._fetch_session_row(conn, current["parent_session_id"])
            if parent is None or parent["id"] in seen:
                break
            seen.add(parent["id"])
            current = parent
        return current

    def _get_descendants(self, conn: sqlite3.Connection, root_id: str) -> List[Dict]:
        """All descendant session rows of root_id (children, grandchildren, ...),
        sorted chronologically by started_at."""
        descendants: List[Dict] = []
        queue = [root_id]
        seen = {root_id}
        while queue:
            current = queue.pop(0)
            cursor = conn.execute(
                f"SELECT {self._SESSION_COLUMNS} FROM sessions "
                "WHERE parent_session_id = ?",
                (current,),
            )
            for row in cursor.fetchall():
                child = dict(row)
                if child["id"] in seen:
                    continue  # cycle guard
                seen.add(child["id"])
                descendants.append(child)
                queue.append(child["id"])
        descendants.sort(key=lambda d: (d.get("started_at") or 0, d["id"]))
        return descendants

    def _normalize_session(
        self, conn: sqlite3.Connection, session: Dict,
        descendants: List[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a session (plus its compression-split descendants,
        merged in chronological started_at order) into the extraction format."""
        messages = self._get_messages(conn, session["id"])

        # Reconstruct compression-split conversations
        descendants = sorted(
            descendants or [],
            key=lambda d: (d.get("started_at") or 0, d.get("id") or ""),
        )
        for child in descendants:
            messages.extend(self._get_messages(conn, child["id"]))

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

        lineage_rows = [session] + descendants
        metadata = {
            "source": session.get("source", "unknown"),
            "model": session.get("model", ""),
            "parent_session_id": session.get("parent_session_id"),
            "tool_call_count": sum(
                (r.get("tool_call_count", 0) or 0) for r in lineage_rows
            ),
            "total_tokens": sum(
                (r.get("input_tokens", 0) or 0) + (r.get("output_tokens", 0) or 0)
                for r in lineage_rows
            ),
        }
        if descendants:
            metadata["merged_session_ids"] = [d["id"] for d in descendants]

        return {
            "session_id": session["id"],
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
            # extraction time, or a new compression-split child appeared).
            changed = [
                s for s in sessions
                if prev_counts.get(s["id"]) != s.get("message_count")
            ]
            if not changed:
                logger.info("No new sessions to extract.")
                return []

            # Resolve each changed session to its root ancestor so
            # compression-split lineages are re-emitted as one merged record —
            # even when the parent itself is not in this incremental batch.
            roots: Dict[str, Dict] = {}
            for session in changed:
                root = self._find_root(conn, session)
                roots.setdefault(root["id"], root)

            extracted = []
            new_counts: Dict[str, int] = {
                s["id"]: s.get("message_count", 0) for s in sessions
            }
            for root in roots.values():
                descendants = self._get_descendants(conn, root["id"])
                normalized = self._normalize_session(conn, root, descendants)
                # min_turns applies to the merged conversation
                if normalized and len(normalized["turns"]) >= self.min_turns:
                    extracted.append(normalized)
                # Record counts for all lineage members so unchanged lineages
                # don't re-trigger next run.
                for row in [root] + descendants:
                    new_counts[row["id"]] = row.get("message_count", 0)

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
