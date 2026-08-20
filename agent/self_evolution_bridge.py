"""Bridge Hermes state.db sessions into the legacy JSON format expected by
hermes-agent-self-evolution's HermesSessionImporter.

The companion repo currently reads ``~/.hermes/sessions/*.json`` while Hermes
Agent stores live conversations in ``state.db``. This module exports selected
sessions from SQLite into one-file-per-session JSON payloads so the external
optimizer can mine real Hermes history without changing the runtime storage
format.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from hermes_constants import get_hermes_home

VALID_EXPORT_ROLES = {"system", "user", "assistant", "tool"}
_DEFAULT_DB_PATH = get_hermes_home() / "state.db"
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WRAPPED_SYSTEM_USER_PATTERNS = (
    "[SYSTEM:",
    "[The user sent an image",
    "[The user sent a voice",
    "[The user sent a video",
    "[The user sent a file",
)


@dataclass
class ExportSummary:
    db_path: str
    output_dir: str
    sessions_exported: int
    messages_exported: int
    sessions_scanned: int
    sessions_skipped: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _safe_session_filename(session_id: str) -> str:
    safe = _FILENAME_SAFE_RE.sub("-", session_id).strip("-.")
    return safe or "session"


def _should_skip_user_message(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in _WRAPPED_SYSTEM_USER_PATTERNS)


def _build_sessions_query(sources: Sequence[str] | None, limit_sessions: int | None) -> tuple[str, list[object]]:
    query = ["SELECT id, source FROM sessions"]
    params: list[object] = []
    if sources:
        placeholders = ", ".join("?" for _ in sources)
        query.append(f"WHERE source IN ({placeholders})")
        params.extend(sources)
    query.append("ORDER BY started_at DESC")
    if limit_sessions is not None:
        query.append("LIMIT ?")
        params.append(limit_sessions)
    return " ".join(query), params


def export_state_db_sessions(
    output_dir: Path,
    db_path: Path = _DEFAULT_DB_PATH,
    *,
    sources: Sequence[str] | None = None,
    limit_sessions: int | None = None,
) -> ExportSummary:
    """Export Hermes sessions into legacy JSON files.

    Args:
        output_dir: Destination directory for ``*.json`` session files.
        db_path: Path to Hermes ``state.db``.
        sources: Optional source filter (e.g. ``["cli", "discord"]``).
        limit_sessions: Optional max number of sessions to export, newest first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        raise FileNotFoundError(f"state.db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        query, params = _build_sessions_query(sources, limit_sessions)
        session_rows = conn.execute(query, params).fetchall()

        sessions_exported = 0
        sessions_skipped = 0
        messages_exported = 0

        for session_row in session_rows:
            message_rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_row["id"],),
            ).fetchall()

            messages = []
            for row in message_rows:
                role = row["role"]
                if role not in VALID_EXPORT_ROLES:
                    continue
                content = row["content"] or ""
                if role == "user" and _should_skip_user_message(content):
                    continue
                messages.append({"role": role, "content": content})

            has_dialogue = any(
                msg["role"] in {"user", "assistant"} and msg["content"].strip()
                for msg in messages
            )
            if not has_dialogue:
                sessions_skipped += 1
                continue

            session_id = session_row["id"]
            payload = {
                "session_id": session_id,
                "source": session_row["source"],
                "messages": messages,
            }
            filename = f"{_safe_session_filename(session_id)}.json"
            (output_dir / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sessions_exported += 1
            messages_exported += len(messages)

        return ExportSummary(
            db_path=str(db_path),
            output_dir=str(output_dir),
            sessions_exported=sessions_exported,
            messages_exported=messages_exported,
            sessions_scanned=len(session_rows),
            sessions_skipped=sessions_skipped,
        )
    finally:
        conn.close()


def _parse_sources(raw_sources: str | None) -> list[str] | None:
    if not raw_sources:
        return None
    sources = [item.strip() for item in raw_sources.split(",") if item.strip()]
    return sources or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Hermes state.db sessions to legacy JSON files for hermes-agent-self-evolution.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory that will receive one JSON file per exported session.",
    )
    parser.add_argument(
        "--db-path",
        default=_DEFAULT_DB_PATH,
        type=Path,
        help=f"Path to Hermes state.db (default: {_DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Optional comma-separated source filter, e.g. cli,discord,telegram.",
    )
    parser.add_argument(
        "--limit-sessions",
        default=None,
        type=int,
        help="Optional max number of sessions to export, newest first.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = export_state_db_sessions(
        output_dir=args.output_dir,
        db_path=args.db_path,
        sources=_parse_sources(args.sources),
        limit_sessions=args.limit_sessions,
    )
    print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
