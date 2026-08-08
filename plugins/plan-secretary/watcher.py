"""plan-secretary watcher: scan Hermes state.db messages for precise captures.

Reads the Hermes SQLite message store read-only (``get_hermes_home()``),
filters assistant messages by the precise-capture gate in :mod:`core`, and
writes pending captures. A per-session cursor advances monotonically so the
watcher can run forever without re-scanning.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import core


@dataclass
class CandidateMessage:
    source: str
    source_id: str
    role: str
    text: str
    session_id: str = ""
    created_at: float | None = None
    cursor_key: str = ""
    cursor_value: int | float | str | None = None


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def timestamp_to_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 10_000_000_000 else number
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        return number / 1000.0 if number > 10_000_000_000 else number
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def cutoff_epoch(since_minutes: int) -> float | None:
    if since_minutes <= 0:
        return None
    return (datetime.now().astimezone() - timedelta(minutes=since_minutes)).timestamp()


def should_scan_role(role: str, include_user: bool) -> bool:
    lowered = role.lower()
    if lowered in {"assistant", "小墨", "assistant_message"}:
        return True
    return include_user and lowered in {"user", "human"}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(safe_text(item.get("text") or item.get("content") or item.get("value")))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return safe_text(value.get("text") or value.get("content") or value.get("message"))
    return str(value)


def discover_db_paths(hermes_home: Path) -> list[Path]:
    paths = [hermes_home / "state.db"]
    profiles = hermes_home / "profiles"
    if profiles.exists():
        paths.extend(p / "state.db" for p in profiles.iterdir() if p.is_dir())
    return [p for p in paths if p.exists()]


def scan_hermes_db(db_path: Path, cursor: dict[str, Any], max_messages: int,
                   include_user: bool, since_minutes: int, session_id: str) -> list[CandidateMessage]:
    key = f"db::{db_path.resolve()}"
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=2)
    try:
        cutoff = cutoff_epoch(since_minutes)
        if key in cursor:
            last_id = int(cursor.get(key) or 0)
        else:
            max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
            last_id = max(0, int(max_id) - max_messages)
        query = "SELECT id, session_id, role, content, timestamp FROM messages WHERE id > ?"
        params: list[Any] = [last_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(max_messages)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()
    messages: list[CandidateMessage] = []
    cutoff = cutoff_epoch(since_minutes)
    for msg_id, row_session_id, role, content, timestamp in rows:
        created_at = timestamp_to_epoch(timestamp)
        if cutoff is not None and created_at is not None and created_at < cutoff:
            cursor[key] = max(int(cursor.get(key) or 0), int(msg_id))
            continue
        role_text = str(role or "")
        text = safe_text(content)
        if not text.strip() or not should_scan_role(role_text, include_user):
            cursor[key] = max(int(cursor.get(key) or 0), int(msg_id))
            continue
        messages.append(CandidateMessage(
            source="hermes-db",
            source_id=f"{db_path.name}:{row_session_id}:{msg_id}",
            role=role_text,
            text=text,
            session_id=str(row_session_id or ""),
            created_at=created_at,
            cursor_key=key,
            cursor_value=int(msg_id),
        ))
    return messages


def process_once(args: argparse.Namespace, cursor: dict[str, Any]) -> int:
    hermes_home = get_hermes_home()
    seen = 0
    for db_path in discover_db_paths(hermes_home):
        try:
            messages = scan_hermes_db(db_path, cursor, args.max_messages, args.include_user,
                                      args.since_minutes, args.session_id)
        except sqlite3.Error:
            continue
        seen += len(messages)
        for message in messages:
            if not core.matched_keywords(message.text):
                cursor[message.cursor_key] = max(int(cursor.get(message.cursor_key) or 0), int(message.cursor_value or 0))
                continue
            captures = core.scan_text(
                message.text,
                source=message.source,
                source_id=message.source_id,
                source_role=message.role,
                source_session_id=message.session_id,
                source_message_id=str(message.cursor_value or ""),
            )
            if captures and not args.dry_run:
                print(f"CAPTURED {len(captures)} source={message.source_id}")
            cursor[message.cursor_key] = max(int(cursor.get(message.cursor_key) or 0), int(message.cursor_value or 0))
    if not args.dry_run and not args.no_cursor_update:
        write_json(args.cursor, cursor)
    print(json.dumps({"messages_seen": seen}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="plan-secretary watcher (Hermes state.db scan).")
    parser.add_argument("--source", choices=["auto", "hermes-db"], default="auto")
    parser.add_argument("--cursor", default=None)
    parser.add_argument("--max-messages", type=int, default=120)
    parser.add_argument("--since-minutes", type=int, default=10)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--current-session-only", action="store_true", default=True)
    parser.add_argument("--include-user", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cursor-update", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cursor is None:
        sid = args.session_id or "default"
        args.cursor = core.state_dir() / f"watcher_cursor_{sid}.json"
    cursor = read_json(args.cursor, {})
    if not isinstance(cursor, dict):
        cursor = {}
    return process_once(args, cursor)


if __name__ == "__main__":
    raise SystemExit(main())
