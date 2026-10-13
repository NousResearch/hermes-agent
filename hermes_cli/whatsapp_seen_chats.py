"""WhatsApp chats the bridge has already seen, for the route editor.

Names come from each profile's channel directory and from gateway session rows.
The route table stores only a JID; this list is how the dashboard shows a group
or contact name without asking the operator to paste one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


def whatsapp_chat_kind(chat_id: str, chat_type: Optional[str] = None) -> str:
    kind = (chat_type or "").strip().lower()
    if kind in {"group", "channel"} or chat_id.lower().endswith("@g.us"):
        return "group"
    return "dm"


def _clean_name(name: Any, chat_id: str) -> Optional[str]:
    text = str(name or "").strip()
    if not text or text == chat_id:
        return None
    return text


def _bare_chat_id(entry_id: str, thread_id: Any) -> str:
    chat_id = str(entry_id or "").strip()
    thread = str(thread_id or "").strip()
    suffix = f":{thread}" if thread else ""
    if suffix and chat_id.endswith(suffix):
        return chat_id[: -len(suffix)]
    return chat_id


def _absorb(
    found: dict[str, dict[str, Any]],
    chat_id: str,
    *,
    name: Optional[str],
    kind: str,
    last_seen: Optional[float],
) -> None:
    chat_id = chat_id.strip()
    if not chat_id:
        return
    current = found.get(chat_id)
    if current is None:
        found[chat_id] = {
            "chat_id": chat_id,
            "name": name,
            "kind": kind,
            "last_seen": last_seen,
        }
        return
    if name and not current.get("name"):
        current["name"] = name
    if last_seen is not None and (
        current.get("last_seen") is None or last_seen > current["last_seen"]
    ):
        current["last_seen"] = last_seen
    if current.get("kind") != "group" and kind == "group":
        current["kind"] = "group"


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _chats_from_directory(home: Path, found: dict[str, dict[str, Any]]) -> None:
    directory = _read_json_dict(home / "channel_directory.json")
    platforms = directory.get("platforms")
    entries = platforms.get("whatsapp") if isinstance(platforms, dict) else None
    aliases = _read_json_dict(home / "channel_aliases.json").get("whatsapp")
    alias_map = aliases if isinstance(aliases, dict) else {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            chat_id = _bare_chat_id(str(entry.get("id") or ""), entry.get("thread_id"))
            alias = _clean_name(alias_map.get(chat_id), chat_id)
            _absorb(
                found,
                chat_id,
                name=alias or _clean_name(entry.get("name"), chat_id),
                kind=whatsapp_chat_kind(chat_id, str(entry.get("type") or "")),
                last_seen=None,
            )
    for chat_id, friendly in alias_map.items():
        _absorb(
            found,
            str(chat_id),
            name=_clean_name(friendly, str(chat_id)),
            kind=whatsapp_chat_kind(str(chat_id)),
            last_seen=None,
        )


def _chats_from_sessions(home: Path, found: dict[str, dict[str, Any]]) -> None:
    db_path = home / "state.db"
    if not db_path.is_file():
        return
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return
    try:
        conn.row_factory = sqlite3.Row
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "chat_id" not in columns or "started_at" not in columns:
            return
        select = ["chat_id", "started_at"]
        for column in ("chat_type", "display_name", "title", "ended_at", "session_key", "origin_json"):
            if column in columns:
                select.append(column)
        matchers: list[str] = []
        if "session_key" in columns:
            matchers.append("IFNULL(session_key, '') LIKE '%:whatsapp:%'")
        if "origin_json" in columns:
            matchers.append("IFNULL(origin_json, '') LIKE '%\"platform\":%whatsapp%'")
        if not matchers:
            return
        query = (
            f"SELECT {', '.join(select)} FROM sessions "
            f"WHERE chat_id IS NOT NULL AND chat_id != '' AND ({' OR '.join(matchers)})"
        )
        for row in conn.execute(query):
            record = {key: row[key] for key in row.keys()}
            if not _row_is_whatsapp(record):
                continue
            chat_id = str(record.get("chat_id") or "").strip()
            ended = record.get("ended_at")
            started = record.get("started_at")
            last_seen = ended if isinstance(ended, (int, float)) else started
            if not isinstance(last_seen, (int, float)):
                last_seen = None
            name = _clean_name(record.get("display_name"), chat_id) or _clean_name(
                record.get("title"), chat_id
            )
            _absorb(
                found,
                chat_id,
                name=name,
                kind=whatsapp_chat_kind(chat_id, str(record.get("chat_type") or "")),
                last_seen=float(last_seen) if last_seen is not None else None,
            )
    except sqlite3.Error:
        return
    finally:
        conn.close()


def _row_is_whatsapp(record: dict[str, Any]) -> bool:
    session_key = str(record.get("session_key") or "")
    if ":whatsapp:" in session_key:
        return True
    origin = str(record.get("origin_json") or "")
    return '"platform": "whatsapp"' in origin or '"platform":"whatsapp"' in origin


def collect_whatsapp_seen_chats(homes: Iterable[Path], *, limit: int = 400) -> list[dict[str, Any]]:
    """Merge seen WhatsApp chats across profile homes. Newest activity first."""
    found: dict[str, dict[str, Any]] = {}
    for home in homes:
        path = Path(home)
        _chats_from_directory(path, found)
        _chats_from_sessions(path, found)
    chats = list(found.values())
    chats.sort(key=lambda chat: (chat.get("last_seen") is not None, chat.get("last_seen") or 0), reverse=True)
    return chats[:limit]
