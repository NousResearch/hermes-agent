"""WhatsApp chats the bridge has already seen, for the route editor.

Names come from each profile's channel directory and from gateway session rows.
The route table stores only a JID; this list is how the dashboard shows a group
or contact name without asking the operator to paste one.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def whatsapp_chat_kind(chat_id: str, chat_type: Optional[str] = None) -> str:
    kind = (chat_type or "").strip().lower()
    if kind in {"group", "channel"} or chat_id.lower().endswith("@g.us"):
        return "group"
    return "dm"


def _jid_local(chat_id: str) -> str:
    return chat_id.split("@", 1)[0]


def _clean_name(name: Any, chat_id: str) -> Optional[str]:
    """A display name. The bridge's placeholder is the JID local part, which is not a title."""
    text = str(name or "").strip()
    local = _jid_local(chat_id)
    if not text or text in {chat_id, local} or text.isdigit():
        return None
    lowered = text.lower()
    if lowered.endswith(("@g.us", "@s.whatsapp.net", "@lid")):
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
    elif name and current.get("name") != name and _clean_name(current.get("name"), chat_id) is None:
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


def _note_message_title(blocked: dict[str, set[str]], chat_id: str, title: Any) -> None:
    """Session titles are the conversation text, not the WhatsApp group subject."""
    text = str(title or "").strip()
    if not text:
        return
    blocked.setdefault(chat_id, set()).add(text)
    local = _jid_local(chat_id)
    if local != chat_id:
        blocked.setdefault(local, set()).add(text)


def _drop_message_titles(found: dict[str, dict[str, Any]], blocked: dict[str, set[str]]) -> None:
    for chat_id, row in found.items():
        banned = set(blocked.get(chat_id, ()))
        banned.update(blocked.get(_jid_local(chat_id), ()))
        if row.get("name") in banned:
            row["name"] = None


def _chats_from_sessions(
    home: Path,
    found: dict[str, dict[str, Any]],
    blocked_titles: dict[str, set[str]],
) -> None:
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
            _note_message_title(blocked_titles, chat_id, record.get("title"))
            name = _clean_name(record.get("display_name"), chat_id)
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


def _fetch_bridge_chats(port: int = 3000, timeout: float = 4) -> Optional[list[dict[str, Any]]]:
    """Group titles from GET /chats.

    None means that route is missing (an older bridge copy on the volume).
    An empty list means the bridge answered and has no subjects right now.
    """
    url = f"http://127.0.0.1:{port}/chats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None if exc.code == 404 else []
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return []
    chats = payload.get("chats") if isinstance(payload, dict) else None
    return [chat for chat in chats if isinstance(chat, dict)] if isinstance(chats, list) else []


def _lookup_bridge_chat(chat_id: str, port: int = 3000, timeout: float = 3) -> Optional[dict[str, Any]]:
    """One group title from GET /chat/:id, which older bridges already serve."""
    url = f"http://127.0.0.1:{port}/chat/{urllib.request.quote(chat_id, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def bridge_subjects_for_chats(
    chat_ids: Iterable[str],
    lookup,
) -> list[dict[str, Any]]:
    """Ask the bridge for each group JID. Digit placeholders are not titles."""
    subjects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in chat_ids:
        chat_id = str(raw or "").strip()
        if not chat_id.lower().endswith("@g.us") or chat_id in seen:
            continue
        seen.add(chat_id)
        try:
            payload = lookup(chat_id)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        name = _clean_name(payload.get("name"), chat_id)
        if not name:
            continue
        subjects.append({"chat_id": chat_id, "name": name, "kind": "group"})
    return subjects


def bridge_chats_for_editor(
    known_chat_ids: Iterable[str],
    listed: Optional[list[dict[str, Any]]],
    lookup,
) -> list[dict[str, Any]]:
    """Prefer GET /chats. When that route is missing, or a routed group is absent, use GET /chat/:id.

    An empty list from /chats means the socket is down. Don't fan out per chat.
    """
    if listed is not None and not listed:
        return []
    listed_chats = [chat for chat in (listed or []) if isinstance(chat, dict)]
    have = {str(chat.get("chat_id") or "").strip() for chat in listed_chats}
    if listed is None:
        missing = list(known_chat_ids)
    else:
        missing = [chat_id for chat_id in known_chat_ids if str(chat_id).strip() not in have]
    return listed_chats + bridge_subjects_for_chats(missing, lookup)


def _apply_bridge_chats(found: dict[str, dict[str, Any]], bridge_chats: Iterable[dict[str, Any]]) -> None:
    for chat in bridge_chats:
        chat_id = str(chat.get("chat_id") or "").strip()
        name = _clean_name(chat.get("name"), chat_id)
        if not chat_id or not name:
            continue
        _absorb(
            found,
            chat_id,
            name=name,
            kind=whatsapp_chat_kind(chat_id, str(chat.get("kind") or "")),
            last_seen=None,
        )
        # Routes and session rows sometimes store the bare number.
        local = _jid_local(chat_id)
        if local in found and local != chat_id:
            found[local]["name"] = name
            if found[local].get("kind") != "group":
                found[local]["kind"] = "group"


def prune_message_title_aliases(home: Path, blocked_titles: dict[str, set[str]]) -> None:
    """Remove aliases that were the session's message title, not a group subject."""
    path = Path(home) / "channel_aliases.json"
    data = _read_json_dict(path)
    current = data.get("whatsapp")
    if not isinstance(current, dict) or not current:
        return
    changed = False
    for chat_id, name in list(current.items()):
        banned = set(blocked_titles.get(str(chat_id), ()))
        banned.update(blocked_titles.get(_jid_local(str(chat_id)), ()))
        if str(name or "").strip() in banned:
            del current[chat_id]
            changed = True
    if not changed:
        return
    data["whatsapp"] = current
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def remember_whatsapp_aliases(home: Path, names: dict[str, str]) -> None:
    """Keep learned group titles in channel_aliases.json so the table still has them later."""
    path = Path(home) / "channel_aliases.json"
    data = _read_json_dict(path)
    current = data.get("whatsapp")
    if not isinstance(current, dict):
        current = {}
    changed = False
    for chat_id, name in names.items():
        cleaned = _clean_name(name, chat_id)
        if not cleaned or current.get(chat_id) == cleaned:
            continue
        current[chat_id] = cleaned
        changed = True
    if not changed:
        return
    data["whatsapp"] = current
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_whatsapp_seen_chats(
    homes: Iterable[Path],
    *,
    limit: int = 400,
    bridge_chats: Optional[Iterable[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Merge seen WhatsApp chats across profile homes. Newest activity first."""
    found: dict[str, dict[str, Any]] = {}
    blocked_titles: dict[str, set[str]] = {}
    home_paths = [Path(home) for home in homes]
    for path in home_paths:
        _chats_from_directory(path, found)
        _chats_from_sessions(path, found, blocked_titles)
    _drop_message_titles(found, blocked_titles)
    for path in home_paths:
        try:
            prune_message_title_aliases(path, blocked_titles)
        except OSError:
            logger.warning("Could not prune WhatsApp chat aliases in %s", path, exc_info=True)
    if bridge_chats:
        _apply_bridge_chats(found, bridge_chats)
    chats = list(found.values())
    chats.sort(key=lambda chat: (chat.get("last_seen") is not None, chat.get("last_seen") or 0), reverse=True)
    return chats[:limit]
