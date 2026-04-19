"""Conversation-oriented dashboard helpers built on top of SessionDB."""

from __future__ import annotations

import re
import time
from typing import Any

from hermes_state import SessionDB


VISIBLE_ROLES = {"user", "assistant"}
CLI_SOURCE = "cli"
ALL_SOURCES = "all"
LINEAGE_TOLERANCE_SECONDS = 3.0
DEFAULT_CONVERSATION_LIMIT = 200
MAX_CONVERSATION_LIMIT = 500
MAX_CONVERSATION_SCAN = 100000
SYNTHETIC_USER_PREFIXES = (
    "[system:",
    "you've reached the maximum number of tool-calling iterations allowed.",
    "you have reached the maximum number of tool-calling iterations allowed.",
)
INTERNAL_PROBE_PREFIXES = (
    "return exactly [silent]",
    "return exactly the text [silent]",
    "reply exactly [silent]",
    "reply exactly the text [silent]",
    "reply with exactly [silent]",
    "reply with exactly flash_provider_ok",
    "reply with exactly flash_ok",
    "reply with exactly mini_ok",
    "reply with exactly hermes_spark_ok",
    "reply with exactly: ok-default",
    "reply with exactly: ok-openai-codex",
)
CONTEXT_COMPACTION_PREFIX = "[CONTEXT COMPACTION — REFERENCE ONLY]"
LEGACY_CONTEXT_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"
SUMMARY_MERGE_SEPARATOR = (
    "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
)
COMPRESSION_END_REASONS = {"compression", "compressed"}


class ConversationNotFoundError(KeyError):
    """Raised when a requested conversation root does not exist."""


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _safe_text(value: Any) -> str:
    return str(value or "")


def _normalize_pagination(limit: Any, offset: Any) -> tuple[int, int]:
    try:
        normalized_limit = int(limit)
    except (TypeError, ValueError):
        normalized_limit = DEFAULT_CONVERSATION_LIMIT

    try:
        normalized_offset = int(offset)
    except (TypeError, ValueError):
        normalized_offset = 0

    normalized_limit = min(max(normalized_limit, 1), MAX_CONVERSATION_LIMIT)
    normalized_offset = max(normalized_offset, 0)
    return normalized_limit, normalized_offset


def _looks_synthetic_user_text(value: Any) -> bool:
    text = _normalize_text(value)
    return any(text.startswith(prefix) for prefix in SYNTHETIC_USER_PREFIXES)


def _visible_message(role: str, content: str) -> bool:
    if role not in VISIBLE_ROLES:
        return False
    text = _safe_text(content).strip()
    if not text:
        return False
    if role == "user" and _looks_synthetic_user_text(text):
        return False
    return True


def _display_excerpt(text: str, width: int = 180) -> str:
    clean = re.sub(r"\s+", " ", _safe_text(text)).strip()
    if not clean:
        return ""
    return clean[:width] + ("…" if len(clean) > width else "")


def _display_title(text: str, width: int = 72) -> str:
    return _display_excerpt(text, width=width)


def _display_seed_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    first_user = next((item for item in messages if item.get("role") == "user"), None)
    if first_user:
        return first_user
    return messages[0] if messages else None


def _repair_conversation_display(db: SessionDB, conversation: dict[str, Any], conversation_ids: list[str]) -> dict[str, Any]:
    title = _safe_text(conversation.get("title")).strip()
    preview = _safe_text(conversation.get("preview")).strip()
    needs_title = not title or _looks_synthetic_user_text(title)
    needs_preview = not preview or _looks_synthetic_user_text(preview)

    if not (needs_title or needs_preview):
        return conversation

    messages = _fetch_visible_messages(db, conversation_ids)
    seed = _display_seed_message(messages)
    if not seed:
        return conversation

    seed_text = _safe_text(seed.get("content")).strip()
    if not seed_text:
        return conversation

    item = dict(conversation)
    if needs_title:
        item["title"] = _display_title(seed_text)
    if needs_preview:
        item["preview"] = _display_excerpt(seed_text)
    return item


def _conversation_snippet(text: str, query: str, width: int = 180) -> str:
    clean = re.sub(r"\s+", " ", _safe_text(text)).strip()
    if not clean:
        return ""

    needle = _normalize_text(query)
    if not needle:
        return clean[:width] + ("…" if len(clean) > width else "")

    lower = clean.lower()
    index = lower.find(needle)
    if index < 0:
        return clean[:width] + ("…" if len(clean) > width else "")

    start = max(0, index - 56)
    end = min(len(clean), index + len(needle) + 88)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(clean) else ""
    return prefix + clean[start:end].strip() + suffix


def _load_session_graph(db: SessionDB) -> tuple[list[str], dict[str, dict[str, Any]], dict[str | None, list[str]]]:
    sessions = db.list_sessions_rich(limit=MAX_CONVERSATION_SCAN, offset=0, include_children=True)
    now = time.time()
    ordered_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str | None, list[str]] = {}

    # SessionDB's public helpers do not expose a graph-wide compaction scan, so
    # this module reads directly from the underlying connection while holding
    # SessionDB's lock. Keep the SQL narrow and regression-tested.
    with db._lock:
        cursor = db._conn.execute(
            """
            SELECT DISTINCT session_id
            FROM messages
            WHERE content LIKE ? OR content LIKE ? OR content LIKE ?
            """,
            (
                f"{CONTEXT_COMPACTION_PREFIX}%",
                f"{LEGACY_CONTEXT_SUMMARY_PREFIX}%",
                f"%{SUMMARY_MERGE_SEPARATOR}%",
            ),
        )
        compaction_session_ids = {row[0] for row in cursor.fetchall()}

    for session in sessions:
        item = dict(session)
        ordered_ids.append(item["id"])
        item["is_active"] = (
            item.get("ended_at") is None
            and (now - item.get("last_active", item.get("started_at", 0))) < 300
        )
        item["has_compaction_artifact"] = item["id"] in compaction_session_ids
        by_id[item["id"]] = item
        children_by_parent.setdefault(item.get("parent_session_id"), []).append(item["id"])

    return ordered_ids, by_id, children_by_parent


def _timing_matches_parent(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    parent_ended = parent.get("ended_at")
    child_started = child.get("started_at")
    if parent_ended is None or child_started is None:
        return False
    return abs(float(child_started) - float(parent_ended)) <= LINEAGE_TOLERANCE_SECONDS


def _root_session(session: dict[str, Any] | None) -> bool:
    if not session or session.get("parent_session_id") is not None:
        return False
    if session.get("source") == "tool":
        return False

    if session.get("source") == CLI_SOURCE:
        title = _safe_text(session.get("title")).strip()
        preview = _normalize_text(session.get("preview"))
        message_count = int(session.get("message_count") or 0)
        if preview and any(preview.startswith(prefix) for prefix in INTERNAL_PROBE_PREFIXES):
            return False
        if not title and not preview:
            return message_count > 0

    return True


def _branch_root_session(session: dict[str, Any] | None, by_id: dict[str, dict[str, Any]]) -> bool:
    if not session or session.get("parent_session_id") is None:
        return False
    if session.get("source") == "tool":
        return False
    parent = by_id.get(session.get("parent_session_id"))
    if not parent:
        return False
    if parent.get("end_reason") == "branched":
        return _timing_matches_parent(parent, session)

    # Gateway-created branches on current main may keep a parent link without
    # marking the parent session as end_reason="branched". Surface those child
    # sessions as display roots when they look like a deliberate branch, while
    # still excluding compression continuations and hidden sources.
    if session.get("has_compaction_artifact"):
        return False
    if parent.get("end_reason") in COMPRESSION_END_REASONS:
        return False
    if parent.get("source") != session.get("source"):
        return False
    if not _safe_text(session.get("title")).strip():
        return False
    started_at = session.get("started_at")
    parent_started = parent.get("started_at")
    if started_at is None or parent_started is None:
        return False
    return float(started_at) >= float(parent_started)


def _display_root_session(session: dict[str, Any] | None, by_id: dict[str, dict[str, Any]]) -> bool:
    return _root_session(session) or _branch_root_session(session, by_id)


def _next_continuation_child(parent: dict[str, Any], by_id: dict[str, dict[str, Any]], children_by_parent: dict[str | None, list[str]]) -> str | None:
    if not parent:
        return None

    # Resuming an older compressed session clears ended_at/end_reason on the
    # parent. Preserve continuation chains by treating untitled, non-display
    # same-source children as implicit continuation candidates in that case.
    if (
        parent.get("end_reason") is None
        and parent.get("ended_at") is None
    ):
        resumed_candidates: list[tuple[float, str]] = []
        for child_id in children_by_parent.get(parent["id"], []):
            child = by_id.get(child_id)
            if not child:
                continue
            if child.get("source") != parent.get("source"):
                continue
            if child.get("has_compaction_artifact"):
                resumed_candidates.append((float(child.get("started_at") or 0), child_id))
                continue
            if _display_root_session(child, by_id):
                continue
            if _safe_text(child.get("title")).strip():
                continue
            resumed_candidates.append((float(child.get("started_at") or 0), child_id))
        if not resumed_candidates:
            return None
        resumed_candidates.sort(key=lambda item: (item[0], item[1]))
        return resumed_candidates[0][1]

    if parent.get("end_reason") not in COMPRESSION_END_REASONS:
        return None

    candidates: list[tuple[int, float, str]] = []
    parent_preview = _normalize_text(parent.get("preview"))
    for child_id in children_by_parent.get(parent["id"], []):
        child = by_id.get(child_id)
        if not child:
            continue
        if child.get("source") != parent.get("source"):
            continue
        if not _timing_matches_parent(parent, child):
            continue
        child_preview = _normalize_text(child.get("preview"))
        preview_score = 0 if (parent_preview and child_preview and child_preview == parent_preview) else 1
        delta = abs(float(child.get("started_at") or 0) - float(parent.get("ended_at") or 0))
        candidates.append((preview_score, delta, child_id))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][2]


def _collect_conversation_chain(root_id: str, by_id: dict[str, dict[str, Any]], children_by_parent: dict[str | None, list[str]]) -> list[str]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    current = by_id.get(root_id)

    while current and current["id"] not in seen:
        chain_ids.append(current["id"])
        seen.add(current["id"])
        next_id = _next_continuation_child(current, by_id, children_by_parent)
        current = by_id.get(next_id) if next_id else None

    return chain_ids


def _collect_subtree(start_id: str, children_by_parent: dict[str | None, list[str]]) -> list[str]:
    stack = [start_id]
    seen: set[str] = set()
    ordered: list[str] = []
    while stack:
        session_id = stack.pop()
        if session_id in seen:
            continue
        seen.add(session_id)
        ordered.append(session_id)
        child_ids = list(children_by_parent.get(session_id, []))
        child_ids.reverse()
        for child_id in child_ids:
            stack.append(child_id)
    return ordered


def _session_depth(session_id: str, by_id: dict[str, dict[str, Any]]) -> int:
    depth = 0
    seen: set[str] = set()
    current = by_id.get(session_id)
    while current:
        parent_id = current.get("parent_session_id")
        if not parent_id or parent_id in seen:
            break
        seen.add(parent_id)
        depth += 1
        current = by_id.get(parent_id)
    return depth


def _ordered_delete_sessions(session_ids: set[str], by_id: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        session_ids,
        key=lambda session_id: (_session_depth(session_id, by_id), session_id),
        reverse=True,
    )


def _aggregate_root_session(root_id: str, by_id: dict[str, dict[str, Any]], children_by_parent: dict[str | None, list[str]]) -> tuple[dict[str, Any], list[str]]:
    chain_ids = _collect_conversation_chain(root_id, by_id, children_by_parent)
    chain_sessions = [by_id[session_id] for session_id in chain_ids if session_id in by_id]
    if not chain_sessions:
        raise ConversationNotFoundError(root_id)

    root = dict(by_id[root_id])
    root["thread_session_count"] = len(chain_sessions)
    root["thread_message_count"] = sum(int(session.get("message_count") or 0) for session in chain_sessions)
    root["last_active"] = max(
        float(session.get("last_active") or session.get("started_at") or 0)
        for session in chain_sessions
    )
    root["is_active"] = any(bool(session.get("is_active")) for session in chain_sessions)
    root["model"] = chain_sessions[-1].get("model") or root.get("model")

    if not _safe_text(root.get("title")).strip():
        for session in reversed(chain_sessions):
            title = _safe_text(session.get("title")).strip()
            if title:
                root["title"] = title
                break

    if not _safe_text(root.get("preview")).strip():
        for session in chain_sessions:
            preview = _safe_text(session.get("preview")).strip()
            if preview:
                root["preview"] = preview
                break

    return root, chain_ids


def _strip_compaction_artifact(content: str) -> str:
    text = _safe_text(content).strip()
    if not text:
        return ""

    stripped = text.lstrip()
    is_summary = stripped.startswith(CONTEXT_COMPACTION_PREFIX) or stripped.startswith(LEGACY_CONTEXT_SUMMARY_PREFIX)
    if not is_summary:
        return text

    if SUMMARY_MERGE_SEPARATOR not in stripped:
        return ""

    return stripped.split(SUMMARY_MERGE_SEPARATOR, 1)[1].strip()


def _fetch_session_visible_messages(db: SessionDB, session_id: str) -> list[dict[str, Any]]:
    # SessionDB exposes get_messages(session_id), but the conversations view
    # needs a tighter projection/order for transcript stitching and search, so
    # it uses a direct read under the shared DB lock.
    with db._lock:
        cursor = db._conn.execute(
            "SELECT id, session_id, role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        )
        rows = cursor.fetchall()

    visible: list[dict[str, Any]] = []
    for row in rows:
        role = _safe_text(row["role"])
        content = _safe_text(row["content"])
        if not _visible_message(role, content):
            continue
        cleaned = _strip_compaction_artifact(content)
        if not cleaned:
            continue
        if role == "user" and _looks_synthetic_user_text(cleaned):
            continue
        visible.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": role,
                "content": cleaned,
                "timestamp": row["timestamp"],
            }
        )
    return visible


def _message_signature(message: dict[str, Any]) -> tuple[str, str]:
    return (_safe_text(message.get("role")), _normalize_text(message.get("content")))



def _prefix_overlap_length(previous_messages: list[dict[str, Any]], current_messages: list[dict[str, Any]]) -> int:
    max_overlap = min(len(previous_messages), len(current_messages))
    if max_overlap <= 0:
        return 0

    previous_signatures = [_message_signature(message) for message in previous_messages]
    current_signatures = [_message_signature(message) for message in current_messages]

    for size in range(max_overlap, 0, -1):
        if previous_signatures[:size] == current_signatures[:size]:
            return size
    return 0



def _suffix_overlap_length(previous_messages: list[dict[str, Any]], current_messages: list[dict[str, Any]]) -> int:
    max_overlap = min(len(previous_messages), len(current_messages))
    if max_overlap <= 0:
        return 0

    previous_signatures = [_message_signature(message) for message in previous_messages]
    current_signatures = [_message_signature(message) for message in current_messages]

    for size in range(max_overlap, 0, -1):
        if previous_signatures[-size:] == current_signatures[:size]:
            return size
    return 0



def _fetch_visible_messages(db: SessionDB, session_ids: list[str]) -> list[dict[str, Any]]:
    if not session_ids:
        return []

    visible: list[dict[str, Any]] = []
    for index, session_id in enumerate(session_ids):
        session_messages = _fetch_session_visible_messages(db, session_id)
        if index == 0:
            visible.extend(session_messages)
            continue

        head_overlap = _prefix_overlap_length(visible, session_messages)
        session_messages = session_messages[head_overlap:]
        tail_overlap = _suffix_overlap_length(visible, session_messages)
        visible.extend(session_messages[tail_overlap:])
    return visible



def _session_search_hit(session: dict[str, Any], query: str) -> bool:
    needle = _normalize_text(query)
    if not needle:
        return True
    haystack = " ".join(
        [
            _safe_text(session.get("title")),
            _safe_text(session.get("preview")),
            _safe_text(session.get("model")),
            _safe_text(session.get("source")),
        ]
    ).lower()
    return needle in haystack


def _search_conversation(db: SessionDB, root_session: dict[str, Any], conversation_ids: list[str], query: str) -> tuple[bool, str]:
    if _session_search_hit(root_session, query):
        return True, ""

    needle = _normalize_text(query)
    for message in _fetch_visible_messages(db, conversation_ids):
        if needle in _normalize_text(message.get("content")):
            return True, _conversation_snippet(message.get("content", ""), query)

    return False, ""


def _resolve_display_root_id(db: SessionDB, conversation_id: str) -> tuple[str, dict[str, dict[str, Any]], dict[str | None, list[str]]]:
    resolved_id = db.resolve_session_id(conversation_id) or conversation_id
    _ordered_ids, by_id, children_by_parent = _load_session_graph(db)
    session = by_id.get(resolved_id)
    if not _display_root_session(session, by_id):
        raise ConversationNotFoundError(conversation_id)
    return resolved_id, by_id, children_by_parent


def _delete_partition_descendants(
    start_id: str,
    by_id: dict[str, dict[str, Any]],
    children_by_parent: dict[str | None, list[str]],
    delete_ids: set[str],
    orphan_ids: set[str],
) -> None:
    stack = [start_id]
    while stack:
        session_id = stack.pop()
        if session_id in delete_ids or session_id in orphan_ids:
            continue
        session = by_id.get(session_id)
        if _display_root_session(session, by_id):
            orphan_ids.add(session_id)
            continue
        delete_ids.add(session_id)
        child_ids = list(children_by_parent.get(session_id, []))
        child_ids.reverse()
        for child_id in child_ids:
            stack.append(child_id)



def _delete_display_conversation(db: SessionDB, root_id: str, by_id: dict[str, dict[str, Any]], children_by_parent: dict[str | None, list[str]]) -> dict[str, int]:
    chain_ids = _collect_conversation_chain(root_id, by_id, children_by_parent)
    if not chain_ids:
        return {"deleted_sessions": 0, "deleted_messages": 0}

    orphan_ids: set[str] = set()
    delete_ids: set[str] = set(chain_ids)

    for chain_id in chain_ids:
        for child_id in children_by_parent.get(chain_id, []):
            if child_id in delete_ids:
                continue
            child = by_id.get(child_id)
            if _display_root_session(child, by_id):
                orphan_ids.add(child_id)
            else:
                _delete_partition_descendants(child_id, by_id, children_by_parent, delete_ids, orphan_ids)

    delete_list = _ordered_delete_sessions(delete_ids, by_id)
    message_placeholders = ",".join("?" for _ in delete_list)

    def _do(conn):
        if orphan_ids:
            orphan_placeholders = ",".join("?" for _ in orphan_ids)
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL WHERE id IN ({orphan_placeholders})",
                list(orphan_ids),
            )

        cursor = conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({message_placeholders})",
            delete_list,
        )
        deleted_messages = cursor.fetchone()[0]
        conn.execute(
            f"DELETE FROM messages WHERE session_id IN ({message_placeholders})",
            delete_list,
        )
        for session_id in delete_list:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return {
            "deleted_sessions": len(delete_list),
            "deleted_messages": deleted_messages,
        }

    # Delete + orphaning must happen atomically. SessionDB does not currently
    # expose a public multi-session mutation for this operation, so we route the
    # write through its transactional helper.
    return db._execute_write(_do)


def list_conversations(q: str = "", source: str = ALL_SOURCES, limit: int = DEFAULT_CONVERSATION_LIMIT, offset: int = 0) -> dict[str, Any]:
    limit, offset = _normalize_pagination(limit, offset)
    source = _safe_text(source).strip() or ALL_SOURCES
    q = _safe_text(q).strip()

    db = SessionDB()
    try:
        ordered_ids, by_id, children_by_parent = _load_session_graph(db)
        root_ids = [
            session_id
            for session_id in ordered_ids
            if _display_root_session(by_id.get(session_id), by_id)
        ]

        conversations: list[tuple[dict[str, Any], list[str]]] = []
        for root_id in root_ids:
            conversation, conversation_ids = _aggregate_root_session(root_id, by_id, children_by_parent)
            conversation = _repair_conversation_display(db, conversation, conversation_ids)
            conversation["snippet"] = ""
            conversation["thread_root_id"] = root_id
            if not _safe_text(conversation.get("title")).strip() and not _safe_text(conversation.get("preview")).strip():
                continue
            conversations.append((conversation, conversation_ids))

        sources = sorted({str(conversation.get("source") or "unknown") for conversation, _ in conversations})
        all_total = len(conversations)
        query = q
        filtered: list[dict[str, Any]] = []

        for conversation, conversation_ids in conversations:
            conversation_source = str(conversation.get("source") or "unknown")
            if source and source != ALL_SOURCES and conversation_source != source:
                continue
            if not query:
                filtered.append(conversation)
                continue
            matched, snippet = _search_conversation(db, conversation, conversation_ids, query)
            if matched:
                item = dict(conversation)
                item["snippet"] = snippet
                filtered.append(item)

        filtered.sort(
            key=lambda item: (
                -(item.get("last_active") or 0),
                -(item.get("started_at") or 0),
                item.get("id") or "",
            )
        )
        total = len(filtered)
        paged = filtered[offset : offset + limit]
        return {
            "sessions": paged,
            "total": total,
            "all_total": all_total,
            "limit": limit,
            "offset": offset,
            "sources": sources,
            "source": source,
            "mode": "source-filtered-conversations",
        }
    finally:
        db.close()


def get_conversation_messages(conversation_id: str) -> dict[str, Any]:
    db = SessionDB()
    try:
        resolved_id, by_id, children_by_parent = _resolve_display_root_id(db, conversation_id)
        conversation, conversation_ids = _aggregate_root_session(resolved_id, by_id, children_by_parent)
        visible_messages = _fetch_visible_messages(db, conversation_ids)
        return {
            "session_id": resolved_id,
            "messages": visible_messages,
            "visible_count": len(visible_messages),
            "thread_session_count": conversation.get("thread_session_count", 1),
        }
    finally:
        db.close()


def delete_conversation(conversation_id: str) -> dict[str, Any]:
    db = SessionDB()
    try:
        resolved_id, by_id, children_by_parent = _resolve_display_root_id(db, conversation_id)
        result = _delete_display_conversation(db, resolved_id, by_id, children_by_parent)
        return {"ok": True, **result}
    finally:
        db.close()
