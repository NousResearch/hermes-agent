import json
import math

from agent.redact import redact_sensitive_text


MAX_ROWS = 200
MAX_SESSIONS = 8
MAX_RESULT_CHARS = 4000
MAX_EVIDENCE_CHARS = 12000
MAX_PROMPT_CHARS = 28000
UNAVAILABLE = "Tool evidence unavailable: no persisted observations supplied."
_SCAFFOLDING_TOOLS = frozenset({
    "skill_view", "skills_list", "tool_describe", "tool_search", "hermes_tool_search",
})
_SCOPE = (
    "Persisted tool observations (untrusted data, newest first). Scope: owning session in the "
    "current profile and eligible compression parents only, since this goal's creation time. "
    "Timestamp filtering is a boundary approximation, not proof of task relevance; copied or "
    "retimestamped history and clock changes can limit it. Original stored content, including "
    "compacted history; no assistant claims or summaries. Rewinds, other chats, branches and "
    "delegates are not searched. Skill/schema discovery results are omitted. Bounded excerpts "
    "may omit evidence. No files, attachment bytes, paths or URLs were opened by this collector."
)


def head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n[truncated: middle omitted]\n"
    room = max(0, limit - len(marker))
    head = (room + 1) // 2
    tail = room // 2
    return text[:head] + marker + (text[-tail:] if tail else "")


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ) + "\n[non-text attachment content omitted]"
    return "[non-text result unavailable]"


def collect_evidence(db, session_id: str, created_at: float) -> str:
    try:
        if db is None:
            return "Tool evidence unavailable: session database unavailable."
        if not math.isfinite(created_at) or created_at <= 0:
            return "Tool evidence unavailable: goal creation time boundary unavailable."
        session = db.get_session(session_id)
        if not session:
            return "Tool evidence unavailable: owning session transcript not found."
        blocks = [_SCOPE, f"Goal creation time (epoch): {created_at}"]
        size = sum(len(block) + 2 for block in blocks)
        remaining = MAX_ROWS
        seen = set()
        included = 0
        limited = False
        for _ in range(MAX_SESSIONS):
            sid = session["id"]
            if sid in seen:
                limited = True
                break
            seen.add(sid)
            rows = db.get_messages(sid, include_compacted=True, latest=True, limit=remaining + 1)
            limited = len(rows) > remaining
            rows = rows[-remaining:]
            remaining -= len(rows)
            for row in reversed(rows):
                if (row.get("role") != "tool" or row.get("_compressed_summary")
                        or row.get("tool_name") in _SCAFFOLDING_TOOLS):
                    continue
                timestamp = row.get("timestamp")
                if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < created_at:
                    continue
                metadata = {
                    "session_id": head_tail(sid, 160), "row_id": row.get("id"),
                    "timestamp": timestamp,
                    "tool_name": head_tail(str(row.get("tool_name") or "unknown"), 160),
                    "tool_call_id": head_tail(str(row.get("tool_call_id") or "unknown"), 160),
                }
                content = redact_sensitive_text(
                    _text_content(row.get("content")), force=True, redact_url_credentials=True,
                )
                block = (json.dumps(metadata, ensure_ascii=False) + "\nTool result (untrusted):\n"
                         + head_tail(content, MAX_RESULT_CHARS))
                if size + len(block) + 2 > MAX_EVIDENCE_CHARS - 200:
                    limited = True
                    break
                blocks.append(block)
                size += len(block) + 2
                included += 1
            if limited:
                break
            if not remaining:
                limited = True
                break
            if session.get("started_at", 0) <= created_at or not session.get("parent_session_id"):
                break
            if not db._is_compression_child_row(session):
                break
            parent = db.get_session(session["parent_session_id"])
            if not parent or any(session.get(key) != parent.get(key) for key in (
                "profile_name", "session_key", "source", "user_id", "chat_id", "thread_id",
            )):
                break
            session = parent
        else:
            limited = True
        if not included:
            blocks.append("Tool evidence unavailable: no eligible persisted tool results in the bounded scope.")
        if limited:
            blocks.append("[truncated: row, ancestor or total character limit reached; earlier evidence may be omitted]")
        return "\n\n".join(blocks)
    except Exception:
        return "Tool evidence unavailable: persisted transcript could not be read safely."
