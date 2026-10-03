"""Query building, vector search, context assembly.

Uses dependency injection and type annotations for testing.
Recommendation format mirrors Hermes system prompt format.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set

import numpy as np

from .config import PREFIX_QUERY

logger = logging.getLogger(__name__)


# --- Protocols for dependency injection ---

class Embedder(Protocol):
    """Protocol for embedding provider."""
    def embed(self, text: str, prefix: str) -> Optional[np.ndarray]: ...


class SkillIndex(Protocol):
    """Protocol for skill index storage."""
    @property
    def conn(self) -> Any: ...
    def embed(self, text: str, prefix: str) -> Optional[np.ndarray]: ...
    def check_fresh(self, names: List[str]) -> List[str]: ...
    def reindex(self, names: List[str]) -> None: ...


# --- Message helpers ---

def msg_role(msg: Any) -> Optional[str]:
    """Extract role from message (dict or object)."""
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)


def msg_content(msg: Any) -> str:
    """Extract text content from message (handles multimodal content)."""
    c = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for item in c:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return ""


def msg_tool_calls(msg: Any) -> List[Any]:
    """Extract tool calls from message."""
    tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    return tcs or []


def tool_call_name(tc: Any) -> Optional[str]:
    """Extract tool name from tool call."""
    if isinstance(tc, dict):
        return tc.get("name") or (tc.get("function") or {}).get("name")
    return getattr(tc, "name", None)


def tool_call_arg(tc: Any, key: str) -> Optional[Any]:
    """Extract argument from tool call."""
    if isinstance(tc, dict):
        args = tc.get("args") or (tc.get("function") or {}).get("arguments") or {}
    else:
        args = getattr(tc, "args", None) or {}
    if isinstance(args, str):
        try:
            import json
            args = json.loads(args)
        except Exception:
            return None
    return args.get(key) if isinstance(args, dict) else None


# --- Tracking functions ---

# Regex for <available_skills> block (matches both system prompt and our injections)
_AVAILABLE_SKILLS_RE = re.compile(r"<available_skills>(.*?)</available_skills>", re.DOTALL)

# Regex for individual skill entries: "    - skill-name: description" or "    - skill-name"
_SKILL_ENTRY_RE = re.compile(r"^\s*-\s+(\S+?)(?:\s*:.*)?$", re.MULTILINE)


def extract_loaded_skills(history: Sequence[Any]) -> Set[str]:
    """Skills actually loaded into context via skill_view."""
    names: Set[str] = set()
    for msg in history or []:
        for tc in msg_tool_calls(msg):
            if tool_call_name(tc) == "skill_view":
                name = tool_call_arg(tc, "name") or tool_call_arg(tc, "skill_name")
                if name:
                    names.add(str(name))
    return names


def extract_available_skills_from_text(text: str) -> Set[str]:
    """Extract skill names from <available_skills> blocks in text.

    Works for both system prompt blocks and our injected blocks.
    """
    names: Set[str] = set()
    for match in _AVAILABLE_SKILLS_RE.finditer(text):
        block = match.group(1)
        for entry_match in _SKILL_ENTRY_RE.finditer(block):
            name = entry_match.group(1).strip()
            if name:
                names.add(name)
    return names


def extract_injected_skills(history: Sequence[Any]) -> Set[str]:
    """Skills from <available_skills> blocks in conversation history.

    Scans all messages for <available_skills> blocks (both system prompt
    and our injections). This is the unified tracking mechanism.
    """
    names: Set[str] = set()
    for msg in history or []:
        content = msg_content(msg)
        if content:
            names |= extract_available_skills_from_text(content)
    return names


def extract_context_skills(history: Sequence[Any]) -> Set[str]:
    """All skills already in context from ALL sources:

    1. Skills loaded via skill_view (tool calls in history)
    2. Skills in <available_skills> blocks (system prompt + our injections)

    ALL are excluded from recommendations.
    """
    loaded = extract_loaded_skills(history)
    available = extract_injected_skills(history)
    return loaded | available


# --- Query building ---

@dataclass
class QueryConfig:
    """Configuration for query building."""
    assistant_truncate: int = 500


# --- Injected block patterns (stripped from ALL text) ---
# Hermes injects these into user/assistant messages; they are noise for embedding.

_INJECTED_RES: List[re.Pattern[str]] = [
    # <available_skills> blocks (system prompt + plugin injections)
    re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL | re.IGNORECASE),
    # <memory-context> blocks (hindsight prefetch)
    re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL | re.IGNORECASE),
    # [OUT-OF-BAND USER MESSAGE ...] blocks (steer messages)
    re.compile(r"\[OUT-OF-BAND USER MESSAGE.*?\[/OUT-OF-BAND USER MESSAGE\]", re.DOTALL),
    # <system-reminder> blocks
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE),
    # [CONTEXT COMPACTION ...] — entire message is the summary
    re.compile(r"\[CONTEXT COMPACTION[^\]]*\].*", re.DOTALL),
    # [Planned todo_list ...] inline markers
    re.compile(r"\[Planned todo_list[^\]]*\]", re.DOTALL),
    # [tool_schemas ...] inline markers
    re.compile(r"\[tool_schemas[^\]]*\]", re.DOTALL),
    # [Planning state preserved ...]
    re.compile(r"\[Planning state preserved[^\]]*\]", re.DOTALL),
    # [ASYNC DELEGATION ...]
    re.compile(r"\[ASYNC DELEGATION.*?\]", re.DOTALL),
    # [Cronjob Response: ...]
    re.compile(r"\[Cronjob Response:.*?\]", re.DOTALL),
    # <system-prompt-reminder>
    re.compile(r"<system-prompt-reminder>.*?</system-prompt-reminder>", re.DOTALL | re.IGNORECASE),
    # [Local command stdout: ...] / [Command message: ...]
    re.compile(r"\[Local command[^\]]*\]", re.DOTALL),
    re.compile(r"\[Command message[^\]]*\]", re.DOTALL),
    # "Potentially relevant skills" header line (our own build_context header, outside tags)
    re.compile(r"Potentially relevant skills[^\n]*:", re.IGNORECASE),
    # Hindsight prefetch truncation notice (inside memory-context, belt-and-suspenders)
    re.compile(r"\[hindsight memory prefetch output truncated[^\]]*\]", re.DOTALL),
    # [Task notification: ...]
    re.compile(r"\[Task notification[^\]]*\]", re.DOTALL),
    # User attachment references
    re.compile(r"\[Attachment:[^\]]*\]", re.DOTALL),
    re.compile(r"\[File:[^\]]*\]", re.DOTALL),
    re.compile(r"\[Image:[^\]]*\]", re.DOTALL),
    re.compile(r"\[Attached file:[^\]]*\]", re.DOTALL),
    re.compile(r"<file[^\>]*>.*?</file>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<attachment[^\>]*>.*?</attachment>", re.DOTALL | re.IGNORECASE),
]

# Markdown noise patterns (stripped for embedding quality — model sees raw tokens)
_MD_RES: List[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),                                    # **bold**
    (re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)"), r"\1"),            # *italic*
    (re.compile(r"```[a-z]*\n?"), ""),                                       # code fence openers
    (re.compile(r"```"), ""),                                                 # code fence closers
    (re.compile(r"`([^`]+)`"), r"\1"),                                       # `inline code`
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),                           # ## headers
]


def _strip_injected(text: str) -> str:
    """Remove ALL injected blocks and markdown noise from text."""
    if not text:
        return ""
    for pattern in _INJECTED_RES:
        text = pattern.sub("", text)
    for pattern, replacement in _MD_RES:
        text = pattern.sub(replacement, text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _last_assistant_text(history: Sequence[Any], max_len: int = 500) -> str:
    """Extract the last assistant's final text answer.

    Hermes stores reasoning in ``reasoning`` field and tool calls in
    ``tool_calls`` field — ``content`` is always the final text answer.
    Scans history in reverse for the last assistant message with non-empty content.
    """
    for msg in reversed(list(history or [])):
        if msg_role(msg) == "assistant":
            content = msg_content(msg)
            if content and content.strip():
                clean = _strip_injected(content)
                if clean:
                    return clean[:max_len]
    return ""


def _last_previous_user_text(history: Sequence[Any], current: str, max_len: int = 150) -> str:
    """Extract the last previous user question (for follow-up disambiguation).

    All injected blocks are stripped; only the user's actual words remain.
    """
    for msg in reversed(list(history or [])):
        if msg_role(msg) == "user":
            content = msg_content(msg)
            clean = _strip_injected(content)
            if clean and clean != current:
                return clean[:max_len]
    return ""


def build_query(
    history: Sequence[Any],
    user_message: str,
    *,
    config: QueryConfig | None = None,
) -> str:
    """Build text query from conversation history and current message.

    Minimal noise design:
    - user_message is the PRIMARY and DOMINANT query component.
    - ALL injected blocks are stripped: memory-context, OUT-OF-BAND,
      system-reminder, CONTEXT COMPACTION, available_skills, todo_list, etc.
    - Last assistant final text answer is included (no reasoning, no tool calls).
    - Last previous user question is included for follow-up disambiguation.
    - Markdown formatting is stripped for embedding quality.
    """
    cfg = config or QueryConfig()
    parts: List[str] = []

    # 1. Primary: user_message (all injected blocks stripped)
    clean_user = _strip_injected(user_message)

    # 2. Last assistant final text (answer/conclusion only, no reasoning/tools)
    clean_assistant = _last_assistant_text(history, cfg.assistant_truncate)

    # 3. Last previous user question (for follow-ups, injected blocks stripped)
    clean_prev_user = _last_previous_user_text(history, clean_user)

    if clean_user:
        parts.append(clean_user)
    if clean_assistant:
        parts.append(f"Previous answer: {clean_assistant}")
    if clean_prev_user:
        parts.append(f"Previous question: {clean_prev_user}")

    return "\n".join(parts).strip()


# --- Retrieval ---

@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    threshold: float = 0.75


def retrieve(
    index: SkillIndex,
    query: str,
    *,
    exclude_names: Set[str] | None = None,
    config: RetrievalConfig | None = None,
) -> List[Dict[str, Any]]:
    """Vector search for relevant skills."""
    cfg = config or RetrievalConfig()
    exclude = exclude_names or set()
    if not query:
        return []

    vec = index.embed(query, PREFIX_QUERY)
    if vec is None:
        return []

    rows = index.conn.execute(
        "SELECT name, description, category, when_to_use, embedding "
        "FROM skills WHERE embedding IS NOT NULL"
    ).fetchall()

    scored: List[Dict[str, Any]] = []
    for name, desc, cat, when, blob in rows:
        if name in exclude or blob is None:
            continue
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape[0] != vec.shape[0]:
            continue
        score = float(np.dot(v, vec))
        if score >= cfg.threshold:
            scored.append({
                "name": name,
                "description": desc,
                "category": cat or "",
                "when_to_use": when or "",
                "score": score,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:cfg.top_k]


def retrieve_bm25(
    index: SkillIndex,
    query: str,
    *,
    exclude_names: Set[str] | None = None,
    config: RetrievalConfig | None = None,
) -> List[Dict[str, Any]]:
    """BM25 fallback search via FTS5.

    Uses hardcoded FTS5 table name to prevent SQL injection.
    """
    cfg = config or RetrievalConfig()
    exclude = exclude_names or set()
    if not query:
        return []

    words = [w.lower().replace('"', '') for w in query.split() if len(w) > 2][:8]
    words = [w for w in words if w]  # remove empty after stripping
    if len(words) < 2:
        return []

    if not getattr(index, "_fts_available", False):
        return []

    match = " OR ".join(f'"{w}"' for w in words)
    try:
        rows = index.conn.execute(
            "SELECT name, description, category, when_to_use, "
            "bm25(skills_fts) AS rank "
            "FROM skills_fts WHERE skills_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (match, cfg.top_k * 3),
        ).fetchall()
    except Exception as e:
        logger.warning("FTS5 query failed: %s", e)
        return []

    if not rows:
        return []

    scored = [
        {"name": n, "description": d, "category": c or "",
         "when_to_use": w or "", "score": -r}
        for n, d, c, w, r in rows
        if n not in exclude
    ]
    if not scored:
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[0]["score"]
    if top <= 0:
        return []

    scored = [s for s in scored if s["score"] >= top * 0.5]

    filtered = []
    for s in scored:
        text = (s["name"] + " " + s["description"] + " " + s["when_to_use"]).lower()
        hits = sum(1 for w in words if w in text)
        if hits >= 2:
            filtered.append(s)

    return filtered[:cfg.top_k]


# --- Context building (unified format with system prompt) ---

def build_context(
    results: List[Dict[str, Any]],
) -> str:
    """Build context string for injection into user message.

    Uses <available_skills> format matching Hermes system prompt.
    No internal marker — tracking via history scanning.

    Args:
        results: List of skill results with name, description, category

    Returns:
        Context string with <available_skills> block.
    """
    if not results:
        return ""

    # Group by category (like Hermes system prompt)
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        cat = r.get("category") or "uncategorized"
        by_category.setdefault(cat, []).append(r)

    lines: List[str] = []
    lines.append("Potentially relevant skills "
                  "(use skill_view to load full content):")
    lines.append("")
    lines.append("<available_skills>")

    for cat in sorted(by_category.keys()):
        entries = by_category[cat]
        lines.append(f"  {cat}:")
        for r in sorted(entries, key=lambda x: x["name"]):
            desc = (r.get("description") or "").strip()
            if len(desc) > 200:
                desc = desc[:197] + "..."
            if desc:
                lines.append(f"    - {r['name']}: {desc}")
            else:
                lines.append(f"    - {r['name']}")

    lines.append("</available_skills>")

    return "\n".join(lines)
