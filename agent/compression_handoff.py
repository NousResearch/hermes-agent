"""Deterministic handoff seeding for compression-exhausted sessions.

This module is intentionally model-free. It is called after the agent has already
failed with ``compression_exhausted=True``; calling another LLM to summarize at
that point would risk the same context overflow loop. Instead we extract bounded,
high-signal snippets from the failed result and seed the fresh session with a
small reference message plus an assistant acknowledgement. The next real user
message can then continue in a clean session without replaying the bloated
transcript.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_HANDOFF_MAX_CHARS = 10_000
_SECTION_MAX_CHARS = 2_400
_PATH_MAX = 24
_TODO_MAX = 18

_REPEATED_CHAR_RE = re.compile(r"([^\s])\1{80,}")
_PATH_RE = re.compile(
    r"(?<![\w@])(?:~|/(?:Users|home|private|tmp|var|Volumes|opt|etc))"
    r"[^\s\"'`<>)]*"
)
_TODO_RE = re.compile(r"^\s*[-*]\s*\[[ xX>\-]?\]\s+.+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,4}\s+.+")


def _plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _squash(text: str, *, limit: int = _SECTION_MAX_CHARS) -> str:
    text = _REPEATED_CHAR_RE.sub(lambda m: f"{m.group(1)}…[repeated]", text)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not blank:
                collapsed.append("")
            blank = True
            continue
        blank = False
        collapsed.append(stripped)
    text = "\n".join(collapsed).strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.65)].rstrip()
    tail = text[-int(limit * 0.25) :].lstrip()
    return f"{head}\n…[gekürzt: {len(text) - len(head) - len(tail)} Zeichen ausgelassen]…\n{tail}".strip()


def _dedupe_keep_order(items: Iterable[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip().rstrip(".,;:")
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _iter_message_texts(agent_result: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str]]:
    for msg in agent_result.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        text = _plain_text(msg.get("content"))
        if text.strip():
            yield msg, text


def _extract_paths(agent_result: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for _msg, text in _iter_message_texts(agent_result):
        for match in _PATH_RE.findall(text):
            cleaned = match.strip().rstrip(".,;:]")
            if cleaned:
                paths.append(cleaned)
    return _dedupe_keep_order(paths, limit=_PATH_MAX)


def _extract_todos(agent_result: dict[str, Any]) -> list[str]:
    todos: list[str] = []
    for _msg, text in _iter_message_texts(agent_result):
        for line in text.splitlines():
            if _TODO_RE.match(line):
                todos.append(_squash(line, limit=400))
    return _dedupe_keep_order(todos, limit=_TODO_MAX)


def _extract_compressed_summary(agent_result: dict[str, Any]) -> str:
    candidates: list[str] = []
    for msg, text in _iter_message_texts(agent_result):
        if msg.get("_compressed_summary") or "CONTEXT COMPACTION" in text:
            keep: list[str] = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    if keep and keep[-1] != "":
                        keep.append("")
                    continue
                # Keep headings and concise content; drop long raw blocks.
                if _HEADING_RE.match(stripped) or len(stripped) <= 260:
                    keep.append(stripped)
            if keep:
                candidates.append("\n".join(keep))
    return _squash("\n\n".join(candidates), limit=_SECTION_MAX_CHARS)


def _extract_recent_user_intent(agent_result: dict[str, Any]) -> str:
    for msg, text in reversed(list(_iter_message_texts(agent_result))):
        if msg.get("role") == "user":
            return _squash(text, limit=1_800)
    return ""


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_compression_handoff_messages(
    agent_result: dict[str, Any],
    *,
    old_session_id: str,
    new_session_id: str,
    profile: str | None = None,
    max_chars: int = _HANDOFF_MAX_CHARS,
) -> list[dict[str, str]]:
    """Return a bounded user/assistant handoff pair for a fresh session.

    The first message is a *reference* user message seeded into the new
    transcript. The second is an assistant acknowledgement. This yields a clean
    prior-history sequence before the next real user turn: user → assistant →
    user, avoiding provider role-order errors.
    """
    profile_name = (profile or "default").strip() or "default"
    old_link = f"@session:{profile_name}/{old_session_id}" if old_session_id else "unknown"
    error = _squash(str(agent_result.get("error") or "Compression exhausted"), limit=700)
    summary = _extract_compressed_summary(agent_result)
    recent = _extract_recent_user_intent(agent_result)
    todos = _extract_todos(agent_result)
    paths = _extract_paths(agent_result)

    sections: list[str] = [
        "[AUTOMATIC CONTEXT HANDOFF — REFERENCE ONLY]",
        "The previous session exceeded the model context window after repeated compression attempts. Hermes started a fresh visible handoff session instead of sending another oversized model request.",
        "",
        "## Session links",
        f"- Previous session: {old_link}",
        f"- New session: {new_session_id or 'unknown'}",
        "",
        "## What carried over",
        "- Compact working state extracted deterministically from the failed turn",
        "- Open todo/path references found in the surviving context",
        "- The old full transcript remains searchable via the previous-session link",
        "",
        "## Not carried over automatically",
        "- Live terminal/process/UI state",
        "- Huge raw tool outputs or media payloads that caused the overflow",
        "- Any unsaved external side effects",
        "",
        "## Failure reason",
        error,
    ]

    if summary:
        sections.extend(["", "## Compact prior summary", summary])
    if todos:
        sections.extend(["", "## Open todos / active work", _format_bullets(todos)])
    if paths:
        sections.extend(["", "## Relevant file/path references", _format_bullets(paths)])
    if recent:
        sections.extend(["", "## Last visible user intent", recent])

    sections.extend(
        [
            "",
            "## Instruction for the assistant reading this history",
            "Treat this as background context only. Do not answer this handoff message directly; wait for the next real user message and continue from the compact state above.",
        ]
    )

    handoff_text = _squash("\n".join(sections), limit=max_chars)
    return [
        {"role": "user", "content": handoff_text},
        {
            "role": "assistant",
            "content": (
                "Handoff übernommen. Ich habe eine neue Session mit kompaktem "
                "Arbeitsstand gestartet und warte auf die nächste Nutzeranweisung."
            ),
        },
    ]
