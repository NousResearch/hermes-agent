"""Todo-snapshot scaffolding that compaction folds into, and strips from, the trailing user row.

A snapshot block runs from ``TODO_INJECTION_HEADER`` to ``TODO_INJECTION_FOOTER``. The footer is what
lets a strip keep user text that came after the block: the pre-call repair
(``_merge_consecutive_users``) can join the next user turn onto a row that already carries a snapshot,
and todo content may itself hold blank lines, so nothing else marks where the block ends (#137652).
"""

from __future__ import annotations

from typing import Any


def _strip_snapshot_text(text: str) -> str:
    """*text* without its first snapshot block. A block with no footer (written before the footer
    existed) ends the text, as it always did."""
    from tools.todo_tool import TODO_INJECTION_FOOTER, TODO_INJECTION_HEADER
    start = text.find(TODO_INJECTION_HEADER)
    if start == -1:
        return text
    end = text.find(TODO_INJECTION_FOOTER, start)
    head = text[:start].rstrip()
    if end == -1:
        return head
    tail = text[end + len(TODO_INJECTION_FOOTER):].strip()
    return f"{head}\n\n{tail}" if head and tail else head or tail


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.
    A surviving block is stale; stripping before re-injection prevents accumulation across boundaries,
    and any user text merged after the block stays."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    if isinstance(content, str):
        return _strip_snapshot_text(content)
    if isinstance(content, list):
        cleaned = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            if TODO_INJECTION_HEADER not in text:
                cleaned.append(part)
            elif stripped := _strip_snapshot_text(text):
                cleaned.append({**part, "text": stripped})
        return cleaned
    return content


def _todo_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Return whether stripping the snapshot leaves no structured content.
    Text snapshots trail a string; structured ones occupy their own text part, so only an empty remainder
    proves the row was scaffolding alone. Text extraction is deliberately not used: image, audio and other
    non-text parts must survive."""
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False
