"""Todo-snapshot block handling at the compaction boundary.

``_fold_todo_snapshot`` appends the live todo list (``TODO_INJECTION_HEADER`` plus task lines, and the
pruned-skill reload notice when skills were pruned) to the trailing user row. At the next boundary that block
is stale and is stripped before the fresh one is folded in.
"""

from __future__ import annotations

from typing import Any

# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"


def _cut_todo_snapshot_block(text: str) -> str:
    """Remove the todo-snapshot block from *text*, keeping any text after it.
    The block is ``TODO_INJECTION_HEADER`` plus its newline-joined task lines, optionally followed by the
    pruned-skill reload paragraph; it ends at the first blank line that starts other text. The pre-call
    consecutive-user repair can join a later user turn after a blank line onto a row carrying a snapshot,
    and that turn must survive the strip (#137652)."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    idx = text.find(TODO_INJECTION_HEADER)
    if idx == -1:
        return text
    head, rest = text[:idx].rstrip(), text[idx + len(TODO_INJECTION_HEADER):]
    end = rest.find("\n\n")
    if end != -1 and rest[end + 2:].startswith(_PRUNED_SKILL_RELOAD_NOTICE_HEADER):
        end = rest.find("\n\n", end + 2)
    tail = _cut_todo_snapshot_block(rest[end:].strip("\n")) if end != -1 else ""
    return f"{head}\n\n{tail}" if head and tail else head or tail


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.
    Snapshots are appended to the trailing user turn, so a surviving header is stale; stripping before
    re-injection prevents accumulation across boundaries. Only the block is removed: user text merged after
    it by the consecutive-user repair is kept."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    if isinstance(content, str):
        if TODO_INJECTION_HEADER not in content:
            return content
        return _cut_todo_snapshot_block(content)
    if isinstance(content, list):
        cleaned = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            if not text or TODO_INJECTION_HEADER not in text:
                cleaned.append(part)
            elif stripped := _cut_todo_snapshot_block(text):
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
