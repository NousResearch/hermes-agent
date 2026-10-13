"""Todo-snapshot lifecycle at the compaction boundary.

Compaction folds the live todo list (``TODO_INJECTION_HEADER`` +
``TodoStore.format_for_injection``) into the trailing user row and strips the
stale copy at the next boundary. The strip is block-scoped: it removes only the
injected block — header, task lines, the optional pruned-skill reload notice —
so a user row the pre-call repair welded on AFTER the snapshot (#137652) is
real conversation and survives.
"""

from __future__ import annotations

import re
from typing import Any

# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"

# One rendered todo line (TodoStore.format_for_injection): optional subtask indent, a status
# marker, then "<id>. text". Keys the snapshot-block scan below.
_TODO_SNAPSHOT_ITEM_LINE = re.compile(r"^\s*- \[[>x ~]\] \S+\. ")


def _todo_snapshot_block_end(text: str, header_start: int) -> int:
    """Offset where the todo-snapshot block starting at ``header_start`` ends.

    The block is the injection header, its ``- [>]`` task lines (indented subtasks included) and the
    optional pruned-skill reload notice paragraph. The first line that is none of those — e.g. a user
    row the pre-call repair welded on AFTER the snapshot (#137652) — ends the block and is kept by
    the caller; the strip must never extend past injected scaffolding into real conversation.
    """
    lines = text[header_start:].split("\n")
    offset = header_start + len(lines[0]) + 1
    in_notice = False
    saw_blank = False
    for line in lines[1:]:
        line_start = offset
        offset += len(line) + 1
        stripped = line.strip()
        if not stripped:
            saw_blank = True
            continue
        if not in_notice and _TODO_SNAPSHOT_ITEM_LINE.match(line):
            continue
        if not in_notice and saw_blank and stripped == _PRUNED_SKILL_RELOAD_NOTICE_HEADER:
            in_notice = True
            saw_blank = False
            continue
        if in_notice and not saw_blank:
            continue
        return line_start
    return len(text)


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.
    Snapshots are appended to the trailing user turn, so a surviving header is stale; stripping before
    re-injection prevents accumulation across boundaries. Only the injected block goes: a user row
    merged after the snapshot (#137652) is real conversation and must survive the strip."""
    from tools.todo_tool import TODO_INJECTION_HEADER

    def _stripped_text(text: str) -> str:
        idx = text.find(TODO_INJECTION_HEADER)
        if idx == -1:
            return text
        head, tail = text[:idx], text[_todo_snapshot_block_end(text, idx):]
        if not tail.strip():
            return head.rstrip()
        if not head.strip():
            return tail.lstrip("\n")
        return head.rstrip() + "\n\n" + tail.lstrip("\n")

    if isinstance(content, str):
        return _stripped_text(content)
    if isinstance(content, list):
        cleaned = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            stripped = _stripped_text(text) if text and TODO_INJECTION_HEADER in text else None
            if stripped is None:
                cleaned.append(part)
            elif stripped:
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


def _pruned_skill_reload_notice(compressed: list) -> str:
    """Reload notice for skills whose bodies were pruned, or ``""``.
    Scans ``[SKILL_PRUNED: ...]`` markers in the post-compression transcript; first-seen order, deduplicated,
    capped at ``_MAX_PRUNED_SKILL_MARKERS``."""
    from agent.context_compressor import _MAX_PRUNED_SKILL_MARKERS, _extract_pruned_skill_names
    from agent.conversation_compression import _message_text

    names: list = []
    for message in compressed:
        if not isinstance(message, dict):
            continue
        for name in _extract_pruned_skill_names(_message_text(message)):
            if name not in names:
                names.append(name)
    del names[_MAX_PRUNED_SKILL_MARKERS:]
    if not names:
        return ""
    calls = "; ".join(f"skill_view(name='{name}')" for name in names)
    return (
        f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
        "The task list above crossed the compression boundary verbatim, but "
        "the skill instructions that governed it were pruned. Before "
        f"executing any preserved task that depends on these skills, reload "
        f"them first: {calls}. After reloading, re-check that each pending "
        "task is still justified — findings recorded before the boundary may have invalidated it."
    )
