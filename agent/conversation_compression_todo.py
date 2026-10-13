"""Todo-snapshot block handling at the compaction boundary.

``_fold_todo_snapshot`` appends the live todo list (``TODO_INJECTION_HEADER`` plus task lines, and the
pruned-skill reload notice when skills were pruned) to the trailing user row. At the next boundary that block
is stale and is stripped before the fresh one is folded in.
"""

from __future__ import annotations

import re
from typing import Any

# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"
# One rendered todo item opener: ``<indent>- <marker> <id>. `` (see TodoStore.format_for_injection).
_TASK_LINE_RE = re.compile(r"^(?:  )*- (\[.\]) \S")


def _todo_snapshot_task_lines_end(lines: list[str]) -> int:
    """Return how many leading *lines* belong to rendered task items.
    ``TodoStore.format_for_injection`` renders each item as ``<indent>- <marker> <id>. <content> (<status>)``.
    Content may span lines (including blank ones), so an item runs from its marker line to the first line ending
    in the status suffix that matches its marker. Anything that does not open an item ends the task list."""
    from tools.todo_tool import _STATUS_MARKERS
    count = 0
    while count < len(lines):
        match = _TASK_LINE_RE.match(lines[count])
        status = next((s for s, m in _STATUS_MARKERS.items() if m == match.group(1)), None) if match else None
        if status is None:
            return count
        suffix = f" ({status})"
        last = next((i for i in range(count, len(lines)) if lines[i].endswith(suffix)), None)
        if last is None:
            return count
        count = last + 1
    return count


def _cut_todo_snapshot_block(text: str) -> str:
    """Remove the todo-snapshot block from *text*, keeping any text after it.
    The block is ``TODO_INJECTION_HEADER`` plus its rendered task items (whose content may itself contain blank
    lines), optionally followed by the pruned-skill reload paragraph. The pre-call consecutive-user repair can
    join a later user turn after a blank line onto a row carrying a snapshot, and that turn must survive the
    strip (#137652)."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    idx = text.find(TODO_INJECTION_HEADER)
    if idx == -1:
        return text
    head = text[:idx].rstrip()
    lines = text[idx + len(TODO_INJECTION_HEADER):].lstrip("\n").split("\n")
    lines = lines[_todo_snapshot_task_lines_end(lines):]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith(_PRUNED_SKILL_RELOAD_NOTICE_HEADER):
        blank = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
        lines = lines[blank:]
    tail = _cut_todo_snapshot_block("\n".join(lines).strip("\n"))
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
