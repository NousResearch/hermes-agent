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
# Length-prefixed item opener from TodoStore.format_for_injection:
# ``<indent>- <marker> <id>. #<N>:<content> (<status>)`` — N is len(content) in characters.
# No ^ anchor: Pattern.match(string, pos) still only lets ^ match at index 0,
# so later items would never match. Line-start is enforced by the caller.
_TASK_ITEM_OPEN_RE = re.compile(r"(?:  )*- (\[.\]) (\S+)\. #(\d+):")


def _consume_todo_snapshot_tasks(text: str) -> str:
    """Consume complete length-prefixed rendered task items from the start of *text*.

    ``TodoStore.format_for_injection`` renders each item as
    ``<indent>- <marker> <id>. #<N>:<content> (<status>)`` where ``content`` is exactly N characters
    (newlines and status-looking suffixes allowed). Malformed or truncated items are not consumed:
    the remainder starts at the failed attempt so surrounding user text stays intact.
    """
    from tools.todo_tool import _STATUS_MARKERS

    marker_to_status = {marker: status for status, marker in _STATUS_MARKERS.items()}
    pos = 0
    while pos < len(text):
        if pos > 0 and text[pos - 1] != "\n":
            break
        match = _TASK_ITEM_OPEN_RE.match(text, pos)
        if not match:
            break
        status = marker_to_status.get(match.group(1))
        if status is None:
            break
        n = int(match.group(3))
        content_start = match.end()
        content_end = content_start + n
        if content_end > len(text):
            break
        expected_suffix = f" ({status})"
        suffix_end = content_end + len(expected_suffix)
        if text[content_end:suffix_end] != expected_suffix:
            break
        pos = suffix_end
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    return text[pos:]


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
    remainder = text[idx + len(TODO_INJECTION_HEADER):].lstrip("\n")
    remainder = _consume_todo_snapshot_tasks(remainder)
    lines = remainder.split("\n") if remainder else []
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
