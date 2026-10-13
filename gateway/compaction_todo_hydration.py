"""Todo-list hydration for the gateway's detached compaction agents."""

import re
from typing import Any

from tools.todo_tool import _STATUS_MARKERS, TODO_INJECTION_HEADER, TodoStore

# The first line of one rendered TodoStore.format_for_injection item: "<indent>- <marker> <id>. <content>".
# Content may span lines; the item ends at the first line that closes with " (<status>)" for its marker
# and is followed by another item's first line or by the end of the block.
_SNAPSHOT_ITEM = re.compile(r"^( *)- (\[.\]) (.+?)\. (.*)$")
_MARKER_STATUS = {marker: status for status, marker in _STATUS_MARKERS.items()}


def _item_start(line: str) -> Any:
    match = _SNAPSHOT_ITEM.match(line)
    return match if match is not None and match.group(2) in _MARKER_STATUS else None


def _row_text(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        content = "\n".join(str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) else ""


def _is_todo_snapshot_row(msg: Any) -> bool:
    """True for a user row carrying a todo snapshot folded in by an earlier compaction."""
    return isinstance(msg, dict) and msg.get("role") == "user" and TODO_INJECTION_HEADER in _row_text(msg)


def _parse_todo_snapshot(text: str) -> list:
    """Read the items back out of a folded snapshot (the block after the header, up to any reload notice).

    Content may hold blank lines and checkbox-shaped lines, so a line opens an item only after the previous
    item's rendered " (<status>)" closed it. A block that does not re-render to itself returns no items:
    the snapshot is then left as it is rather than restored from a guess."""
    from agent.conversation_compression import _PRUNED_SKILL_RELOAD_NOTICE_HEADER

    block = text[text.find(TODO_INJECTION_HEADER) + len(TODO_INJECTION_HEADER):]
    block = block.split("\n\n" + _PRUNED_SKILL_RELOAD_NOTICE_HEADER, 1)[0].strip("\n")
    items: list = []
    parents: list = []  # parents[depth] = id of the latest item rendered at that depth
    lines: list = []  # content lines of the item still open
    rows = block.split("\n")
    for n, line in enumerate(rows):
        if not lines:
            match = _item_start(line)
            if match is None:
                return []
            depth = len(match.group(1)) // 2
            del parents[depth:]
            item = {"id": match.group(3), "status": _MARKER_STATUS[match.group(2)]}
            if depth and len(parents) == depth:
                item["parent"] = parents[-1]
            parents.append(item["id"])
            line = match.group(4)
        lines.append(line)
        suffix = f" ({item['status']})"
        if line.endswith(suffix) and (n + 1 == len(rows) or _item_start(rows[n + 1]) is not None):
            lines[-1] = line[:-len(suffix)]
            item["content"] = "\n".join(lines)
            items.append(item)
            lines = []
    check = TodoStore()
    try:
        check.restore(items)
    except ValueError:  # an empty content item: not something the renderer emits
        return []
    if lines or check.format_for_injection() != TODO_INJECTION_HEADER + "\n" + block:
        return []
    return items


def hydrate_todos_for_compaction(agent: Any, messages: list) -> Any:
    """Return *agent* after replaying the transcript's latest todo state into its ``TodoStore``.

    Session hygiene and manual ``/compress`` compact with a freshly built agent whose store is empty.
    A live turn hydrates the store from history before it can compact; without the same replay the
    boundary has no task list to re-inject, and the list is lost once its tool rows are summarised.
    Only rows after the newest folded snapshot count: that snapshot is newer than any todo result
    before it (often one kept in the protected head), so replaying an older result would replace it.
    With no todo result after it, the snapshot itself is the latest state: restore its items so the
    boundary re-folds them even when the summariser drops the snapshot row.
    """
    hydrate = getattr(agent, "_hydrate_todo_store", None)
    if callable(hydrate):
        start = next((i + 1 for i in range(len(messages) - 1, -1, -1) if _is_todo_snapshot_row(messages[i])), 0)
        hydrate(messages[start:])
        latest = getattr(agent, "_latest_todo_response", None)
        if start and callable(latest) and latest(messages[start:]) is None:
            items = _parse_todo_snapshot(_row_text(messages[start - 1]))
            if items:
                agent._todo_store.restore(items)
    return agent
