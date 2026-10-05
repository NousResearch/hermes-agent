"""Backend-independent active plan pointer at the compaction boundary."""

import json
import posixpath
import re

from agent.plan_prompt import PLAN_PROMPT_HEADER

PLAN_POINTER_HEADER = "[Active plan from /plan: "
_POINTER_SUFFIX = ". Re-read it before continuing the planned work.]"
_POINTER_LINE = re.compile(
    r"^" + re.escape(PLAN_POINTER_HEADER) + r"(.+\.md)" + re.escape(_POINTER_SUFFIX) + r"$",
    re.MULTILINE,
)
_POINTER_PARAGRAPH = re.compile(
    r"(?:\n\n" + _POINTER_LINE.pattern + r"|" + _POINTER_LINE.pattern + r"(?:\n\n)?)", re.MULTILINE,
)


def _strip_plan_pointer(content):
    """Remove only pointer paragraphs and their single insertion separator."""
    if isinstance(content, str):
        return _POINTER_PARAGRAPH.sub("", content)
    if not isinstance(content, list):
        return content
    cleaned = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            stripped = _strip_plan_pointer(text)
            if stripped != text:
                if stripped:
                    cleaned.append({**part, "text": stripped})
                continue
        cleaned.append(part)
    return cleaned


def _plan_write_path(messages):
    """Newest write in the latest /plan exchange, bounded by real user input."""
    from agent.context_compressor import _extract_tool_call_name_and_args
    from agent.conversation_compression import _is_real_user_message, _message_text
    from tools.tool_search_catalog import TOOL_CALL_NAME
    from tools.tool_search_validation import normalize_tool_call_entries

    pointers = [(i, path) for i, row in enumerate(messages) if row.get("role") == "user"
                for path in _POINTER_LINE.findall(_message_text(row))]
    pointer_index, path = pointers[-1] if pointers else (-1, None)
    plan_index = next((i for i in range(len(messages) - 1, -1, -1)
                       if messages[i].get("role") == "user"
                       and _message_text(messages[i]).startswith(PLAN_PROMPT_HEADER)), None)
    if plan_index is None:
        return path
    end = next((i for i in range(plan_index + 1, len(messages)) if _is_real_user_message(messages[i])), len(messages))
    calls = (call for row in reversed(messages[max(plan_index, pointer_index) + 1:end])
             if row.get("role") == "assistant" for call in reversed(row.get("tool_calls") or []))
    for call in calls:
        name, raw_args = _extract_tool_call_name_and_args(call)
        try:
            args = json.loads(raw_args)
        except (ValueError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        entries, error = normalize_tool_call_entries(args) if name == TOOL_CALL_NAME else (
            [{"name": name, "arguments": args}], None
        )
        if error or len(entries) != 1:
            continue
        entry = entries[0]
        candidate = entry["arguments"].get("path") if entry["name"] == "write_file" else None
        if isinstance(candidate, str) and re.search(
            r"(?:^|/)\.hermes/plans/[^\r\n]+\.md\Z", posixpath.normpath(candidate.replace("\\", "/"))
        ):
            return candidate
    return path


def _insert_plan_pointer(content, pointer):
    """Insert one paragraph ahead of todo scaffolding without trimming user text."""
    from agent.context_compressor import _append_text_to_content
    from tools.todo_tool import TODO_INJECTION_HEADER

    if isinstance(content, str):
        before, marker, after = content.partition(TODO_INJECTION_HEADER)
        if marker:
            separator = "\n\n" if before and not before.endswith("\n\n") else ""
            return before + separator + pointer + "\n\n" + marker + after
        return content + ("\n\n" if content else "") + pointer
    for i, part in enumerate(content):
        if isinstance(part, dict) and part.get("type") == "text" and TODO_INJECTION_HEADER in part.get("text", ""):
            return [*content[:i], {**part, "text": _insert_plan_pointer(part["text"], pointer)}, *content[i + 1:]]
    return _append_text_to_content(content, pointer)


def _fold_plan_pointer(agent, messages: list, compressed: list) -> None:
    """Refresh one pointer from pre-compaction history without reading the file."""
    from agent.conversation_compression import _message_text, _replace_message_content

    fallback = [path for row in compressed if row.get("role") == "user"
                for path in _POINTER_LINE.findall(_message_text(row))]
    path = _plan_write_path(messages) or (fallback[-1] if fallback else None)
    if path is None:
        return
    removed = False
    for i in range(len(compressed) - 1, -1, -1):
        row = compressed[i]
        if row.get("role") != "user":
            continue
        content = row.get("content")
        cleaned = _strip_plan_pointer(content)
        if cleaned == content:
            continue
        if not cleaned:
            compressed.pop(i)
            removed = True
        else:
            _replace_message_content(row, cleaned)
            row.pop("_plan_pointer_synthetic", None)
    if removed:
        agent._repair_message_sequence(compressed)
    pointer = f"{PLAN_POINTER_HEADER}{path}{_POINTER_SUFFIX}"
    if compressed and compressed[-1].get("role") == "user":
        tail = compressed[-1]
        content = tail.get("content") or ""
        _replace_message_content(tail, _insert_plan_pointer(content, pointer))
        if not content:
            tail["_plan_pointer_synthetic"] = True
    else:
        compressed.append({"role": "user", "content": pointer, "_plan_pointer_synthetic": True})
