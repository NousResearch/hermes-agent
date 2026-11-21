"""Carry the active /plan file in the handoff without reading backend-owned files."""

import json
import posixpath
import re

from agent.plan_prompt import PLAN_PROMPT_HEADER

PLAN_POINTER_HEADER = "[Active plan from /plan: "
_POINTER_SUFFIX = ". Re-read it before continuing the planned work.]"
_POINTER_LINE = re.compile(
    r"^" + re.escape(PLAN_POINTER_HEADER) + r"([^\r\n]+\.md)" + re.escape(_POINTER_SUFFIX) + r"$",
    re.MULTILINE,
)


def _written_plan_path(call):
    """Resolve a direct or deferred file write using the normal tool-call parser."""
    from agent.context_compressor import _extract_tool_call_name_and_args
    from tools.tool_search_catalog import TOOL_CALL_NAME
    from tools.tool_search_validation import normalize_tool_call_entries

    name, raw_args = _extract_tool_call_name_and_args(call)
    try:
        args = json.loads(raw_args)
    except (ValueError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    entries, error = normalize_tool_call_entries(args) if name == TOOL_CALL_NAME else (
        [{"name": name, "arguments": args}], None
    )
    if error or len(entries) != 1:
        return None
    entry = entries[0]
    path = entry["arguments"].get("path") if entry["name"] == "write_file" else None
    if isinstance(path, str) and re.search(
        r"(?:^|/)\.hermes/plans/[^\r\n]+\.md\Z", posixpath.normpath(path.replace("\\", "/")),
    ):
        return path
    return None


def with_plan_pointer(summary: str, messages: list) -> str:
    """Reattach one pointer, including when a prepared summary omitted the writing turn.

    Only writes within a /plan exchange count; a later ordinary reference write
    must not replace the plan. Existing handoffs keep the pointer across passes.
    """
    from agent.context_compressor import is_compaction_summary_message
    from agent.conversation_compression import _is_real_user_message, _message_text

    path = None
    planning = False
    for row in messages:
        text = _message_text(row)
        if is_compaction_summary_message(row):
            pointers = _POINTER_LINE.findall(text)
            if pointers:
                path = pointers[-1]
        if row.get("role") == "user" and text.startswith(PLAN_PROMPT_HEADER):
            planning = True
        elif _is_real_user_message(row):
            planning = False
        if planning and row.get("role") == "assistant":
            for call in row.get("tool_calls") or []:
                path = _written_plan_path(call) or path
    if path is None:
        return summary
    # The summarizer may have copied an older pointer; replace it deterministically.
    summary = _POINTER_LINE.sub("", summary).rstrip()
    return f"{summary}\n\n{PLAN_POINTER_HEADER}{path}{_POINTER_SUFFIX}"
