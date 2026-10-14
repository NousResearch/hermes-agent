"""Bounded tool-result retention and ``{{tool_result:<call_id>.<field>}}`` reference
resolution (#78061): a later tool call can consume a prior tool's structured output
without the model re-emitting the payload.

``remember_tool_result`` keeps raw results keyed by tool call id, bounded three ways:
per-result bytes, entry count, and a total retention budget (oldest evicted first).
``tool_result_reference_hint`` advertises a usable reference path to the model via a
sibling ``_tool_result_hint`` key on the tool message - kept out of ``content`` so the
content stays a parseable JSON string; each wire path folds the hint into the
model-visible text at request-build time (``fold_tool_result_hint``).
``resolve_tool_result_refs`` substitutes references in tool arguments before dispatch.

References resolve against results committed before dispatch: earlier turns, earlier
batches, and earlier sequential calls of the same batch. Concurrent same-batch
siblings are not yet committed, so a reference to one passes through untouched.
A bare ``{{tool_result:<call_id>}}`` (no dot) resolves to the whole stored payload;
list indices follow Python semantics (negative indices included). Unresolvable
references always pass through untouched - never a crash.
"""

import json
import re
from typing import Any

_TOOL_RESULT_REF_RE = re.compile(r"\{\{tool_result:([^{}]+)\}\}")
_MAX_TOOL_RESULT_REF_BYTES = 20 * 1024 * 1024  # per-result cap
_MAX_TOOL_RESULT_REFS = 32  # entry cap
_MAX_TOOL_RESULT_REF_TOTAL_BYTES = 64 * 1024 * 1024  # total budget; oldest evicted first


def remember_tool_result(agent, tool_call_id: str, result: Any) -> None:
    """Keep bounded structured results available for later tool arguments."""
    if not tool_call_id:
        return
    try:
        payload = json.loads(result) if isinstance(result, str) else result
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size > _MAX_TOOL_RESULT_REF_BYTES:
            return
    except (TypeError, ValueError):
        return
    refs = getattr(agent, "_tool_result_refs", None)
    if refs is None:
        refs = agent._tool_result_refs = {}
    refs.pop(tool_call_id, None)  # re-remembering moves the entry to the end (recency)
    refs[tool_call_id] = (payload, size)
    total = sum(entry[1] for entry in refs.values())
    while refs and (len(refs) > _MAX_TOOL_RESULT_REFS or total > _MAX_TOOL_RESULT_REF_TOTAL_BYTES):
        _, dropped_size = refs.pop(next(iter(refs)))
        total -= dropped_size


def tool_result_reference_hint(tool_call_id: str, result: Any) -> str:
    """Return a compact hint for structured results reusable by later tools."""
    if not tool_call_id:
        return ""
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, (dict, list)):
        return ""
    # Reveal a real path so the model can form a working reference. For an
    # MCP wrapper {"result": ..., "structuredContent": ...} both values may
    # be JSON strings (double-encoded) - prefer structuredContent, which is
    # the machine-oriented branch and the only one that can hold a real
    # container; fall back to result otherwise. Show a path that reaches a
    # real container: dive through string leaves (double-encoding) so the
    # advertised path is usable, not one that lands on a JSON blob string.
    if isinstance(payload, dict) and payload:
        top_key = ("structuredContent" if "structuredContent" in payload
                   else next(iter(payload)))
        top_value = payload[top_key]
        if isinstance(top_value, str):
            try:
                top_value = json.loads(top_value)
            except (TypeError, ValueError):
                pass
        # Dive one more level through a double-encoded container so the hint
        # lands on a real structure, e.g. structuredContent.result.folders.
        if isinstance(top_value, (dict, list)) and top_value:
            if isinstance(top_value, dict):
                inner_key = next(iter(top_value))
                inner_value = top_value[inner_key]
            else:
                inner_key = 0
                inner_value = top_value[0]
            if isinstance(inner_value, str):
                try:
                    inner_value = json.loads(inner_value)
                except (TypeError, ValueError):
                    pass
            ref = (f"{{{{tool_result:{tool_call_id}.{top_key}.{inner_key}.field}}}}"
                   if isinstance(inner_value, (dict, list))
                   else f"{{{{tool_result:{tool_call_id}.{top_key}.{inner_key}}}}}")
        elif isinstance(top_value, (dict, list)):
            ref = f"{{{{tool_result:{tool_call_id}.{top_key}.field}}}}"
        else:
            ref = f"{{{{tool_result:{tool_call_id}.{top_key}}}}}"
    elif isinstance(payload, list) and payload:
        # A top-level list: the first index is a resolvable path.
        ref = f"{{{{tool_result:{tool_call_id}.0}}}}"
    else:
        # Empty containers: `.field` is a replace-me placeholder the model swaps
        # for a real leaf name (an empty payload has no resolvable path to show).
        ref = f"{{{{tool_result:{tool_call_id}.field}}}}"
    return f"\n\n[Reusable tool result: {ref}]"


def fold_tool_result_hint(content: Any, hint: Any, *, part_type: str = "text") -> Any:
    """Fold a sibling-key tool-result hint into model-visible content: append to a string,
    or add a text part to a block list. No-op for an empty hint or other content shapes."""
    if not hint:
        return content
    if isinstance(content, str):
        return content + str(hint)
    if isinstance(content, list):
        return [*content, {"type": part_type, "text": str(hint).strip()}]
    return content


def _resolve_tool_result_ref(agent, value: Any) -> Any:
    if not isinstance(value, str) or "{{tool_result:" not in value:
        return value
    refs = getattr(agent, "_tool_result_refs", {})

    def resolve(match):
        call_id, _, path = match.group(1).partition(".")
        if call_id not in refs:
            return match.group(0)
        current = refs[call_id][0]
        for part in path.split(".") if path else []:
            # MCP tool results are double-encoded: the wrapper stores
            # {"result": "<json-string>", "structuredContent": ...}, so a
            # walk may land on a string containing JSON. Re-parse it once
            # so chained paths like .result.data resolve against the real
            # payload; unresolvable refs pass through untouched (same
            # contract as the walk below).
            if isinstance(current, str):
                try:
                    current = json.loads(current)
                except (TypeError, ValueError):
                    return match.group(0)
            try:
                current = current[int(part)] if isinstance(current, list) else current[part]
            except (KeyError, IndexError, TypeError, ValueError):
                return match.group(0)
        return current if isinstance(current, str) else json.dumps(current, ensure_ascii=False)

    if value.startswith("{{tool_result:") and value.endswith("}}"):
        match = _TOOL_RESULT_REF_RE.fullmatch(value)
        return resolve(match) if match else value
    return _TOOL_RESULT_REF_RE.sub(resolve, value)


def resolve_tool_result_refs(agent, value: Any) -> Any:
    """Resolve ``{{tool_result:call_id.field}}`` recursively in tool args."""
    if isinstance(value, dict):
        return {key: resolve_tool_result_refs(agent, item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_tool_result_refs(agent, item) for item in value]
    return _resolve_tool_result_ref(agent, value)
