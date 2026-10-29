"""OpenAI-shape bridge shared by Hermes' ACP clients.

ACP has no OpenAI-style ``tools``/``tool_calls`` channel, so Hermes' tool schemas travel INTO the
prompt as text (:func:`render_tool_bridge_sections`) and calls are parsed back OUT of the response
text (:func:`extract_tool_calls_from_text`). Clients differ only in WHICH tools they forward
(``allowlist``): a CLI with no tools of its own forwards everything; an autonomous agent with its own
read/edit/execute tools forwards only Hermes' agent-level tools, since re-offering overlapping ones
makes Hermes redo finished work.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

TOOL_CALL_OPEN_TAG = "<tool_call>"
TOOL_CALL_CLOSE_TAG = "</tool_call>"

TOOL_CALL_CONTRACT = (
    "Available tools (OpenAI function schema). "
    "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
    "containing id/type/function{name,arguments}. arguments must be a JSON string."
)

__all__ = [
    "TOOL_CALL_CLOSE_TAG",
    "TOOL_CALL_CONTRACT",
    "TOOL_CALL_OPEN_TAG",
    "StreamChunks",
    "build_openai_tool_call",
    "completion_to_stream_chunks",
    "extract_tool_calls_from_text",
    "render_tool_bridge_sections",
    "tool_specs_from_openai_tools",
]


class StreamChunks(list):
    """Chunk list that also carries response-level attributes (e.g. ``hermes_projected_messages``)
    Hermes reads off the ``create`` result; a plain list would drop them on the stream path."""


def completion_to_stream_chunks(completion: SimpleNamespace) -> StreamChunks:
    """Re-shape a one-shot ACP response as OpenAI stream chunks (data chunk + usage chunk); response-level
    attributes other than choices/usage/model are copied onto the result."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = [
            SimpleNamespace(
                index=index, id=getattr(tool_call, "id", None), type=getattr(tool_call, "type", "function"),
                function=SimpleNamespace(name=getattr(tool_call.function, "name", None),
                                         arguments=getattr(tool_call.function, "arguments", None)),
            )
            for index, tool_call in enumerate(message.tool_calls)
        ]
    delta = SimpleNamespace(
        role="assistant", content=message.content or None, tool_calls=tool_call_deltas,
        reasoning_content=getattr(message, "reasoning_content", None), reasoning=getattr(message, "reasoning", None),
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model, usage=None,
    )
    usage_chunk = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
    chunks = StreamChunks([data_chunk, usage_chunk])
    for key, value in vars(completion).items():
        if key not in ("choices", "usage", "model"):
            setattr(chunks, key, value)
    return chunks


def build_openai_tool_call(*, call_id: str, name: str, arguments: str) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id, call_id=call_id, response_item_id=None, type="function",
        function=Function(name=name, arguments=arguments),
    )


def _named_function(container: Any) -> tuple[dict[str, Any], str] | None:
    """``(fn, stripped name)`` from ``container["function"]`` when it is a dict with a non-blank name, else None."""
    fn = container.get("function") if isinstance(container, dict) else None
    name = fn.get("name") if isinstance(fn, dict) else None
    return (fn, name.strip()) if isinstance(name, str) and name.strip() else None


def tool_specs_from_openai_tools(
    tools: list[dict[str, Any]] | None, *, allowlist: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten OpenAI ``tools`` into ``{name, description, parameters}`` specs; malformed entries are skipped."""
    allowed = {str(n).strip() for n in allowlist} if allowlist is not None else None
    specs: list[dict[str, Any]] = []
    for t in tools or []:
        named = _named_function(t)
        if named is None or (allowed is not None and named[1] not in allowed):
            continue
        fn, name = named
        specs.append({"name": name, "description": fn.get("description", ""), "parameters": fn.get("parameters", {})})
    return specs


def render_tool_bridge_sections(
    tools: list[dict[str, Any]] | None, tool_choice: Any = None, *, allowlist: Iterable[str] | None = None,
) -> list[str]:
    """Prompt sections carrying the forwarded tool schemas + choice hint (empty list when neither applies)."""
    specs = tool_specs_from_openai_tools(tools, allowlist=allowlist)
    sections: list[str] = []
    if specs:
        sections.append(TOOL_CALL_CONTRACT + "\n" + json.dumps(specs, ensure_ascii=False))
    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")
    return sections


def _find_matching_brace(text: str, start: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``start``, or -1 when unbalanced.

    String- and escape-aware, so braces inside JSON string values or nested objects don't close the
    object early. This is what the old lazy ``\\{.*?\\}`` regex could not do: it stopped at the first
    ``}`` that happened to be followed by ``</tool_call>``.
    """
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_tool_call_blocks(text: str) -> list[tuple[int, int, str]]:
    """Every ``<tool_call>...</tool_call>`` block → ``(start, end, body_json)`` in text order.

    The body runs from the opening tag's first ``{`` through its balanced ``}``, and the closing tag
    must follow that body. Scanning braces FIRST is what makes a ``</tool_call>`` or ``}`` sitting
    inside a JSON string value harmless: it is skipped as string content, and only the real tag
    after the balanced object terminates the block.
    """
    blocks: list[tuple[int, int, str]] = []
    pos = 0
    while True:
        open_at = text.find(TOOL_CALL_OPEN_TAG, pos)
        if open_at < 0:
            return blocks
        brace_start = open_at + len(TOOL_CALL_OPEN_TAG)
        while brace_start < len(text) and text[brace_start] in " \t\n\r":
            brace_start += 1
        if brace_start >= len(text) or text[brace_start] != "{":
            pos = open_at + 1
            continue
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end < 0:
            pos = open_at + 1
            continue
        close_at = brace_end + 1
        while close_at < len(text) and text[close_at] in " \t\n\r":
            close_at += 1
        if not text.startswith(TOOL_CALL_CLOSE_TAG, close_at):
            pos = open_at + 1
            continue
        span_end = close_at + len(TOOL_CALL_CLOSE_TAG)
        blocks.append((open_at, span_end, text[brace_start:brace_end + 1]))
        pos = span_end


def _looks_like_bare_tool_call(obj: Any) -> bool:
    """True when ``obj`` has the exact OpenAI ``tool_call`` shape the removed fallback regex matched."""
    if not isinstance(obj, dict):
        return False
    call_id = obj.get("id")
    return (
        isinstance(call_id, str)
        and bool(call_id.strip())
        and obj.get("type") == "function"
        and isinstance(obj.get("function"), dict)
    )


def _extract_bare_tool_calls(text: str) -> list[tuple[int, int, str]]:
    """Every brace-balanced JSON object shaped like a tool call → ``(start, end, raw_json)`` in text order."""
    found: list[tuple[int, int, str]] = []
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            return found
        end = _find_matching_brace(text, start)
        if end < 0:
            return found
        raw = text[start:end + 1]
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if _looks_like_bare_tool_call(obj):
            found.append((start, end + 1, raw))
            cursor = end + 1
        else:
            cursor = start + 1


def _parse_tool_call(raw_json: str, ordinal: int) -> ChatCompletionMessageToolCall | None:
    """One ``<tool_call>`` JSON body → tool call, or None when malformed. Missing id → ``acp_call_<ordinal>``."""
    try:
        obj = json.loads(raw_json)
    except Exception:
        return None
    named = _named_function(obj)
    if named is None:
        return None
    fn, fn_name = named
    fn_args = fn.get("arguments", "{}")
    if not isinstance(fn_args, str):
        fn_args = json.dumps(fn_args, ensure_ascii=False)
    call_id = obj.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = f"acp_call_{ordinal}"
    return build_openai_tool_call(call_id=call_id, name=fn_name, arguments=fn_args)


def extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    """Pull ``<tool_call>`` blocks out of an ACP response → ``(tool_calls, cleaned_text)`` with the consumed blocks
    removed so the assistant message doesn't show raw JSON. Bare-JSON fallback runs only when no XML block parsed."""
    if not isinstance(text, str) or not text.strip():
        return [], ""
    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []
    for start, end, body in _extract_tool_call_blocks(text):
        call = _parse_tool_call(body, len(extracted) + 1)
        if call is not None:
            extracted.append(call)
        consumed_spans.append((start, end))
    if not extracted:
        for start, end, body in _extract_bare_tool_calls(text):
            call = _parse_tool_call(body, len(extracted) + 1)
            if call is not None:
                extracted.append(call)
            consumed_spans.append((start, end))
    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])
    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned
