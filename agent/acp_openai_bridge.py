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
import re
from types import SimpleNamespace
from typing import Any, Iterable

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL
)

TOOL_CALL_CONTRACT = (
    "Available tools (OpenAI function schema). "
    "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
    "containing id/type/function{name,arguments}. arguments must be a JSON string."
)

__all__ = [
    "TOOL_CALL_BLOCK_RE", "TOOL_CALL_JSON_RE", "TOOL_CALL_CONTRACT", "StreamChunks", "build_openai_tool_call",
    "tool_specs_from_openai_tools", "render_tool_bridge_sections", "extract_tool_calls_from_text",
    "completion_to_stream_chunks",
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
    for pattern, group in ((TOOL_CALL_BLOCK_RE, 1), (TOOL_CALL_JSON_RE, 0)):
        for m in pattern.finditer(text):
            call = _parse_tool_call(m.group(group), len(extracted) + 1)
            if call is not None:
                extracted.append(call)
            consumed_spans.append((m.start(), m.end()))
        if extracted:
            break
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
