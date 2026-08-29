"""OpenAI-shape bridge shared by Hermes' ACP clients.

An ACP agent (``copilot --acp``, and the ACP CLIs that reach Hermes as
providers) speaks the Agent Client Protocol, which has no OpenAI-style
``tools``/``tool_calls`` channel: a prompt is text, and a response is text plus
the agent's *own* tool notifications. Hermes' agentic surface — ``memory``,
``todo``, ``skill_manage`` and friends — is dispatched from OpenAI-shaped
``tool_calls``, so on an ACP provider it can only work if the schemas travel
*into* the prompt as text and the calls are parsed back *out* of the response
text.

``agent/copilot_acp_client.py`` already carried a private copy of that bridge.
This module is that code, lifted verbatim into one place so every ACP client
shares it instead of re-deriving the wire contract:

* :func:`render_tool_bridge_sections` — prompt sections describing the
  forwarded tools and the ``<tool_call>{...}</tool_call>`` contract.
* :func:`extract_tool_calls_from_text` — parse those blocks back into
  ``ChatCompletionMessageToolCall`` objects and return the response text with
  the blocks stripped.
* :func:`completion_to_stream_chunks` — re-shape a one-shot ACP response as
  OpenAI stream chunks for callers that asked for ``stream=True`` (an ACP turn
  is inherently one-shot from Hermes' perspective).

The one axis clients differ on is *which* tools they forward, so
``render_tool_bridge_sections`` takes an optional allowlist. A CLI with no tools
of its own (Copilot) forwards everything Hermes offers; a CLI that is an
autonomous agent with its own read/edit/execute tools must forward only Hermes'
agent-level tools, because re-offering the overlapping ones makes Hermes re-run
work the agent already finished.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Iterable

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)

# The contract sentence shared by every ACP client: how to emit a call.
TOOL_CALL_CONTRACT = (
    "Available tools (OpenAI function schema). "
    "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
    "containing id/type/function{name,arguments}. arguments must be a JSON string."
)

__all__ = [
    "TOOL_CALL_BLOCK_RE",
    "TOOL_CALL_JSON_RE",
    "TOOL_CALL_CONTRACT",
    "StreamChunks",
    "build_openai_tool_call",
    "tool_specs_from_openai_tools",
    "render_tool_bridge_sections",
    "extract_tool_calls_from_text",
    "completion_to_stream_chunks",
]


class StreamChunks(list):
    """Stream chunks that can still carry response-level attributes.

    Hermes reads provider-level extras off the object returned by
    ``chat.completions.create`` (e.g. ``hermes_projected_messages``, consumed by
    ``agent/provider_projection.py``). A plain list of chunks would silently drop
    them on the ``stream=True`` path, so ACP clients return this instead and copy
    the extras onto it.
    """


def completion_to_stream_chunks(completion: SimpleNamespace) -> StreamChunks:
    """Convert a one-shot ACP response into OpenAI-style stream chunks.

    Response-level attributes other than ``choices``/``usage``/``model`` are
    copied onto the returned object so nothing a caller reads off the completion
    is lost when it asked to stream.
    """
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=getattr(message, "reasoning_content", None),
        reasoning=getattr(message, "reasoning", None),
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    chunks = StreamChunks([data_chunk, usage_chunk])
    for key, value in vars(completion).items():
        if key not in ("choices", "usage", "model"):
            setattr(chunks, key, value)
    return chunks


def build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def tool_specs_from_openai_tools(
    tools: list[dict[str, Any]] | None,
    *,
    allowlist: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Flatten OpenAI ``tools`` into ``{name, description, parameters}`` specs.

    Malformed entries are skipped. When ``allowlist`` is given, only tools whose
    name is in it survive — that is how a client forwards just Hermes'
    agent-level tools instead of the whole toolset.
    """
    allowed = {str(n).strip() for n in allowlist} if allowlist is not None else None
    specs: list[dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if allowed is not None and name not in allowed:
            continue
        specs.append(
            {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return specs


def render_tool_bridge_sections(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
    *,
    allowlist: Iterable[str] | None = None,
) -> list[str]:
    """Prompt sections that carry the forwarded tool schemas + choice hint.

    Returns an empty list when no tool survives filtering and no choice hint was
    requested, so callers can splice the result into their section list
    unconditionally.
    """
    specs = tool_specs_from_openai_tools(tools, allowlist=allowlist)
    sections: list[str] = []
    if specs:
        sections.append(
            TOOL_CALL_CONTRACT + "\n" + json.dumps(specs, ensure_ascii=False)
        )
    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")
    return sections


def extract_tool_calls_from_text(
    text: str,
) -> tuple[list[ChatCompletionMessageToolCall], str]:
    """Pull ``<tool_call>`` blocks out of an ACP response.

    Returns ``(tool_calls, cleaned_text)`` where ``cleaned_text`` is the
    response with the consumed blocks removed, so the assistant message doesn't
    show raw JSON to the user.
    """
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

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
