"""The single gate for request-body debug dumps.

Three places write a dump: the preflight hook, the terminal non-retryable-4xx path, and the
retries-exhausted path. Only the preflight one was gated on ``HERMES_DUMP_REQUESTS``; the two
error paths wrote unconditionally, so a failing provider left a full request body — system
prompt, whole conversation, every tool schema — in ``logs_dir`` with no switch to turn it off.

Two properties keep that from coming back:

* the gate lives on the writer (``dump_api_request_debug``), so a call site cannot write an
  ungated dump however it is reached;
* ``diagnostics.request_dump.content`` in ``config.yaml`` decides how much of the body is
  kept. The default, ``redacted``, keeps the shape — roles, tool names, sizes, error type —
  because the shape is what diagnoses a rejected request, and the conversation in it is the
  user's data. Behaviour belongs in config, not in a second ``HERMES_*`` env var.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Optional

from utils import env_var_enabled

DUMP_ENV = "HERMES_DUMP_REQUESTS"
CONTENT_KEY = "diagnostics.request_dump.content"

CONTENT_REDACTED = "redacted"
CONTENT_FULL = "full"

# Body keys that carry request configuration rather than conversation. Anything else is
# summarized by type and size instead of copied, so a text-bearing field added later cannot
# leak by default.
_STRUCTURAL_KEYS = frozenset({
    "model", "max_tokens", "max_output_tokens", "max_completion_tokens", "temperature",
    "top_p", "top_k", "n", "stream", "tool_choice", "parallel_tool_calls", "presence_penalty",
    "frequency_penalty", "seed", "stop_sequences", "reasoning", "reasoning_effort",
    "response_format", "service_tier", "store", "metadata",
})

# Text-bearing keys that carry conversation, not configuration: keep the size, drop the text.
_TEXT_KEYS = frozenset({"input", "system", "prompt", "instructions", "suffix", "context"})


def request_dump_enabled() -> bool:
    """True only when ``HERMES_DUMP_REQUESTS`` is truthy. Unset means no file is written."""
    return env_var_enabled(DUMP_ENV)


def request_dump_content_mode(config: Optional[Mapping[str, Any]] = None) -> str:
    """``full`` or ``redacted``, read from ``diagnostics.request_dump.content``.

    Anything unrecognized falls back to ``redacted``: a typo must not silently turn on
    full-body dumps, so only the exact word ``full`` does that. ``config`` is injectable for
    tests; production reads the live ``config.yaml``.
    """
    if config is None:
        from hermes_cli.config import load_config
        config = load_config() or {}
    section = config.get("diagnostics")
    entry = section.get("request_dump") if isinstance(section, Mapping) else None
    raw = str((entry or {}).get("content") or "") if isinstance(entry, Mapping) else ""
    return CONTENT_FULL if raw.strip().lower() == CONTENT_FULL else CONTENT_REDACTED


def dump_body_for_debug(api_kwargs: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The body to write for one dump, or ``None`` when dumps are off.

    One function decides both questions — whether a dump happens and how much of the body
    survives — so a call site cannot answer either one differently.
    """
    if not request_dump_enabled():
        return None
    body = {k: v for k, v in copy.deepcopy(dict(api_kwargs)).items() if v is not None and k != "timeout"}
    if request_dump_content_mode() != CONTENT_FULL:
        body = redact_request_body(body)
    return body


def _size_of(value: Any) -> str:
    try:
        return f"<{type(value).__name__} {len(json.dumps(value, default=str))} chars>"
    except (TypeError, ValueError):
        return f"<{type(value).__name__}>"


def _summarize_content(content: Any) -> Any:
    if isinstance(content, str):
        return f"<text {len(content)} chars>"
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                parts.append({"type": type(part).__name__})
                continue
            part_type = part.get("type") or ("text" if "text" in part else "unknown")
            entry: dict[str, Any] = {"type": part_type}
            text = part.get("text")
            if isinstance(text, str):
                entry["chars"] = len(text)
            if part_type in {"image_url", "input_image", "image"}:
                entry["image"] = "<image>"
            parts.append(entry)
        return parts
    if content is None:
        return None
    return _size_of(content)


def _summarize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        return {"type": type(message).__name__}
    summary: dict[str, Any] = {"role": message.get("role")}
    if "content" in message:
        summary["content"] = _summarize_content(message.get("content"))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        names: list[Optional[str]] = []
        for call in tool_calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            names.append(function.get("name") if isinstance(function, Mapping) else None)
        summary["tool_calls"] = names
    # Identifiers name the shape of the exchange; they are not conversation text.
    for key in ("name", "tool_call_id"):
        if key in message:
            summary[key] = message.get(key)
    return summary


def _summarize_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        return {"type": type(tool).__name__}
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    parameters = function.get("parameters")
    return {
        "type": tool.get("type", "function"),
        "name": function.get("name"),
        "parameters_chars": len(json.dumps(parameters, default=str)) if parameters else 0,
    }


def redact_request_body(body: Any) -> Any:
    """Strip conversation text from a request body, keeping the shape a diagnosis needs."""
    if not isinstance(body, Mapping):
        return _size_of(body)

    redacted: dict[str, Any] = {}
    for key, value in body.items():
        if key == "messages" and isinstance(value, list):
            redacted["message_count"] = len(value)
            redacted["messages"] = [_summarize_message(message) for message in value]
        elif key == "tools" and isinstance(value, list):
            redacted["tool_count"] = len(value)
            redacted["tools"] = [_summarize_tool(tool) for tool in value]
        elif key in _TEXT_KEYS:
            redacted[key] = _size_of(value)
        elif key in _STRUCTURAL_KEYS or isinstance(value, (bool, int, float)) or value is None:
            redacted[key] = value
        else:
            redacted[key] = _size_of(value)
    return redacted
