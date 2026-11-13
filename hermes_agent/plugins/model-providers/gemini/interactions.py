"""Gemini Interactions API wire (``POST {base}/interactions``), translated to and from the OpenAI chat shape.

Stateless by default (``store: false``): every request carries the whole conversation as typed steps, the
same contract as ``generateContent``, so compaction, branching and model switches keep working and nothing
is retained server-side. Thought signatures live in dedicated ``thought`` steps; they ride back to Hermes
in ``extra_content.google.thought_signature`` (the slot the generateContent path already uses) and are
replayed as ``thought`` steps ahead of the output they preceded.

Long decode (a total output budget above one request's cap) is the one stored path: Google continues a
``requires_continuation`` hop only from a stored interaction (``previous_interaction_id`` +
``continuation_token``), so those requests set ``store: true``.

Spec: https://aistudio.google.com/api-reference/interactions.openapi.json
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Iterator, Optional

from hermes_agent.agent.gemini_native_adapter import (
    _FREE_TIER_GUIDANCE,
    _SKIP_SIGNATURE,
    GeminiAPIError,
    _response_text,
    gemini_http_error,
    _coerce_content_to_text,
    _effective_gemini_max_output_tokens,
    _envelope,
    _function_declaration,
    _looks_like_json_schema,
    _make_stream_chunk,
    _normalize_thinking_config,
    _text_of,
    _tool_call_extra_from_part,
    _tool_call_extra_signature,
    _tool_call_id,
    _tool_call_ns,
)
from hermes_agent.agent.gemini_schema import prepare_gemini_tool_parameters

logger = logging.getLogger(__name__)

# One request's output cap (streamed hops get 64K; unary 32K). A budget above it needs continuation hops.
SINGLE_REQUEST_OUTPUT_CAP = 65_536
_THINKING_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}
_TOOL_CHOICES = {"auto": "auto", "required": "any", "none": "none"}
_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif", "image/gif", "image/bmp", "image/tiff"}
# Interaction.status -> OpenAI finish_reason (requires_continuation is consumed by the hop loop).
_STATUS_FINISH = {
    "completed": "stop", "requires_action": "stop", "incomplete": "length", "budget_exceeded": "length",
    "requires_continuation": "length", "cancelled": "stop",
}
# google.rpc status names an ErrorEvent / failed interaction may carry -> the HTTP status they stand for,
# so hermes_agent.agent.error_classifier treats an in-band 429 like a wire 429.
# Interactions spells them snake_case (live: ``too_many_requests``, ``service_unavailable``, ``invalid_request``).
_RPC_HTTP_STATUS = {
    "INVALID_ARGUMENT": 400, "INVALID_REQUEST": 400, "FAILED_PRECONDITION": 400, "OUT_OF_RANGE": 400,
    "UNAUTHENTICATED": 401, "PERMISSION_DENIED": 403, "NOT_FOUND": 404, "RESOURCE_EXHAUSTED": 429,
    "TOO_MANY_REQUESTS": 429, "CANCELLED": 499, "INTERNAL": 500, "UNKNOWN": 500, "UNAVAILABLE": 503,
    "SERVICE_UNAVAILABLE": 503, "DEADLINE_EXCEEDED": 504,
}
_HTTP_ERROR_CODES = {401: "gemini_unauthorized", 429: "gemini_rate_limited", 404: "gemini_model_not_found"}


# ── OpenAI messages -> Interactions steps ────────────────────────────────────
def _image_content(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    """``image`` content for an ``image_url`` item carrying a ``data:`` URL; None otherwise (remote URLs
    are dropped, as on the generateContent path)."""
    url = (item.get("image_url") or {}).get("url") or ""
    if item.get("type") != "image_url" or not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        header, encoded = url.split(",", 1)
        mime = header.split(":", 1)[1].split(";", 1)[0].strip().lower()
        data = base64.b64encode(base64.b64decode(encoded)).decode("ascii")
    except (ValueError, IndexError):  # malformed data URL (binascii.Error is a ValueError)
        return None
    return {"type": "image", "mime_type": mime, "data": data} if mime in _IMAGE_MIME_TYPES else None


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        text = _coerce_content_to_text(content)
        return [{"type": "text", "text": text}] if text else []
    blocks: list[dict[str, Any]] = []
    for item in content:
        text = _text_of(item)
        if text:
            blocks.append({"type": "text", "text": text})
        elif isinstance(item, dict) and (image := _image_content(item)):
            blocks.append(image)
    return blocks


def _call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) and raw else (raw if isinstance(raw, dict) else {})
    except json.JSONDecodeError:
        args = {"_raw": raw}
    return args if isinstance(args, dict) else {"_value": args}


def _function_result(message: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    call_id = str(message.get("tool_call_id") or "")
    raw = message.get("content")
    text = _coerce_content_to_text(raw)
    try:
        parsed = json.loads(text) if text.strip().startswith(("{", "[")) else None
    except json.JSONDecodeError:
        parsed = None
    images = [b for b in _content_blocks(raw) if b["type"] == "image"] if isinstance(raw, list) else []
    if images:
        result: Any = [*([{"type": "text", "text": text}] if text else []), *images]
    elif isinstance(parsed, dict) and not _looks_like_json_schema(parsed):
        # Same rule as generateContent: a result that is itself a JSON Schema ($ref pointers) goes as text.
        result = parsed
    else:
        result = text
    step = {"type": "function_result", "call_id": call_id, "result": result}
    if name := names.get(call_id) or message.get("name"):
        step["name"] = str(name)
    return step


def _assistant_steps(message: dict[str, Any], names: dict[str, str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    replayed: set[str] = set()

    def thought(sig: Optional[str]) -> None:
        if sig and sig not in replayed:
            replayed.add(sig)
            steps.append({"type": "thought", "signature": sig})

    tool_calls = [tc for tc in (message.get("tool_calls") or []) if isinstance(tc, dict)]
    text_blocks = [b for b in _content_blocks(message.get("content")) if b["type"] == "text"]
    if text_blocks:
        thought(_tool_call_extra_signature(message))
        steps.append({"type": "model_output", "content": text_blocks})
    for tool_call in tool_calls:
        name = str((tool_call.get("function") or {}).get("name") or "")
        call_id = _tool_call_id(tool_call) or f"call_{uuid.uuid4().hex[:12]}"
        names[call_id] = name
        thought(_tool_call_extra_signature(tool_call))
        steps.append({"type": "function_call", "id": call_id, "name": name, "arguments": _call_arguments(tool_call)})
    if not text_blocks and not tool_calls:
        thought(_tool_call_extra_signature(message))
    # A turn that calls tools must open with a thought step or the request is rejected (HTTP 400). One
    # from another provider has no signature to replay, so it opens with the placeholder Google accepts
    # in its place, as on generateContent.
    if tool_calls and steps[0]["type"] != "thought":
        steps.insert(0, {"type": "thought", "signature": _SKIP_SIGNATURE})
    return steps


def build_input_steps(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """``(steps, system_instruction)`` for a stateless request: the whole history as typed steps."""
    system: list[str] = []
    steps: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        if role in ("system", "developer"):
            system.append(_coerce_content_to_text(message.get("content")))
        elif role in ("tool", "function"):
            steps.append(_function_result(message, names))
        elif role == "assistant":
            steps.extend(_assistant_steps(message, names))
        elif blocks := _content_blocks(message.get("content")):
            if steps and steps[-1]["type"] == "user_input":
                steps[-1]["content"].extend(blocks)  # adjacent user turns fold into one input step
            else:
                steps.append({"type": "user_input", "content": blocks})
    joined = "\n".join(part for part in system if part).strip()
    return steps, joined or None


def _tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for tool in tools if isinstance(tools, list) else []:
        if decl := _function_declaration(tool, json_schema=True):
            entry = {"type": "function", "name": decl["name"]}
            if "description" in decl:
                entry["description"] = decl["description"]
            if "parametersJsonSchema" in decl:
                entry["parameters"] = decl["parametersJsonSchema"]
            out.append(entry)
    return out


def _tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return _TOOL_CHOICES.get(tool_choice)
    name = (tool_choice.get("function") or {}).get("name") if isinstance(tool_choice, dict) else None
    return {"allowed_tools": {"mode": "any", "tools": [name]}} if isinstance(name, str) and name else None


def _response_format(response_format: Any) -> Optional[dict[str, Any]]:
    if not isinstance(response_format, dict) or response_format.get("type") not in ("json_object", "json_schema"):
        return None
    spec = response_format.get("json_schema") if response_format.get("type") == "json_schema" else None
    schema = spec.get("schema") if isinstance(spec, dict) else None
    out = {"type": "text", "mime_type": "application/json"}
    return {**out, "schema": prepare_gemini_tool_parameters(schema)} if isinstance(schema, dict) else out


def build_interactions_request(
    *, model: str, messages: list[dict[str, Any]], tools: Any = None, tool_choice: Any = None,
    temperature: Optional[float] = None, max_tokens: Optional[int] = None, top_p: Optional[float] = None,
    stop: Any = None, thinking_config: Any = None, response_format: Any = None,
) -> dict[str, Any]:
    steps, system_instruction = build_input_steps(messages)
    thinking = _normalize_thinking_config(thinking_config) or {}
    generation: dict[str, Any] = {}
    for key, value in (("temperature", temperature), ("top_p", top_p)):
        if value is not None:
            generation[key] = value
    generation["max_output_tokens"] = _effective_gemini_max_output_tokens(max_tokens, thinking_config)
    if stop:
        generation["stop_sequences"] = stop if isinstance(stop, list) else [str(stop)]
    if thinking.get("thinkingLevel") in _THINKING_LEVELS:
        generation["thinking_level"] = thinking["thinkingLevel"]
    if isinstance(thinking.get("includeThoughts"), bool):
        generation["thinking_summaries"] = "auto" if thinking["includeThoughts"] else "none"
    interaction_tools = _tools(tools)
    choice = _tool_choice(tool_choice) if interaction_tools else None
    if choice is not None:
        generation["tool_choice"] = choice
    request: dict[str, Any] = {"model": model, "input": steps, "store": False}
    if system_instruction:
        request["system_instruction"] = system_instruction
    if interaction_tools:
        request["tools"] = interaction_tools
    fmt = _response_format(response_format)
    # Forced function calling and JSON output conflict on Gemini; the tools win (as on generateContent).
    if fmt and not (choice == "any" or isinstance(choice, dict)):
        request["response_format"] = fmt
    request["generation_config"] = generation
    if generation["max_output_tokens"] > SINGLE_REQUEST_OUTPUT_CAP:
        request["store"] = True  # continuation hops resume only from a stored interaction
    return request


def continuation_request(request: dict[str, Any], interaction_id: str, token: str) -> dict[str, Any]:
    """The next long-decode hop: the same request continued from the stored hop."""
    return {**request, "previous_interaction_id": interaction_id, "continuation_token": token}


# ── Interaction -> OpenAI response ───────────────────────────────────────────
def _count(usage: dict[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def usage_namespace(usage: Any) -> SimpleNamespace:
    """Interactions ``Usage`` -> OpenAI usage. ``total_output_tokens`` excludes thoughts (the spec example
    reports 500 output + 200 thought = 750 total with 50 input), so thoughts fold into completion tokens
    as on the generateContent path. ``initial_*`` (the first hop's prompt, i.e. the context size) is set
    only for a merged multi-hop usage, as on generateContent: Google reports ``initial_input_tokens`` on
    every response, and passing that through alone would make the context buckets read as uncached."""
    usage = usage if isinstance(usage, dict) else {}
    thoughts = _count(usage, "total_thought_tokens")
    extra = ({"initial_prompt_tokens": _count(usage, "initial_input_tokens"), "initial_cached_tokens": _count(usage, _INITIAL_CACHED)}
             if _INITIAL_CACHED in usage else {})
    return SimpleNamespace(
        prompt_tokens=_count(usage, "total_input_tokens"), completion_tokens=_count(usage, "total_output_tokens") + thoughts,
        total_tokens=_count(usage, "total_tokens"), prompt_tokens_details=SimpleNamespace(cached_tokens=_count(usage, "total_cached_tokens")),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=thoughts), **extra,
    )


_INITIAL_CACHED = "initial_cached_tokens"  # set by merge_usage only: marks a multi-hop usage


def merge_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum every hop's counts (billing); ``initial_*`` is hop 1's input and its cached part (context size)."""
    keys = ("total_input_tokens", "total_cached_tokens", "total_output_tokens", "total_tool_use_tokens",
            "total_thought_tokens", "total_tokens")
    merged = {key: sum(_count(u, key) for u in usages) for key in keys if any(key in u for u in usages)}
    if usages:
        merged["initial_input_tokens"] = _count(usages[0], "total_input_tokens")
        merged[_INITIAL_CACHED] = _count(usages[0], "total_cached_tokens")
    return merged


def interaction_error(errors: Any, *, fallback: str = "Gemini interaction failed") -> GeminiAPIError:
    first = next((e for e in errors if isinstance(e, dict)), {}) if isinstance(errors, list) else (errors if isinstance(errors, dict) else {})
    rpc = str(first.get("code") or "").strip()
    message = str(first.get("message") or "").strip() or fallback
    status = _RPC_HTTP_STATUS.get(rpc.upper()) or (int(rpc) if rpc.isdigit() else None)
    return GeminiAPIError(
        f"Gemini interaction error ({rpc or 'error'}): {message}",
        code=_HTTP_ERROR_CODES.get(status or 0, f"gemini_http_{status}" if status else "gemini_interaction_failed"),
        status_code=status, details={"status": rpc, "message": message},
    )


def http_error(response: Any, *, body_text: Optional[str], api_key: str, base_url: str) -> GeminiAPIError:
    """An Interactions HTTP error. Its body is ``{"error": {"code": "too_many_requests", "message": ...}}``
    (a snake_case string code, no ``status``), and some front-end rejections come as a one-element list of the
    generateContent body; both are normalized so ``gemini_http_error`` keeps its status mapping, Retry-After
    and key guidance. Interactions words free-tier exhaustion "on Free Tier", which the generateContent check
    (``free_tier``) misses, so that guidance is added here."""
    text = (_response_text(response) if body_text is None else body_text) or ""
    try:
        parsed = json.loads(text) if text else None
    except ValueError:
        parsed = None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict) and not err.get("status") and isinstance(err.get("code"), str):
        parsed = {"error": {**err, "status": err["code"]}}
    if isinstance(parsed, dict):
        text = json.dumps(parsed)
    error = gemini_http_error(response, body_text=text, api_key=api_key, base_url=base_url)
    message = str((err or {}).get("message") or "") if isinstance(err, dict) else ""
    if error.status_code == 429 and "free tier" in message.lower() and _FREE_TIER_GUIDANCE not in str(error):
        return GeminiAPIError(str(error) + _FREE_TIER_GUIDANCE, code=error.code, status_code=error.status_code,
                              response=error.response, retry_after=error.retry_after, details=error.details)
    return error


def _thought_text(step: dict[str, Any]) -> str:
    return "".join(c.get("text") or "" for c in step.get("summary") or [] if isinstance(c, dict) and c.get("type", "text") == "text")


def translate_interaction(interactions: list[dict[str, Any]], model: str) -> SimpleNamespace:
    """One chat.completion from one interaction, or from every hop of a long decode (in order)."""
    last = interactions[-1]
    if str(last.get("status") or "") == "failed":
        raise interaction_error(last.get("errors"))
    text: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[SimpleNamespace] = []
    signature: Optional[str] = None  # the latest thought's signature: what the next output or call rides
    text_signature: Optional[str] = None  # the signature that preceded the text, kept apart from a later call's
    for interaction in interactions:
        for step in interaction.get("steps") or []:
            kind = step.get("type") if isinstance(step, dict) else None
            if kind == "thought":
                reasoning.append(_thought_text(step))
                signature = step.get("signature") if isinstance(step.get("signature"), str) and step.get("signature") else signature
            elif kind == "model_output":
                pieces = [c.get("text") or "" for c in step.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
                text.extend(pieces)
                text_signature = (signature or text_signature) if any(pieces) else text_signature
            elif kind == "function_call" and step.get("name"):
                arguments = json.dumps(step.get("arguments") or {}, ensure_ascii=False)
                call_id = str(step.get("id") or f"call_{uuid.uuid4().hex[:12]}")
                extra = _tool_call_extra_from_part({"thoughtSignature": signature})
                tool_calls.append(_tool_call_ns(str(step["name"]), arguments, len(tool_calls), call_id, extra))
    status = str(last.get("status") or "completed")
    finish = "tool_calls" if tool_calls else _STATUS_FINISH.get(status, "stop")
    usages = [i.get("usage") or {} for i in interactions]
    usage = usage_namespace(merge_usage(usages) if len(usages) > 1 else usages[0])
    joined_reasoning = "".join(reasoning) or None
    message = SimpleNamespace(
        role="assistant", content="".join(text), tool_calls=tool_calls or None, reasoning=joined_reasoning,
        reasoning_content=joined_reasoning, reasoning_details=None,
        extra_content=_tool_call_extra_from_part({"thoughtSignature": text_signature if any(text) else (None if tool_calls else signature)}),
    )
    return _envelope(model, "chat.completion", SimpleNamespace(index=0, message=message, finish_reason=finish), usage)


# ── Streaming ────────────────────────────────────────────────────────────────
class StreamState:
    """Per-turn stream state, shared across long-decode hops (they form one model turn)."""

    def __init__(self) -> None:
        self.steps: dict[int, dict[str, Any]] = {}  # step index -> {"type", tool slot fields}
        self.tool_calls = 0
        self.signature: Optional[str] = None  # the latest thought's signature
        self.output_signature: Optional[str] = None  # the one already sent as the text's (message-level) signature
        self.interaction_id: Optional[str] = None
        self.status: Optional[str] = None
        self.continuation_token: Optional[str] = None
        self.hop_usage: list[dict[str, Any]] = []
        self.offset = 0  # step indexes restart per hop
        self.hop_done = False  # this hop delivered ``interaction.completed``
        self.completed_hops = 0
        self.checkpoint_usage: Optional[dict[str, Any]] = None  # usage at this hop's latest mid-stream checkpoint

    def next_hop(self) -> None:
        self.offset = max(self.steps, default=-1) + 1
        self.status = self.continuation_token = self.checkpoint_usage = None
        self.hop_done = False

    def close_dropped_hop(self) -> None:
        """A hop that dropped mid-stream never reports its final usage: count it up to its last checkpoint,
        which is what Google bills for a dropped stream."""
        if not self.hop_done and self.checkpoint_usage is not None:
            self.hop_usage.append(self.checkpoint_usage)
            self.checkpoint_usage = None


def _tool_delta(model: str, slot: dict[str, Any], arguments: str, *, first: bool, signature: Optional[str]) -> Any:
    delta = {"index": slot["index"], "id": slot["id"], "name": slot["name"] if first else "", "arguments": arguments,
             "extra_content": _tool_call_extra_from_part({"thoughtSignature": signature}) if first else None}
    return _make_stream_chunk(model=model, tool_call_delta=delta)


def _token(source: dict[str, Any]) -> Optional[str]:
    token = source.get("continuation_token")
    return token if isinstance(token, str) and token else None


def _on_error(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    raise interaction_error(event.get("error"), fallback="Gemini stream error")


def _on_created(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    state.interaction_id = (event.get("interaction") or {}).get("id") or state.interaction_id
    return []


def _start_thought(step: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    if isinstance(step.get("signature"), str) and step["signature"]:
        state.signature = step["signature"]
    text = _thought_text(step)
    return [_make_stream_chunk(model=model, reasoning=text)] if text else []


def _output_signature(model: str, state: StreamState) -> list[Any]:
    """The message-level signature chunk for text: the signature of the thought that preceded this output.
    A thought that precedes a function call rides that call instead, so it is not sent here."""
    if not state.signature or state.signature == state.output_signature:
        return []
    state.output_signature = state.signature
    return [_make_stream_chunk(model=model, text_signature=state.signature)]


def _start_output(step: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    return _output_signature(model, state) + [
        _make_stream_chunk(model=model, content=b["text"]) for b in step.get("content") or []
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]


def _start_call(step: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    entry.update(index=state.tool_calls, id=str(step.get("id") or f"call_{uuid.uuid4().hex[:12]}"), name=str(step.get("name") or ""))
    state.tool_calls += 1
    args = step.get("arguments")
    initial = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) and args else ""
    entry["sent"] = bool(initial)
    return [_tool_delta(model, entry, initial, first=True, signature=state.signature)]


_STEP_START = {"thought": _start_thought, "model_output": _start_output, "function_call": _start_call}


def _on_step_start(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    step = event.get("step") or {}
    entry = state.steps[state.offset + int(event.get("index") or 0)] = {"type": step.get("type")}
    handler = _STEP_START.get(step.get("type"))
    return handler(step, entry, model, state) if handler else []


def _delta_text(delta: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    if not delta.get("text"):
        return []
    key = "reasoning" if entry.get("type") == "thought" else "content"
    return [_make_stream_chunk(model=model, **{key: delta["text"]})]


def _delta_summary(delta: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    # A summary item with no ``type`` is read as text (the wire sends ``type``; this only tolerates its absence).
    content = delta.get("content") or {}
    ok = isinstance(content, dict) and content.get("type", "text") == "text" and content.get("text")
    return [_make_stream_chunk(model=model, reasoning=content["text"])] if ok else []


def _delta_signature(delta: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    if not (isinstance(delta.get("signature"), str) and delta["signature"]):
        return []
    state.signature = delta["signature"]
    return _output_signature(model, state) if entry.get("type") == "model_output" else []


def _delta_arguments(delta: dict[str, Any], entry: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    if entry.get("type") != "function_call" or not delta.get("arguments"):
        return []
    entry["sent"] = True
    return [_tool_delta(model, entry, str(delta["arguments"]), first=False, signature=None)]


_STEP_DELTA = {"text": _delta_text, "thought_summary": _delta_summary, "thought_signature": _delta_signature,
               "arguments_delta": _delta_arguments}


def _on_step_delta(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    entry = state.steps.get(state.offset + int(event.get("index") or 0), {})
    delta = event.get("delta") or {}
    handler = _STEP_DELTA.get(delta.get("type"))
    return handler(delta, entry, model, state) if handler else []


def _on_step_stop(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    entry = state.steps.get(state.offset + int(event.get("index") or 0), {})
    if entry.get("type") != "function_call" or entry.get("sent"):
        return []
    entry["sent"] = True
    return [_tool_delta(model, entry, "{}", first=False, signature=None)]  # a no-argument call


def _on_status(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    state.status = event.get("status") or state.status
    state.interaction_id = event.get("interaction_id") or state.interaction_id
    if token := _token(event):
        state.continuation_token = token
        state.checkpoint_usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    return []


def _on_completed(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    interaction = event.get("interaction") or {}
    state.interaction_id = interaction.get("id") or state.interaction_id
    state.status = str(interaction.get("status") or state.status or "completed")
    state.continuation_token = _token(interaction) or state.continuation_token
    if state.status == "failed":
        raise interaction_error(interaction.get("errors"))
    state.hop_done, state.completed_hops = True, state.completed_hops + 1
    state.hop_usage.append(interaction.get("usage") or {})
    if state.status == "requires_continuation":
        return []
    usage = merge_usage(state.hop_usage) if len(state.hop_usage) > 1 else state.hop_usage[0]
    chunk = _make_stream_chunk(model=model, finish_reason="tool_calls" if state.tool_calls else _STATUS_FINISH.get(state.status, "stop"))
    chunk.usage = usage_namespace(usage)
    return [chunk]


_EVENTS = {
    "error": _on_error, "interaction.created": _on_created, "step.start": _on_step_start, "step.delta": _on_step_delta,
    "step.stop": _on_step_stop, "interaction.status_update": _on_status, "interaction.completed": _on_completed,
}


def translate_stream_event(event: dict[str, Any], model: str, state: StreamState) -> list[Any]:
    """OpenAI chunks for one Interactions SSE event. ``interaction.completed`` with a final status yields the
    finish chunk (usage attached); a ``requires_continuation`` completion only records the hop."""
    handler = _EVENTS.get(event.get("event_type"))
    return handler(event, model, state) if handler else []


def event_token(event: dict[str, Any]) -> Optional[str]:
    """The continuation token an event carries: a mid-stream checkpoint (``interaction.status_update``) or a
    hop's end (``interaction.completed`` with ``requires_continuation``)."""
    return _token(event) or _token(event.get("interaction") or {})


def finish_ceiling(model: str, state: StreamState) -> Any:
    """Finish chunk when the hop ceiling stops a long decode that still asks for continuation."""
    chunk = _make_stream_chunk(model=model, finish_reason="tool_calls" if state.tool_calls else "length")
    chunk.usage = usage_namespace(merge_usage(state.hop_usage))
    return chunk


def iter_events(lines: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """SSE payloads whose ``data`` is an ``InteractionSseEvent`` (``event_type`` keyed)."""
    for payload in lines:
        if isinstance(payload, dict) and payload.get("event_type"):
            yield payload
