"""Direct LLM client — Anthropic SDK for Claude, OpenAI SDK for everything else.

Single entry point `completion(**kwargs)` accepts OpenAI-shaped kwargs the rest
of the codebase already builds and returns a SimpleNamespace shaped like an
OpenAI ChatCompletion response so callers don't have to branch.

Routing:
- model starts with "claude" or provider == "anthropic"  → Anthropic Messages API
- otherwise                                              → OpenAI Chat Completions
  (works for OpenAI proper, OpenRouter, vLLM, rvLLM, any OpenAI-compatible server)

No silent fallback between providers. Auth errors raise immediately.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_anthropic(model: str, api_base: Optional[str]) -> bool:
    if model and model.lower().startswith(("claude", "anthropic/")):
        return True
    if api_base and "api.anthropic.com" in api_base.lower():
        return True
    return False


# -----------------------------------------------------------------------------
# OpenAI-shape → Anthropic-shape translation
# -----------------------------------------------------------------------------

def _to_anthropic_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn:
            continue
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out or None


def _to_anthropic_tool_choice(tc):
    if tc in (None, "auto"):
        return None
    if tc == "required":
        return {"type": "any"}
    if tc == "none":
        return {"type": "none"}
    if isinstance(tc, dict) and tc.get("type") == "function":
        return {"type": "tool", "name": tc["function"]["name"]}
    return None


def _content_to_blocks(content, cache_marker=None):
    """Coerce OpenAI-shape content (str|list[dict]) into Anthropic blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        block = {"type": "text", "text": content}
        if cache_marker:
            block["cache_control"] = cache_marker
        return [block]
    blocks = []
    for piece in content:
        if isinstance(piece, dict):
            pt = piece.get("type")
            if pt == "text":
                b = {"type": "text", "text": piece.get("text", "")}
                if piece.get("cache_control"):
                    b["cache_control"] = piece["cache_control"]
                blocks.append(b)
            elif pt == "image_url":
                # OpenAI vision shape -> Anthropic image block
                url = piece.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # data URL: data:image/png;base64,XXXX
                    header, _, b64 = url.partition(",")
                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    })
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            elif pt in ("tool_use", "tool_result", "image", "thinking"):
                blocks.append(piece)
            else:
                blocks.append({"type": "text", "text": str(piece)})
        else:
            blocks.append({"type": "text", "text": str(piece)})
    if cache_marker and blocks:
        # apply to the last block (mirrors prompt_caching.py behaviour)
        blocks[-1]["cache_control"] = cache_marker
    return blocks


def _messages_to_anthropic(messages: List[dict]):
    """Returns (system: str|list, msgs: list[dict])."""
    system_blocks: List[dict] = []
    out: List[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        cache_marker = m.get("cache_control")
        content = m.get("content")

        if role == "system":
            for b in _content_to_blocks(content, cache_marker):
                system_blocks.append(b)
            i += 1
            continue

        if role == "tool":
            # Merge consecutive tool messages into a single user turn.
            blocks = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tm = messages[i]
                tc_id = tm.get("tool_call_id") or tm.get("id")
                tc_content = tm.get("content")
                if isinstance(tc_content, list):
                    inner = tc_content
                else:
                    inner = [{"type": "text", "text": "" if tc_content is None else str(tc_content)}]
                block = {"type": "tool_result", "tool_use_id": tc_id, "content": inner}
                if tm.get("cache_control"):
                    block["cache_control"] = tm["cache_control"]
                blocks.append(block)
                i += 1
            out.append({"role": "user", "content": blocks})
            continue

        if role == "assistant":
            blocks = _content_to_blocks(content)
            tool_calls = m.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                args_str = fn.get("arguments", "") or "{}"
                try:
                    parsed = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception:
                    parsed = {"_raw": args_str}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": parsed,
                })
            if not blocks:
                # Anthropic rejects empty assistant turns; insert a stub.
                blocks = [{"type": "text", "text": ""}]
            out.append({"role": "assistant", "content": blocks})
            i += 1
            continue

        # user (default)
        out.append({"role": "user", "content": _content_to_blocks(content, cache_marker) or [{"type": "text", "text": ""}]})
        i += 1

    system = system_blocks if len(system_blocks) > 1 else (system_blocks[0]["text"] if system_blocks and "cache_control" not in system_blocks[0] else system_blocks)
    return system, out


# -----------------------------------------------------------------------------
# Anthropic-shape → OpenAI-shape response
# -----------------------------------------------------------------------------

def _anthropic_to_openai_response(message, model: str) -> SimpleNamespace:
    """Convert an Anthropic Message object into the OpenAI ChatCompletion-shape
    callers consume."""
    content_parts: List[str] = []
    tool_calls: List[SimpleNamespace] = []
    reasoning_text = None

    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            content_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls.append(SimpleNamespace(
                id=getattr(block, "id", ""),
                type="function",
                function=SimpleNamespace(
                    name=getattr(block, "name", ""),
                    arguments=json.dumps(getattr(block, "input", {}) or {}),
                ),
            ))
        elif btype == "thinking":
            reasoning_text = (reasoning_text or "") + (getattr(block, "thinking", "") or "")

    finish_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_map.get(getattr(message, "stop_reason", "end_turn"), "stop")

    msg = SimpleNamespace(
        role="assistant",
        content="".join(content_parts) if content_parts else None,
        tool_calls=tool_calls or None,
        function_call=None,
        reasoning_content=reasoning_text,
    )
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    usage_obj = getattr(message, "usage", None)
    if usage_obj is not None:
        usage = SimpleNamespace(
            prompt_tokens=getattr(usage_obj, "input_tokens", 0),
            completion_tokens=getattr(usage_obj, "output_tokens", 0),
            total_tokens=(getattr(usage_obj, "input_tokens", 0) + getattr(usage_obj, "output_tokens", 0)),
            cache_creation_input_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0),
            cache_read_input_tokens=getattr(usage_obj, "cache_read_input_tokens", 0),
        )
    else:
        usage = None
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def completion(
    *,
    model: str,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    tool_choice=None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    stream: bool = False,
    stream_options: Optional[dict] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    extra_body: Optional[dict] = None,
    thinking: Optional[dict] = None,
    on_text: Optional[Callable[[str], None]] = None,
    on_tool_use: Optional[Callable[[int, str, str], None]] = None,
    **_ignored: Any,
) -> SimpleNamespace:
    """Synchronous LLM completion.

    Returns a SimpleNamespace mimicking OpenAI ChatCompletion. Streaming, when
    requested, is reassembled into the same shape; per-token callbacks
    (`on_text`, `on_tool_use`) are invoked while the stream runs.
    """
    api_base = api_base or base_url or None
    if max_completion_tokens is not None and max_tokens is None:
        max_tokens = max_completion_tokens

    if _is_anthropic(model, api_base):
        return _anthropic_completion(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            stream=stream, timeout=timeout, api_key=api_key, api_base=api_base,
            extra_headers=extra_headers, thinking=thinking,
            on_text=on_text, on_tool_use=on_tool_use,
        )

    return _openai_completion(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice,
        temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty, max_tokens=max_tokens,
        stream=stream, stream_options=stream_options, timeout=timeout,
        api_key=api_key, api_base=api_base, extra_headers=extra_headers,
        extra_body=extra_body, on_text=on_text, on_tool_use=on_tool_use,
    )


# -----------------------------------------------------------------------------
# Anthropic path
# -----------------------------------------------------------------------------

def _anthropic_completion(
    *, model, messages, tools, tool_choice, temperature, top_p, max_tokens,
    stream, timeout, api_key, api_base, extra_headers, thinking,
    on_text, on_tool_use,
):
    import anthropic  # type: ignore

    # Strip "anthropic/" prefix some callers attach.
    if model.lower().startswith("anthropic/"):
        model = model.split("/", 1)[1]

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise anthropic.AuthenticationError("ANTHROPIC_API_KEY is not set")

    client_kwargs: Dict[str, Any] = {"api_key": key}
    if api_base and "api.anthropic.com" not in api_base.lower():
        client_kwargs["base_url"] = api_base
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    client = anthropic.Anthropic(**client_kwargs)

    system, msgs = _messages_to_anthropic(messages)

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens if max_tokens is not None else 8192,
    }
    if system:
        create_kwargs["system"] = system
    a_tools = _to_anthropic_tools(tools)
    if a_tools:
        create_kwargs["tools"] = a_tools
    a_tool_choice = _to_anthropic_tool_choice(tool_choice)
    if a_tool_choice:
        create_kwargs["tool_choice"] = a_tool_choice
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    if thinking:
        create_kwargs["thinking"] = thinking
    if extra_headers:
        create_kwargs["extra_headers"] = extra_headers

    if not stream:
        msg = client.messages.create(**create_kwargs)
        return _anthropic_to_openai_response(msg, model)

    # Streaming: use messages.stream() and reassemble.
    text_parts: List[str] = []
    tool_state: Dict[int, Dict[str, str]] = {}
    reasoning_text = ""
    final_message = None

    with client.messages.stream(**create_kwargs) as s:
        for event in s:
            etype = getattr(event, "type", "")
            if etype == "text":
                token = getattr(event, "text", "") or ""
                if token:
                    text_parts.append(token)
                    if on_text:
                        on_text(token)
            elif etype == "input_json":
                # tool input partial json
                # event has .partial_json and .snapshot
                idx = getattr(event, "index", 0)
                slot = tool_state.setdefault(idx, {"id": "", "name": "", "args": ""})
                delta = getattr(event, "partial_json", "") or ""
                slot["args"] += delta
                if on_tool_use:
                    on_tool_use(idx, "", delta)
            elif etype == "content_block_start":
                block = getattr(event, "content_block", None)
                if block is not None and getattr(block, "type", "") == "tool_use":
                    idx = getattr(event, "index", 0)
                    slot = tool_state.setdefault(idx, {"id": "", "name": "", "args": ""})
                    slot["id"] = getattr(block, "id", "") or slot["id"]
                    slot["name"] = getattr(block, "name", "") or slot["name"]
                    if on_tool_use:
                        on_tool_use(idx, slot["name"], "")
            elif etype == "thinking":
                reasoning_text += getattr(event, "thinking", "") or ""
        final_message = s.get_final_message()

    if final_message is not None:
        resp = _anthropic_to_openai_response(final_message, model)
        # Streaming reasoning may not be on final_message; preserve what we saw.
        if reasoning_text and not resp.choices[0].message.reasoning_content:
            resp.choices[0].message.reasoning_content = reasoning_text
        return resp

    # Fallback (should not happen): build response from accumulated state.
    tool_calls = []
    for idx in sorted(tool_state):
        slot = tool_state[idx]
        tool_calls.append(SimpleNamespace(
            id=slot["id"], type="function",
            function=SimpleNamespace(name=slot["name"], arguments=slot["args"] or "{}"),
        ))
    msg = SimpleNamespace(
        role="assistant",
        content="".join(text_parts) or None,
        tool_calls=tool_calls or None,
        function_call=None,
        reasoning_content=reasoning_text or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=msg, finish_reason="tool_calls" if tool_calls else "stop")],
        usage=None, model=model,
    )


# -----------------------------------------------------------------------------
# OpenAI-compatible path
# -----------------------------------------------------------------------------

def _openai_completion(
    *, model, messages, tools, tool_choice, temperature, top_p, presence_penalty,
    frequency_penalty, max_tokens, stream, stream_options, timeout, api_key,
    api_base, extra_headers, extra_body, on_text, on_tool_use,
):
    import openai  # type: ignore

    # Strip provider-prefixes some callers attach (e.g. "openrouter/...").
    if "/" in model and model.lower().startswith(("openrouter/", "openai/", "together_ai/", "groq/")):
        model = model.split("/", 1)[1]

    key = api_key or os.getenv("OPENAI_API_KEY", "").strip() or "sk-no-key"
    client_kwargs: Dict[str, Any] = {"api_key": key}
    if api_base:
        client_kwargs["base_url"] = api_base
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    client = openai.OpenAI(**client_kwargs)

    create_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        create_kwargs["tools"] = tools
    if tool_choice is not None:
        create_kwargs["tool_choice"] = tool_choice
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if top_p is not None:
        create_kwargs["top_p"] = top_p
    if presence_penalty is not None:
        create_kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        create_kwargs["frequency_penalty"] = frequency_penalty
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if extra_headers:
        create_kwargs["extra_headers"] = extra_headers
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    if not stream:
        resp = client.chat.completions.create(**create_kwargs)
        return resp  # SDK object is already the shape callers expect

    create_kwargs["stream"] = True
    if stream_options:
        create_kwargs["stream_options"] = stream_options
    else:
        create_kwargs["stream_options"] = {"include_usage": True}

    text_parts: List[str] = []
    tool_state: Dict[int, Dict[str, str]] = {}
    finish_reason = None
    usage = None
    model_name = None

    for chunk in client.chat.completions.create(**create_kwargs):
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if getattr(chunk, "model", None):
                model_name = chunk.model
            continue
        delta = chunk.choices[0].delta
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if getattr(chunk, "model", None):
            model_name = chunk.model
        if getattr(delta, "content", None):
            text_parts.append(delta.content)
            if on_text:
                on_text(delta.content)
        if getattr(delta, "tool_calls", None):
            for tc_delta in delta.tool_calls:
                idx = getattr(tc_delta, "index", 0)
                slot = tool_state.setdefault(idx, {"id": "", "name": "", "args": ""})
                if getattr(tc_delta, "id", None):
                    slot["id"] = tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
                        if on_tool_use:
                            on_tool_use(idx, slot["name"], fn.arguments)

    tool_calls = None
    if tool_state:
        tool_calls = []
        for idx in sorted(tool_state):
            slot = tool_state[idx]
            tool_calls.append(SimpleNamespace(
                id=slot["id"], type="function",
                function=SimpleNamespace(name=slot["name"], arguments=slot["args"] or "{}"),
            ))

    msg = SimpleNamespace(
        role="assistant",
        content="".join(text_parts) or None,
        tool_calls=tool_calls,
        function_call=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=msg, finish_reason=finish_reason or "stop")],
        usage=usage, model=model_name or model,
    )
