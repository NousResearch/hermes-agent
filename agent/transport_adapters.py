"""Protocol adapters for non-OpenAI chat transports.

These adapters expose an OpenAI chat-completions-like interface so Hermes can
keep its tool loop and message store stable while routing requests to providers
that speak Anthropic Messages or Google GenerateContent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import httpx


def _ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_ns(v) for v in value]
    return value


class _CompletionsNamespace:
    def __init__(self, adapter: Any):
        self.create = adapter.create


class _ChatNamespace:
    def __init__(self, adapter: Any):
        self.completions = _CompletionsNamespace(adapter)


class _BaseTransportClient:
    def __init__(self, *, base_url: str, api_key: str, default_headers: Optional[dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._client = httpx.Client(headers=self._default_headers)
        self.chat = _ChatNamespace(self)

    def close(self):
        self._client.close()

    def create(self, **kwargs):
        raise NotImplementedError


class AnthropicMessagesClient(_BaseTransportClient):
    """Expose Anthropic Messages as chat.completions.create()."""

    def create(self, **kwargs):
        timeout = kwargs.pop("timeout", None)
        payload = _to_anthropic_request(kwargs)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        response = self._client.post(
            f"{self.base_url}/messages",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return _from_anthropic_response(response.json(), kwargs.get("model"))


class GoogleGenerateContentClient(_BaseTransportClient):
    """Expose Google GenerateContent as chat.completions.create()."""

    def create(self, **kwargs):
        timeout = kwargs.pop("timeout", None)
        model = (kwargs.get("model") or "").strip()
        if not model:
            raise ValueError("Google transport requires a non-empty model id.")
        payload = _to_google_generate_content_request(kwargs)
        response = self._client.post(
            f"{self.base_url}/models/{model}:generateContent",
            json=payload,
            headers={"x-goog-api-key": self.api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        return _from_google_generate_content_response(response.json(), model)


# -- Anthropic conversion -------------------------------------------------


def _normalize_openai_tools(tools: Any) -> Optional[list[dict[str, Any]]]:
    if not isinstance(tools, list):
        return None
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {}) if tool.get("type") == "function" else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        normalized.append(
            {
                "name": name.strip(),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return normalized or None


def _coerce_json_object(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except Exception:
            return {"value": stripped}
    return raw if raw is not None else {}


def _to_anthropic_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = kwargs.get("messages") or []
    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = content if isinstance(content, str) else ""
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue

        if role == "user":
            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append({"type": "text", "text": part["text"]})
                if parts:
                    anthropic_messages.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"type": "text", "text": content})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function", {})
                    name = fn.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.get("id") or tool_call.get("call_id") or "tool_call",
                            "name": name.strip(),
                            "input": _coerce_json_object(fn.get("arguments", {})),
                        }
                    )
            if parts:
                anthropic_messages.append({"role": "assistant", "content": parts})
            continue

        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": str(message.get("content", "") or ""),
                        }
                    ],
                }
            )

    payload: dict[str, Any] = {
        "model": kwargs.get("model"),
        "messages": anthropic_messages,
        "stream": False,
        "max_tokens": int(kwargs.get("max_tokens") or 32000),
    }
    if system_blocks:
        payload["system"] = system_blocks
    if kwargs.get("temperature") is not None:
        payload["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        payload["top_p"] = kwargs["top_p"]
    tools = _normalize_openai_tools(kwargs.get("tools"))
    if tools:
        payload["tools"] = tools
    return payload


def _from_anthropic_response(data: dict[str, Any], model: Optional[str]) -> Any:
    blocks = data.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "tool_use":
            tool_calls.append(
                SimpleNamespace(
                    id=block.get("id") or "toolu_generated",
                    type="function",
                    function=SimpleNamespace(
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {}), ensure_ascii=False),
                    ),
                )
            )

    stop_reason = data.get("stop_reason")
    finish_reason = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "content_filter": "content_filter",
    }.get(stop_reason, None)

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    cached_tokens = usage.get("cache_read_input_tokens")
    usage_ns = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None,
    )

    message = SimpleNamespace(
        role="assistant",
        content="".join(text_parts),
        tool_calls=tool_calls or None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=data.get("id"),
        object="chat.completion",
        created=0,
        model=data.get("model") or model,
        choices=[choice],
        usage=usage_ns,
    )


# -- Google conversion ----------------------------------------------------


def _tool_name_map(messages: Iterable[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            fn = tool_call.get("function", {})
            name = fn.get("name")
            call_id = tool_call.get("id") or tool_call.get("call_id")
            if isinstance(name, str) and name and isinstance(call_id, str) and call_id:
                mapping[call_id] = name
    return mapping


def _to_google_generate_content_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = kwargs.get("messages") or []
    call_names = _tool_name_map(messages)
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append({"text": content})
            continue

        if role == "user":
            if isinstance(content, str) and content:
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = [
                    {"text": part.get("text")}
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                ]
                if parts:
                    contents.append({"role": "user", "parts": parts})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                parts.append({"text": content})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function", {})
                    name = fn.get("name")
                    if not isinstance(name, str) or not name.strip():
                        continue
                    parts.append(
                        {
                            "functionCall": {
                                "name": name.strip(),
                                "args": _coerce_json_object(fn.get("arguments", {})),
                            }
                        }
                    )
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            call_id = message.get("tool_call_id", "")
            name = call_names.get(call_id, "tool")
            result_text = str(message.get("content", "") or "")
            try:
                payload = json.loads(result_text)
            except Exception:
                payload = {"content": result_text}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": payload,
                            }
                        }
                    ],
                }
            )

    generation_config: dict[str, Any] = {}
    if kwargs.get("temperature") is not None:
        generation_config["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        generation_config["topP"] = kwargs["top_p"]
    if kwargs.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = kwargs["max_tokens"]

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if generation_config:
        payload["generationConfig"] = generation_config

    tools = _normalize_openai_tools(kwargs.get("tools"))
    if tools:
        payload["tools"] = [{"functionDeclarations": tools}]
    return payload


def _from_google_generate_content_response(data: dict[str, Any], model: str) -> Any:
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") or {}
    parts = content.get("parts") or []

    text_parts: list[str] = []
    tool_calls: list[Any] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
        function_call = part.get("functionCall") or part.get("function_call")
        if isinstance(function_call, dict):
            arguments = function_call.get("args", {})
            if isinstance(arguments, str):
                arg_text = arguments
            else:
                arg_text = json.dumps(arguments, ensure_ascii=False)
            tool_call = SimpleNamespace(
                id=function_call.get("id") or f"call_google_{index}",
                type="function",
                function=SimpleNamespace(
                    name=function_call.get("name", ""),
                    arguments=arg_text,
                ),
            )
            tool_calls.append(tool_call)

    finish_reason = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
    }.get(candidate.get("finishReason"), None)
    if tool_calls:
        finish_reason = "tool_calls"

    usage = data.get("usageMetadata") or {}
    prompt_tokens = usage.get("promptTokenCount")
    completion_tokens = usage.get("candidatesTokenCount")
    reasoning_tokens = usage.get("thoughtsTokenCount")
    cached_tokens = usage.get("cachedContentTokenCount")
    usage_ns = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.get("totalTokenCount") or ((prompt_tokens or 0) + (completion_tokens or 0)),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None,
    )

    message = SimpleNamespace(
        role="assistant",
        content="".join(text_parts),
        tool_calls=tool_calls or None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        id=data.get("responseId") or "google-generate-content",
        object="chat.completion",
        created=0,
        model=model,
        choices=[choice],
        usage=usage_ns,
    )
