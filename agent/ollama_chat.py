"""Ollama's native chat adapter for local custom-provider endpoints.

The OpenAI-compatible endpoint cannot set ``num_ctx``. Ollama's Qwen renderer
then drops old user turns when the default context fills, often reporting the
misleading ``no user query found in messages`` error during a tool continuation.
The native endpoint accepts ``options.num_ctx`` and uses tool-name replies.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

_DEFAULT_CONTEXT = 4096
_OUTPUT_RUNWAY = 2048


def ollama_chat_url(base_url: str) -> str:
    """Map an OpenAI-compatible base URL to Ollama's native ``/api/chat`` route."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/chat", "", ""))


def _tool_name_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            call_id, name = call.get("id"), function.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def _native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = _tool_name_by_id(messages)
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        item = {"role": role, "content": message.get("content") or ""}
        if role == "assistant":
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if reasoning:
                item["thinking"] = reasoning
            calls = []
            for call in message.get("tool_calls") or ():
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, TypeError) as exc:
                        raise ValueError("Ollama tool call arguments must be valid JSON") from exc
                calls.append({"type": "function", "function": {
                    "name": function.get("name", ""), "arguments": arguments,
                }})
            if calls:
                item["tool_calls"] = calls
        elif role == "tool":
            tool_name = names.get(message.get("tool_call_id"))
            if tool_name:
                item["tool_name"] = tool_name
        converted.append(item)
    return converted


def _native_payload(api_kwargs: dict[str, Any], *, num_ctx: int, stream: bool) -> dict[str, Any]:
    extra = api_kwargs.get("extra_body") or {}
    options = dict(extra.get("options") or {})
    options["num_ctx"] = num_ctx
    for source, target in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed"),
                           ("max_tokens", "num_predict"), ("max_completion_tokens", "num_predict")):
        if source in api_kwargs:
            options[target] = api_kwargs[source]
    payload: dict[str, Any] = {
        "model": api_kwargs["model"], "messages": _native_messages(api_kwargs.get("messages") or []),
        "stream": stream, "options": options,
    }
    if api_kwargs.get("tools"):
        payload["tools"] = api_kwargs["tools"]
    if extra.get("think") is not None:
        payload["think"] = extra["think"]
    else:
        effort = api_kwargs.get("reasoning_effort")
        if isinstance(effort, str) and effort:
            payload["think"] = False if effort == "none" else effort
    return payload


def _context_for_request(agent: Any, api_kwargs: dict[str, Any]) -> int:
    from agent.model_metadata import estimate_request_tokens_rough

    messages = api_kwargs.get("messages") or []
    estimate = estimate_request_tokens_rough(messages, tools=api_kwargs.get("tools"))
    available = int(getattr(agent, "_ollama_num_ctx", 0) or 0)
    desired = max(_DEFAULT_CONTEXT, estimate + _OUTPUT_RUNWAY)
    return min(desired, available) if available > 0 else desired


def _tool_call(call: dict[str, Any], index: int) -> SimpleNamespace:
    function = call.get("function") or {}
    arguments = function.get("arguments", {})
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:12]}", type="function", index=index,
        function=SimpleNamespace(name=function.get("name"), arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def _choice(message: dict[str, Any], finish_reason: str) -> Any:
    calls = message.get("tool_calls") or ()
    converted = [_tool_call(call, index) for index, call in enumerate(calls) if isinstance(call, dict)]
    normalized_reason = "tool_calls" if converted else finish_reason or "stop"
    response_message = SimpleNamespace(
        role="assistant", content=message.get("content") or None, tool_calls=converted or None,
        reasoning_content=message.get("thinking") or None, reasoning_details=None,
    )
    return SimpleNamespace(index=0, message=response_message, finish_reason=normalized_reason)


def _usage(payload: dict[str, Any]) -> Any:
    prompt = payload.get("prompt_eval_count", 0) or 0
    completion = payload.get("eval_count", 0) or 0
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=prompt + completion)


def _headers(agent: Any) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = getattr(agent, "api_key", None)
    if isinstance(key, str) and key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def create_completion(
    agent: Any,
    api_kwargs: dict[str, Any],
    *,
    timeout: Any,
    stream: bool = False,
    http_client: Any = None,
) -> Any:
    """Call Ollama ``/api/chat`` and expose OpenAI-shaped result/stream objects."""
    import httpx

    base_url = str(getattr(agent, "base_url", "") or "")
    url = ollama_chat_url(base_url)
    payload = _native_payload(api_kwargs, num_ctx=_context_for_request(agent, api_kwargs), stream=stream)
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    request = client.build_request("POST", url, json=payload, headers=_headers(agent), timeout=timeout)
    response = client.send(request, stream=stream)
    try:
        response.raise_for_status()
    except Exception:
        response.read()
        response.close()
        if owns_client:
            client.close()
        raise
    if stream:
        return OllamaChatStream(client, response, owns_client=owns_client)
    try:
        body = response.json()
    finally:
        response.close()
        if owns_client:
            client.close()
    message = body.get("message") or {}
    choice = _choice(message, body.get("done_reason") or "stop")
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex[:16]}", model=body.get("model") or api_kwargs["model"],
        choices=[choice], usage=_usage(body),
    )


class OllamaChatStream:
    """NDJSON-to-OpenAI-chunk iterator consumed by the existing stream accumulator."""

    def __init__(self, client: Any, response: Any, *, owns_client: bool = True) -> None:
        self._client, self._response = client, response
        self._owns_client = owns_client
        self.response = response
        self._closed = False
        self._tool_ids: dict[int, str] = {}
        self._saw_tool_calls = False

    def __iter__(self) -> Iterator[Any]:
        for line in self._response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message") or {}
            content = message.get("content") or None
            thinking = message.get("thinking") or None
            calls = message.get("tool_calls") or ()
            self._saw_tool_calls = self._saw_tool_calls or bool(calls)
            tool_deltas = []
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                call_id = self._tool_ids.setdefault(index, f"call_{uuid.uuid4().hex[:12]}")
                tool_deltas.append(SimpleNamespace(
                    index=index, id=call_id, type="function",
                    function=SimpleNamespace(
                        name=function.get("name"),
                        arguments=json.dumps(function.get("arguments", {}), ensure_ascii=False),
                    ),
                ))
            if content or thinking or tool_deltas:
                delta = SimpleNamespace(role="assistant", content=content, reasoning_content=thinking,
                                         reasoning=None, tool_calls=tool_deltas or None, refusal=None)
                yield SimpleNamespace(id="chatcmpl-ollama", model=chunk.get("model"),
                                      choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)], usage=None)
            if chunk.get("done"):
                finish = "tool_calls" if self._saw_tool_calls else (chunk.get("done_reason") or "stop")
                yield SimpleNamespace(id="chatcmpl-ollama", model=chunk.get("model"),
                                      choices=[SimpleNamespace(
                                          index=0, delta=SimpleNamespace(), finish_reason=finish,
                                      )],
                                      usage=None)
                usage = _usage(chunk)
                yield SimpleNamespace(id="chatcmpl-ollama", model=chunk.get("model"), choices=[], usage=usage)
                break

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._response.close()
            if self._owns_client:
                self._client.close()
