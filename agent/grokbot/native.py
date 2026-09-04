#!/usr/bin/env python3
"""Native Hermes OpenAI-client facade over Grok Bot InferenceService/Stream.

This is the in-process adapter (no localhost proxy). Hermes loads it from
``agent.transports.grokbot`` and treats ``api_mode: grokbot`` like MoA:
the facade *is* ``client.chat.completions.create``.

Wire: POST https://api2.cursor.sh/aiserver.v1.InferenceService/Stream
      Content-Type: application/connect+proto
Auth: ~/.grokbot/session.json (PKCE). Never print tokens.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Iterator

from agent.grokbot import client as gc


def openai_tool_calls(raw: list[dict] | None) -> list[SimpleNamespace]:
    out = []
    for tc in raw or []:
        fn = tc.get("function") or {}
        out.append(SimpleNamespace(
            id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            index=len(out),
            function=SimpleNamespace(
                name=fn.get("name") or "",
                arguments=fn.get("arguments") or "{}",
            ),
        ))
    return out


class _Usage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    prompt_tokens_details = None
    prompt_cache_hit_tokens = 0


class _Message:
    def __init__(self, content: str | None, tool_calls: list | None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or None
        self.refusal = None
        self.reasoning_content = None
        self.reasoning = None


class _Choice:
    def __init__(self, *, message=None, delta=None, finish_reason=None):
        self.index = 0
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, model: str, content: str | None, tool_calls: list | None,
                 finish_reason: str):
        self.id = f"grokbot-{uuid.uuid4().hex[:16]}"
        self.object = "chat.completion"
        self.model = model or "grok-4.6"
        self.choices = [_Choice(
            message=_Message(content, tool_calls),
            finish_reason=finish_reason,
        )]
        self.usage = _Usage()


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"
        self.reasoning_content = None


class _Chunk:
    def __init__(self, model: str, *, content=None, tool_calls=None,
                 finish_reason=None, usage=None):
        self.id = f"grokbot-chunk-{uuid.uuid4().hex[:12]}"
        self.object = "chat.completion.chunk"
        self.model = model or "grok-4.6"
        self.usage = usage
        delta = _Delta(content=content, tool_calls=tool_calls)
        self.choices = [_Choice(delta=delta, finish_reason=finish_reason)]


class GrokbotStream:
    """Iterator that looks enough like an OpenAI streaming response."""

    def __init__(self, chunks: list):
        self._chunks = chunks
        self.response = None
        self.final_response = None

    def __iter__(self) -> Iterator:
        return iter(self._chunks)

    def close(self) -> None:
        return None


def _run_infer(messages: list[dict], model: str, tools: list | None) -> dict:
    prompt, history = gc.openai_messages_to_history(messages)
    if not prompt and not history:
        prompt = "hello"
    return gc.infer(
        prompt,
        model or gc.DEFAULT_MODEL,
        history=history or None,
        tools=tools or None,
    )


def _to_response(result: dict, model: str) -> _Response:
    tcs = openai_tool_calls(result.get("tool_calls"))
    text = (result.get("text") or "") or None
    if tcs:
        finish = "tool_calls"
    else:
        finish = "stop"
    routed = result.get("model") or model
    return _Response(routed, text, tcs or None, finish)


def _to_stream(result: dict, model: str) -> GrokbotStream:
    routed = result.get("model") or model or "grok-4.6"
    chunks: list = []
    text = result.get("text") or ""
    if text:
        chunks.append(_Chunk(routed, content=text))
    tcs = openai_tool_calls(result.get("tool_calls"))
    if tcs:
        chunks.append(_Chunk(routed, tool_calls=tcs))
        chunks.append(_Chunk(routed, finish_reason="tool_calls", usage=_Usage()))
    else:
        chunks.append(_Chunk(routed, finish_reason="stop", usage=_Usage()))
    return GrokbotStream(chunks)


class _Completions:
    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        model = kwargs.get("model") or gc.DEFAULT_MODEL
        tools = kwargs.get("tools") or []
        stream = bool(kwargs.get("stream"))
        result = _run_infer(messages, model, tools)
        if stream:
            return _to_stream(result, model)
        return _to_response(result, model)


class GrokBotClient:
    """Drop-in for ``openai.OpenAI`` on the chat.completions.create path."""

    def __init__(self, agent: Any = None):
        self._agent = agent
        self.chat = SimpleNamespace(completions=_Completions())
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed


def build_grokbot_client(agent: Any = None) -> GrokBotClient:
    return GrokBotClient(agent=agent)


class GrokbotTransport:
    """Hermes ProviderTransport for api_mode=grokbot.

    Kwargs stay OpenAI-shaped; the facade owns Connect/protobuf. Normalization
    reuses ChatCompletionsTransport so the agent loop sees the usual objects.
    """

    @property
    def api_mode(self) -> str:
        return "grokbot"

    def convert_messages(self, messages, **kwargs):
        return messages

    def convert_tools(self, tools):
        return tools

    def build_kwargs(self, model, messages, tools=None, **params):
        out = {"model": model, "messages": messages}
        if tools:
            out["tools"] = tools
        return out

    def _cc(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        return ChatCompletionsTransport()

    def normalize_response(self, response, **kwargs):
        return self._cc().normalize_response(response, **kwargs)

    def validate_response(self, response) -> bool:
        return self._cc().validate_response(response)

    def extract_cache_stats(self, response):
        return None

    def map_finish_reason(self, raw_reason: str) -> str:
        return raw_reason
