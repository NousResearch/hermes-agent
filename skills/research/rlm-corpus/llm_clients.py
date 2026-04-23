"""LLM client abstractions for RLM root and sub calls.

Three concrete clients:

* ``AnthropicClient`` — native Anthropic Messages API
* ``OpenAIClient`` — native OpenAI Chat Completions API
* ``OMLXClient`` — OpenAI-compatible local endpoint (Ollama, mlx-lm server, vLLM, etc.)

The abstract ``LLMClient`` deliberately exposes only ``chat(messages, system, temperature, max_tokens)``
so callers can swap backends without touching engine code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        import anthropic  # lazy import

        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------


class _OpenAICompatibleClient:
    def __init__(self, model: str, *, api_key: str | None, base_url: str | None) -> None:
        import openai  # lazy import

        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy"),
            base_url=base_url,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        for m in messages:
            payload.append({"role": m.role, "content": m.content})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class OpenAIClient(_OpenAICompatibleClient):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model, api_key=api_key, base_url=None)


class OMLXClient(_OpenAICompatibleClient):
    """OpenAI-compatible local endpoint (mlx-lm server / vLLM / Ollama OpenAI shim)."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_client(
    endpoint: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    endpoint = endpoint.lower().strip()
    if endpoint == "anthropic":
        return AnthropicClient(model, api_key=api_key)
    if endpoint == "openai":
        return OpenAIClient(model, api_key=api_key)
    if endpoint == "omlx":
        if not base_url:
            raise ValueError("omlx endpoint requires base_url")
        return OMLXClient(model, base_url=base_url, api_key=api_key)
    raise ValueError(f"unknown LLM endpoint: {endpoint!r}")
