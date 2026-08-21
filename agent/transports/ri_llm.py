"""RecursiveIntell llm-pipeline transport for Hermes.

Provides a Rust-backed LLM calling interface that replaces raw httpx/openai
calls. Falls back gracefully if the native extension is not installed.

Usage::

    from agent.transports.ri_llm import RiPipeline

    pipe = RiPipeline("http://localhost:11434", "llama3.2:3b")
    result = pipe.call("What is 2+2?")
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_NATIVE_AVAILABLE = False
try:
    from llm_pipeline._native import LlmConfig, Pipeline as _NativePipeline

    _NATIVE_AVAILABLE = True
except ImportError:
    logger.debug("llm-pipeline native extension not available; using None")


class RiLlmConfig:
    """Python-side mirror of the Rust LlmConfig."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking: bool = False,
        json_mode: bool = False,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.json_mode = json_mode

    def _to_native(self):
        if not _NATIVE_AVAILABLE:
            return None
        return LlmConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            json_mode=self.json_mode,
        )


class RiPipeline:
    """Rust-backed LLM pipeline for Hermes transport."""

    def __init__(self, url: str, model: str, *, config: RiLlmConfig | None = None):
        self.url = url
        self.model = model
        self.config = config or RiLlmConfig()
        self._native: _NativePipeline | None = None
        if _NATIVE_AVAILABLE:
            self._native = _NativePipeline(
                url, model, config=self.config._to_native()
            )

    @property
    def available(self) -> bool:
        return self._native is not None

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        config: RiLlmConfig | None = None,
    ) -> str:
        """Call the LLM and return the raw response text."""
        if self._native is None:
            raise RuntimeError(
                "llm-pipeline native extension is not installed. "
                "Install with: pip install llm-pipeline"
            )
        native_config = config._to_native() if config else None
        return self._native.call(prompt, system=system, config=native_config)

    def call_structured(
        self,
        prompt: str,
        json_schema: str,
        *,
        system: str | None = None,
        config: RiLlmConfig | None = None,
    ) -> str:
        """Call the LLM with a JSON schema constraint."""
        if self._native is None:
            raise RuntimeError("llm-pipeline native extension is not installed")
        native_config = config._to_native() if config else None
        return self._native.call_structured(
            prompt, json_schema, system=system, config=native_config
        )

    def __repr__(self) -> str:
        status = "native" if self.available else "unavailable"
        return f"RiPipeline(url={self.url}, model={self.model}, {status})"


# ── Phase 2: RiChatCompletionsTransport (universal) ───────────────
#
# Plugs into the chat_completion_helpers dispatch. Active by default when
# the native extension is available and not runtime-disabled. Provider-
# agnostic — works for any OpenAI-compatible provider.
# OpenAI-compatible provider. Set HERMES_RI_PIPELINE_PROVIDERS to
# a comma-separated whitelist to restrict (e.g. 'ollama-launch,deepseek').
# Falls through to the stock httpx/openai path on any error.

import json as _json
import os as _os
from types import SimpleNamespace as _SimpleNamespace


def _should_use_ri_pipeline(agent) -> bool:
    """Return True when the RiPipeline fast path should be used.

    Active by default when the native extension is available.
    Provider-agnostic — works for any OpenAI-compatible provider.
    Set HERMES_RI_PIPELINE=0 to disable, or HERMES_RI_PIPELINE_PROVIDERS
    to a comma-separated whitelist (e.g. 'ollama-launch,deepseek').
    If no env whitelist is set, agent._ri_pipeline_enabled and
    agent._ri_pipeline_providers from config.yaml determine eligibility.
    """
    if _os.environ.get("HERMES_RI_PIPELINE") == "0":
        return False
    if not _NATIVE_AVAILABLE:
        return False

    if not bool(getattr(agent, "_ri_pipeline_enabled", True)):
        return False

    whitelist = _os.environ.get("HERMES_RI_PIPELINE_PROVIDERS")
    if whitelist:
        allowed = _normalize_ri_pipeline_provider_list(whitelist)
        return str(getattr(agent, "provider", "")).strip().lower() in allowed

    config_whitelist = _normalize_ri_pipeline_provider_list(
        getattr(agent, "_ri_pipeline_providers", [])
    )
    if config_whitelist:
        return str(getattr(agent, "provider", "")).strip().lower() in config_whitelist

    return True


def _normalize_ri_pipeline_provider_list(raw_providers):
    """Normalize a provider whitelist input into a lowercase set."""
    if raw_providers is None:
        return set()
    if isinstance(raw_providers, str):
        values = raw_providers.split(",")
    elif isinstance(raw_providers, (list, tuple, set)):
        values = raw_providers
    else:
        return set()

    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def ri_pipeline_chat_completion(agent, api_kwargs: dict):
    """Run a chat-completion request through the Rust llm-pipeline.

    Accepts the same kwargs shape as ``client.chat.completions.create()``
    and returns an OpenAI-compatible response namespace so the rest of
    the agent loop is unchanged.
    """
    model = api_kwargs.get("model", agent.model)
    messages = api_kwargs.get("messages", [])
    base_url = getattr(agent, "base_url", "http://localhost:11434/v1")

    # Inject API key into environment for the Rust pipeline.
    # The Rust OpenAiBackend reads OPENAI_API_KEY from the environment.
    _prev_key = _os.environ.get("OPENAI_API_KEY")
    agent_api_key = getattr(agent, "api_key", None)
    if callable(agent_api_key):
        try:
            agent_api_key = agent_api_key()
        except Exception:
            agent_api_key = None
    if agent_api_key and isinstance(agent_api_key, str) and agent_api_key.strip():
        _os.environ["OPENAI_API_KEY"] = agent_api_key
    try:
        return _ri_chat_completion_impl(
            agent, api_kwargs, model, messages, base_url
        )
    finally:
        if _prev_key is not None:
            _os.environ["OPENAI_API_KEY"] = _prev_key
        elif "OPENAI_API_KEY" in _os.environ:
            del _os.environ["OPENAI_API_KEY"]


def _ri_chat_completion_impl(agent, api_kwargs, model, messages, base_url):
    # Build a text prompt from the messages list (basic: system + user + assistant)
    system_prompt = ""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multimodal content: extract text parts only
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            system_prompt = content
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            prompt_parts.append(f"Tool output: {content}")

    full_prompt = "\n".join(prompt_parts)

    # Try to extract tool schemas for structured output
    tools = api_kwargs.get("tools") or api_kwargs.get("functions")
    use_json_mode = bool(tools and not api_kwargs.get("stream"))

    pipe = RiPipeline(base_url, model)
    if not pipe.available:
        raise RuntimeError("RiPipeline native extension not available")

    if use_json_mode:
        # Tool-calling: use call_structured with a JSON schema for tool_choice
        tool_names = [t.get("function", {}).get("name", "tool") for t in tools]
        json_schema = _json.dumps({
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": tool_names},
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
        })
        raw = pipe.call_structured(full_prompt, json_schema, system=system_prompt or None)
    else:
        raw = pipe.call(full_prompt, system=system_prompt or None)

    # Parse tool calls if present (basic JSON extraction)
    tool_calls = None
    content = raw
    if use_json_mode and raw.strip():
        try:
            parsed = _json.loads(raw)
            tool_name = parsed.get("tool", "")
            tool_args = parsed.get("arguments", {})
            if tool_name:
                import uuid as _uuid
                tool_calls = [{
                    "id": f"call_{_uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": _json.dumps(tool_args),
                    },
                }]
                content = None  # Tool call — no text content
        except _json.JSONDecodeError:
            pass

    # Approximate token counts (rough character-based estimate)
    prompt_chars = len(full_prompt) + len(system_prompt)
    completion_chars = len(raw)

    # Build response namespace matching OpenAI shape
    message = _SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=[_SimpleNamespace(**tc) for tc in tool_calls] if tool_calls else None,
    )
    choice = _SimpleNamespace(
        index=0,
        message=message,
        finish_reason="tool_calls" if tool_calls else "stop",
    )
    usage = _SimpleNamespace(
        prompt_tokens=max(1, prompt_chars // 4),
        completion_tokens=max(1, completion_chars // 4),
        total_tokens=max(2, (prompt_chars + completion_chars) // 4),
    )
    return _SimpleNamespace(choices=[choice], usage=usage, model=model)
