"""Anthropic Messages API shim for CommandCode.

CommandCode exposes a public OpenAI-compatible catalog at
``https://api.commandcode.ai/provider/v1/models`` and a bearer-authenticated
Anthropic-compatible ``/v1/messages`` route under the same provider root.
Hermes' native Anthropic transport is built around providers with either
Anthropic's native auth semantics or hardcoded third-party compatibility
rules, so this module provides two building blocks:

- ``build_commandcode_anthropic_profile``: a provider profile factory for the
  ``commandcode-anthropic`` profile.
- ``CommandCodeAnthropicShim``: a tiny adapter that presents an
  ``Anthropic.messages.create(...)``-like surface over an OpenAI-compatible
  ``chat.completions`` client.

The shim keeps the implementation self-contained so future runtime wiring can
opt into it without duplicating message / tool conversion logic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from providers.base import ProviderProfile

COMMANDCODE_ANTHROPIC_BASE_URL = "https://api.commandcode.ai/provider"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
    return str(value)


def _flatten_system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = _coerce_text(block.get("text"))
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _tool_result_to_openai_message(block: dict[str, Any]) -> dict[str, Any]:
    payload = block.get("content")
    if isinstance(payload, list):
        payload = "\n".join(
            _coerce_text(item.get("text"))
            for item in payload
            if isinstance(item, dict) and item.get("type") == "text"
        )
    elif isinstance(payload, dict):
        payload = payload.get("text") or json.dumps(payload)
    elif not isinstance(payload, str):
        payload = json.dumps(payload if payload is not None else "")
    return {
        "role": "tool",
        "tool_call_id": str(block.get("tool_use_id") or block.get("id") or "tool"),
        "content": str(payload or ""),
    }


def _assistant_block_to_tool_call(block: dict[str, Any], index: int) -> dict[str, Any]:
    arguments = block.get("input")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {})
    return {
        "id": str(block.get("id") or f"toolu_{index}"),
        "type": "function",
        "function": {
            "name": str(block.get("name") or f"tool_{index}"),
            "arguments": arguments,
        },
    }


def anthropic_messages_to_openai(
    *,
    messages: list[dict[str, Any]],
    system: Any = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages blocks to OpenAI chat messages."""
    converted: list[dict[str, Any]] = []
    system_text = _flatten_system_text(system)
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            converted.append({"role": role, "content": _coerce_text(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []

        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "text")
            if block_type == "text":
                text = _coerce_text(block.get("text"))
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use" and role == "assistant":
                tool_calls.append(_assistant_block_to_tool_call(block, index))
            elif block_type == "tool_result" and role == "user":
                tool_messages.append(_tool_result_to_openai_message(block))
            elif block_type == "thinking":
                # OpenAI chat.completions has no first-class thinking block input.
                continue

        if role == "assistant" and tool_calls:
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            if text_parts:
                assistant_message["content"] = "\n".join(text_parts)
            converted.append(assistant_message)
        elif text_parts or role != "assistant":
            converted.append({"role": role, "content": "\n".join(text_parts)})

        converted.extend(tool_messages)

    return converted


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "tool"),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted or None


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice in (None, "auto", "none"):
        return tool_choice
    if tool_choice in ("any", "required"):
        return "required"
    if isinstance(tool_choice, str):
        return {"type": "function", "function": {"name": tool_choice}}
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "").lower()
        if choice_type in {"auto", "none"}:
            return choice_type
        if choice_type in {"any", "required"}:
            return "required"
        if choice_type == "tool":
            return {
                "type": "function",
                "function": {"name": str(tool_choice.get("name") or "tool")},
            }
    return None


def openai_response_to_anthropic(response: Any) -> Any:
    """Convert a chat.completions response object to an Anthropic-like message."""
    choice = (getattr(response, "choices", None) or [None])[0]
    message = getattr(choice, "message", None)
    content: list[Any] = []

    if message is not None:
        message_content = getattr(message, "content", None)
        if isinstance(message_content, str) and message_content:
            content.append(SimpleNamespace(type="text", text=message_content))

        for index, tool_call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(tool_call, "function", None)
            raw_arguments = getattr(function, "arguments", "{}") if function is not None else "{}"
            try:
                parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except Exception:
                parsed_arguments = {}
            content.append(
                SimpleNamespace(
                    type="tool_use",
                    id=str(getattr(tool_call, "id", None) or f"toolu_{index}"),
                    name=str(getattr(function, "name", None) or f"tool_{index}"),
                    input=parsed_arguments or {},
                )
            )

    finish_reason = str(getattr(choice, "finish_reason", "stop") or "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
    }
    usage = getattr(response, "usage", None)
    anthropic_usage = None
    if usage is not None:
        anthropic_usage = SimpleNamespace(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0)
            or ((getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)),
        )

    return SimpleNamespace(
        id=str(getattr(response, "id", "") or ""),
        model=getattr(response, "model", None),
        content=content,
        stop_reason=stop_reason_map.get(finish_reason, "end_turn"),
        usage=anthropic_usage,
        raw=response,
    )


class CommandCodeAnthropicShim:
    """Expose ``messages.create()`` over an OpenAI-compatible client.

    The shim intentionally supports the subset Hermes relies on today:
    system prompts, text messages, tool definitions, tool-choice routing, and
    assistant tool calls. Thinking blocks are ignored on input because the
    chat.completions wire format has no equivalent first-class field.
    """

    def __init__(self, openai_client: Any, default_model: str | None = None):
        self._client = openai_client
        self._default_model = default_model
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model") or self._default_model
        if not model:
            raise ValueError("CommandCodeAnthropicShim requires a model")

        response = self._client.chat.completions.create(
            model=model,
            messages=anthropic_messages_to_openai(
                messages=kwargs.get("messages") or [],
                system=kwargs.get("system"),
            ),
            tools=anthropic_tools_to_openai(kwargs.get("tools")),
            tool_choice=anthropic_tool_choice_to_openai(kwargs.get("tool_choice")),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
        )
        return openai_response_to_anthropic(response)


def build_commandcode_anthropic_profile(
    *,
    env_vars: tuple[str, ...],
    fallback_models: tuple[str, ...],
    models_url: str,
) -> ProviderProfile:
    """Build the ``commandcode-anthropic`` provider profile.

    ``base_url`` intentionally omits ``/v1`` because the Anthropic SDK appends
    ``/v1/messages``. The public model catalog remains the OpenAI-compatible
    ``/provider/v1/models`` endpoint.
    """
    return ProviderProfile(
        name="commandcode-anthropic",
        aliases=("commandcode_claude", "commandcode-anthropic-messages"),
        api_mode="anthropic_messages",
        env_vars=env_vars,
        display_name="CommandCode (Anthropic Messages)",
        description="CommandCode bearer-auth Anthropic Messages compatibility route",
        signup_url="https://commandcode.ai/",
        base_url=COMMANDCODE_ANTHROPIC_BASE_URL,
        models_url=models_url,
        auth_type="api_key",
        fallback_models=fallback_models,
        default_aux_model="claude-haiku-4-5-20251001",
    )


__all__ = [
    "COMMANDCODE_ANTHROPIC_BASE_URL",
    "CommandCodeAnthropicShim",
    "anthropic_messages_to_openai",
    "anthropic_tools_to_openai",
    "anthropic_tool_choice_to_openai",
    "openai_response_to_anthropic",
    "build_commandcode_anthropic_profile",
]
