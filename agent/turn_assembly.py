"""Helpers for API-call-time turn assembly."""

from __future__ import annotations

from typing import Any

from agent.memory_manager import build_memory_context_block


def build_user_context_blocks(
    memory_context: str = "",
    plugin_user_context: str = "",
) -> list[str]:
    blocks: list[str] = []

    if memory_context:
        memory_block = build_memory_context_block(memory_context)
        if memory_block:
            blocks.append(memory_block)

    if plugin_user_context:
        blocks.append(plugin_user_context)

    return blocks


def apply_user_turn_context(
    message: dict[str, Any],
    memory_context: str = "",
    plugin_user_context: str = "",
) -> dict[str, Any]:
    api_message = message.copy()
    if api_message.get("role") != "user":
        return api_message

    content = api_message.get("content", "")
    if not isinstance(content, str):
        return api_message

    blocks = build_user_context_blocks(memory_context, plugin_user_context)
    if blocks:
        api_message["content"] = content + "\n\n" + "\n\n".join(blocks)

    return api_message


def compose_effective_system_prompt(
    base_system: str = "",
    ephemeral_system_prompt: str | None = None,
) -> str:
    effective_system = base_system or ""
    if ephemeral_system_prompt:
        effective_system = (effective_system + "\n\n" + ephemeral_system_prompt).strip()
    return effective_system


def inject_prefill_messages(
    api_messages: list[dict[str, Any]],
    prefill_messages: list[dict[str, Any]] | None,
    effective_system: str,
) -> list[dict[str, Any]]:
    if not prefill_messages:
        return api_messages

    injected_messages = list(api_messages)
    sys_offset = 1 if effective_system else 0
    for idx, prefill_message in enumerate(prefill_messages):
        injected_messages.insert(sys_offset + idx, prefill_message.copy())
    return injected_messages
