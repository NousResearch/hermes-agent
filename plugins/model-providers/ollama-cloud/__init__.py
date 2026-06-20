"""Ollama Cloud provider profile.

Ollama Cloud's API only supports strict user/assistant alternation — it
does not accept ``tool``-role messages.  When Hermes makes a tool call,
the resulting ``tool``-role messages cause HTTP 400:
"Conversation roles must alternate user/assistant/user/assistant..."

``OllamaCloudProfile.prepare_messages`` converts tool-role messages into
plain ``user`` messages so the conversation stays within the alternation
constraint that Ollama Cloud enforces.
"""

import copy
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class OllamaCloudProfile(ProviderProfile):
    """Ollama Cloud — collapse tool-role messages for strict alternation."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool-role messages to user-role messages.

        Ollama Cloud only accepts user/assistant alternation.  We:
        1. Strip ``tool_calls`` from assistant messages (the model won't
           generate them anyway since we don't send ``tools``).
        2. Convert ``tool``-role messages to ``user``-role with the tool
           output embedded as readable text.
        3. Merge consecutive user messages that result from the conversion.
        """
        prepared = copy.deepcopy(messages)
        if not prepared:
            return prepared

        result: list[dict[str, Any]] = []
        for msg in prepared:
            if not isinstance(msg, dict):
                result.append(msg)
                continue

            role = msg.get("role")

            # Strip tool_calls from assistant messages — Ollama Cloud
            # won't generate them and the field confuses strict parsers.
            if role == "assistant":
                msg.pop("tool_calls", None)
                msg.pop("function_call", None)
                result.append(msg)
                continue

            # Convert tool-role messages to user-role.
            if role == "tool":
                tool_name = msg.get("name", "tool")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Multimodal tool result — extract text parts only.
                    text_parts = [
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                new_msg = {
                    "role": "user",
                    "content": f"[Tool result from `{tool_name}`]:\n{content}",
                }
                result.append(new_msg)
                continue

            result.append(msg)

        # Merge consecutive same-role messages (happens when multiple
        # tool results are converted to user-role in a row).
        merged: list[dict[str, Any]] = []
        for msg in result:
            if (
                merged
                and merged[-1].get("role") == msg.get("role") == "user"
                and isinstance(merged[-1].get("content"), str)
                and isinstance(msg.get("content"), str)
            ):
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        return merged


ollama_cloud = OllamaCloudProfile(
    name="ollama-cloud",
    aliases=("ollama_cloud",),
    default_aux_model="nemotron-3-nano:30b",
    env_vars=("OLLAMA_API_KEY",),
    base_url="https://ollama.com/v1",
)

register_provider(ollama_cloud)
