"""Qwen Portal provider profile."""

import copy
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class QwenProfile(ProviderProfile):
    """Qwen Portal — message normalization, vl_high_resolution, metadata top-level."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize content to list-of-dicts format.

        Inject cache_control on system message.

        Matches the behavior of run_agent.py:_qwen_prepare_chat_messages().
        """
        prepared = copy.deepcopy(messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # Inject cache_control on the last part of the system message.
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[-1], dict)
                ):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def build_extra_body(
        self, *, session_id: str | None = None, **context
    ) -> dict[str, Any]:
        return {"vl_high_resolution_images": True}

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        qwen_session_metadata: dict | None = None,
        **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Qwen metadata goes to top-level api_kwargs, not extra_body."""
        top_level = {}
        if qwen_session_metadata:
            top_level["metadata"] = qwen_session_metadata
        return {}, top_level


qwen = QwenProfile(
    name="qwen-oauth",
    aliases=("qwen", "qwen-portal", "qwen-cli"),
    env_vars=("QWEN_API_KEY",),
    base_url="https://portal.qwen.ai/v1",
    auth_type="oauth_external",
    default_max_tokens=65536,
)

register_provider(qwen)
