"""Qwen Portal provider profile."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class QwenProfile(ProviderProfile):
    """Qwen Portal — message normalization, vl_high_resolution, metadata top-level."""

    @staticmethod
    def _copy_part_if_request_mutable(part: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            copied = dict(part)
            copied["image_url"] = dict(image_url)
            return copied, True
        return part, False

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize content to list-of-dicts format.

        Inject cache_control on system message.

        Matches the behavior of run_agent.py:_qwen_prepare_chat_messages().
        """
        if not messages:
            return []

        prepared = list(messages)
        system_idx: int | None = None

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if system_idx is None and msg.get("role") == "system":
                system_idx = idx
            content = msg.get("content")
            if isinstance(content, str):
                msg_copy = dict(msg)
                msg_copy["content"] = [{"type": "text", "text": content}]
                prepared[idx] = msg_copy
            elif isinstance(content, list):
                normalized_parts = []
                changed = False
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                        changed = True
                    elif isinstance(part, dict):
                        normalized_part, copied = self._copy_part_if_request_mutable(part)
                        normalized_parts.append(normalized_part)
                        changed = changed or copied
                    else:
                        changed = True
                if normalized_parts and changed:
                    msg_copy = dict(msg)
                    msg_copy["content"] = normalized_parts
                    prepared[idx] = msg_copy

        # Inject cache_control on the last part of the system message.
        if system_idx is not None:
            msg = prepared[system_idx]
            if isinstance(msg, dict):
                content = msg.get("content")
                if (
                    isinstance(content, list)
                    and content
                    and isinstance(content[-1], dict)
                ):
                    msg_copy = dict(msg)
                    content_copy = list(content)
                    content_copy[-1] = dict(content_copy[-1])
                    content_copy[-1]["cache_control"] = {"type": "ephemeral"}
                    msg_copy["content"] = content_copy
                    prepared[system_idx] = msg_copy

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
