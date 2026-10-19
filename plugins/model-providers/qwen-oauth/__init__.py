"""Qwen Portal provider profile."""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _normalize_parts(content: list) -> list | None:
    """List content -> list-of-dict parts (str -> text part, image_url dicts copied,
    other junk dropped). None when nothing changed (copy-on-write)."""
    parts, changed = [], False
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            changed = True
        elif isinstance(part, dict):
            if isinstance(part.get("image_url"), dict):
                part = {**part, "image_url": dict(part["image_url"])}
                changed = True
            parts.append(part)
        else:
            changed = True
    return parts if parts and changed else None


class QwenProfile(ProviderProfile):
    """Qwen Portal — message normalization, vl_high_resolution, metadata top-level."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize content to list-of-dicts and inject cache_control on the (first)
        system message. Copy-on-write: only touched messages/parts are copied."""
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
                prepared[idx] = {**msg, "content": [{"type": "text", "text": content}]}
            elif isinstance(content, list):
                parts = _normalize_parts(content)
                if parts is not None:
                    prepared[idx] = {**msg, "content": parts}
        if system_idx is not None:
            msg = prepared[system_idx]
            content = msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content_copy = list(content)
                content_copy[-1] = {**content_copy[-1], "cache_control": {"type": "ephemeral"}}
                prepared[system_idx] = {**msg, "content": content_copy}
        return prepared

    def build_extra_body(self, *, session_id: str | None = None, **context) -> dict[str, Any]:
        return {"vl_high_resolution_images": True}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, qwen_session_metadata: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Qwen metadata goes to top-level api_kwargs, not extra_body."""
        return {}, {"metadata": qwen_session_metadata} if qwen_session_metadata else {}


qwen = QwenProfile(
    name="qwen-oauth", aliases=("qwen", "qwen-portal", "qwen-cli"), env_vars=("QWEN_API_KEY",),
    base_url="https://portal.qwen.ai/v1", auth_type="oauth_external", default_max_tokens=65536,
)

register_provider(qwen)
