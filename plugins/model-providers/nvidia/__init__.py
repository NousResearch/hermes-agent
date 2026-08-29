"""NVIDIA NIM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class NvidiaProviderProfile(ProviderProfile):
    """NVIDIA NIM accepts a stricter ToolMessage schema than most OpenAI-compatible APIs."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        needs_sanitize = any(
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and ("name" in msg or "tool_name" in msg)
            for msg in messages
        )
        if not needs_sanitize:
            return messages

        # Copy-on-write: shallow outer-list copy, then a shallow dict copy
        # only for the role:"tool" messages that actually need a field
        # dropped. Avoids recursively deep-copying every message's content
        # (including large tool outputs and attachments) for a turn that
        # only ever needs to touch two top-level keys on a handful of
        # messages. Matches the pattern already used by the shared
        # sanitizer in agent/transports/chat_completions.py and by
        # QwenProfile.prepare_messages().
        sanitized = list(messages)
        for idx, msg in enumerate(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "tool"
                and ("name" in msg or "tool_name" in msg)
            ):
                msg_copy = dict(msg)
                msg_copy.pop("name", None)
                msg_copy.pop("tool_name", None)
                sanitized[idx] = msg_copy
        return sanitized


nvidia = NvidiaProviderProfile(
    name="nvidia",
    aliases=("nvidia-nim",),
    env_vars=("NVIDIA_API_KEY",),
    display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference",
    signup_url="https://build.nvidia.com/",
    fallback_models=(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.3-70b-instruct",
    ),
    base_url="https://integrate.api.nvidia.com/v1",
    default_max_tokens=16384,
)

register_provider(nvidia)
