"""NVIDIA NIM provider profile."""

import copy
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

        sanitized = copy.deepcopy(messages)
        for msg in sanitized:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                msg.pop("name", None)
                msg.pop("tool_name", None)
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
