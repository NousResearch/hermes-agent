"""NVIDIA NIM provider profile."""

import logging
from typing import Any

from agent.nim_reasoning import (
    is_glm5_nim_model,
    is_nemotron_3_ultra_nim_model,
    is_nim_thinking_model,
    normalize_nim_reasoning_effort,
)
from providers import register_provider
from providers.base import ProviderProfile


logger = logging.getLogger(__name__)


class NvidiaProviderProfile(ProviderProfile):
    """NVIDIA NIM accepts a stricter ToolMessage schema than most OpenAI-compatible APIs."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict) or not is_nim_thinking_model(model):
            return {}, {}

        enabled = reasoning_config.get("enabled", True)
        raw_effort, nim_effort = normalize_nim_reasoning_effort(reasoning_config.get("effort"))
        if enabled is False or raw_effort == "none":
            if is_glm5_nim_model(model):
                template = {"enable_thinking": False, "clear_thinking": False}
            elif is_nemotron_3_ultra_nim_model(model):
                template = {"enable_thinking": False}
            else:
                template = {"thinking": False}
            return {"chat_template_kwargs": template}, {}

        if is_nemotron_3_ultra_nim_model(model):
            template = {"enable_thinking": True}
            if nim_effort in {"low", "medium"}:
                template["medium_effort"] = True
            return {"chat_template_kwargs": template}, {}

        if raw_effort not in {"low", "medium", "high"}:
            logger.info(
                "NIM reasoning_effort: clamping %s → %s (NIM supports low/medium/high)",
                raw_effort,
                nim_effort,
            )

        if is_glm5_nim_model(model):
            template = {
                "enable_thinking": True,
                "clear_thinking": False,
                "reasoning_effort": nim_effort,
            }
        else:
            template = {"thinking": True, "reasoning_effort": nim_effort}
        return {"chat_template_kwargs": template}, {}
    @staticmethod
    def _needs_strip(msg: Any) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool" and ("name" in msg or "tool_name" in msg)

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy-on-write: only tool messages that lose a field are copied
        (no deep copy of large tool outputs); untouched input returned as-is."""
        if not any(self._needs_strip(msg) for msg in messages):
            return messages
        return [
            {k: v for k, v in msg.items() if k not in ("name", "tool_name")} if self._needs_strip(msg) else msg
            for msg in messages
        ]


nvidia = NvidiaProviderProfile(
    name="nvidia", aliases=("nvidia-nim",), env_vars=("NVIDIA_API_KEY",), display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference", signup_url="https://build.nvidia.com/",
    fallback_models=("nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.3-70b-instruct"),
    base_url="https://integrate.api.nvidia.com/v1", default_max_tokens=16384,
)

register_provider(nvidia)
