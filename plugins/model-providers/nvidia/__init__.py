"""NVIDIA NIM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class NvidiaProviderProfile(ProviderProfile):
    """NVIDIA NIM accepts a stricter ToolMessage schema than most OpenAI-compatible APIs."""

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

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reasoning controls for NIM's nemotron/glm families.

        Verified against the live API (2026-10-22): nemotron-3.5-lightning emits ~130 hidden
        reasoning tokens for a trivial prompt unless disabled; with chat_template_kwargs.thinking=false
        it emits 0 and answers in 17 completion tokens. Aux calls (titles, compression, triage)
        pass reasoning_config={'enabled': False}; without this hook that disable never reached the
        wire, so every title call burned its whole 512-token budget on hidden reasoning.
        """
        if not reasoning_config:
            return {}, {}
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        enabled = reasoning_config.get("enabled")
        if enabled is not None:
            extra_body["chat_template_kwargs"] = {"thinking": bool(enabled)}
        effort = reasoning_config.get("effort")
        if effort:
            top_level["reasoning_effort"] = effort
        return extra_body, top_level


nvidia = NvidiaProviderProfile(
    name="nvidia", aliases=("nvidia-nim", "nim", "build-nvidia", "nemotron"), env_vars=("NVIDIA_API_KEY",), display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference", signup_url="https://build.nvidia.com/",
    fallback_models=("nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.3-70b-instruct"),
    base_url="https://integrate.api.nvidia.com/v1", default_max_tokens=16384,
)

register_provider(nvidia)
