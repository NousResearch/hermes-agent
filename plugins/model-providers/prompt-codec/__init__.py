"""prompt-codec densify proxy provider profile.

Routes Hermes chat completions through a local OpenAI-compatible densify
proxy (default http://127.0.0.1:8787/v1) that rewrites prompts with
fence-safe rules + optional Ollama/MLX refine before they hit paid APIs.

This does NOT replace Hermes ContextCompressor / compression: — densify
runs on every outbound request; context compaction still runs mid-session.

Install proxy: https://github.com/jwaynelowry/prompt-codec
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class PromptCodecProfile(ProviderProfile):
    """OpenAI-compatible densify sidecar (chat_completions transport)."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Forward reasoning_effort=none so thinking-capable upstream models
        # (and the local densify encoder when Hermes probes) do not burn the
        # output budget. Harmless if the endpoint ignores the field.
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if reasoning_config and isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            enabled = reasoning_config.get("enabled", True)
            if effort == "none" or enabled is False:
                top_level["reasoning_effort"] = "none"
            elif effort:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


prompt_codec = PromptCodecProfile(
    name="prompt_codec",
    aliases=(
        "prompt-codec",
        "densify",
        "promptcodec",
    ),
    display_name="Prompt Codec (local densify)",
    description=(
        "Local densify proxy on :8787 — shrinks outbound prompts before paid "
        "APIs. Requires prompt-codec + Ollama/MLX. Stacks with Hermes compression."
    ),
    signup_url="https://github.com/jwaynelowry/prompt-codec",
    env_vars=("X_API_KEY",),
    base_url="http://127.0.0.1:8787/v1",
    default_aux_model="",
)

register_provider(prompt_codec)
