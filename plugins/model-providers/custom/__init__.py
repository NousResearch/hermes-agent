"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances. Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
"""

from typing import Any

from agent.models_dev import infer_semantic_provider_for_model
from agent.reasoning_efforts import resolve_reasoning_effort
from providers import register_provider
from providers.base import ProviderProfile


_OPENAI_FAMILY_CUSTOM_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
_OPENAI_FAMILY_CUSTOM_EFFORT_ALIASES = {"max": "xhigh"}


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        # Disable thinking when reasoning is turned off; otherwise, relay
        # OpenAI-family custom endpoints inherit OpenAI's top-level
        # reasoning_effort contract from the model slug (e.g. gpt-5.5).
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                extra_body["think"] = False
            elif _effort:
                model = str(ctx.get("model") or "")
                if infer_semantic_provider_for_model("custom", model) == "openai":
                    _wire_effort = resolve_reasoning_effort(
                        _effort,
                        allowed=_OPENAI_FAMILY_CUSTOM_REASONING_EFFORTS,
                        aliases=_OPENAI_FAMILY_CUSTOM_EFFORT_ALIASES,
                    )
                    if _wire_effort:
                        top_level["reasoning_effort"] = _wire_effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not self.base_url:
            return None
        return super().fetch_models(api_key=api_key, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(
        "ollama",
        "local",
        "vllm",
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
    ),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
)

register_provider(custom)
