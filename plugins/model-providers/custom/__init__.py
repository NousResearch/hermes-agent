"""Custom / Ollama (local) provider profile.

Covers any endpoint registered as provider="custom", including local
Ollama instances. Key quirks:
  - ollama_num_ctx → extra_body.options.num_ctx (local context window)
  - reasoning_config disabled → extra_body.think = False
  - reasoning_config enabled → extra_body.reasoning passthrough (#55276)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Effort levels accepted by the OpenAI-compatible reasoning format.
_REASONING_EFFORT_LEVELS = {"minimal", "low", "medium", "high", "xhigh"}


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false, num_ctx, and reasoning passthrough."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        # Ollama context window
        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)

            if _effort == "none" or _enabled is False:
                # Disable thinking (Ollama convention)
                extra_body["think"] = False
            elif _enabled and _effort:
                # Pass through reasoning config for OpenAI-compatible
                # backends (vLLM, Qwen3, etc.) that accept the standard
                # extra_body.reasoning format.  Previously this was
                # silently dropped for custom providers, leaving users
                # with zero effect from their reasoning_effort setting.
                # (#55276)
                extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": _effort if _effort in _REASONING_EFFORT_LEVELS else "medium",
                }

        return extra_body, {}

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Custom/Ollama: base_url is user-configured; fetch if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


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
    # Without this, no max_tokens is sent and Ollama falls back to its internal
    # num_predict=128, truncating responses after a few tokens (#39281). This is
    # only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
