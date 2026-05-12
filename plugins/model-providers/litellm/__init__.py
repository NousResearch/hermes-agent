"""LiteLLM Proxy — first-class provider.

LiteLLM is an OpenAI-compatible multi-provider gateway that routes
requests to 100+ backends (Anthropic, OpenAI, Ollama, vLLM, z.ai, …).
Because LiteLLM handles backend detection, parameter filtering, and
fallback chains internally, Hermes never needs to probe what sits
behind it.

Configuration (config.yaml)::

    model:
      default: glm-5.1
      provider: litellm
      base_url: http://192.168.1.2:4000/v1

Or interactively::

    hermes model  # → pick "LiteLLM Proxy"

Environment variables:
  LITELLM_API_KEY  — API key for the LiteLLM proxy (optional if the
                     proxy does not require authentication)
  LITELLM_BASE_URL — Override the base URL (optional; can also be set
                     via model.base_url in config.yaml)
"""

from providers import register_provider
from providers.base import ProviderProfile


litellm = ProviderProfile(
    name="litellm",
    aliases=("litellm-proxy",),
    api_mode="chat_completions",
    env_vars=("LITELLM_API_KEY", "LITELLM_BASE_URL"),
    base_url="",  # user-configured via LITELLM_BASE_URL or hermes model
    models_url="",  # falls back to {base_url}/models (OpenAI-compatible)
    auth_type="api_key",
    display_name="LiteLLM Proxy",
    description="Multi-provider gateway — 100+ backends behind a single OpenAI-compatible API",
)

register_provider(litellm)
