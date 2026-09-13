"""Kyma API provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


kyma = ProviderProfile(
    name="kyma", aliases=("kyma-api",), display_name="Kyma API",
    description="Kyma API: open and frontier models behind one endpoint, uptime measured per model",
    signup_url="https://kymaapi.com/keys", env_vars=("KYMA_API_KEY",),
    base_url="https://api.kymaapi.com/v1", auth_type="api_key",
    default_aux_model="deepseek-v4-flash",
    # Live catalog ids as of 2026-09-04 (GET https://api.kymaapi.com/v1/models).
    # The full catalog is fetched dynamically; this is only the picker fallback.
    fallback_models=(
        "qwen-3.6-plus", "qwen-3-coder", "kimi-k2.6", "glm-5.2",
        "deepseek-v3", "claude-sonnet-4-6", "gemini-3.5-flash-lite",
    ),
)

register_provider(kyma)
