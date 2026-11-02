"""Scaleway Generative APIs provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

scaleway = ProviderProfile(
    name="scaleway", aliases=("scw",), display_name="Scaleway",
    description="Scaleway Generative APIs — European OpenAI-compatible inference (pay-per-use)",
    signup_url="https://console.scaleway.com/iam/api-keys", env_vars=("SCW_SECRET_KEY", "SCALEWAY_BASE_URL"),
    base_url="https://api.scaleway.ai/v1", auth_type="api_key",
    # Chat Completions accepts image content in messages, tool results included.
    supports_vision=True,
    default_aux_model="mistral-small-3.2-24b-instruct-2506",
    # [0] is the setup default. Serverless models that tool-call on the
    # chat-completions wire; gpt-oss-120b is deliberately absent — Scaleway
    # serves its tool-calling on /v1/responses only (Responses API, beta).
    fallback_models=(
        "deepseek-v4.1-flash-1017",
        "glm-5.3", "glm-5.2",
        "qwen3.5-397b-a17b", "qwen3.8-27b", "qwen3.6-35b-a3b",
        "mistral-medium-3.5-128b", "mistral-small-3.2-24b-instruct-2506",
        "gemma-4-26b-a4b-it", "llama-3.3-70b-instruct", "qwen3-235b-a22b-instruct-2507",
    ),
)

register_provider(scaleway)
