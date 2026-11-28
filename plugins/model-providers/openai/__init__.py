"""OpenAI provider profile (ChatGPT models via API key)."""

from providers import register_provider
from providers.base import ProviderProfile

openai = ProviderProfile(
    name="openai",
    aliases=("openai-api", "gpt"),
    api_mode="chat_completions",
    env_vars=("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    base_url="https://api.openai.com/v1",
    auth_type="api_key",
    supports_streaming=True,
    supports_vision=True,
    default_aux_model="gpt-5-mini",
    # No hardcoded model list: OpenAI's /v1/models catalog is the source of truth,
    # so an empty picker beats silently routing to a retired model.
    fallback_models=(),
)

register_provider(openai)
