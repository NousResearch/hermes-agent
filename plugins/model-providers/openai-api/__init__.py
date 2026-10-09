"""OpenAI direct API provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

openai_api = ProviderProfile(
    name="openai-api",
    display_name="OpenAI API", description="OpenAI API (api.openai.com, API key)",
    api_mode="codex_responses",
    env_vars=("OPENAI_API_KEY",),
    base_url="https://api.openai.com/v1",
    base_url_env_var="OPENAI_BASE_URL",
)

register_provider(openai_api)
