"""LLMRouter provider profile.

LLMRouter (https://llmrouter.sh) is a hosted, OpenAI-compatible LLM gateway.
Sending ``model="auto"`` routes each request to the cheapest model that meets
the query's needs; sending a vendor-slugged model id (e.g.
``anthropic/claude-sonnet-4.6``) pins to that model directly. Auth is a
developer API key (``llmr_sk_...``) via ``Authorization: Bearer``.
"""

from providers import register_provider
from providers.base import ProviderProfile

llmrouter = ProviderProfile(
    name="llmrouter",
    aliases=("llm-router", "llmr"),
    display_name="LLMRouter",
    description='LLMRouter — model="auto" routes to the cheapest model that fits',
    signup_url="https://llmrouter.sh/signup",
    env_vars=("LLMROUTER_API_KEY", "LLMROUTER_BASE_URL"),
    base_url="https://llmrouter.sh/v1",
    auth_type="api_key",
    default_aux_model="auto",
    fallback_models=(
        "auto",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3-flash-preview",
    ),
)

register_provider(llmrouter)
