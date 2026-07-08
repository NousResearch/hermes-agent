"""TokenLab provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


_TOKENLAB_AGENTIC_MODELS = (
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-fable-5",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "grok-4.3",
    "grok-4-fast",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "minimax-m3",
)


tokenlab = ProviderProfile(
    name="tokenlab",
    display_name="TokenLab",
    description=(
        "TokenLab multi-model gateway (OpenAI-compatible /v1; native "
        "Responses, Anthropic Messages, and Gemini endpoints available)"
    ),
    signup_url="https://tokenlab.sh",
    env_vars=("TOKENLAB_API_KEY", "TOKENLAB_BASE_URL"),
    base_url="https://api.tokenlab.sh/v1",
    models_url="https://api.tokenlab.sh/v1/models",
    auth_type="api_key",
    default_aux_model="gemini-3.1-flash-lite",
    fallback_models=_TOKENLAB_AGENTIC_MODELS,
)


register_provider(tokenlab)
