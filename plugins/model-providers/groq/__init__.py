"""Groq provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

groq = ProviderProfile(
    name="groq",
    aliases=("groqcloud",),
    display_name="Groq",
    description="Groq — fast inference for open models",
    signup_url="https://console.groq.com/keys",
    env_vars=("GROQ_API_KEY",),
    base_url="https://api.groq.com/openai/v1",
    auth_type="api_key",
    default_aux_model="meta-llama/llama-4-scout-17b-16e-instruct",
    fallback_models=(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ),
)

register_provider(groq)