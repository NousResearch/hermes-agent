"""Groq Cloud provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

groq = ProviderProfile(
    name="groq",
    aliases=("groq-cloud", "groqcloud"),
    env_vars=("GROQ_API_KEY",),
    display_name="Groq",
    description="Groq Cloud — fast OpenAI-compatible inference",
    signup_url="https://console.groq.com/keys",
    base_url="https://api.groq.com/openai/v1",
    default_aux_model="llama-3.1-8b-instant",
    fallback_models=(
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ),
)

register_provider(groq)
