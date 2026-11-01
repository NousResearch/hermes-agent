"""Leanroute AI Gateway provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


leanroute = ProviderProfile(
    name="leanroute",
    display_name="Leanroute",
    description="Leanroute: OpenAI-compatible AI gateway for 100+ models across 14 providers",
    signup_url="https://leanroute.dev/dashboard/keys",
    env_vars=("LEANROUTE_API_KEY",),
    base_url="https://api.leanroute.dev/v1",
    auth_type="api_key",
    default_aux_model="deepseek/deepseek-v4-1-flash",
    fallback_models=(
        "deepseek/deepseek-v4-1-flash",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.6-luna",
        "google/gemini-3.8-flash",
        "groq/openai/gpt-oss-120b",
        "deepseek/deepseek-v4-pro",
    ),
)

register_provider(leanroute)
