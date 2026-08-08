"""SambaNova provider profile.

SambaNova provides an OpenAI-compatible API endpoint for their AI models.
This profile enables users to use SambaNova as a model provider in Hermes.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile

sambanova = ProviderProfile(
    name="sambanova",
    aliases=("sambanova-ai", "sambanovaai"),
    env_vars=("SAMBANOVA_API_KEY", "SAMBANOVA_BASE_URL"),
    display_name="SambaNova",
    description="SambaNova — AI acceleration platform with OpenAI-compatible API",
    signup_url="https://cloud.sambanova.ai/",
    base_url="https://api.sambanova.ai/v1",
    models_url="https://api.sambanova.ai/v1/models",
    auth_type="api_key",
    default_aux_model="gemma-4-31B-it",
    fallback_models=(
        "MiniMax-M2.7",
        "gemma-4-31B-it",
        "gpt-oss-120b",
    ),
)

register_provider(sambanova)