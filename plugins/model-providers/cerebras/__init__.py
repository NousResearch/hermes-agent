"""Cerebras provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


cerebras = ProviderProfile(
    name="cerebras",
    aliases=("cerebras-ai",),
    api_mode="chat_completions",
    env_vars=("CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
    display_name="Cerebras",
    description="Cerebras Inference — OpenAI-compatible high-speed models",
    signup_url="https://cloud.cerebras.ai/",
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
)

register_provider(cerebras)
