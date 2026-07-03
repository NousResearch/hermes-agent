"""Kenari provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


kenari = ProviderProfile(
    name="kenari",
    display_name="Kenari",
    description="Kenari: Indonesian OpenAI-compatible AI gateway billed in Rupiah (IDR); one kn- API key covers Claude, GPT, DeepSeek, GLM, Kimi and more",
    signup_url="https://kenari.id/login?next=/keys",
    env_vars=("KENARI_API_KEY", "KENARI_BASE_URL"),
    base_url="https://kenari.id/v1",
    auth_type="api_key",
    default_aux_model="glm-5-2",
    fallback_models=(
        "glm-5-2",
        "claude-sonnet-5",
        "gpt-5-5",
        "deepseek-v4-pro",
        "kimi-k2-7-code",
        "minimax-m3",
    ),
)

register_provider(kenari)
