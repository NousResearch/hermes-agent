"""NovitaAI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


novita = ProviderProfile(
    name="novita", aliases=("novita-ai", "novitaai"), display_name="NovitaAI",
    description="NovitaAI — AI-native cloud for builders and agents",
    signup_url="https://novita.ai/settings/key-management", env_vars=("NOVITA_API_KEY", "NOVITA_BASE_URL"),
    base_url="https://api.novita.ai/openai/v1", auth_type="api_key",
    default_aux_model="deepseek/deepseek-v4-flash",
    fallback_models=(
        "deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro", "zai-org/glm-5.3",
        "moonshotai/kimi-k3", "minimax/minimax-m3", "qwen/qwen3.8-flash",
    ),
)

register_provider(novita)
