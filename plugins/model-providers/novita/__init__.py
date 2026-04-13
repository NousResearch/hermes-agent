"""NovitaAI provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


novita = ProviderProfile(
    name="novita",
    aliases=("novita-ai", "novitaai"),
    display_name="NovitaAI",
    description="NovitaAI — 90+ models, pay-per-use",
    signup_url="https://novita.ai/settings/key-management",
    env_vars=("NOVITA_API_KEY", "NOVITA_BASE_URL"),
    base_url="https://api.novita.ai/openai/v1",
    auth_type="api_key",
    default_aux_model="deepseek/deepseek-v3-0324",
    fallback_models=(
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.7",
        "zai-org/glm-5",
        "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b-fp8",
    ),
)

register_provider(novita)
