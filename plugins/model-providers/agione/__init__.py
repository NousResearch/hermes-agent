"""AGIone provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


agione = ProviderProfile(
    name="agione",
    aliases=("agi-one", "agione-pro"),
    display_name="AGIone",
    description="AGIone — OpenAI-compatible multi-vendor inference API",
    signup_url="https://agione.pro/",
    env_vars=("AGIONE_API_KEY", "AGIONE_BASE_URL"),
    base_url="https://agione.pro/hyperone/xapi/api/v1",
    models_url="https://agione.pro/hyperone/xapi/api/models",
    auth_type="api_key",
    default_aux_model="deepseek/deepseek-v4-pro/d3462",
    fallback_models=(
        "deepseek/deepseek-v4-pro/d3462",
        "openai/GPT-5.5/c6fbe",
        "anthropic/Claude-opus-4.7/a4d5d",
    ),
)

register_provider(agione)
