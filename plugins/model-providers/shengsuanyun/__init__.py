from providers import register_provider
from providers.base import ProviderProfile

shengsuanyun = ProviderProfile(
    name="shengsuanyun",
    aliases=("ssy", "sheng-suan-yun"),
    display_name="ShengSuanYun",
    description="胜算云 — 多模型云端 API",
    signup_url="https://shengsuanyun.com/",
    env_vars=("SHENGSUANYUN_API_KEY", "SHENGSUANYUN_BASE_URL"),
    base_url="https://router.shengsuanyun.com/api/v1",
    models_url="https://router.shengsuanyun.com/api/v1/models",
    auth_type="api_key",
    fallback_models=(
        "anthropic/claude-opus-4.7",
        "anthropic/claude-opus-4.5",
        "anthropic/claude-opus-4.6",
        "openai/gpt-5.1",
        "openai/gpt-5.4",
        "openai/gpt-5.3-chat",
        "google/gemini-3-flash",
        "google/gemini-2.5-flash"
    ),
    default_aux_model="google/gemini-2.5-pro",
)

register_provider(shengsuanyun)
