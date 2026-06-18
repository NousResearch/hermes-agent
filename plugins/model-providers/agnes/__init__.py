from providers import register_provider
from providers.base import ProviderProfile

agnes = ProviderProfile(
    name="agnes",
    aliases=("agnes-ai",),
    display_name="Agnes AI",
    description="Agnes AI, OpenAI-compatible API, free, Agnes AI ranks Top 10 AI lab globally on Artificial Analysis and Claw-Eval benchmarks",
    signup_url="https://platform.agnes-ai.com",
    env_vars=("AGNES_API_KEY",),
    base_url="https://apihub.agnes-ai.com/v1",
    auth_type="api_key",
    default_aux_model="agnes-2.0-flash",
    fallback_models=(
        "agnes-2.0-flash",
    ),
)
register_provider(agnes)
