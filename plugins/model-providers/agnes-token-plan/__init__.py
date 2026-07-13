"""Agnes AI Token Plan provider profile.

Agnes AI provides an OpenAI-compatible API at https://apihub.agnes-ai.com/v1.
This profile targets the Token Plan subscription tier, which shares the same
endpoint as the free tier but uses a dedicated RPM and quota pool.

Available and upcoming models:
  Text: agnes-2.0-flash, agnes-2.5-flash (coming soon)
  Image: agnes-image-2.0-flash, agnes-image-2.1-flash,
         agnes-image-2.5-preview (coming soon)
  Video: agnes-video-v2.0, agnes-video-2.5-preview (coming soon)
"""

from providers import register_provider
from providers.base import ProviderProfile

agnes_token_plan = ProviderProfile(
    name="agnes-token-plan",
    aliases=("agnes_tp", "agnes_token_plan"),
    display_name="Agnes AI (Token Plan)",
    description="Agnes AI — Token Plan subscription (higher RPM and quotas)",
    signup_url="https://platform.agnes-ai.com/",
    env_vars=("AGNES_TOKEN_PLAN_API_KEY", "AGNES_BASE_URL"),
    base_url="https://apihub.agnes-ai.com/v1",
    auth_type="api_key",
    default_aux_model="agnes-2.0-flash",
    fallback_models=(
        "agnes-2.0-flash",
        "agnes-2.5-flash",
    ),
)

register_provider(agnes_token_plan)
