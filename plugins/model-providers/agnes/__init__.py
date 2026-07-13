"""Agnes AI (Free / Default) provider profile.

Agnes AI provides an OpenAI-compatible API at https://apihub.agnes-ai.com/v1.
This profile targets the free / default access tier, which shares the same
endpoint as the Token Plan but uses a separate RPM and quota pool.

Available and upcoming models:
  Text: agnes-2.0-flash, agnes-2.5-flash (coming soon)
  Image: agnes-image-2.0-flash, agnes-image-2.1-flash,
         agnes-image-2.5-preview (coming soon)
  Video: agnes-video-v2.0, agnes-video-2.5-preview (coming soon)

For higher RPM limits, use the `agnes-token-plan` provider instead.
"""

from providers import register_provider
from providers.base import ProviderProfile

agnes = ProviderProfile(
    name="agnes",
    aliases=("agnes-ai", "agnes_free"),
    display_name="Agnes AI (Free)",
    description="Agnes AI — free omni-modal AI API (OpenAI-compatible)",
    signup_url="https://platform.agnes-ai.com/",
    env_vars=("AGNES_API_KEY", "AGNES_BASE_URL"),
    base_url="https://apihub.agnes-ai.com/v1",
    auth_type="api_key",
    default_aux_model="agnes-2.0-flash",
    fallback_models=(
        "agnes-2.0-flash",
        "agnes-2.5-flash",
    ),
)

register_provider(agnes)
