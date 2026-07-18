"""BytePlus provider profile.

OpenAI-compatible chat-completions endpoint. Auto-registered via
``register_provider`` so auth, ``hermes model``, runtime resolution, and
OPTIONAL_ENV_VARS pick it up without core edits.
"""

from providers import register_provider
from providers.base import ProviderProfile


byteplus = ProviderProfile(
    name="byteplus",
    aliases=('byteplus-ark',),
    display_name="BytePlus",
    description="BytePlus (international ARK) — OpenAI-compatible model API",
    signup_url="https://console.byteplus.com/",
    env_vars=('BYTEPLUS_API_KEY', 'BYTEPLUS_BASE_URL'),
    base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="deepseek-v3-2-251201",
    fallback_models=('deepseek-v3-2-251201', 'seed-1-8-251228'),
)

register_provider(byteplus)
