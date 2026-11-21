"""Token Kiosk provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

token_kiosk = ProviderProfile(
    name="token-kiosk",
    aliases=("tokenkiosk", "token_kiosk", "tokenrouter"),
    display_name="Token Kiosk",
    description="Token Kiosk — multi-model AI routing gateway",
    signup_url="https://token-kiosk.gaib.ai",
    env_vars=("TOKEN_KIOSK_API_KEY", "TOKEN_KIOSK_BASE_URL"),
    base_url="https://api-token-kiosk.gaib.ai/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="gemini-2.5-flash",
    fallback_models=(
        "claude-3-5-sonnet",
        "gpt-4o",
        "deepseek-r1",
        "gemini-2.5-flash",
    ),
)

register_provider(token_kiosk)
