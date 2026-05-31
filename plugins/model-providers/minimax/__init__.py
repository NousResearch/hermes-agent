"""MiniMax provider profiles (international + China).

Both use anthropic_messages api_mode — their inference_base_url
ends with /anthropic which triggers auto-detection to anthropic_messages.
"""

from providers import register_provider
from providers.base import ProviderProfile

minimax = ProviderProfile(
    name="minimax",
    aliases=("mini-max",),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic",
    auth_type="api_key",
    default_aux_model="MiniMax-M2.7",
)

minimax_cn = ProviderProfile(
    name="minimax-cn",
    aliases=("minimax-china", "minimax_cn"),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",),
    base_url="https://api.minimaxi.com/anthropic",
    auth_type="api_key",
    default_aux_model="MiniMax-M2.7",
)

minimax_oauth = ProviderProfile(
    name="minimax-oauth",
    aliases=("minimax_oauth", "minimax-oauth-io"),
    api_mode="anthropic_messages",
    display_name="MiniMax (OAuth)",
    description="MiniMax via OAuth browser flow — no API key required",
    signup_url="https://api.minimax.io/",
    env_vars=(),  # OAuth — tokens in auth.json, not env
    base_url="https://api.minimax.io/anthropic",
    auth_type="oauth_external",
    default_aux_model="MiniMax-M2.7-highspeed",
)

register_provider(minimax)
register_provider(minimax_cn)
register_provider(minimax_oauth)
