"""MiniMax provider profiles (international + China + OAuth).

All three use anthropic_messages api_mode — their inference ``base_url``
ends with ``/anthropic`` which triggers auto-detection to anthropic_messages.

The catalog endpoint lives at the OpenAI-compatible ``/v1/models`` path
on the same host (NOT under ``/anthropic``), so we set ``models_url``
explicitly.  Without this, the default ``ProviderProfile.fetch_models``
hits ``<base_url>/models`` → ``https://api.minimax.io/anthropic/models``,
which 404s, and the live catalog never makes it into the /model picker.
"""

from providers import register_provider
from providers.base import ProviderProfile

# OpenAI-compat /v1/models catalog endpoints, paired with the inference
# base URLs above.  Keep these in sync if MiniMax ever moves the catalog.
_MINIMAX_INTL_MODELS_URL = "https://api.minimax.io/v1/models"
_MINIMAX_CN_MODELS_URL = "https://api.minimaxi.com/v1/models"

minimax = ProviderProfile(
    name="minimax",
    aliases=("mini-max",),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_API_KEY",),
    base_url="https://api.minimax.io/anthropic",
    models_url=_MINIMAX_INTL_MODELS_URL,
    auth_type="api_key",
    default_aux_model="MiniMax-M2.7",
)

minimax_cn = ProviderProfile(
    name="minimax-cn",
    aliases=("minimax-china", "minimax_cn"),
    api_mode="anthropic_messages",
    env_vars=("MINIMAX_CN_API_KEY",),
    base_url="https://api.minimaxi.com/anthropic",
    models_url=_MINIMAX_CN_MODELS_URL,
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
    models_url=_MINIMAX_INTL_MODELS_URL,
    auth_type="oauth_external",
    default_aux_model="MiniMax-M2.7-highspeed",
)

register_provider(minimax)
register_provider(minimax_cn)
register_provider(minimax_oauth)
