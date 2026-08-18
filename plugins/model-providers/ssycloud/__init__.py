"""SSYCloud (ShengSuanYun / 胜算云) provider profile.

SSYCloud exposes a multi-vendor catalog through an OpenAI-compatible
Chat Completions API.  Native model IDs keep their vendor prefix, for example
``deepseek/deepseek-v4-flash``.
"""

from providers import register_provider
from providers.base import ProviderProfile


ssycloud = ProviderProfile(
    name="ssycloud",
    aliases=("shengsuanyun", "ssy-cloud"),
    display_name="SSYCloud (胜算云)",
    description="SSYCloud — 300+ LLM and multimodel API router",
    signup_url="https://console.shengsuanyun.com/user/keys",
    env_vars=("SSYCLOUD_API_KEY",),
    base_url="https://router.shengsuanyun.com/api/v1",
    api_mode="chat_completions",
    auth_type="api_key",
    # Official docs use this agentic, tool-capable model as the default.
    # The live /v1/models catalog remains authoritative when reachable.
    fallback_models=("deepseek/deepseek-v4-flash",),
    default_aux_model="deepseek/deepseek-v4-flash",
)

register_provider(ssycloud)
