"""LM Studio local inference provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

lmstudio = ProviderProfile(
    name="lmstudio",
    aliases=("lm-studio", "lm_studio"),
    display_name="LM Studio", description="LM Studio (Local desktop app with built-in model server)",
    env_vars=("LM_API_KEY",),
    base_url="http://127.0.0.1:1234/v1",
    base_url_env_var="LM_BASE_URL",
)

register_provider(lmstudio)
