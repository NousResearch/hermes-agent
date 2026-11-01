"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

xiaomi = ProviderProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=True,  # re-verified 2026-09-25: api.xiaomimimo.com accepts list-type tool content with image_url (mimo-v2.5 and mimo-v2.6-flash); the #41072 veto's 400 "text is not set" no longer reproduces
)

register_provider(xiaomi)
