"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

xiaomi = ProviderProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    # Probe-able: api.xiaomimimo.com/v1/models answers 200 with a valid key (it only
    # 401s for an invalid one) — verified 2026-09-25 with a live pay-as-you-go key.
    # While this was False, `hermes doctor` skipped the probe and printed a synthetic
    # "key configured" ok, so a broken key or endpoint looked healthy.
    supports_health_check=True,
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
