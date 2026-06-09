"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

# XIAOMI_BASE_URL is declared alongside XIAOMI_API_KEY in env_vars so the
# auto-bridge in hermes_cli.auth populates PROVIDER_REGISTRY[*].base_url_env_var
# and users can repoint the inference endpoint (e.g. to a region-routed
# token-plan gateway such as https://token-plan-sgp.xiaomimimo.com/v1).
xiaomi = ProviderProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY", "XIAOMI_BASE_URL"),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni and mimo-v2.5 are vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
    fallback_models=(
        # Current token-plan catalog (June 2026). V2-Pro/V2-Omni auto-route
        # to V2.5; Flash/UltraSpeed require separate tier enrollment.
        "mimo-v2.5-pro",   # flagship reasoning, 1M ctx, deep thinking
        "mimo-v2.5",       # omni: text + image + audio + video, 1M ctx
        "mimo-v2-pro",     # legacy → auto-routes to v2.5 (deprecates 2026-06-30)
        "mimo-v2-omni",    # legacy → auto-routes to v2.5 (deprecates 2026-06-30)
    ),
)

register_provider(xiaomi)
