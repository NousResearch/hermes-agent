"""xAI (Grok) provider profile."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

xai = ProviderProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), api_mode="codex_responses", env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1", auth_type="api_key",
    default_headers={"User-Agent": f"Hermes-Agent/{_HERMES_VERSION}"},
    # Day-zero metadata for the direct API, before external catalogs catch up.
    model_capabilities={
        "grok-4.7": {
            "context_window": 500_000,
            "supports_vision": True,
            "supports_tools": True,
            "supports_reasoning": True,
        },
    },
)

register_provider(xai)
