"""Cloudflare AI Gateway provider profile.

This profile targets Cloudflare's OpenAI-compatible `/compat` endpoint in
BYOK / Unified Billing mode. Provider keys stay in Cloudflare; Hermes only
stores and sends the Cloudflare AI Gateway token.
"""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


cloudflare_ai_gateway = ProviderProfile(
    name="cloudflare-ai-gateway",
    aliases=("cloudflare", "cf-ai-gateway", "cf-aig", "cloudflare-aig"),
    display_name="Cloudflare AI Gateway",
    description="Cloudflare AI Gateway (BYOK / Unified Billing)",
    signup_url="https://dash.cloudflare.com/?to=/:account/ai/ai-gateway",
    env_vars=(
        "CLOUDFLARE_AI_GATEWAY_TOKEN",
        "CF_AIG_TOKEN",
        "CLOUDFLARE_AI_GATEWAY_BASE_URL",
    ),
    base_url="https://gateway.ai.cloudflare.com/v1",
    models_url="",
    supports_health_check=False,
    default_headers={
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    },
    default_aux_model="workers-ai/@cf/moonshotai/kimi-k2.6",
)


register_provider(cloudflare_ai_gateway)
