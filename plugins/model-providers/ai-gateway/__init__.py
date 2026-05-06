"""Vercel AI Gateway provider profile.

AI Gateway routes to multiple backends. Hermes sends attribution
headers and full reasoning config passthrough.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class VercelAIGatewayProfile(ProviderProfile):
    """Vercel AI Gateway — attribution headers + reasoning passthrough."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = True,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        if supports_reasoning and reasoning_config is not None:
            extra_body["reasoning"] = dict(reasoning_config)
        elif supports_reasoning:
            extra_body["reasoning"] = {"enabled": True, "effort": "medium"}
        return extra_body, {}


vercel = VercelAIGatewayProfile(
    name="ai-gateway",
    aliases=("vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"),
    env_vars=("AI_GATEWAY_API_KEY",),
    base_url="https://ai-gateway.vercel.sh/v1",
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
    },
    default_aux_model="google/gemini-3-flash",
)

register_provider(vercel)
