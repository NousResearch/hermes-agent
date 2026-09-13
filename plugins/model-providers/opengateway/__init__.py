"""OpenGateway provider profile.

OpenGateway is an OpenAI-compatible gateway, so the shared chat-completions
transport is the correct integration point. Apitopia adds gateway-local
fallbacks, repetition protection, and Kimi ultrafast output clamping; those
are deliberately not injected here because direct Hermes requests have a
different ownership boundary and must preserve the provider contract.
"""

from providers import register_provider
from providers.base import ProviderProfile


opengateway = ProviderProfile(
    name="opengateway",
    display_name="OpenGateway",
    description="OpenGateway — OpenAI-compatible gateway for production LLM routing",
    signup_url="https://opengateway.ai/api-keys",
    env_vars=("OPENGATEWAY_API_KEY",),
    base_url="https://apis.opengateway.ai/v1",
    fallback_models=(
        "moonshotai/kimi-k3-ultrafast",
        "z-ai/glm-5.2-ultrafast",
    ),
)

register_provider(opengateway)
