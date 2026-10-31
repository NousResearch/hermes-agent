"""API Route provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


api_route = ProviderProfile(
    name="api-route",
    aliases=("api_route", "apiroute"),
    env_vars=("API_ROUTE_API_KEY",),
    display_name="API Route",
    description="API Route — unified OpenAI-compatible API",
    signup_url="https://www.api-route.com",
    base_url="https://global.api-route.com/v1",
    models_url="https://global.api-route.com/v1/models",
)

register_provider(api_route)
