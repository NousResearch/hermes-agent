"""TrueFoundry AI Gateway provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

truefoundry = ProviderProfile(
    name="truefoundry",
    aliases=("tfy", "truefoundry-gateway"),
    display_name="TrueFoundry",
    description="TrueFoundry AI Gateway (OpenAI-compatible proxy, 1000+ models, BYO key)",
    signup_url="https://gateway.truefoundry.ai",
    env_vars=("TFY_API_KEY", "TFY_BASE_URL"),
    base_url="",
    auth_type="api_key",
    fallback_models=(),
)

register_provider(truefoundry)
