"""Apiário Dev provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

apiario_dev = ProviderProfile(
    name="apiario-dev",
    env_vars=("APIARIO_API_KEY",),
    display_name="Apiário Dev",
    description="Provedor oficial do Apiário Dev",
    fallback_models=(
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
    ),
    base_url="https://api.apiario.dev/v1", 
)

register_provider(apiario_dev)