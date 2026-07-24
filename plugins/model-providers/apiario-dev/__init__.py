"""Apiário Dev provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

apiario_dev = ProviderProfile(
    name="apiario-dev",
    env_vars=("APIARIO_API_KEY",),
    display_name="Apiário Dev",
    description="Provedor oficial do Apiário Dev",
    fallback_models=(
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5.5",
    ),
    base_url="https://api.apiario.dev/v1", 
)

register_provider(apiario_dev)