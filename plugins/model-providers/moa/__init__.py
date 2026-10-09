"""Mixture of Agents virtual provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

moa = ProviderProfile(
    name="moa",
    display_name="Mixture of Agents",
    description="Mixture of Agents (named presets; aggregator acts after reference models)",
    auth_type="virtual",
    base_url="moa://local",
    supports_health_check=False,
    supports_model_listing=False,
)

register_provider(moa)
