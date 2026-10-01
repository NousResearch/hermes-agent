"""Kilo Code provider profile."""

from providers import register_provider
from providers.model_normalizers import VendorQualifiedProviderProfile

kilocode = VendorQualifiedProviderProfile(
    name="kilocode", aliases=("kilo-code", "kilo", "kilo-gateway"), display_name="Kilo Code",
    description="Kilo Code (Kilo Gateway API)", env_vars=("KILOCODE_API_KEY",), base_url="https://api.kilo.ai/api/gateway",
    base_url_env_var="KILOCODE_BASE_URL", is_aggregator=True,
    default_aux_model="google/gemini-3.6-flash",
)

register_provider(kilocode)
