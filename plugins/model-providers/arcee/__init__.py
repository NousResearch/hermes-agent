"""Arcee AI provider profile."""

from providers import register_provider
from providers.model_normalizers import MatchingPrefixProviderProfile

arcee = MatchingPrefixProviderProfile(
    name="arcee", aliases=("arcee-ai", "arceeai"), display_name="Arcee AI",
    description="Arcee AI (Trinity models, direct API)", signup_url="https://chat.arcee.ai/",
    env_vars=("ARCEEAI_API_KEY",), base_url="https://api.arcee.ai/api/v1",
    base_url_env_var="ARCEE_BASE_URL",
)

register_provider(arcee)
