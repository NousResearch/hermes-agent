"""xAI (Grok) provider profile."""

from hermes_cli.version_info import get_version_info
from providers import register_provider
from providers.model_normalizers import MatchingPrefixProviderProfile

xai = MatchingPrefixProviderProfile(
    name="xai", aliases=("grok", "x-ai", "x.ai"), display_name="xAI",
    description="xAI Grok (Direct API)", signup_url="https://console.x.ai/", api_mode="codex_responses",
    env_vars=("XAI_API_KEY",), base_url="https://api.x.ai/v1", base_url_env_var="XAI_BASE_URL",
    auth_type="api_key",
    default_headers={"User-Agent": f"Hermes-Agent/{get_version_info().base_version}"},
)

register_provider(xai)
