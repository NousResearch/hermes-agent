"""Sakana AI Fugu provider profile."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

sakana = ProviderProfile(
    name="sakana",
    aliases=("sakana-ai", "fugu"),
    display_name="Sakana AI",
    description="Sakana AI Fugu — multi-agent LLM API",
    signup_url="https://console.sakana.ai/",
    env_vars=("SAKANA_API_KEY", "SAKANA_BASE_URL"),
    base_url="https://api.sakana.ai/v1",
    auth_type="api_key",
    # Attribution so Sakana can identify traffic from Hermes Agent.
    # The generic profile.default_headers fallback in run_agent.py and
    # agent/auxiliary_client.py picks this up at client construction time.
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    fallback_models=(
        "fugu",
        "fugu-ultra",
    ),
)

register_provider(sakana)
