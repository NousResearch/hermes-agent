"""BharatRouter provider profile. India-resident, OpenAI-compatible gateway
(Krutrim, Sarvam + 140+ global models on one key). Keys are ``br-...``; the
live catalog is discovered from ``{base_url}/models`` (GET /v1/models)."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


bharatrouter = ProviderProfile(
    name="bharatrouter", aliases=("bharat-router", "br"), display_name="BharatRouter",
    description="BharatRouter — India-resident OpenAI-compatible gateway (Krutrim, Sarvam + global)",
    signup_url="https://bharatrouter.com/console", env_vars=("BHARATROUTER_API_KEY",),
    base_url="https://api.bharatrouter.com/v1", auth_type="api_key",
    # Attribution headers (canonical Hermes set); via default_headers so they
    # survive switch_model and credential rotation.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    },
)

register_provider(bharatrouter)
