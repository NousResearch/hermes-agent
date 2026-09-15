"""Fireworks AI provider profile. Models are addressed by full catalog ID
(``accounts/fireworks/models/<slug>``), tracking fw-ai/fireconnect ``setup-cli``."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


fireworks = ProviderProfile(
    name="fireworks", aliases=("fireworks-ai", "fw"), display_name="Fireworks AI",
    description="Fireworks AI — OpenAI-compatible direct model API",
    signup_url="https://app.fireworks.ai/settings/users/api-keys", env_vars=("FIREWORKS_API_KEY",),
    base_url="https://api.fireworks.ai/inference/v1", auth_type="api_key",
    # Attribution headers (canonical Hermes set); via default_headers so they
    # survive switch_model and credential rotation.
    default_headers={
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    },
    default_aux_model="accounts/fireworks/models/glm-5p2",
    # Picker safety net when the live catalog fetch fails.
    fallback_models=(
        "accounts/fireworks/models/kimi-k2p6", "accounts/fireworks/models/glm-5p2",
        "accounts/fireworks/models/kimi-k2p7-code",
    ),
)

register_provider(fireworks)
