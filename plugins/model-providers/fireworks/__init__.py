"""Fireworks AI provider profile. Models are addressed by full catalog ID
(``accounts/fireworks/models/<slug>``), tracking fw-ai/fireconnect ``setup-cli``."""

from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


class FireworksProfile(ProviderProfile):
    """Map Hermes reasoning controls onto Fireworks' OpenAI-compatible wire."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Fireworks rejects the nested ``extra_body.reasoning`` fallback used
        # by some OpenAI-compatible gateways.  It accepts the standard
        # top-level ``reasoning_effort=none`` control for disabling thinking.
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            return {}, {"reasoning_effort": "none"}
        return {}, {}


fireworks = FireworksProfile(
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
