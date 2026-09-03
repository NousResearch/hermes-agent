"""OpenCode Free provider profile.

OpenCode's free model tier on the Zen relay (https://opencode.ai/zen/v1).
KEYLESS: the relay serves free-tier models anonymously and rejects any
Authorization bearer it doesn't recognize with 401 — so this provider
never sends a credential at all (the runtime resolver pins the keyless
placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). No OpenCode account needed.
Select via ``hermes model`` or ``/model free``.
"""

from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

# Attribution headers, same values as the opencode-zen/go profiles, plus the
# empty Authorization override that keeps the SDK's "Bearer <placeholder>"
# off the wire (the free tier 401s any unrecognized bearer).
_KEYLESS_HEADERS = {
    "Authorization": "",
    "HTTP-Referer": "https://hermes-agent.nousresearch.com",
    "X-Title": "Hermes Agent",
    "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
}


class OpenCodeFreeProfile(ProviderProfile):
    """OpenCode Free — keyless, with Ox Alpha reasoning controls.

    Ox Alpha (x-preview-f-free) is reachable through this provider as well
    as opencode-zen; both share the same wire contract (reasoning_effort
    accepts exactly low/high/max — anything else 400s). The translation
    lives in the zen plugin; resolve it through the registered zen profile's
    module so the two providers can never drift.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            import sys

            from providers import get_provider_profile

            zen_profile = get_provider_profile("opencode-zen")
            zen_module = sys.modules[type(zen_profile).__module__]
            return zen_module._build_ox_alpha_reasoning_extras(reasoning_config, model)
        except Exception:
            return {}, {}


opencode_free = OpenCodeFreeProfile(
    name="opencode-free",
    aliases=("free", "opencode_free"),
    env_vars=(),  # keyless — nothing to configure
    base_url="https://opencode.ai/zen/v1",
    display_name="OpenCode Free",
    description="OpenCode free models — keyless, no account needed",
    default_headers=dict(_KEYLESS_HEADERS),
    # laguna is the fastest non-UA-gated free model; big-pickle 429s every
    # client except the opencode CLI's own User-Agent (verified 2026-08-21).
    default_aux_model="laguna-s-2.1-free",
)

register_provider(opencode_free)
