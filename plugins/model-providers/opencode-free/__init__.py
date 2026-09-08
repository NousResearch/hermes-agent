"""OpenCode Free provider profile: the free tier on the Zen relay (https://opencode.ai/zen/v1).

KEYLESS: the relay serves free-tier models anonymously and 401s any bearer it
doesn't recognize, so this provider never sends a credential (the runtime
resolver pins the keyless placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). Select via ``/model free``.
"""

from typing import Any
from providers import register_provider
from providers.base import ProviderProfile


class OpenCodeFreeProfile(ProviderProfile):
    """OpenCode Free — keyless, with Ox Alpha reasoning controls.

    Ox Alpha (x-preview-f-free) is also reachable via opencode-zen with the same wire
    contract; the translation lives in the zen plugin and is resolved through the
    registered zen profile's module so the two providers can never drift.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            import sys

            from providers import get_provider_profile

            zen_module = sys.modules[type(get_provider_profile("opencode-zen")).__module__]
            return zen_module._build_ox_alpha_reasoning_extras(reasoning_config, model)
        except Exception:
            return {}, {}


opencode_free = OpenCodeFreeProfile(
    name="opencode-free", aliases=("free", "opencode_free"),
    env_vars=(),  # keyless — nothing to configure
    base_url="https://opencode.ai/zen/v1", display_name="OpenCode Free",
    description="OpenCode free models — keyless, no account needed",
    # Attribution headers (same values as opencode-zen/go) plus the empty Authorization
    # override that keeps the SDK's "Bearer <placeholder>" off the wire (free tier 401s it).
    # User-Agent must be opencode/<semver>: the Zen relay 429s other UAs (issue #42074).
    # The zero-key runtime path (agent_init / auxiliary_client) builds these via
    # hermes_cli.models.opencode_zen_free_headers(); the session affinity is added
    # per-request by agent.opencode_affinity.merge_opencode_session_headers. This dict
    # keeps the same contract for consumers that read profile.default_headers directly.
    default_headers={
        "Authorization": "",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": "opencode/1.18.25",
    },
    # laguna is the fastest non-UA-gated free model; big-pickle 429s every
    # client except the opencode CLI's own User-Agent.
    default_aux_model="laguna-s-2.1-free",
)

register_provider(opencode_free)
