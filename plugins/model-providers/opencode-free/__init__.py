"""OpenCode Free provider profile: the free tier on the Zen relay (https://opencode.ai/zen/v1).

KEYLESS: the relay serves free-tier models anonymously and 401s any bearer it
doesn't recognize, so this provider never sends a credential (the runtime
resolver pins the keyless placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). Select via ``/model free``.
"""

from typing import Any

from agent.reasoning_effort import ox_alpha_reasoning_extras
from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


class OpenCodeFreeProfile(ProviderProfile):
    """OpenCode Free — keyless, with Ox Alpha reasoning controls.

    Ox Alpha (x-preview-f-free) is also reachable via opencode-zen with the same wire
    contract; both profiles call ``agent.reasoning_effort.ox_alpha_reasoning_extras``.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return ox_alpha_reasoning_extras(reasoning_config, model)


opencode_free = OpenCodeFreeProfile(
    name="opencode-free", aliases=("free", "opencode_free"),
    env_vars=(),  # keyless — nothing to configure
    base_url="https://opencode.ai/zen/v1", display_name="OpenCode Free",
    description="OpenCode free models — keyless, no account needed",
    # Attribution headers (same values as opencode-zen/go) plus the empty Authorization
    # override that keeps the SDK's "Bearer <placeholder>" off the wire (free tier 401s it).
    default_headers={
        "Authorization": "",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    },
    # laguna is the fastest non-UA-gated free model; big-pickle 429s every
    # client except the opencode CLI's own User-Agent.
    default_aux_model="laguna-s-2.1-free",
)

register_provider(opencode_free)
