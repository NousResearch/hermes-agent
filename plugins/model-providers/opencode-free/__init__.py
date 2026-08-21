"""OpenCode Free provider profile.

OpenCode's free model tier on the Zen relay (https://opencode.ai/zen/v1).
KEYLESS: the relay serves free-tier models anonymously and rejects any
Authorization bearer it doesn't recognize with 401 — so this provider
never sends a credential at all (the runtime resolver pins the keyless
placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). No OpenCode account needed.
Select via ``hermes model`` or ``/model free``.
"""

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

opencode_free = ProviderProfile(
    name="opencode-free",
    aliases=("free", "opencode_free"),
    env_vars=(),  # keyless — nothing to configure
    base_url="https://opencode.ai/zen/v1",
    display_name="OpenCode Free",
    description="OpenCode free models — keyless, no account needed",
    default_headers=dict(_KEYLESS_HEADERS),
    default_aux_model="big-pickle",
)

register_provider(opencode_free)
