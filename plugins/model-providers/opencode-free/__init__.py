"""OpenCode Free provider profile.

Provides access to OpenCode's free model tier via the Zen endpoint.
No paid API key required — activate via OPENCODE_FREE_API_KEY env var
or select explicitly via ``/model free``.
"""

from providers import register_provider
from providers.base import ProviderProfile

opencode_free = ProviderProfile(
    name="opencode-free",
    aliases=("free", "opencode_free"),
    env_vars=("OPENCODE_FREE_API_KEY",),
    base_url="https://opencode.ai/zen/v1",
    display_name="OpenCode Free",
    description="OpenCode free models (no API key required)",
    default_aux_model="big-pickle",
)

register_provider(opencode_free)
