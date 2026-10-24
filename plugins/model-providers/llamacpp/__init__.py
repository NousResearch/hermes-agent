"""llama.cpp / llama-swap provider profile.

Registers the ``llamacpp`` provider so llama.cpp endpoints - a bare
``llama-server`` or a llama-swap proxy - get their own profile instead
of riding the generic ``custom`` profile. Bundled discovery imports
``custom`` before this plugin (sorted order, last-writer-wins), so the
self-claimed aliases repoint the lookup here. A user plugin of the same
name under ``$HERMES_HOME/plugins/model-providers/`` overrides this
bundled copy; removing both restores stock resolution.
"""

import logging

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger("providers.llamacpp")


class LlamaCppProfile(ProviderProfile):
    """llama.cpp servers, bare or behind llama-swap."""

    # Custom-provider entries named llamacpp / llama-swap activate this
    # profile: providers.resolve_provider_profile does its requested-first
    # lookup only for profiles carrying this opt-in.
    activates_on_requested_provider = True

    def probe_server_caps(self, *, base_url=None, model=None, timeout=3.0):
        """Detect server kind and served-template capabilities (read-only).

        Never dispatches a request that could start a non-resident model
        on llama-swap - see probe.py for the safety contract.
        """
        from . import probe

        return probe.probe_model(base_url or self.base_url, model, timeout=timeout)


llamacpp = LlamaCppProfile(
    name="llamacpp",
    # The three llama.cpp spellings are self-claimed on purpose: they steal
    # the aliases the bundled custom profile registered, which is what makes
    # lookups by those names resolve here. Runtime resolution already treats
    # the three as one provider, so the profile lookup must too, or
    # `provider: llama.cpp` would silently lose every feature of this
    # profile. "llama-swap" is new and ours alone.
    aliases=(
        "llamacpp",
        "llama.cpp",
        "llama-cpp",
        "llama-swap",
    ),
    display_name="llama.cpp",
    description="Local llama.cpp server (bare llama-server or llama-swap proxy)",
    env_vars=(),  # local endpoint - no API key required
    base_url="",  # user-configured
    auth_type="api_key",
    # Parity with the custom profile this shadows: a generous floor used
    # only when the user has not set model.max_tokens.
    default_max_tokens=65536,
)

register_provider(llamacpp)
