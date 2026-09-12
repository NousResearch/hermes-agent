"""OpenCode Free provider profile: the free tier on the Zen relay (https://opencode.ai/zen/v1).

KEYLESS: the relay serves free-tier models anonymously and 401s any bearer it
doesn't recognize, so this provider never sends a credential (the runtime
resolver pins the keyless placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). Select via ``/model free``.

The free tier is also gated on the request identifying as the OpenCode client:
the canonical Hermes attribution headers draw HTTP 429 ``FreeUsageLimitError``,
the OpenCode fingerprint gets 200 (#106495), so the header set below is that
fingerprint rather than the Hermes attribution used by keyed zen/go routes.
"""

from typing import Any

from hermes_cli.opencode_client_headers import opencode_client_headers
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
    # Attribution-free keyless identity: the empty Authorization override keeps the SDK's
    # "Bearer <placeholder>" off the wire (the free tier 401s it), and the rest is the OpenCode
    # client fingerprint the Zen relay's free tier requires instead of the Hermes attribution
    # headers — which it 429s (FreeUsageLimitError, #106495). Runtime owner of the same set:
    # hermes_cli.models.opencode_zen_free_headers (both read hermes_cli.opencode_client_headers
    # so the copies cannot drift).
    default_headers={"Authorization": "", **opencode_client_headers()},
    # laguna is the fastest non-UA-gated free model; big-pickle only answers the
    # opencode CLI's own User-Agent, which the fingerprint above now carries.
    default_aux_model="laguna-s-2.1-free",
)

register_provider(opencode_free)
