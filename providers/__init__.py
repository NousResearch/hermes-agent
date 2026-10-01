"""Canonical provider domain API.

Provider identity is defined by :class:`ProviderProfile`. The registry is the
single authority for effective provider declarations; discovery mechanics are
internal to the package.

Provider profiles may be supplied by bundled plugins, the currently bound
HERMES_HOME, pip entry points, or the documented legacy single-file plugin
boundary. Consumers should use this package API rather than registry state.
"""

from __future__ import annotations

from providers.base import ProviderProfile
from providers.registry import (
    get_provider_profile,
    list_providers,
    provider_source,
    register_provider,
    routed_model_rejects_vision_tool_messages,
)
from providers.model_normalizers import vendor_for_model
from providers.github import COPILOT_EDITOR_VERSION, copilot_request_headers
from providers.configured import (
    ConfiguredProvider,
    configured_custom_identity,
    expand_direct_api_alias,
    match_configured_provider,
    resolves_to_custom_provider,
)
from providers.route_identity import (
    is_actual_route,
    is_foreign_provider_endpoint,
    normalize_route_base_url,
)
from providers.opencode import (
    normalize_opencode_base_url,
    normalize_opencode_model_id,
    opencode_provider_family,
)
from providers.identity import (
    ResolvedProvider,
    custom_provider_aliases,
    custom_provider_slug,
    get_provider_label,
    is_aggregator,
    is_routing_aggregator,
    normalize_provider,
)
from providers.routing import (
    InvocationRequest,
    InvocationRoute,
    RuntimeKind,
    canonicalize_api_mode,
    endpoint_api_mode,
    is_external_process_provider,
    resolve_invocation_route,
)

# Load the internal discovery module so registry calls and package submodule
# identity are stable; discovery itself remains lazy and performs no scan here.
from providers import discovery as _discovery  # noqa: E402,F401


__all__ = [
    "ProviderProfile",
    "ResolvedProvider",
    "ConfiguredProvider",
    "register_provider",
    "get_provider_profile",
    "list_providers",
    "provider_source",
    "routed_model_rejects_vision_tool_messages",
    "normalize_provider",
    "get_provider_label",
    "is_aggregator",
    "is_routing_aggregator",
    "custom_provider_slug",
    "custom_provider_aliases",
    "match_configured_provider",
    "configured_custom_identity",
    "resolves_to_custom_provider",
    "expand_direct_api_alias",
    "vendor_for_model",
    "COPILOT_EDITOR_VERSION",
    "copilot_request_headers",
    "is_actual_route",
    "is_foreign_provider_endpoint",
    "normalize_route_base_url",
    "opencode_provider_family",
    "normalize_opencode_model_id",
    "normalize_opencode_base_url",
    "InvocationRequest",
    "InvocationRoute",
    "RuntimeKind",
    "canonicalize_api_mode",
    "endpoint_api_mode",
    "is_external_process_provider",
    "resolve_invocation_route",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

_PLUGIN_COMPAT_LAZY = {
    "OMIT_TEMPERATURE": ("providers.base", "OMIT_TEMPERATURE"),
}


def __getattr__(name):  # PEP 562 - lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    from hermes_cli.plugin_compat import warn_once

    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
