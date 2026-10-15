"""``request_overrides`` re-derivation when ``switch_model`` changes provider.

Moved out of ``agent/agent_runtime_helpers.py`` (file-line ratchet).
"""

from __future__ import annotations

import copy


def _apply_switched_provider_request_overrides(agent, new_provider):
    """Re-derive the switched-to provider's ``request_overrides`` (custom_providers ``extra_body``).
    Matches by provider key, base_url AND model (same rule as
    ``agent_init._merge_custom_provider_extra_body``) so a different model at the same endpoint
    never inherits another's ``extra_body``. Stale ``extra_body`` cleared; ``service_tier``/``speed`` kept.
    The whole previous ``extra_body`` goes, whatever its provenance; the new route's is config-owned."""
    from agent.agent_init_extra_body import _custom_provider_extra_body_for_agent
    # Prefer the init-time cache (agent._custom_providers); reload only if absent.
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers is None:
        try:
            from hermes_cli.config import load_config, get_compatible_custom_providers
            custom_providers = get_compatible_custom_providers(load_config())
        except Exception:
            custom_providers = []
    new_extra_body = _custom_provider_extra_body_for_agent(
        provider=new_provider, model=getattr(agent, "model", "") or "",
        base_url=getattr(agent, "base_url", "") or "", custom_providers=custom_providers or [],
    )
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    overrides.pop("extra_body", None)  # always drop the previous provider's extra_body
    if new_extra_body:
        overrides["extra_body"] = copy.deepcopy(new_extra_body)
    agent.request_overrides = overrides
    agent._custom_provider_injected_extra_body = copy.deepcopy(new_extra_body or {})
