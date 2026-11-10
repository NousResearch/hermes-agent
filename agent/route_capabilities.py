"""Boolean model capabilities a user declares for the agent's LIVE route.

A provider entry declares one per model (``models.<id>.<capability>``) or for every model it
serves (``capabilities: {<capability>: ...}``). Both are re-read from the live route on every call,
so fallback activation, primary restore and model switches never carry a decision to another route.
"""

from __future__ import annotations

from typing import Any, Optional

_LITERAL_THINK_TAGS_CAPABILITY = "literal_think_tags"


def configured_route_capability(agent: Any, capability: str) -> Optional[bool]:
    """Explicit boolean for *capability* on the agent's live route, or ``None`` when undeclared."""
    # Returns None on unreadable config, so no guard is needed here.
    from hermes_cli.config import _entries_for_route, get_custom_provider_model_capability

    base_url = str(getattr(agent, "base_url", "") or "")
    custom_providers = getattr(agent, "_custom_providers", None)
    configured = get_custom_provider_model_capability(
        model=str(getattr(agent, "model", "") or ""),
        base_url=base_url,
        capability=capability,
        custom_providers=custom_providers,
    )
    if configured is not None:
        return configured
    # The provider-level ``capabilities:`` block, re-read on the live route like the per-model key
    # so it holds on CLI/TUI (no constructor ``capabilities=``) and survives /model switches.
    # Several entries may share one endpoint (LiteLLM/vLLM router): only the live ``custom:<slug>``
    # entry counts; without one, every flagged sibling on the endpoint must opt in. Startup, gateway
    # and TUI keep ``provider="custom"`` and the named identity in ``requested_provider``.
    from hermes_cli.providers import custom_provider_slug

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    requested = str(getattr(agent, "requested_provider", "") or "").strip().lower()
    ids = {custom_provider_slug(p) for p in (provider, requested) if p.startswith("custom:")}
    entries = list(_entries_for_route(base_url, custom_providers, None))
    live = [e for e in entries
            if custom_provider_slug(e.get("name"), e.get("provider_key")) in ids]
    flags = {v for e in live or entries
             if isinstance(v := (e.get("capabilities") or {}).get(capability), bool)}
    return flags == {True} if flags else None


def literal_think_tags_capability(agent: Any) -> bool:
    """True when the live route opted in to ``literal_think_tags``: its server returns reasoning out
    of band (``reasoning_content`` / ``reasoning``) and ends thinking by token id, so ``<think>`` /
    ``</think>`` text in ``content`` is answer text that the reasoning-tag scrubbers must keep.

    Explicit opt-in only; ``None`` agents (history and compression helpers) keep stripping.
    """
    if agent is None:
        return False
    return configured_route_capability(agent, _LITERAL_THINK_TAGS_CAPABILITY) is True
