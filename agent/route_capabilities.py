"""Boolean model capabilities a user declares for the agent's LIVE route.

A provider entry declares one per model (``models.<id>.<capability>``) or for every model it
serves (``capabilities: {<capability>: ...}``). Both are re-read from the live route on every call,
so fallback activation, primary restore and model switches never carry a decision to another route.
"""

from __future__ import annotations

from typing import Any, Optional

_LITERAL_THINK_TAGS_CAPABILITY = "literal_think_tags"
# Runtime provider values that name no particular saved entry.
_GENERIC_PROVIDER_IDS = frozenset({"", "auto", "custom"})


def _entry_decision(entry: dict[str, Any], model: str, capability: str) -> Optional[bool]:
    """One entry's explicit answer: its ``models.<model>`` key, else its ``capabilities:`` block."""
    from hermes_cli.config import _route_model_cfg

    for block in (_route_model_cfg(entry, model), entry.get("capabilities")):
        value = block.get(capability) if isinstance(block, dict) else None
        if isinstance(value, bool):
            return value
    return None


def configured_route_capability(agent: Any, capability: str) -> Optional[bool]:
    """Explicit boolean for *capability* on the agent's live route, or ``None`` when undeclared.

    Only the live provider entry decides: the entry on this endpoint whose name, config key or
    ``custom:<slug>`` matches the agent's ``provider`` / ``requested_provider`` (bare names count,
    as the runtime resolves them), so a sibling sharing the endpoint never speaks for it. Entries
    that match together must agree, and an explicit false wins. A ``custom:`` identity with no entry
    on this endpoint is undeclared. With no named identity (``provider="custom"``, an alias such as
    ``vllm``), every entry on the endpoint counts and an explicit false still wins.
    """
    # Returns no entries on unreadable config, so no guard is needed here.
    from hermes_cli.config import _entries_for_route
    from hermes_cli.providers import custom_provider_aliases

    model = str(getattr(agent, "model", "") or "")
    entries = list(_entries_for_route(str(getattr(agent, "base_url", "") or ""),
                                      getattr(agent, "_custom_providers", None), None))
    names = {str(getattr(agent, attr, "") or "").strip().lower().replace(" ", "-")
             for attr in ("provider", "requested_provider")} - _GENERIC_PROVIDER_IDS
    live = [e for e in entries
            if names & custom_provider_aliases(str(e.get("name") or ""), str(e.get("provider_key") or ""))]
    if live or any(name.startswith("custom:") for name in names):
        decisions = {_entry_decision(e, model, capability) for e in live}
        return False if False in decisions else (True if decisions == {True} else None)
    decisions = {_entry_decision(e, model, capability) for e in entries} - {None}
    return (False not in decisions) if decisions else None


def literal_think_tags_capability(agent: Any) -> bool:
    """True when the live route opted in to ``literal_think_tags``: its server returns reasoning out
    of band (``reasoning_content`` / ``reasoning``) and ends thinking by token id, so ``<think>`` /
    ``</think>`` text in ``content`` is answer text that the reasoning-tag scrubbers must keep.

    Explicit opt-in only; ``None`` agents (history and compression helpers) keep stripping.
    """
    if agent is None:
        return False
    return configured_route_capability(agent, _LITERAL_THINK_TAGS_CAPABILITY) is True
