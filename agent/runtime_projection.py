"""The live main runtime of an agent, as one auxiliary ``main_runtime`` snapshot.

An auxiliary call rebuilds its client and re-decides the wire policy from the snapshot it is
handed, and an explicit snapshot replaces the context-local runtime instead of merging with it.
So every producer must hand over the whole route: the named owner (``requested_provider`` — a
named relay resolves to ``provider: custom``), the full endpoint with the tenant query its SDK
client split off, the effective model, the sanitized capability map and the conversation id.
Building it here once keeps the compressor, the feasibility probe, the review fork and the TUI
one-shot from each dropping a different piece.
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional

_IDENTITY_FIELDS = ("model", "provider", "requested_provider", "api_key", "api_mode", "auth_mode", "session_id")
# The fields a ContextCompressor holds for its own summary route.
_ROUTE_FIELDS = ("model", "provider", "base_url", "api_key", "api_mode")


def live_main_runtime(agent: Any) -> Dict[str, Any]:
    """The agent's live route: owner, full endpoint (query included), model, capabilities, session.

    Read off the agent alone, never off ambient context state, so a snapshot cannot borrow
    qualification from another session's runtime. A callable ``api_key`` (Entra ID token
    provider) is kept as-is."""
    from agent.turn_context import live_route_base_url

    runtime: Dict[str, Any] = {key: getattr(agent, key, "") or "" for key in _IDENTITY_FIELDS}
    runtime["base_url"] = live_route_base_url(agent)
    capabilities = getattr(agent, "capabilities", None)
    runtime["capabilities"] = {
        key: value for key, value in (capabilities.items() if isinstance(capabilities, dict) else ())
        if isinstance(key, str) and isinstance(value, bool)
    }
    return runtime


def bind_route_owner(engine: Any, agent: Any) -> None:
    """Let a context engine read its host agent's live route for its own summary calls."""
    try:
        engine._route_owner = weakref.ref(agent)
    except (AttributeError, TypeError):  # slotted plugin engine / non-weakrefable double
        pass


def compressor_main_runtime(compressor: Any) -> Dict[str, Any]:
    """``main_runtime`` for a compressor's summary call on its own route.

    The compressor holds only model/provider/SDK-clean URL/key/API mode. When its host agent is on
    that same route (every ``update_model`` caller passes the agent's fields), the agent's live
    projection replaces them, so the summary request carries the owner, query, capabilities and
    session the main turn uses. A compressor on any other route keeps its five fields unqualified."""
    route = {key: getattr(compressor, key, "") or "" for key in _ROUTE_FIELDS}
    owner_ref = getattr(compressor, "_route_owner", None)
    owner: Optional[Any] = owner_ref() if callable(owner_ref) else None
    if owner is None or not _same_route(route, owner):
        return route
    return live_main_runtime(owner)


def _same_route(route: Dict[str, Any], agent: Any) -> bool:
    def clean(value: Any) -> str:
        return str(value or "").strip().rstrip("/")

    return all(
        clean(route[key]) == clean(getattr(agent, key, ""))
        for key in ("model", "provider", "base_url", "api_mode")
    )
