"""Common per-request managed-route guard (design §5, §12).

``_enforce_kanban_routing_receipt`` in ``cli.py`` proves the guard ONCE, before
the turn's first inference call. That leaves every LATER request in the same
managed turn (tool-loop iterations, same-turn continuations) unchecked: a
policy revoked mid-turn, or a code path that swaps the agent's client/model
between iterations (credential rotation, fallback activation), could reach
the provider boundary again with no re-validation. Design §12 calls for a
"best-effort revocation generation check before each subsequent managed
request" — this module is that check, run from the conversation loop itself
so it applies to every adapter that sets the two agent attributes below,
not just the Kanban CLI bootstrap path.

Pure orchestration glue: no provider clients, no credentials, no network.
Delegates the actual validation to ``hermes_cli.kanban_model_routing`` (the
receipt-store + policy-revocation logic already lives there — this module
does not re-implement it) and ``agent.model_selection_guard`` (route/reasoning
equality). Adapters not yet wired to a receipt (unmanaged agents,
delegation/MoA before their own adapters land) are a complete no-op — the
agent simply carries no ``_managed_routing_receipt_id``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_loop")


def managed_route_receipt_id(agent: Any) -> Optional[str]:
    """The receipt id this agent's turn was authorized under, or ``None`` for
    an unmanaged/unwired agent (complete no-op for every existing behavior)."""
    return getattr(agent, "_managed_routing_receipt_id", None) or None


def enforce_managed_route_per_request(
    agent: Any, *, record_outcome: bool = False, outcome_kind: str = "routing_reverified",
) -> Optional[str]:
    """Re-validate the actually-constructed route against the receipted decision,
    immediately before EACH request a managed agent is about to send (§12: "no
    silent escape through parent inheritance", "it never substitutes another
    model"). No-op (returns ``None``) for an agent with no receipt wired.

    Returns ``None`` on success, or a short redacted reason string on failure —
    callers must treat any non-``None`` return as fail-closed: end the attempt
    before transmitting content, never retry the same call with a different
    route and never fall through to an unmanaged path.

    ``record_outcome``: pass True only for the FIRST request of a turn (the
    startup call in cli.py already records ``routing_started`` there; this
    function defaults to not re-recording so per-iteration re-checks don't
    grow the outcome log once per model call).
    """
    receipt_id = managed_route_receipt_id(agent)
    if receipt_id is None:
        return None
    hermes_home = getattr(agent, "_managed_routing_home", None)
    if not hermes_home:
        # A receipt id with no origin home is a wiring bug, not "unmanaged" —
        # fail closed rather than guess a default store that might belong to
        # a different profile.
        logger.error(
            "guided-routing per-request guard: agent carries a routing receipt "
            "(%s) with no origin home recorded; refusing this request",
            receipt_id,
        )
        return "missing_routing_origin_home"

    from agent.model_selection_types import RoutingBlocked
    from hermes_cli.kanban_model_routing import enforce_worker_route

    # Same field derivation cli.py's startup check uses (see
    # ``requested_provider`` comment in ``_enforce_kanban_routing_receipt``):
    # ``agent.provider`` is the canonicalized transport family, so a rebuilt/
    # rotated/fallback client that changed the WIRE identity but kept the same
    # canonical family (e.g. two different custom providers both "custom")
    # would slip past a check keyed on ``agent.provider`` alone.
    actual_provider = (getattr(agent, "requested_provider", "") or agent.provider or "").strip()
    actual_model = (getattr(agent, "model", "") or "").strip()
    actual_endpoint = getattr(agent, "base_url", None) or None
    actual_reasoning = _requested_effort(agent)
    try:
        enforce_worker_route(
            hermes_home, receipt_id,
            actual_provider=actual_provider, actual_model=actual_model,
            actual_endpoint=actual_endpoint, actual_reasoning=actual_reasoning,
            record_outcome=record_outcome, outcome_kind=outcome_kind,
        )
    except RoutingBlocked as exc:
        logger.error(
            "guided-routing per-request guard blocked a managed request "
            "(receipt=%s): %s", receipt_id, exc,
        )
        return str(exc.reason)
    return None


def _requested_effort(agent: Any) -> Optional[str]:
    from agent.reasoning_effort import requested_effort

    return requested_effort(getattr(agent, "reasoning_config", None))


def clear_managed_route_receipt(agent: Any) -> None:
    """Drop the per-request guard's wiring (defensive: never re-check a stale
    receipt against a later, unmanaged/differently-managed reuse of the same
    long-lived agent object)."""
    agent._managed_routing_receipt_id = None
    agent._managed_routing_home = None
