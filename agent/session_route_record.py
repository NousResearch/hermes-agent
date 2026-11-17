"""The session's requested vs effective model, kept durably in ``sessions.model_config``.

A provider fallback is turn-scoped (``restore_primary_runtime`` tries the primary again next turn), and
until now it was visible only as a transient status line and in the logs: reopening the session, or
looking at it from another window, showed the model that was asked for, not the one answering. The
fallback walk records the switch here, and the next turn that starts off the fallback route marks it
inactive (keeping it as ``last_fallback``), so every session reader sees which route really ran.

One ``effective_route`` key, merged into ``model_config`` (lineage markers and every other key
survive); session rows lift it to the top level (``hermes_state.SessionDB._session_row_dict``) so the
list endpoints, which strip ``model_config``, still carry it. A metadata write only: the system prompt,
the tool surface and the cached prefix are untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hermes_cli.observability import shared_metrics_events

logger = logging.getLogger(__name__)

KEY = "effective_route"
ROUTE_FALLBACK = "fallback"
ROUTE_PRIMARY = "primary"


def _session_db(agent) -> Any:
    if getattr(agent, "_persist_disabled", False) or not getattr(agent, "session_id", None):
        return None
    return getattr(agent, "_session_db", None)


def _write(agent, record: dict[str, Any]) -> None:
    db = _session_db(agent)
    if db is None:
        return
    try:
        db.patch_session_model_config(agent.session_id, {KEY: record})
    except Exception:  # health: allow BLE001 -- a metadata write must never fail the model switch
        logger.debug("effective route record not written", exc_info=True)


def _reason_value(reason: Any) -> str:
    return str(getattr(reason, "value", reason) or "")


def _route(model: Any, provider: Any) -> dict[str, str]:
    return {"model": str(model or ""), "provider": str(provider or "")}


def record_fallback_switch(agent, reason: Any, requested: tuple[str, str], effective: tuple[str, str]) -> None:
    """The fallback walk bound ``effective`` (model, provider) in place of ``requested``: count it
    (opt-in shared metrics) and record it on the session row."""
    shared_metrics_events.record_fallback(from_provider=requested[1], to_provider=effective[1], reason=reason)
    # The requested side is the first route this turn attempted (a voice turn's own model included),
    # never a later chain link: the first switch of the turn pins it, ``settle_effective_route`` resets it.
    first = getattr(agent, "_turn_requested_route", None)
    if not isinstance(first, dict):
        first = agent._turn_requested_route = _route(*requested)
    _write(agent, {
        "active": True, "route_source": ROUTE_FALLBACK, "reason": _reason_value(reason),
        "requested": dict(first), "effective": _route(*effective), "at": time.time(),
    })


def settle_effective_route(agent) -> None:
    """Turn start, after ``restore_primary_runtime``: reconcile the stored record with the route this
    turn starts on. Read every turn (another agent may have written it), written only on a change."""
    if getattr(agent, "_provider_fallback_active", False) is True:
        # Still on a fallback (the restore was gated by a cooldown): a further failover this turn is
        # still a fallback from what the session asked for.
        primary = getattr(agent, "_primary_runtime", None) or {}
        agent._turn_requested_route = _route(primary.get("model"), primary.get("provider"))
        return
    agent._turn_requested_route = None
    if getattr(agent, "_fallback_activated", False) is True:
        return  # a route planned before the turn (init-time fallback, voice turn): not a recovery
    db = _session_db(agent)
    if db is None:
        return
    try:
        stored = db.get_session_model_config_value(agent.session_id, KEY)
    except Exception:  # health: allow BLE001 -- a metadata read must never fail the turn
        logger.debug("effective route record not read", exc_info=True)
        return
    if not (isinstance(stored, dict) and stored.get("active")):
        return
    current = _route(getattr(agent, "model", ""), getattr(agent, "provider", ""))
    if current == stored.get("effective") and current != stored.get("requested"):
        return  # resumed onto the fallback route itself: the requested model has not come back
    _write(agent, {
        "active": False, "route_source": ROUTE_PRIMARY, "requested": current, "effective": current,
        "last_fallback": {k: stored.get(k) for k in ("requested", "effective", "reason", "at")},
        "at": time.time(),
    })
