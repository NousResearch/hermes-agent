"""The session's requested vs effective model, kept durably in ``sessions.model_config``.

A provider fallback is turn-scoped (``restore_primary_runtime`` tries the primary again next turn), and
until now it was visible only as a transient status line and in the logs: reopening the session, or
looking at it from another window, showed the model that was asked for, not the one answering. The
fallback walk records the switch here, and the next turn that runs on the requested route marks it
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


def record_fallback_switch(agent, reason: Any, requested: tuple[str, str], effective: tuple[str, str]) -> None:
    """The fallback walk bound ``effective`` (model, provider) in place of ``requested``: count it
    (opt-in shared metrics) and record it on the session row."""
    shared_metrics_events.record_fallback(from_provider=requested[1], to_provider=effective[1], reason=reason)
    # The first link of a chain is what the user asked for; a later link failing over again must not
    # overwrite it with the previous fallback.
    primary = getattr(agent, "_primary_runtime", None) or {}
    req_model = str(primary.get("model") or requested[0] or "")
    req_provider = str(primary.get("provider") or requested[1] or "")
    agent._effective_route_recorded = True
    _write(agent, {
        "active": True, "route_source": ROUTE_FALLBACK, "reason": _reason_value(reason),
        "requested": {"model": req_model, "provider": req_provider},
        "effective": {"model": str(effective[0] or ""), "provider": str(effective[1] or "")},
        "at": time.time(),
    })


def settle_effective_route(agent) -> None:
    """Turn start, after ``restore_primary_runtime``: a turn back on the requested route clears the
    stored fallback. Checked when this agent recorded one, and once per agent otherwise (a gateway
    restart or an evicted agent never runs the restore, so its row would keep naming the fallback)."""
    # Still on a fallback (restore gated by a cooldown), or on a route chosen before the turn (an
    # init-time fallback, a voice turn): nothing to settle.
    if getattr(agent, "_provider_fallback_active", False) is True or getattr(agent, "_fallback_activated", False) is True:
        return
    recorded = getattr(agent, "_effective_route_recorded", None)
    if recorded is False:
        return
    agent._effective_route_recorded = False
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
    model, provider = str(getattr(agent, "model", "") or ""), str(getattr(agent, "provider", "") or "")
    _write(agent, {
        "active": False, "route_source": ROUTE_PRIMARY,
        "requested": {"model": model, "provider": provider},
        "effective": {"model": model, "provider": provider},
        "last_fallback": {k: stored.get(k) for k in ("requested", "effective", "reason", "at")},
        "at": time.time(),
    })
