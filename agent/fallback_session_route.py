"""Keep a provider fallback turn-scoped across a session reload (#152333).

A fallback swaps the agent's model and provider for the rest of the turn, and the next turn
restores the primary (``restore_primary_runtime``). That restore lives in the process. The
session row is another matter: the first accounted API call writes the route that served it
(``sessions.model`` / ``billing_provider``), the gateway's per-turn sync writes the agent's live
route, and the TUI persists ``agent.model``. A session restored from that row after an idle
teardown or a restart came back on the fallback, with no provider error and no notice, and kept
billing it after the primary recovered.

So the fallback records where the row's model came from: ``model_config._fallback_route`` names
the fallback model and the route the session ran before it (its primary, which may itself be an
explicit ``/model`` pick). Every resume reader passes the row through
``fallback_restored_row``, which puts that pre-fallback route back while the row still holds
the fallback's model. If the primary is still rate limited, its first request fails again and
the ordinary fallback runs with its usual notice (``Primary retry eligible in ~N s``).

An explicit model pick clears the marker (``update_session_model``, a local policy model
selection, and any persist of a live agent that is not on a fallback), so a user's choice of the
same model the fallback used still survives a reload. Gateway rows written before this marker
existed carry ``gateway_runtime.fallback_active``; those restore the profile's configured model.
Credentials are never stored here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

FALLBACK_ROUTE_KEY = "_fallback_route"
_ROUTE_KEYS = ("provider", "base_url", "api_mode")


def mark_fallback_route(agent: Any, fallback_model: str) -> None:
    """Record on the session row that its route now comes from a provider fallback (best effort)."""
    if getattr(agent, "_persist_disabled", False):
        return
    session_id = getattr(agent, "session_id", None)
    patcher = getattr(getattr(agent, "_session_db", None), "patch_session_model_config", None)
    if not session_id or not callable(patcher) or not fallback_model:
        return
    rt = getattr(agent, "_primary_runtime", None) or {}
    primary_model = str(rt.get("model") or "").strip()
    primary: dict[str, Any] = {}
    if primary_model:
        from hermes_cli.runtime_provider_custom import session_provider_identity
        primary = {
            "model": primary_model,
            "provider": session_provider_identity(
                rt.get("provider"), rt.get("requested_provider"), rt.get("base_url"), primary_model) or None,
            "base_url": rt.get("base_url") or None,
            "api_mode": rt.get("api_mode") or None,
        }
        primary = {k: v for k, v in primary.items() if v}
    try:
        patcher(session_id, {FALLBACK_ROUTE_KEY: {"model": str(fallback_model), "primary": primary}})
    except Exception:
        logger.debug("fallback route marker could not be stored on the session row", exc_info=True)


def _parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def fallback_primary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """The pre-fallback route (``{}`` = the profile's configured one) when *row*'s stored model
    came from a provider fallback, else None."""
    if not isinstance(row, dict):
        return None
    config = _parse(row.get("model_config"))
    stored_model = str(row.get("model") or "").strip()
    marker = config.get(FALLBACK_ROUTE_KEY)
    if isinstance(marker, dict) and stored_model and str(marker.get("model") or "").strip() == stored_model:
        primary = marker.get("primary")
        return dict(primary) if isinstance(primary, dict) else {}
    runtime = config.get("gateway_runtime")
    if isinstance(runtime, dict) and runtime.get("fallback_active") is True:
        return {}
    return None


def fallback_restored_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """*row* with a fallback-derived route replaced by the route the session ran before the
    fallback; *row* itself when its route is the session's own."""
    primary = fallback_primary(row)
    if primary is None or not isinstance(row, dict):
        return row
    config = _parse(row.get("model_config"))
    for key in ("gateway_runtime", "model", FALLBACK_ROUTE_KEY, *_ROUTE_KEYS):
        config.pop(key, None)
    route = {k: primary[k] for k in _ROUTE_KEYS if primary.get(k)}
    if route.get("provider"):
        config["gateway_runtime"] = dict(route)
        config.update(route)
    restored = dict(row)
    restored.update(model=primary.get("model") or None, model_config=json.dumps(config) if config else None,
                    billing_provider=None, billing_base_url=None)
    logger.info("Session %s: stored model %r came from a provider fallback; restoring %r",
                row.get("id") or "", row.get("model"), primary.get("model") or "the configured model")
    return restored
