"""Per-request ``model_options`` decoding, lifted from ``platforms/api_server.py``.

Keeps the api_server facade under its frozen line cap (root AGENTS.md god-file
guidance: move code into a topical sibling). ``api_server.py`` re-exports
``_request_service_tier`` and ``_request_reasoning_config`` so existing importers
are unaffected; the helpers those two read stay in the facade and are imported
lazily (call time) to avoid an import cycle.
"""

from typing import Any, Dict, Optional


def _request_reasoning_config(model_options: Any) -> Optional[dict[str, Any]]:
    """Translate model_options (structured ``reasoning`` or legacy ``reasoning_effort``) into
    AIAgent reasoning_config; unknown effort values are ignored, never raised."""
    if not isinstance(model_options, dict):
        return None
    reasoning = model_options.get("reasoning")
    enabled: Any = None
    effort: Any = None
    if isinstance(reasoning, dict):
        enabled = reasoning.get("enabled")
        effort = reasoning.get("effort")
    effort = model_options.get("reasoning_effort", effort)
    effort_norm = str(effort).strip().lower() if effort is not None else ""
    if enabled is False or effort_norm == "none":
        return {"enabled": False}
    from gateway.platforms.api_server import _REASONING_EFFORTS

    if effort_norm in _REASONING_EFFORTS and effort_norm != "none":
        return {"enabled": True, "effort": effort_norm}
    if enabled is True:
        return {"enabled": True}
    return None


def _request_service_tier(model_options: Any) -> Any:
    """Return a per-request service_tier override or _REQUEST_OPTION_MISSING."""
    from gateway.platforms.api_server import _REQUEST_OPTION_MISSING, _clean_request_string, _coerce_request_bool

    if not isinstance(model_options, dict):
        return _REQUEST_OPTION_MISSING
    if "service_tier" in model_options:
        raw_tier = model_options.get("service_tier")
        return _clean_request_string(raw_tier) if isinstance(raw_tier, str) else raw_tier
    if "fast" in model_options:
        return "priority" if _coerce_request_bool(model_options.get("fast"), default=False) else None
    return _REQUEST_OPTION_MISSING


def _request_skip_memory(model_options: Any) -> bool:
    """True when the request opts this run out of the external memory provider
    (``{"memory": false}`` or ``{"skip_memory": true}``): no prefetch, no memory-context
    injection and no turn sync for that run. Built-in MEMORY.md/USER.md stay. Anything
    else (absent, malformed) keeps memory on."""
    if not isinstance(model_options, dict):
        return False
    from gateway.platforms.api_server import _coerce_request_bool

    if "skip_memory" in model_options and _coerce_request_bool(model_options.get("skip_memory"), default=False):
        return True
    if "memory" in model_options:
        return not _coerce_request_bool(model_options.get("memory"), default=True)
    return False


def _request_memory_kwargs(model_options: Any, memory_sessions: Any, session_id: str) -> dict[str, Any]:
    """The run's ``skip_memory``/``memory_manager`` agent kwargs. An opted-out run gets no manager and
    leaves the session's parked one in *memory_sessions* untouched for the next normal turn; otherwise
    the parked manager is checked out (#120116) so its queued recall reaches this turn."""
    if _request_skip_memory(model_options):
        return {"skip_memory": True, "memory_manager": None}
    return {"skip_memory": False, "memory_manager": memory_sessions.checkout(session_id)}
