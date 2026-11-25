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


def _request_agent_overrides(
    body: Any, *, virtual_model: Optional[str] = None, allow_bare_model: bool = True
) -> dict[str, Any]:
    """Extract per-request model/provider/options for _run_agent.

    The virtual model (``hermes-agent``) means "gateway default". A bare ``model`` without
    ``provider`` is honored only when ``allow_bare_model`` (generic clients hardcode "gpt-4o";
    OpenAI-compatible handlers pass the ``direct_model_requests`` opt-in, Hermes-native
    endpoints always allow it). An explicit ``provider`` is always honored.
    """
    from gateway.platforms.api_server import _clean_request_string

    if not isinstance(body, dict):
        return {}
    overrides: dict[str, Any] = {}
    provider = _clean_request_string(body.get("provider"))
    if provider:
        overrides["requested_provider"] = provider
    model = _clean_request_string(body.get("model"))
    if model and model != virtual_model and (provider or allow_bare_model):
        overrides["requested_model"] = model
    model_options = body.get("model_options")
    if isinstance(model_options, dict):
        overrides["model_options"] = dict(model_options)
    return overrides
