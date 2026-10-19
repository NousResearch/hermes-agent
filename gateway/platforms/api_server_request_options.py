"""Per-request ``model_options`` decoding, lifted from ``platforms/api_server.py``.

Keeps the api_server facade under its frozen line cap (root AGENTS.md god-file
guidance: move code into a topical sibling). ``api_server.py`` re-exports
``_request_service_tier``, ``_request_reasoning_config`` and ``_coerce_request_bool`` so existing importers
are unaffected; the helpers those two read stay in the facade and are imported
lazily (call time) to avoid an import cycle.
"""

from typing import Any, Dict, Optional

_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


def _coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like payload values; only explicit bool-ish scalars count (some
    frontends send ``"false"`` for ``stream``, which is truthy), else ``default``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        return False if normalized in _FALSE_REQUEST_BOOL_STRINGS else default
    return bool(value) if isinstance(value, (int, float)) else default


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
    from gateway.platforms.api_server import _REQUEST_OPTION_MISSING, _clean_request_string

    if not isinstance(model_options, dict):
        return _REQUEST_OPTION_MISSING
    if "service_tier" in model_options:
        raw_tier = model_options.get("service_tier")
        return _clean_request_string(raw_tier) if isinstance(raw_tier, str) else raw_tier
    if "fast" in model_options:
        return "priority" if _coerce_request_bool(model_options.get("fast"), default=False) else None
    return _REQUEST_OPTION_MISSING
