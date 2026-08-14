"""Server-side observability read off the resolved provider profile: the server's own generation speed
(``timings`` on a completed response) and, for swap proxies, which models are resident. Only profiles that opt
in ever set anything; every other provider records None and the status-bar segments stay hidden. Methods on
AIAgent forward here (see run_agent.py)."""

from __future__ import annotations

from typing import Any


def _resolved_profile(agent):
    """Requested-provider-first profile resolution, matching the transport's profile activation."""
    from providers import resolve_provider_profile
    return resolve_provider_profile(agent.provider, getattr(agent, "requested_provider", None))


def profile_surfaces_server_timings(agent) -> bool:
    """True when the resolved profile opts into ``surfaces_server_timings``. Non-opted providers resolve False,
    so no speed figure is recorded and their display is unchanged."""
    try:
        profile = _resolved_profile(agent)
    except Exception:
        return False
    return bool(getattr(profile, "surfaces_server_timings", False))


def capture_server_timings(agent, response: Any) -> None:
    """Record the server's own generation speed off a completed response as ``agent.last_server_tps``.

    Reads ``timings.predicted_per_second`` from ``model_extra`` (the streaming accumulator forwards the block
    on its response mimic). Re-assigned on every completed call, so a response without timings clears it
    instead of leaving a stale number.
    """
    tps = None
    try:
        if profile_surfaces_server_timings(agent):
            extra = getattr(response, "model_extra", None)
            timings = extra.get("timings") if isinstance(extra, dict) else None
            if isinstance(timings, dict):
                value = timings.get("predicted_per_second")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    tps = float(value)
    except Exception:
        tps = None
    agent.last_server_tps = tps


def capture_server_residency(agent) -> None:
    """Record which models the endpoint's swap proxy has resident as ``agent.last_server_residency``.

    Only profiles exposing ``resident_models`` report anything; all others record None and the indicator
    stays hidden.
    """
    residency = None
    try:
        getter = getattr(_resolved_profile(agent), "resident_models", None)
        if callable(getter):
            resident = getter(base_url=agent.base_url)
            if resident is not None:
                residency = tuple(str(m) for m in resident)
    except Exception:
        residency = None
    agent.last_server_residency = residency
