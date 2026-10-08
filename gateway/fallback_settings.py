"""Coherent, profile-owned fallback settings for long-lived interactive surfaces."""
from __future__ import annotations

import logging
from pathlib import Path

from hermes_cli.fallback_config import get_fallback_auto_activate, get_fallback_chain
from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)
FallbackSettings = tuple[list | None, bool]


def read_fallback_settings(config_path: Path, cache: dict[str, FallbackSettings]) -> FallbackSettings:
    """Publish chain and policy together; a failed first read has no automatic authority.

    Never borrow the runner's last slot: that slot may belong to another profile. Missing files
    are intentional removal; malformed/in-progress files retain only this home's last good pair.
    """
    from hermes_cli.config_effective import load_user_config_effective
    key = hermes_home_key(config_path.parent)
    try:
        cfg = load_user_config_effective(config_path, fail_closed=True) if config_path.exists() else {}
        settings = get_fallback_chain(cfg) or None, get_fallback_auto_activate(cfg)
    except Exception:
        logger.debug("Fallback config read failed; keeping this profile's last known-good settings", exc_info=True)
        return cache.get(key, (None, False))
    cache[key] = settings
    return settings


def api_fallback_kwargs(config: dict, confirmed_runtime_lock: bool) -> dict:
    """API requests never have an interactive consent surface, including hosted rooms."""
    return {
        "fallback_model": None if confirmed_runtime_lock else get_fallback_chain(config),
        "fallback_auto_activate": False if confirmed_runtime_lock else get_fallback_auto_activate(config),
        "fallback_selection_interactive": False,
    }


def require_automatic_override_fallback(runner, config: dict | None, error: Exception) -> None:
    """Unavailable session selections may substitute the default route only in automatic mode."""
    automatic = get_fallback_auto_activate(config) and runner._refresh_fallback_settings()[1]
    if not automatic:
        raise error


def stamp_api_runtime(agent, runtime: dict, model: str, locked, session_override, request_override, fallback_notice) -> None:
    """Keep the API's selected route provenance, including automatic pre-agent substitution."""
    route_source = ("session_model_lock" if locked else "session_model_override" if session_override
                    else "raw_request" if request_override else "global")
    if fallback_notice and route_source == "global":
        agent._fallback_bootstrap_active = True
    agent._hermes_api_runtime = {
        "provider": runtime.get("provider") or getattr(agent, "provider", "") or "",
        "model": getattr(agent, "model", None) or model, "route_source": route_source}
