"""``delegation.profiles`` / ``default_profile`` / ``agent_routing`` startup validation.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``validate_config_structure`` calls it.
The parse rules live in the resolver (``agent/delegation_model_routing.py``) — one source of truth.
"""

from __future__ import annotations

from typing import Any

_NO_MENU = ("delegation.agent_routing is enabled but no delegation.profiles are configured; "
            "model_profile will not be exposed.")
_NO_MENU_HINT = ("Add at least one profile under delegation.profiles in config.yaml, or set "
                 "delegation.agent_routing: false")


def validate_delegation_profiles(config: dict[str, Any], issues: list) -> None:
    """A malformed profile otherwise fails only at the first delegate_task spawn; surface it at
    startup / `hermes config check` instead."""
    from hermes_cli.config import _issue

    delegation_cfg = config.get("delegation")
    if not isinstance(delegation_cfg, dict):
        return
    if not (delegation_cfg.get("profiles") or delegation_cfg.get("default_profile")):
        # legacy config — nothing profile-shaped to validate, but a routing gate with no menu
        # behind it silently exposes nothing: warn instead of staying quiet.
        if delegation_cfg.get("agent_routing"):
            _issue(issues, "warning", _NO_MENU, _NO_MENU_HINT)
        return
    from agent.delegation_model_routing import parse_profiles, profile_config_errors

    if delegation_cfg.get("agent_routing"):
        try:
            routable = parse_profiles(delegation_cfg)
        except ValueError:
            routable = None  # malformed — the error issues below already tell the story
        if routable == {}:
            _issue(issues, "warning", _NO_MENU, _NO_MENU_HINT)
    for message in profile_config_errors(delegation_cfg):
        _issue(issues, "error", message,
               "Fix the entry under delegation.profiles in config.yaml — each profile needs at "
               "minimum a 'model', and default_profile must name a configured profile")
