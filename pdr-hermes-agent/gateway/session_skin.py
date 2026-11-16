"""The owning profile's resolved skin for a canonical client's readiness.

Main's TUI booted its theme from ``gateway.ready.skin`` (``tui_gateway/entry.py`` / ``ws.py``, both
``change_watcher.resolve_skin()``). On the session authority the client publishes its own ready after
``runtime.describe``, so the describe reply carries the same payload, resolved for the profile that
owns this authority (``display.skin`` and ``<home>/skins/*.yaml`` are per profile).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolved_skin() -> dict:
    """``resolve_skin()``'s dict for the profile in scope (the dispatcher's ``owner_scope``); ``{}``
    when the skin engine fails, which the client treats as the default theme, as main did."""
    try:
        from hermes_cli.config_effective import load_user_config_effective
        from hermes_cli.skin_engine import get_active_skin, init_skin_from_config
        from hermes_constants import get_hermes_home
        init_skin_from_config(load_user_config_effective(get_hermes_home() / "config.yaml"))
        skin = get_active_skin()
        return {
            "name": skin.name, "colors": skin.colors,
            "light_colors": skin.light_colors, "dark_colors": skin.dark_colors,
            "branding": skin.branding, "banner_logo": skin.banner_logo,
            "banner_hero": skin.banner_hero, "tool_prefix": skin.tool_prefix,
            "help_header": (skin.branding or {}).get("help_header", ""),
            "customCSS": skin.custom_css}
    except Exception:  # health: allow BLE001 -- main's resolve_skin: a broken skin boots the default theme
        logger.debug("runtime.describe: skin resolution failed", exc_info=True)
        return {}
