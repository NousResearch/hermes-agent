"""Product name for gateway-authored, user-facing text.

Messenger notices (pairing DM, home-channel prompt, ...) used to hardcode
"Hermes".  A deployment that ships a custom skin with its own
``branding.agent_name`` should see that name in chat too, so the catalog
strings carry a ``{brand}`` placeholder and this helper fills it.
"""

from __future__ import annotations

_STOCK_AGENT_NAME = "Hermes Agent"
_STOCK_SHORT_NAME = "Hermes"


def agent_display_name() -> str:
    """Return the active skin's ``agent_name`` for chat messages.

    The gateway never calls ``init_skin_from_config()`` -- that is a CLI
    startup step -- so ``get_active_skin()`` alone would always report the
    stock skin here.  Resolve ``display.skin`` from config.yaml directly.
    The stock skin keeps the short "Hermes" the notices always used, so
    default installs see no change; only a custom skin swaps the name.
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.skin_engine import get_active_skin, load_skin

        skin = get_active_skin()
        display = load_config_readonly().get("display") or {}
        configured = (
            str(display.get("skin") or "").strip() if isinstance(display, dict) else ""
        )
        if configured and configured != getattr(skin, "name", ""):
            skin = load_skin(configured)
        name = str(skin.get_branding("agent_name", _STOCK_AGENT_NAME) or "").strip()
    except Exception:
        name = ""
    if not name or name == _STOCK_AGENT_NAME:
        return _STOCK_SHORT_NAME
    return name[:64]


__all__ = ["agent_display_name"]
