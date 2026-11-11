"""Plugin-declared Telegram UI tools (``provides_tools``), importable without the adapter."""

from __future__ import annotations

from plugins.platforms.telegram.telegram_ui_tool import SCHEMA, telegram_ui


def _telegram_configured() -> bool:
    """Static capability gate: a Telegram bot token exists in this profile's scope."""
    try:
        from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
        return bool(str(_get_scoped_secret("TELEGRAM_BOT_TOKEN") or "").strip())
    except Exception:
        return False


def register_tools(ctx) -> None:
    """Register the session-scoped Telegram UI tool in the ``telegram`` toolset."""
    ctx.register_tool(name="telegram_ui", toolset="telegram", schema=SCHEMA, handler=telegram_ui,
                      is_async=True, check_fn=_telegram_configured,
                      description=SCHEMA["description"], emoji="✈️")
