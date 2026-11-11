"""Bot short description as an opt-in online/offline indicator (``extra.status_indicator``).

The short description is the bot's profile text (profile page, share links), so the indicator
borrows it without destroying it (#78784). The operator's own text is captured once, before the
first "Online" write, and kept per bot in the profile home: a crash leaves the status text up, and
a later boot must not read "Online" back as the original. Online/Offline behave as documented
while the indicator is on; turning it off puts the captured text back on the next connect.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .adapter import TelegramAdapter

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_STATE_FILE = "telegram_status_description.json"
_TELEGRAM_SHORT_DESCRIPTION_CAP = 120


def _state_path(adapter: TelegramAdapter) -> Path:
    # ``_update_receipt_dir`` is the owning profile's home, resolved when the adapter was built.
    return Path(adapter._update_receipt_dir) / _STATE_FILE


def _bot_key(bot: Any) -> str:
    return str(getattr(bot, "id", None) or getattr(bot, "username", None) or "bot")


def _captures(adapter: TelegramAdapter) -> dict:
    from utils import read_json_or_empty
    return read_json_or_empty(_state_path(adapter))


def _write_captures(adapter: TelegramAdapter, data: dict) -> None:
    from utils import atomic_json_write
    atomic_json_write(_state_path(adapter), data)


async def _capture_original(adapter: TelegramAdapter, bot: Any) -> None:
    """Remember the operator's short description. Read on every connect, so text the operator set
    in BotFather since the last run replaces the old capture. A live text equal to one of our
    status strings is not the operator's (an earlier run, a crash, or a build before this fix wrote
    it): the capture is kept as it is, and nothing wrong is ever restored."""
    current = getattr(await bot.get_my_short_description(), "short_description", "") or ""
    if current in (adapter._status_online_text, adapter._status_offline_text):
        return
    data = _captures(adapter)
    if data.get(_bot_key(bot)) != current:
        data[_bot_key(bot)] = current
        _write_captures(adapter, data)


async def _restore_original(adapter: TelegramAdapter, bot: Any) -> None:
    """Indicator off: put the captured text back once and forget it."""
    data = _captures(adapter)
    original = data.get(_bot_key(bot))
    if not isinstance(original, str):
        return
    await bot.set_my_short_description(short_description=original[:_TELEGRAM_SHORT_DESCRIPTION_CAP])
    del data[_bot_key(bot)]
    _write_captures(adapter, data)
    logger.info("[%s] status_indicator is off: restored the bot's own short description", adapter.name)


async def set_status_indicator(adapter: TelegramAdapter, online: bool) -> None:
    """Connect (``online``) / clean disconnect. Best-effort: failures are debug-logged, never raised."""
    bot = adapter._bot
    if bot is None:
        return
    from .adapter import _redact_telegram_error_text
    enabled = bool(getattr(adapter, "_status_indicator_enabled", False))
    try:
        if not enabled:
            if online:
                await _restore_original(adapter, bot)
            return
        if online:
            await _capture_original(adapter, bot)
    except Exception as e:  # health: allow BLE001 -- best-effort profile text; never blocks connect/disconnect
        logger.debug("[%s] Bot short description capture/restore failed: %s", adapter.name,
                     _redact_telegram_error_text(e), exc_info=True)
        if not enabled:
            return
    text = (adapter._status_online_text if online else adapter._status_offline_text)[:_TELEGRAM_SHORT_DESCRIPTION_CAP]
    try:
        await bot.set_my_short_description(short_description=text)
        logger.info("[%s] Set bot status indicator to %r", adapter.name, text)
    except Exception as e:  # health: allow BLE001 -- best-effort profile text; never blocks connect/disconnect
        logger.debug("[%s] Failed to set bot status indicator to %r: %s", adapter.name, text,
                     _redact_telegram_error_text(e), exc_info=True)
