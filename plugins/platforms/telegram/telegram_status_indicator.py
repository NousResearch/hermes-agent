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


def _status_text(adapter: TelegramAdapter, online: bool) -> str:
    return (adapter._status_online_text if online else adapter._status_offline_text)[:_TELEGRAM_SHORT_DESCRIPTION_CAP]


def _is_ours(adapter: TelegramAdapter, entry: dict, live: str) -> bool:
    """Hermes wrote ``live``: it is in the saved list of texts we wrote (survives a changed
    ``status_online``/``status_offline``), or it is a current status text (a build before this fix
    wrote it and kept no list)."""
    return live in entry.get("written", ()) or live in (_status_text(adapter, True), _status_text(adapter, False))


async def _capture_original(adapter: TelegramAdapter, bot: Any) -> None:
    """Remember the operator's short description. Read on every connect, so text the operator set
    in BotFather since the last run replaces the old capture; text Hermes wrote never does. Raises
    when the read or the save fails, and the caller then writes no status text this run."""
    live = getattr(await bot.get_my_short_description(), "short_description", "") or ""
    data = _captures(adapter)
    entry = data.get(_bot_key(bot)) or {}
    if not _is_ours(adapter, entry, live):
        entry["original"] = live
    # Record every text before it is written, so a crash between the save and the write still
    # leaves a live text this file names as ours.
    entry["written"] = sorted({*entry.get("written", ()), _status_text(adapter, True), _status_text(adapter, False)})
    data[_bot_key(bot)] = entry
    _write_captures(adapter, data)


async def _restore_original(adapter: TelegramAdapter, bot: Any) -> None:
    """Indicator off: put the captured text back once and forget it, unless the operator replaced
    the status text in BotFather since; their newer text is kept."""
    data = _captures(adapter)
    entry = data.get(_bot_key(bot))
    if not isinstance(entry, dict):
        return
    live = getattr(await bot.get_my_short_description(), "short_description", "") or ""
    original = entry.get("original")
    if isinstance(original, str) and _is_ours(adapter, entry, live):
        await bot.set_my_short_description(short_description=original[:_TELEGRAM_SHORT_DESCRIPTION_CAP])
        logger.info("[%s] status_indicator is off: restored the bot's own short description", adapter.name)
    del data[_bot_key(bot)]
    _write_captures(adapter, data)


async def set_status_indicator(adapter: TelegramAdapter, online: bool) -> None:
    """Connect (``online``) / clean disconnect. Best-effort: failures are debug-logged, never raised.
    No status text is written in a run whose capture failed: it would overwrite the only copy of the
    operator's text."""
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
        # Re-read before Offline too: a BotFather edit made while connected is saved, not overwritten.
        if online or getattr(adapter, "_status_capture_saved", False):
            adapter._status_capture_saved = False
            await _capture_original(adapter, bot)
            adapter._status_capture_saved = True
    except Exception as e:  # health: allow BLE001 -- best-effort profile text; never blocks connect/disconnect
        logger.debug("[%s] Bot short description capture/restore failed: %s", adapter.name,
                     _redact_telegram_error_text(e), exc_info=True)
    if not enabled or not getattr(adapter, "_status_capture_saved", False):
        return
    text = _status_text(adapter, online)
    try:
        await bot.set_my_short_description(short_description=text)
        logger.info("[%s] Set bot status indicator to %r", adapter.name, text)  # health: allow HX017 -- configured status string
    except Exception as e:  # health: allow BLE001 -- best-effort profile text; never blocks connect/disconnect
        logger.debug("[%s] Failed to set bot status indicator to %r: %s", adapter.name, text,
                     _redact_telegram_error_text(e), exc_info=True)
