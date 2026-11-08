"""Telegram outbound send policy shared by the gateway adapter and the standalone sender
(``hermes send``, the ``send_message`` tool, standalone cron delivery): the push-notification
mode and the visually-empty guard, so both lanes make the same decision for the same text."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_NOTIFICATION_MODES = ("all", "important")

# Code points that render as nothing: zero-width space/non-joiner/joiner, word joiner, invisible
# operators, BOM, soft hyphen, Mongolian vowel separator, and the bidi marks/embeddings/isolates.
# A message made only of these (plus whitespace) passes ``str.strip()`` and lands as an empty bubble.
_INVISIBLE = str.maketrans("", "", (
    "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad\u180e"
    "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"))


def is_visually_empty(text: str | None) -> bool:
    """True when ``text`` has no visible glyph (empty, whitespace, or invisible code points only)."""
    return not text or not text.translate(_INVISIBLE).strip()


def resolve_notifications_mode() -> str:
    """``all`` or ``important``: ``HERMES_TELEGRAM_NOTIFICATIONS``, else the active profile's
    ``display.platforms.telegram.notifications``; unknown values warn and use ``important``."""
    mode = os.getenv("HERMES_TELEGRAM_NOTIFICATIONS", "").strip().lower()
    if not mode:
        from hermes_cli.config import cfg_get, load_config_readonly
        raw = cfg_get(load_config_readonly(), "display", "platforms", "telegram", "notifications")
        mode = "" if raw is None else str(raw).strip().lower()
    if not mode:
        return "important"
    if mode not in _NOTIFICATION_MODES:
        logger.warning("Unknown telegram notifications mode '%s', defaulting to 'important' (valid: %s)",
                       mode, ", ".join(_NOTIFICATION_MODES))
        return "important"
    return mode


def notification_kwargs(mode: str, notify: bool | None) -> dict:
    """Bot API kwargs for one send: silent under ``important`` unless the caller asks to ``notify``."""
    return {} if mode != "important" or notify else {"disable_notification": True}


def _standalone_adapter(bot, extra: dict):
    """An unconnected ``TelegramAdapter`` over the standalone ``bot`` (no polling, no lock), so an
    out-of-process send reuses the gateway's gates, routing and fallbacks verbatim."""
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    adapter = TelegramAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))
    adapter._bot = bot
    adapter._notifications_mode = resolve_notifications_mode()
    return adapter


def _metadata(thread_id, notify) -> dict:
    return {**({"thread_id": str(thread_id)} if thread_id is not None else {}), **({"notify": True} if notify else {})}


async def send_rich_standalone(bot, extra: dict, chat_id, content: str, *, thread_id=None, notify=None):
    """The gateway's ``sendRichMessage`` fast-path for an out-of-process send: the same gates
    (``rich_messages`` opt-in, constructs MarkdownV2 degrades, Desktop crash/garble shapes, size cap,
    capability latch) and routing. ``None`` = not eligible or permanently rejected, so the caller
    sends legacy MarkdownV2; otherwise the adapter's ``SendResult``. A transient failure is returned,
    not resent as legacy, because the rich request may have landed."""
    adapter = _standalone_adapter(bot, extra)
    if not adapter._should_attempt_rich(content):
        return None
    return await adapter._try_send_rich(str(chat_id), content, None, _metadata(thread_id, notify))


async def send_album_standalone(bot, extra: dict, chat_id, paths: list[str], *, thread_id=None, notify=None):
    """Photos as native albums (``sendMediaGroup``, 10 per chunk) through the adapter's album sender.
    A refused chunk is handed back in ``raw_response["refused"]`` (local paths) for the caller's own
    per-file upload, which reports each failure; a lost reply is ``ambiguous`` and never resent."""
    from urllib.parse import quote, unquote
    from plugins.platforms.telegram.telegram_album import send_photo_album
    adapter = _standalone_adapter(bot, extra)
    # The caller hands absolute, existence-checked paths; quote() matches the album sender's unquote().
    images = [("file://" + quote(p), None) for p in paths]
    result = await send_photo_album(adapter, str(chat_id), images, _metadata(thread_id, notify), per_photo_fallback=False)
    if result.raw_response:
        result.raw_response["refused"] = [unquote(url[7:]) for url, _ in result.raw_response["refused"]]
    return result
