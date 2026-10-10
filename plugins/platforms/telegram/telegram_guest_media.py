"""Telegram guest-mode media delivery (Bot API 10.0): the ``deliver_<token>`` flow.

A guest chat's single reply goes through ``answerGuestQuery``, which takes an inline result, not an
upload. So a media send in a guest chat is staged instead: the file is uploaded to
``TELEGRAM_HOME_CHANNEL`` to mint a ``file_id``, and when the turn completes the stub is edited with a
"tap to send" button whose inline query is ``deliver_<token>``. Tapping it issues a fresh guest query
that is answered at once with the cached media, with no LLM pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_TG_SEND_METHODS = {"photo": "send_photo", "audio": "send_audio", "video": "send_video", "document": "send_document"}
_READY_SUFFIX = "✅ Ready — tap to receive"


def _tg():
    """The adapter module, read at call time: it rebinds the Telegram classes on a lazy install."""
    from plugins.platforms.telegram import adapter
    return adapter


def guest_media_root() -> Path:
    """The directory guest media staging is confined to (``HERMES_HOME/cache``).

    The delivery note only *asks* the model to stage files there; this is the enforced boundary. A
    guest-triggered turn coerced into emitting ``MEDIA:`` for any other host path (credentials,
    config) is rejected before the file is ever read.
    """
    from hermes_constants import get_hermes_home
    return (get_hermes_home() / "cache").resolve()


def guest_media_dir() -> str:
    """The directory the delivery note tells the model to save guest media into."""
    return str(guest_media_root() / "videos")


def resolve_guest_media_path(local_path: str, tg_type: str) -> tuple[Optional[tuple], Optional[str]]:
    """Validate a local media path for staging: ``((resolved_path, tg_type, mtime), None)`` or ``(None, error)``.

    Blocking (filesystem calls), so callers run it in a thread. A sandbox-visible cache path is
    translated to its host path first, then contained under :func:`guest_media_root`. The mtime in
    the cache key makes a file regenerated at the same path re-stage instead of reusing a stale id.
    """
    from tools.credential_files import from_agent_visible_cache_path
    try:
        resolved = Path(from_agent_visible_cache_path(local_path)).resolve()
    except (OSError, ValueError) as exc:  # e.g. an embedded null byte in a model-emitted path
        return None, f"guest_media: path validation failed: {exc}"
    if not resolved.is_relative_to(guest_media_root()):
        return None, f"guest_media: path outside the allowed staging directory: {local_path}"
    try:
        mtime = os.path.getmtime(resolved)
    except OSError:
        return None, f"guest_media: file not found: {local_path}"
    return (str(resolved), tg_type, mtime), None


def guest_delivery_note() -> str:
    """Per-turn channel prompt for a guest chat: no direct Bot API calls, files go through ``MEDIA:``."""
    try:
        media_dir = guest_media_dir()
    except Exception:  # health: allow BLE001 -- the prompt must still be built when HERMES_HOME can't be resolved
        logger.debug("guest media dir unresolved; using the default", exc_info=True)
        media_dir = "~/.hermes/cache/videos"
    return (
        "**Delivery constraint (this session only):** You are responding to "
        "a @mention in a group chat where the bot is not a member. "
        "Direct Bot API calls (sendVideo, sendPhoto, sendDocument, sendAudio, "
        "curl to api.telegram.org, etc.) to this chat will fail with "
        "\"Forbidden: bot is not a member\" — do NOT attempt them.\n"
        "To deliver a file:\n"
        f"1. Save it to `{media_dir}/<filename>` "
        "(this is the same path from inside the sandbox and on the host — "
        "do not use a temporary or any other container-local directory).\n"
        f"2. Output `MEDIA: {media_dir}/<filename>` — the exact same path. "
        "The platform will upload it to Telegram and deliver it to the chat automatically."
    )


def _staging_chat_id() -> tuple[Optional[int], Optional[str]]:
    staging = os.environ.get("TELEGRAM_HOME_CHANNEL")
    if not staging:
        return None, "guest_no_staging: TELEGRAM_HOME_CHANNEL not configured"
    try:
        return int(staging), None
    except (ValueError, TypeError):
        return None, "guest_no_staging: TELEGRAM_HOME_CHANNEL is not a valid chat id"


class TelegramGuestMediaMixin:
    """Staging and ``deliver_<token>`` answering for guest-chat media."""

    def _init_guest_media_state(self) -> None:
        # Per-turn staged media: chat_id → {file_id, media_kind}; last write wins, consumed at OPC.
        self._guest_turn_media: dict[str, dict] = {}
        # (resolved_path, tg_type[, mtime]) → file_id, so repeat deliveries don't re-upload.
        self._guest_file_id_cache: dict[tuple, str] = {}

    async def _guest_media_send(self, chat_id: str, tg_type: str, local_path: str,
                                caption: Optional[str] = None) -> SendResult:
        """Stage media to ``TELEGRAM_HOME_CHANNEL`` and record its ``file_id`` for delivery at OPC.

        ``tg_type``: "photo" | "audio" | "video" | "document". ``local_path`` may also be an http(s)
        URL, which Telegram fetches itself. *caption* is unused: the staged copy is never shown.
        """
        if not self._bot:
            return SendResult(success=False, error="guest_media: bot not available")
        staging_id, error = _staging_chat_id()
        if error:
            return SendResult(success=False, error=error)
        source: Union[str, Path]
        if local_path.startswith(("http://", "https://")):
            cache_key: Optional[tuple] = (local_path, tg_type)
            source = local_path
        else:
            cache_key, error = await asyncio.to_thread(resolve_guest_media_path, local_path, tg_type)
            if error:
                logger.warning("[%s] guest media rejected (chat=%s): %s", self.name, chat_id, error)
                return SendResult(success=False, error=error)
            source = Path(cache_key[0])
        file_id = self._guest_file_id_cache.get(cache_key)
        if not file_id:
            file_id, error = await self._guest_stage_upload(staging_id, tg_type, source)
            if error:
                return SendResult(success=False, error=error)
            self._guest_file_id_cache[cache_key] = file_id
        self._guest_turn_media[str(chat_id)] = {"file_id": file_id, "media_kind": tg_type}
        logger.info("[%s] guest media staged for OPC delivery (type=%s chat=%s)", self.name, tg_type, chat_id)
        return SendResult(success=True, message_id="staged")

    async def _guest_stage_upload(self, staging_id: int, tg_type: str,
                                  source: Union[str, Path]) -> tuple[Optional[str], Optional[str]]:
        """Upload *source* to the staging chat; ``(file_id, None)`` or ``(None, error)``."""
        send = getattr(self._bot, _TG_SEND_METHODS.get(tg_type, "send_document"))
        try:
            msg = await send(staging_id, **{tg_type: source, "disable_notification": True})
        except Exception as exc:  # health: allow BLE001 -- every staging failure becomes a SendResult; logged redacted
            error = _tg()._redact_telegram_error_text(exc)
            logger.warning("[%s] guest media staging failed (type=%s): %s", self.name, tg_type, error)
            return None, error
        media = msg.photo[-1] if tg_type == "photo" and msg.photo else getattr(msg, tg_type, None)
        file_id = getattr(media, "file_id", None)
        if not file_id:
            return None, "guest_media: staging upload returned no file_id"
        return file_id, None

    async def _guest_send_images(self, chat_id: str, images: list[tuple]) -> SendResult:
        """Stage each image of a batch; the last staged one is what the turn delivers."""
        from urllib.parse import unquote
        delivered, last_error = False, None
        for url, alt in images:
            path = unquote(url[len("file://"):]) if url.startswith("file://") else url
            result = await self._guest_media_send(chat_id, "photo", path, alt or None)
            delivered = delivered or result.success
            last_error = last_error if result.success else result.error
        return SendResult(success=delivered, error=None if delivered else last_error)

    async def _guest_answer_deliver_query(self, guest_query_id: str, token: str, chat_id: str) -> None:
        """Answer a ``deliver_<token>`` query: the cached media, or an error card for an unknown/expired token."""
        from tools.guest_mode_tool import resolve_token
        entry = resolve_token(token)
        if not entry:
            result = _tg().InlineQueryResultArticle(
                id="reply", title="Something went wrong",
                input_message_content=_tg().InputTextMessageContent("⚠️ Sorry, something went wrong. Please try again."),
            )
            if await self._answer_guest_query(guest_query_id, result, log_label="deliver expired"):
                logger.info("[%s] guest deliver: unknown or expired token (chat=%s)", self.name, chat_id)
            return
        kind = entry["media_kind"]
        cached_cls = {
            "photo": _tg().InlineQueryResultCachedPhoto, "video": _tg().InlineQueryResultCachedVideo,
            "audio": _tg().InlineQueryResultCachedAudio, "document": _tg().InlineQueryResultCachedDocument,
        }[kind]
        kwargs = {"id": "delivery", f"{kind}_file_id": entry["file_id"]}
        if kind != "photo":
            kwargs["title"] = kind.capitalize()
        if await self._answer_guest_query(guest_query_id, cached_cls(**kwargs), log_label="deliver"):
            logger.info("[%s] guest deliver (type=%s chat=%s)", self.name, kind, chat_id)

    async def _guest_edit_ready_button(self, imi: str, plain: str, turn_media: dict, chat_id: str) -> None:
        """OPC for a turn that staged media: edit the stub with the reply and a ``deliver_<token>`` button."""
        from tools.guest_mode_tool import mint_token
        token = mint_token(turn_media["file_id"], turn_media["media_kind"])
        text = f"{plain[:3900]}\n\n{_READY_SUFFIX}" if plain else _READY_SUFFIX
        markup = _tg().InlineKeyboardMarkup([[
            _tg().InlineKeyboardButton("📥 Tap to send here", switch_inline_query_current_chat=f"deliver_{token}"),
        ]])
        await self._bot.edit_message_text(text=text[:4096], inline_message_id=imi, reply_markup=markup)
        logger.info("[%s] guest OPC media button (chat=%s kind=%s)", self.name, chat_id, turn_media["media_kind"])
