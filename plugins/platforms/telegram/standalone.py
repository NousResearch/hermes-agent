"""Out-of-process Telegram send (``hermes send``, the ``send_message`` tool, standalone cron delivery):
a one-shot Bot API client built like the gateway adapter's, with the adapter's formatting, chunking,
notification mode, rich-message fast-path, albums and fallbacks. Wired as the plugin's
``standalone_sender_fn``; core reaches it only through the platform registry.

``_media_caption_split``, the media extension sets and the result/redaction helpers are small local
copies of ``tools.send_message_senders``' (private core names a plugin may not import)."""

import asyncio
import contextlib
import logging
import os
import re
from types import SimpleNamespace

from agent.redact import redact_sensitive_text
from plugins.platforms.telegram.rich_code_regions import strip_code
from plugins.platforms.telegram.telegram_send_errors import SAME_REQUEST_ATTEMPTS, is_entity_parse_error
from pm import install_hint

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_VOICE_EXTS = {".ogg", ".opus"}
_TELEGRAM_SEND_AUDIO_EXTS = {".mp3", ".m4a"}  # sendAudio accepts only these; other audio -> sendVoice / document
# Captionable on the media bubble; voice/audio notes excluded (a caption there reads as a separate label).
_CAPTIONABLE_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | {".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx", ".zip"}
_TELEGRAM_CAPTION_LIMIT = 1024  # photo/video/document captions
_NO_DELIVERABLE = "No deliverable text or media remained after processing MEDIA tags"

_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\s]+)", re.IGNORECASE)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\s*=\s*([^\s,;]+)", re.IGNORECASE)


def _sanitize_error_text(text) -> str:
    """Redact secrets from error text before surfacing it to users/models."""
    redacted = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", redact_sensitive_text(text))
    return _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", redacted)


def _error(message: str) -> dict:
    return {"error": _sanitize_error_text(message)}


def _success(chat_id, warnings=None, **fields) -> dict:
    return {"success": True, "platform": "telegram", "chat_id": chat_id, **fields,
            **({"warnings": warnings} if warnings else {})}


def _media_caption_split(text, media_files):
    """``(caption, "")`` when the text rides on the media bubble as its caption: exactly one
    captionable file (not a voice/audio note) and text within Telegram's caption limit, else
    ``(None, text)`` (multi-file caption association is ambiguous)."""
    stripped = (text or "").strip()
    media = media_files or []
    if (not stripped or len(media) != 1 or len(stripped) > _TELEGRAM_CAPTION_LIMIT or media[0][1]
            or os.path.splitext(media[0][0])[1].lower() not in _CAPTIONABLE_EXTS):
        return None, text
    return stripped, ""


def _telegram_retry_delay(exc: Exception, attempt: int) -> float | None:
    """Retry delay in seconds, or None when final, from ``telegram_send_errors.send_error_kind``:
    ``retry_after`` wins; a connect or pool failure (the request never left) and Telegram's own 5xx
    JSON answer back off; a timeout, a 504, a reset after the write or an unreadable 5xx body may
    have posted the message, and a refusal or a local error will not change, so both are final."""
    from plugins.platforms.telegram import telegram_send_errors as kinds
    kind = kinds.send_error_kind(exc)
    if kind == kinds.FLOOD:
        try:
            return max(float(exc.retry_after), 0.0)
        except (TypeError, ValueError):
            return 1.0
    return kinds.same_request_backoff(attempt) if kind in (kinds.UNSENT, kinds.SERVER_ERROR) else None


async def _send_telegram_message_with_retry(bot, *, attempts: int = SAME_REQUEST_ATTEMPTS, **kwargs):
    """``bot.send_message`` with bounded retries on transient failures."""
    for attempt in range(attempts):
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:
            delay = _telegram_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts - 1:
                raise
            logger.warning("Transient Telegram send failure (attempt %d/%d), retrying in %.1fs: %s",
                           attempt + 1, attempts, delay, _sanitize_error_text(exc))
            await asyncio.sleep(delay)


def _is_telegram_thread_not_found(error: Exception) -> bool:
    """Mirror of the gateway adapter's ``_is_thread_not_found_error``.

    Matches the gateway adapter's ``_is_thread_not_found_error`` for the standalone ``_send_telegram`` path
    (issue #27012).
    """
    return "thread not found" in str(error).lower()


def _telegram_fallback_ips(extra):
    """The configured ``extra.fallback_ips`` or the seed IPs (no DoH discovery on a one-shot send);
    ``[]`` when ``HERMES_TELEGRAM_DISABLE_FALLBACK_IPS`` opts out, as for the gateway."""
    if os.getenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return []
    from plugins.platforms.telegram.telegram_network import SEED_FALLBACK_IPS, parse_fallback_ip_env
    # TELEGRAM_FALLBACK_IPS reaches ``extra["fallback_ips"]`` through the gateway env bridge.
    configured = extra.get("fallback_ips") or ""
    if not isinstance(configured, str):
        configured = ",".join(str(ip) for ip in configured)
    return parse_fallback_ip_env(configured) or list(SEED_FALLBACK_IPS)


def _telegram_bot(token, extra=None):
    """Bot built by the gateway adapter's request builder (``telegram_network.build_ptb_requests``:
    pool size, keepalive limits, timeouts, TLS verify): ``extra.base_url``/``base_file_url``
    (self-hosted or compatible Bot API), TELEGRAM_PROXY / ``extra.proxy_url`` checked against the
    server actually reached, else the fallback-IP transport on api.telegram.org. Falls back to PTB
    defaults (same URLs) on failure."""
    from telegram import Bot
    from gateway.platforms.base import describe_proxy_error, resolve_proxy_url, safe_proxy_url_for_log
    proxy = None
    extra = extra or {}
    base_url = extra.get("base_url")
    urls = {"base_url": base_url, "base_file_url": extra.get("base_file_url") or base_url} if base_url else {}
    try:
        from telegram.request import HTTPXRequest
        from plugins.platforms.telegram.telegram_network import build_ptb_requests, proxy_target_hosts
        fallback_ips = [] if base_url else _telegram_fallback_ips(extra)  # a custom server is reached directly
        proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=proxy_target_hosts(base_url, fallback_ips),
                                  configured=extra.get("proxy_url"))
        if proxy:
            logger.info("send_message: standalone Telegram send routed through proxy %s", safe_proxy_url_for_log(proxy))
        request, updates_request = build_ptb_requests(HTTPXRequest, fallback_ips=fallback_ips, proxy_url=proxy)
        return Bot(token=token, request=request, get_updates_request=updates_request, **urls)
    except Exception as request_err:
        logger.warning("send_message: failed to build the Telegram request (%s), using PTB defaults",
                       describe_proxy_error(request_err, proxy) or _sanitize_error_text(request_err),
                       exc_info=describe_proxy_error(request_err, proxy) is None)
    return Bot(token=token, **urls)


def _telegram_thread_kwargs(thread_id):
    """Topic id -> ``message_thread_id`` kwargs. Forum "General" is thread "1" inbound but
    the Bot API rejects message_thread_id=1, so it maps to no thread — same as the adapter."""
    if thread_id is None:
        return {}
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
        effective = TelegramAdapter._message_thread_id_for_send(str(thread_id))
    except Exception:  # adapter import failed (python-telegram-bot missing): explicit mapping
        effective = None if str(thread_id) == "1" else int(thread_id)
    return {} if effective is None else {"message_thread_id": effective}


def _strip_mdv2_safe(text):
    """Strip MarkdownV2 escapes for the plain-text fallback; identity if unavailable."""
    try:
        from plugins.platforms.telegram.adapter import _strip_mdv2
        return _strip_mdv2(text)
    except Exception:
        return text


async def _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **kwargs):
    """Bot API media method by extension: photo (unless forced document), video, voice note,
    sendAudio (MP3/M4A only), else document."""
    kind = next((k for exts, k in ((() if force_document else _IMAGE_EXTS, "photo"), (_VIDEO_EXTS, "video"),
                                    (_VOICE_EXTS if is_voice else (), "voice"), (_TELEGRAM_SEND_AUDIO_EXTS, "audio"))
                 if ext in exts), "document")
    return await getattr(bot, f"send_{kind}")(chat_id=chat_id, **{kind: f}, **kwargs)


def _is_telegram_parse_error(error) -> bool:
    return is_entity_parse_error(error)


async def _telegram_send_text_chunk(bot, chat_id, chunk, parse_mode, has_html, text_kwargs):
    """One text chunk with adapter-matching fallbacks: thread-not-found -> retry without
    ``message_thread_id`` (dropped from ``text_kwargs`` for later chunks too); parse failure
    -> plain text. A rejected HTML chunk (an unsupported tag beside supported ones) first
    retries as MarkdownV2, so its Markdown still renders."""
    async def send(text, mode):
        return await _send_telegram_message_with_retry(bot, chat_id=chat_id, text=text, parse_mode=mode, **text_kwargs)
    try:
        return await send(chunk, parse_mode)
    except Exception as md_error:
        # Thread not found — retry without message_thread_id so the message still delivers (matching the
        # gateway adapter's fallback behaviour, issue #27012).
        if _is_telegram_thread_not_found(md_error) and text_kwargs.get("message_thread_id") is not None:
            logger.warning("Thread %s not found in _send_telegram, retrying without message_thread_id",
                           text_kwargs.pop("message_thread_id"))
            return await send(chunk, parse_mode)
        if not _is_telegram_parse_error(md_error):
            raise
        logger.warning("Parse mode %s failed in _send_telegram, falling back: %s",
                       parse_mode, _sanitize_error_text(md_error))
        if has_html and (mdv2 := _telegram_mdv2(chunk, max_units=4096)) is not None:
            try:
                return await send(mdv2, "MarkdownV2")
            except Exception as mdv2_error:
                if not _is_telegram_parse_error(mdv2_error):
                    raise
        return await send(chunk if has_html else _strip_mdv2_safe(chunk), None)


async def _telegram_send_one_media(bot, chat_id, media_path, is_voice, *, caption, parse_mode, has_html,
                                   thread_kwargs, force_document):
    """Upload one file with adapter-matching fallbacks (thread-not-found -> no
    ``message_thread_id``; caption parse failure -> plain caption); retries re-seek the file."""
    ext = os.path.splitext(media_path)[1].lower()
    voice_note = ext in _VOICE_EXTS and is_voice
    # ``caption`` is only set for a single captionable file, so this never double-captions
    # a multi-file send or a voice note.
    media_kwargs = {**thread_kwargs, **({"caption": caption, "parse_mode": parse_mode}
                                        if caption is not None and not voice_note else {})}
    if voice_note or ext in _TELEGRAM_SEND_AUDIO_EXTS:
        with contextlib.suppress(Exception):
            from plugins.platforms.telegram.adapter import _probe_voice_duration_seconds
            duration = await asyncio.to_thread(_probe_voice_duration_seconds, media_path)
            if duration is not None:
                media_kwargs["duration"] = duration
    thumb_path = None
    if ext in _VIDEO_EXTS:
        # Telegram processes only small video uploads itself; past that the message carries no
        # geometry and no thumbnail, so clients draw a square tile (see the adapter helpers).
        # Keyed on the extension because ``_telegram_send_media`` routes every video extension to
        # ``sendVideo`` regardless of ``force_document`` (which only forces images to documents).
        with contextlib.suppress(Exception):
            from plugins.platforms.telegram.adapter import _probe_video_geometry, _video_thumbnail_jpeg
            geometry = await asyncio.to_thread(_probe_video_geometry, media_path)
            if geometry:
                media_kwargs.update(geometry)
                thumb_path = await asyncio.to_thread(_video_thumbnail_jpeg, media_path, geometry.get("duration"))
                if thumb_path:
                    media_kwargs["thumbnail"] = thumb_path
    from plugins.platforms.telegram.telegram_media_budget import scaled_budgets
    # The adapter's media budgets (#133164): the Bot's 20s read timeout cuts off Telegram's
    # post-upload processing, so a delivered file came back "Timed out" and got resent.
    media_kwargs.update(scaled_budgets([media_path, thumb_path], 0.0)[0])
    try:
        with open(media_path, "rb") as f:  # noqa: ASYNC230 -- file handle is streamed to the upload; a local open() is non-blocking in practice
            try:
                return await _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **media_kwargs)
            except Exception as media_err:
                if _is_telegram_thread_not_found(media_err) and media_kwargs.get("message_thread_id"):
                    logger.warning("Thread %s not found for media send, retrying without message_thread_id",
                                   media_kwargs.pop("message_thread_id"))
                elif media_kwargs.get("parse_mode") and is_entity_parse_error(media_err, ("parse", "caption")):
                    logger.warning("Caption parse failed for media send, retrying plain: %s",
                                   _sanitize_error_text(media_err))
                    media_kwargs.pop("parse_mode", None)
                    if not has_html and media_kwargs.get("caption"):
                        media_kwargs["caption"] = _strip_mdv2_safe(media_kwargs["caption"])
                else:
                    raise
                f.seek(0)
                return await _telegram_send_media(bot, chat_id, f, ext, is_voice, force_document, **media_kwargs)
    finally:
        if thumb_path:
            with contextlib.suppress(OSError):
                os.remove(thumb_path)


# The complete set of tags Telegram's HTML parse mode understands; anything else in angle
# brackets (``<CoT>``, ``List<int>``, ``<name>``) is prose and must take the MarkdownV2 branch.
_TELEGRAM_HTML_TAGS = ("b", "strong", "i", "em", "u", "ins", "s", "strike", "del", "a",
                       "code", "pre", "blockquote", "tg-spoiler", "span", "tg-emoji")
# Opening tag (optionally with attributes) or closing tag. The name must end at ``>`` or
# whitespace, so ``<b>`` never matches ``<body>``; longest-first keeps the alternation readable.
# Quoted attribute values may hold ``>`` (``href="a>b"``): the tag ends at the first unquoted one.
_TELEGRAM_HTML_TAG_ALT = "|".join(sorted(_TELEGRAM_HTML_TAGS, key=len, reverse=True))
_TELEGRAM_HTML_TAG_RE = re.compile(
    rf"""</?(?:{_TELEGRAM_HTML_TAG_ALT})(?:\s(?:[^>"']++|"[^"]*+"|'[^']*+')*+)?>""", re.IGNORECASE)


def _telegram_album_paths(media_files, force_document, caption):
    """The paths when the batch is two or more existing still photos and nothing else (no caption,
    voice, GIF or forced document, which keep the per-file path), else ``[]``."""
    if force_document or caption is not None or len(media_files) < 2:
        return []
    ok = all(not voice and os.path.splitext(path)[1].lower() in _IMAGE_EXTS - {".gif"} and os.path.exists(path)
             for path, voice in media_files)
    return [os.path.abspath(path) for path, _ in media_files] if ok else []


# Telegram rejects the whole message on one stray ``<`` / ``>`` / ``&`` outside a supported tag.
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def _escape_non_telegram_markup(text):
    """Escape everything between Telegram's supported tags, so a model-invented ``<service>``
    reads as text instead of failing the parse (and the rest of the formatting with it)."""
    def esc(chunk):
        parts = _HTML_ENTITY_RE.split(chunk)
        entities = _HTML_ENTITY_RE.findall(chunk)
        out = [p.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for p in parts]
        return "".join(x for pair in zip(out, entities + [""]) for x in pair)
    pieces, pos = [], 0
    for tag in _TELEGRAM_HTML_TAG_RE.finditer(text):
        pieces += [esc(text[pos:tag.start()]), tag.group(0)]
        pos = tag.end()
    return "".join(pieces) + esc(text[pos:])


def _telegram_format(message):
    """``(formatted, parse_mode, has_html)``: text using a Telegram HTML tag outside Markdown code
    is sent as HTML (any other markup escaped); otherwise Markdown -> MarkdownV2 via the adapter's
    ``format_message``."""
    from telegram.constants import ParseMode
    # A tag shown as Markdown code (any backtick width, as the MarkdownV2 path reads it) is an example.
    if _TELEGRAM_HTML_TAG_RE.search(strip_code(message)):
        return _escape_non_telegram_markup(message), ParseMode.HTML, True
    mdv2 = _telegram_mdv2(message)
    return (message if mdv2 is None else mdv2), ParseMode.MARKDOWN_V2, False  # None: send as-is


def _telegram_mdv2(text, *, max_units=None):
    """``text`` as the adapter's MarkdownV2 (``format_message``), or ``None`` when the adapter is
    unavailable or the escaped text exceeds ``max_units`` UTF-16 units."""
    from gateway.platforms.base import utf16_len
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ImportError:
        return None
    formatted = TelegramAdapter.__new__(TelegramAdapter).format_message(text)
    return None if max_units is not None and utf16_len(formatted) > max_units else formatted


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False,
                         force_document=False, extra=None, notify=None):
    """One-shot Telegram Bot API send over the platform's ``extra`` (base_url, proxy, fallback IPs,
    ``rich_messages``), under the adapter's push-notification mode (``notify`` rings regardless);
    parse failures fall back to plain text."""
    if str(thread_id or "").strip() and not str(thread_id).strip().lstrip("-").isdigit():
        # ``<chat>:<topic name>`` parses for every caller, but only the live gateway's cron router
        # resolves a name (ensure_dm_topic): the Bot API has no topic lookup by name (#80483).
        return _error(f"Telegram topic '{thread_id}' is a name, not a thread id: a direct send cannot "
                      "resolve named topics. Use telegram:<chat_id>:<numeric thread id>, or deliver "
                      "through a cron job while the gateway is running.")
    try:
        from plugins.platforms.telegram.telegram_outbound import (
            is_visually_empty, notification_kwargs, resolve_notifications_mode, send_album_standalone,
            send_rich_standalone)
        if is_visually_empty(message):
            message = ""  # invisible-only text would land as an empty bubble
        formatted, send_parse_mode, _has_html = _telegram_format(message)
        bot = _telegram_bot(token, extra)
        from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
        from gateway.platforms.base import BasePlatformAdapter, utf16_len
        # Telegram accepts a numeric chat_id OR an @username string; never force-int.
        # See #13206.
        int_chat_id = normalize_telegram_chat_id(chat_id)
        media_files = media_files or []
        thread_kwargs = {**_telegram_thread_kwargs(thread_id),
                         **notification_kwargs(resolve_notifications_mode(), notify)}
        # disable_web_page_preview is only valid for send_message, not media sends.
        text_kwargs = {**thread_kwargs, **({"disable_web_page_preview": True} if disable_link_previews else {})}
        last_msg, warnings, _tg_caption = None, [], None
        if message and not media_files:
            rich = await send_rich_standalone(bot, {**(extra or {}), "disable_link_previews": disable_link_previews},
                                              chat_id, message, thread_id=thread_id, notify=notify)
            if rich is not None:
                return (_success(chat_id, message_id=rich.message_id) if rich.success
                        else _error(f"Telegram send failed: {rich.error}"))
        # MEDIA caption rides on the bubble as its *formatted* caption; formatting can inflate a
        # raw <1024 string past Telegram's cap, so re-check in UTF-16 units.
        _cap, _ = _media_caption_split(message, media_files)
        if _cap is not None and utf16_len(formatted) <= _TELEGRAM_CAPTION_LIMIT:
            _tg_caption, formatted = formatted, ""  # suppress the separate text send below
        # Chunk *after* formatting, in UTF-16 units: escaping can push a raw-<4096 message over.
        chunks = BasePlatformAdapter.truncate_message(formatted, 4096, len_fn=utf16_len) if formatted.strip() else []
        if len(chunks) > 1 and not _has_html:
            # truncate_message appends a raw " (1/2)" suffix after format_message escaped the text;
            # escape the MarkdownV2-special parentheses, as the live adapter does (#74004).
            from plugins.platforms.telegram.adapter import _separate_chunk_indicator_from_fence
            chunks = [
                _separate_chunk_indicator_from_fence(re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk))
                for chunk in chunks
            ]
        for chunk in chunks:
            last_msg = await _telegram_send_text_chunk(bot, int_chat_id, chunk, send_parse_mode, _has_html, text_kwargs)
        if album := _telegram_album_paths(media_files, force_document, _tg_caption):
            result = await send_album_standalone(bot, extra or {}, chat_id, album, thread_id=thread_id, notify=notify)
            detail = result.raw_response or {}
            warnings += [_sanitize_error_text(f"Failed to send Telegram photo {path}: {err}")
                         for path, err in detail.get("failed", ())]
            if detail.get("ambiguous"):
                warnings.append("Telegram never confirmed a photo album; it may already be in the chat, so it was not resent")
            if not result.success and not detail.get("failed") and not detail.get("refused"):
                warnings.append(_sanitize_error_text(f"Failed to send Telegram photo album: {result.error}"))
            # A refused album goes photo by photo below (topic fallback, a warning per failure).
            media_files = [(path, False) for path in detail.get("refused", ())]
            last_msg = last_msg or (SimpleNamespace(message_id=None) if result.success else None)
        for media_path, is_voice in media_files:
            if not os.path.exists(media_path):
                warnings.append(f"Media file not found, skipping: {media_path}")
                logger.warning(warnings[-1])
                # Caption mode suppressed the text send; the file is gone, so deliver the words alone.
                if _tg_caption is not None and last_msg is None:
                    try:
                        last_msg = await _send_telegram_message_with_retry(
                            bot, chat_id=int_chat_id, text=_tg_caption, parse_mode=send_parse_mode, **text_kwargs)
                        _tg_caption = None  # delivered — don't re-caption a later file
                    except Exception as _cap_err:
                        logger.warning("Telegram caption-fallback send failed for missing media: %s",
                                       _sanitize_error_text(_cap_err))
                continue
            try:
                last_msg = await _telegram_send_one_media(
                    bot, int_chat_id, media_path, is_voice, caption=_tg_caption, parse_mode=send_parse_mode,
                    has_html=_has_html, thread_kwargs=thread_kwargs, force_document=force_document)
            except Exception as e:
                from plugins.platforms.telegram.telegram_send_errors import may_have_delivered
                maybe = " (Telegram may still deliver it; do not resend)" if may_have_delivered(e) else ""
                warnings.append(_sanitize_error_text(f"Failed to send media {media_path}: {e}{maybe}"))
                logger.error(warnings[-1])
        if last_msg is None:
            return {"error": _NO_DELIVERABLE, **({"warnings": warnings} if warnings else {})}
        return _success(chat_id, warnings, **({"message_id": str(last_msg.message_id)}
                                                          if last_msg.message_id is not None else {}))
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: "
                f"{install_hint('telegram')}"}
    except Exception as e:
        return _error(f"Telegram send failed: {e}")


async def standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False,
                          notify=None, **_ignored):
    """``standalone_sender_fn``: the target profile's token (``pconfig.token``, else its scoped
    ``TELEGRAM_BOT_TOKEN``) and ``extra`` (base_url, proxy, fallback IPs, rich_messages, link
    previews); ``notify`` rings under the ``important`` notification mode (cron.delivery.notify)."""
    token = getattr(pconfig, "token", None)
    if not token:
        from agent.secret_scope import get_secret  # profile-scoped: never borrow another profile's token
        token = get_secret("TELEGRAM_BOT_TOKEN", "") or ""
    extra = getattr(pconfig, "extra", None) or {}
    return await _send_telegram(
        token, chat_id, message, media_files=media_files, thread_id=thread_id,
        disable_link_previews=bool(extra.get("disable_link_previews")), force_document=force_document,
        extra=extra, notify=notify)
