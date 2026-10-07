"""Bot API 10.2/10.3 rich media: documents/photos/videos in ONE ``sendRichMessage``.

``InputRichMessage.media`` (10.2) embeds media referenced from the ``markdown``/``html`` body
via ``tg://document?id=`` / ``tg://photo?id=`` / ``tg://video?id=`` / ``tg://audio?id=`` links.
Each entry is ``InputRichMessageMedia(id, media)`` where ``media`` is an ``InputMedia*``:

* an existing Telegram ``file_id`` → plain string media (no upload, reusable);
* a remote HTTP(S) URL → plain string media (Telegram fetches it);
* a NEW local file → a PTB ``InputFile(attach=True)`` whose generated ``attach://<id>`` URI
  replaces the file_id in the message body, and whose bytes ride the SAME multipart request.

PTB 22.8 extracts InputFile values at the TOP LEVEL of api_kwargs, not inside arbitrary
dicts. The builders therefore hoist upload objects to top-level request parameters and
leave attach:// references inside the JSON media entries. Safe generated part names keep
caption and file bytes in one sendRichMessage request.

Also here: the raw-content detector for 10.3 rich constructs (``<blockquote expandable>``,
``<table compact>``/bordered/striped, ``tg-document``/``tg-photo``/``tg-video``/``tg-audio``
blocks and ``tg://…?id=`` links) and ``build_rich_message`` — the ``telegram_rich_message``
metadata facade that validates exactly one of ``html``/``markdown``/``blocks`` plus the media
list.

Sends follow the adapter's established rich contract: permanent/capability error → ``None``
(caller falls back to legacy), transient/unknown → an ambiguous NON-retryable ``SendResult``
(the request may have landed — never re-send, never duplicate).
"""

from __future__ import annotations

import pathlib
import re
import uuid
from typing import IO, Any, Callable, Dict, List, Optional, Union

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

# Bot API 10.2/10.3 contracts ---------------------------------------------------------
_RICH_FORMAT_KEYS = ("blocks", "html", "markdown")
_TG_MEDIA_SCHEMES = ("photo", "video", "document", "audio")
# InputRichMessageMedia.id: 1-64 chars of [A-Za-z0-9_-].
_MEDIA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Media links referenced from markdown/html bodies ("reuse or upload").
_TG_MEDIA_LINK_RE = re.compile(
    r"tg://(?P<kind>" + "|".join(_TG_MEDIA_SCHEMES) + r")\?id=(?P<id>[A-Za-z0-9_-]+)")
# HTML media blocks whose content forces rich rendering (10.3 added tg-document).
_HTML_MEDIA_BLOCK_RE = re.compile(
    r"<(?:img|video|audio|tg-document)\b[^>]*\bsrc=(\"|')(?P<src>(?:(?!\1).)*)\1[^>]*>",
    re.IGNORECASE)
_HTML_BLOCKQUOTE_EXPANDABLE_RE = re.compile(r"<blockquote\b[^>]*\bexpandable\b[^>]*>", re.IGNORECASE)
_HTML_TABLE_ATTR_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_HTML_TABLE_ATTRS_RE = re.compile(r"^bordered$|^striped$|^compact$", re.IGNORECASE)
# Hard caps the payload builders enforce before any network call.
_RICH_MEDIA_MAX_ITEMS = 50  # Bot API: up to 50 media attachments per rich message
_RICH_MAX_CHARS = 32768


class RichMediaError(ValueError):
    """A malformed ``telegram_rich_message`` metadata payload — caller error, not transport."""


def new_media_id(prefix: str = "m") -> str:
    """Safe unique ``InputRichMessageMedia.id`` / multipart part name (matches
    ``[A-Za-z0-9_-]{1,64}``; the uuid hex alone is 32 chars)."""
    return f"{prefix}{uuid.uuid4().hex}"[:64]


def rich_media_spec(kind: str, media: Any, media_id: Optional[str] = None) -> Dict[str, Any]:
    """One media item spec: ``{'kind', 'media', 'id'}`` with a validated/generated id.

    ``media`` is a Telegram ``file_id`` or HTTP(S) URL STRING (reuse path — sent verbatim,
    no upload), or a path / open file handle / bytes (new upload — becomes an ``InputFile``
    at payload time). Ambiguous path STRINGS are rejected: a local upload must be passed as
    ``pathlib.Path`` or an open handle so it can never be mistaken for a file_id.
    """
    kind = str(kind or "").strip().lower()
    if kind not in ("photo", "video", "document", "audio"):
        raise RichMediaError(f"unsupported rich media kind: {kind!r}")
    media_id = media_id or new_media_id("m")
    if not _MEDIA_ID_RE.match(media_id):
        raise RichMediaError(f"invalid rich media id: {media_id!r}")
    return {"kind": kind, "media": media, "id": media_id}


def rich_media_document(
    file: "Union[str, os.PathLike, IO[bytes]]" = None, *, file_id: Optional[str] = None,
    url: Optional[str] = None, caption: str = "", media_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Caption + document in ONE rich message: ``{content, media}`` spec.

    Exactly one source: a local ``file`` (path-like or open handle — NEVER an ambiguous string
    guessed as file_id), a remote ``url``, or a reusable Telegram ``file_id``. ``caption`` is
    the rich body (markdown/HTML formatting applies); ``media_id`` is validated/generated.
    The body references the media as ``tg://document?id=<media_id>`` (see
    :func:`media_ref_html`), so the caption and the attachment share one message.
    """
    source = _exactly_one_source(file=file, file_id=file_id, url=url)
    if source == "file" and isinstance(file, str):
        file = pathlib.Path(file)  # path-like ⇒ upload, never a file_id look-alike
    media = file_id if source == "file_id" else (url if source == "url" else file)
    return {"content": caption or "", "media": rich_media_spec("document", media, media_id)}


def media_ref_html(kind: str, media_id: str, caption: Optional[str] = None) -> str:
    """HTML media block referencing an embedded media id, e.g.
    ``<figure><tg-document src="tg://document?id=m..."></tg-document><figcaption>…</figcaption></figure>``.

    ``caption`` is inserted verbatim (caller pre-escapes); omit it when the rich body
    already carries the prose.
    """
    kind = str(kind).strip().lower()
    tag = {"document": "tg-document", "photo": "img", "video": "video", "audio": "audio"}.get(kind)
    if tag is None:
        raise RichMediaError(f"unsupported rich media kind: {kind!r}")
    if tag == "img":
        block = f'<img src="tg://{kind}?id={media_id}"/>'
    else:
        block = f'<{tag} src="tg://{kind}?id={media_id}"></{tag}>'
    if caption:
        return f"<figure>{block}<figcaption>{caption}</figcaption></figure>"
    return block


def _exactly_one_source(**fields: Any) -> str:
    present = [name for name, value in fields.items() if value]
    if not present:
        raise RichMediaError("exactly one of file/file_id/url is required")
    if len(present) > 1:
        raise RichMediaError(f"mutually exclusive sources given: {', '.join(sorted(present))}")
    return present[0]


# --- raw-content detection (10.3 grammar) ---------------------------------------------

def content_needs_rich_media(content: str) -> bool:
    """True when raw agent content contains Bot API 10.2/10.3 rich media/quote/table grammar
    that legacy MarkdownV2 cannot represent natively:

    * ``tg://photo|video|document|audio?id=…`` links (embedded media);
    * HTML media blocks ``<tg-document>``/``<img>``/``<video>``/``<audio>`` with a non-empty src;
    * ``<blockquote expandable>`` — 10.3 expandable/collapsible quotations;
    * ``<table …>`` carrying a ``bordered``/``striped``/``compact`` attribute — 10.3 table styles
      (compact = smaller cell indents; ``<table compact>`` alone must trigger rich rendering).
    """
    if not content:
        return False
    if _TG_MEDIA_LINK_RE.search(content):
        return True
    if _HTML_BLOCKQUOTE_EXPANDABLE_RE.search(content):
        return True
    for m in _HTML_MEDIA_BLOCK_RE.finditer(content):
        if m.group("src").strip():
            return True
    for m in _HTML_TABLE_ATTR_RE.finditer(content):
        attrs = re.findall(r"[A-Za-z:]+(?:=\"[^\"]*\")?", m.group(0))
        if any(_HTML_TABLE_ATTRS_RE.match(a.split("=")[0].strip()) for a in attrs if "=" not in a):
            return True
    return False


# --- telegram_rich_message metadata facade --------------------------------------------

def _normalize_media_list(media: Any) -> List[Dict[str, Any]]:
    if media is None:
        return []
    if isinstance(media, dict):
        media = [media]
    if not isinstance(media, (list, tuple)):
        raise RichMediaError("telegram_rich_message.media must be a list of media specs")
    out: List[Dict[str, Any]] = []
    for item in media:
        if not isinstance(item, dict):
            raise RichMediaError(f"media item must be a dict, got {type(item).__name__}")
        if "kind" not in item or "media" not in item:
            raise RichMediaError("each media item requires 'kind' and 'media'")
        out.append({**item, **rich_media_spec(item["kind"], item["media"], item.get("id"))})
    if len(out) > _RICH_MEDIA_MAX_ITEMS:
        raise RichMediaError(f"too many media attachments: {len(out)} > {_RICH_MEDIA_MAX_ITEMS}")
    return out


def build_rich_message(
    content: str, metadata: Optional[Dict[str, Any]], normalizer: Optional[Callable[[str], str]] = None,
) -> Optional[Dict[str, Any]]:
    """``InputRichMessage`` dict from content + ``metadata['telegram_rich_message']``.

    Metadata shape: ``{'html' | 'markdown' | 'blocks': ..., 'media': [...]}``.
    Exactly ONE of html/markdown/blocks must be present (official InputRichMessage contract);
    ``media`` is the optional 10.2 embedded-media list. Without the metadata key this returns
    ``None`` so callers keep their default (raw-markdown) rich behavior — an explicit opt-in.

    ``normalizer`` (optional) is applied to ``markdown``/``html`` content (e.g. the adapter's
    ``_rich_normalize_linebreaks``); ``blocks`` pass through untouched. Raises
    :class:`RichMediaError` on contract violations — malformed metadata is a caller bug and
    must fail loudly rather than silently degrade to a different format.
    """
    spec = (metadata or {}).get("telegram_rich_message")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise RichMediaError("telegram_rich_message must be an object")
    given = [k for k in _RICH_FORMAT_KEYS if spec.get(k) is not None]
    if len(given) != 1:
        raise RichMediaError(
            f"telegram_rich_message must specify exactly one of html/markdown/blocks, got {given or 'none'}")
    fmt = given[0]
    payload: Dict[str, Any] = {}
    if fmt == "blocks":
        blocks = spec[fmt]
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise RichMediaError("telegram_rich_message.blocks must be a non-empty list")
        payload["blocks"] = list(blocks)
    else:
        text = spec[fmt]
        if not isinstance(text, str) or not text.strip():
            raise RichMediaError(f"telegram_rich_message.{fmt} must be a non-empty string")
        if len(text) > _RICH_MAX_CHARS:
            raise RichMediaError(f"telegram_rich_message.{fmt} exceeds {_RICH_MAX_CHARS} chars")
        if normalizer is not None:
            text = normalizer(text)
        payload[fmt] = text
    media = _normalize_media_list(spec.get("media"))
    if media:
        payload["media"] = [_rich_media_entry(m) for m in media]
    return payload


def _rich_media_entry(spec: Dict[str, Any]) -> Dict[str, Any]:
    """One ``InputRichMessageMedia`` entry as a JSON-able dict: ``{'id', 'media': {...}}``.

    The official wire shape for the reuse path is exactly ``{"type": <kind>, "media": <ref>}``
    where ``<ref>`` is a Telegram ``file_id`` or HTTP(S) URL — no PTB class is involved, so
    this stays import-mock friendly. A NEW upload (``media`` = path / open file handle / bytes)
    builds a PTB ``InputFile(attach=True)`` whose ``attach://<id>`` URI replaces the file in
    the entry; the returned ``_input_file`` must be hoisted to a TOP-LEVEL ``api_kwargs``
    parameter named by the media id (see :func:`hoist_rich_media_uploads`) — PTB 22.8's
    ``RequestParameter`` only hoists ``InputFile``s sitting at parameter-VALUE level, never
    nested inside the ``rich_message`` dict, and ``json.dumps`` rejects them. Verified against
    real PTB 22.8: top-level ``api_kwargs['<id>'] = InputFile`` serializes to
    ``"<id>": "attach://<id>"`` in the JSON params PLUS a multipart part named ``<id>``,
    so caption body and file bytes share ONE ``sendRichMessage`` request.

    String ``media`` values are ALWAYS references (file_id/URL) — callers pass open file
    handles or ``pathlib.Path`` for new uploads, never a path string, so local paths are
    never misinterpreted as file ids.
    """
    from telegram import InputFile

    def upload(value: Any, name: str, filename: Optional[str] = None) -> Any:
        if isinstance(value, pathlib.Path):
            with value.open("rb") as file:
                result = InputFile(file, filename=filename or value.name, attach=True)
        else:
            result = InputFile(value, filename=filename, attach=True)
        result.attach_name = name
        return result

    media = spec["media"]
    properties = {key: value for key, value in spec.items() if key not in {"kind", "media", "id", "filename"}}
    entry: Dict[str, Any] = {"id": spec["id"]}
    if isinstance(media, str):
        reference = media
    else:
        file = upload(media, spec["id"], spec.get("filename"))
        reference = file.attach_uri
        entry["_input_file"] = file
    thumbnail = properties.get("thumbnail")
    if thumbnail is not None:
        thumb = upload(pathlib.Path(thumbnail) if isinstance(thumbnail, str) else thumbnail,
                       new_media_id("thumb"))
        properties["thumbnail"] = thumb.attach_uri
        entry["_thumbnail_file"] = thumb
    entry["media"] = {"type": spec["kind"], "media": reference, **properties}
    return entry


def hoist_rich_media_uploads(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Move the ``_input_file`` markers of ``payload['rich_message']['media']`` entries to
    TOP-LEVEL parameters named by each media id (in place). Reusable for the parent adapter's
    own payload builder (e.g. alongside ``_rich_payload_base``): any ``InputRichMessage``
    carrying a ``media`` list built by :func:`_rich_media_entry`/:func:`build_rich_message``
    can be finalized with this one call."""
    rich_message = payload.get("rich_message")
    media = rich_message.get("media") if isinstance(rich_message, dict) else None
    if not media:
        return payload
    for entry in media:
        input_file = entry.pop("_input_file", None)
        if input_file is not None:
            payload[entry["id"]] = input_file
        thumbnail = entry.pop("_thumbnail_file", None)
        if thumbnail is not None:
            payload[thumbnail.attach_name] = thumbnail
    return payload

# --- transport mixin ------------------------------------------------------------------

class TelegramRichMediaMixin:
    """Mixin for ``TelegramAdapter``: rich-media send/payload on top of the adapter's routing,
    flag coercion, error classifiers and latching. One ``sendRichMessage`` per media answer;
    permanent rejection → ``None`` (legacy owns the send), transient → ambiguous non-retryable
    (never re-send). Callers translate ``_RichMediaTransient`` into a ``SendResult`` via
    :meth:`_rich_media_transient_result`."""

    def _rich_media_transient_result(self, exc: "_RichMediaTransient") -> "SendResult":
        """Ambiguous transient failure → non-retryable SendResult (request may have landed;
        redacted error, RetryAfter honored). Mirrors the adapter's rich-control contract."""
        from gateway.platforms.base import SendResult
        return SendResult(
            success=False, error=exc.safe_text, retryable=False, retry_after=exc.retry_after,
            raw_response={"ambiguous": True, "what": "sendRichMessage (rich media)"})

    # The payload/transport core below composes the host adapter's routing helpers
    # (``_metadata_thread_id``, ``_thread_kwargs_for_send``,
    # ``_reply_to_message_id_for_send``, ``_notification_kwargs``) and classifiers
    # (``_is_rich_fallback_error``, ``_is_rich_capability_error``, ``_rich_send_disabled``,
    # ``_bot``); routing/thread/reply and notifications mirror ``_media_send_kwargs``.

    def _rich_media_flag_enabled(self) -> bool:
        """``extra.rich_messages`` opt-in (default off = legacy media paths)."""
        coerce = getattr(self, "_coerce_bool_extra", None)
        if callable(coerce):
            return bool(coerce("rich_messages", False))
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        value = extra.get("rich_messages", False)
        return str(value).strip().lower() in {"true", "1", "yes", "on"} if isinstance(value, str) else bool(value)

    def _rich_media_active(self) -> bool:
        """Rich media needs the opt-in, a live bot with ``do_api_request``, and no latched-off."""
        bot = getattr(self, "_bot", None)
        return bool(
            self._rich_media_flag_enabled()
            and bot is not None
            and getattr(bot, "do_api_request", None) is not None
            and not getattr(self, "_rich_send_disabled", False))

    async def _send_rich_media(
        self, chat_id: Any, content: str, media_specs: List[Dict[str, Any]], reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Any]:
        """ONE ``sendRichMessage`` carrying ``content`` + ``media_specs``.

        Returns the raw message-like result (dict or object with ``message_id``), or ``None``
        for a permanent/capability rejection (caller falls back to legacy) or an inactive
        transport (flag off / no bot / latched). Transient/ambiguous failures raise
        :class:`_RichMediaTransient` — the request may have reached Telegram, so the caller
        must NOT re-send (no duplicates); the host converts it to a non-retryable ambiguous
        ``SendResult``.
        """
        from plugins.platforms.telegram.adapter import (  # lazy: adapter imports this module
            _MEDIA_SEND_DEADLINE, _MEDIA_SEND_READ_TIMEOUT, _await_with_thread_deadline, _redact_telegram_error_text)
        if not self._rich_media_active():
            return None
        payload = self._rich_media_payload(chat_id, content, media_specs, reply_to, metadata)
        if payload is None:
            return None
        bot = self._bot
        try:
            msg = await _await_with_thread_deadline(
                bot.do_api_request("sendRichMessage", api_kwargs=payload, read_timeout=_MEDIA_SEND_READ_TIMEOUT),
                timeout=_MEDIA_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as exc:
            if getattr(self, "_is_rich_fallback_error", None) and self._is_rich_fallback_error(exc):
                if getattr(self, "_is_rich_capability_error", None) and self._is_rich_capability_error(exc):
                    self._rich_send_disabled = True
                return None
            raise _RichMediaTransient(exc, _redact_telegram_error_text(exc)) from exc
        message_id = msg.get("message_id") if isinstance(msg, dict) else getattr(msg, "message_id", None)
        if isinstance(msg, dict) and message_id is None:
            message_id = (msg.get("result") or {}).get("message_id")
        if message_id is None:
            raise _RichMediaTransient(RuntimeError("sendRichMessage response without message_id"), "no message_id")
        record = getattr(self, "_record_rich_sent", None)
        if callable(record):
            await record(chat_id, message_id, content)
        return msg

    def _rich_media_payload(
        self, chat_id: Any, content: str, media_specs: List[Dict[str, Any]], reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """``sendRichMessage`` api_kwargs: chat/routing/notification/reply + InputRichMessage.

        ``None`` = skip (caller falls back) — used when routing says legacy owns the send.
        Media entries are JSON-able ``InputRichMessageMedia`` dicts; each NEW upload's
        ``InputFile`` is added as a top-level ``api_kwargs`` parameter named by its media id,
        so PTB 22.8 serializes ``"<id>": "attach://<id>"`` and the file bytes land in the same
        multipart request — ONE ``sendRichMessage`` for caption + media, no second message.
        The body keeps its ``tg://…?id=<media_id>`` links.
        """
        metadata = metadata or {}
        thread_id = self._metadata_thread_id(metadata)
        chat_id_norm = normalize_telegram_chat_id(chat_id)
        reply_to_mode = getattr(self, "_reply_to_mode", "on")
        reply_to_id = self._reply_to_message_id_for_send(reply_to, metadata, reply_to_mode=reply_to_mode)
        payload: Dict[str, Any] = {"chat_id": chat_id_norm}
        thread_kwargs = self._thread_kwargs_for_send(
            chat_id_norm, thread_id, metadata, reply_to_message_id=reply_to_id, reply_to_mode=reply_to_mode)
        payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        payload.update(self._notification_kwargs(metadata))
        if reply_to_id is not None:
            payload["reply_parameters"] = {"message_id": reply_to_id}
        rich_message: Dict[str, Any] = build_rich_message(content, metadata) or {"markdown": content}
        if "markdown" in rich_message:
            from plugins.platforms.telegram.adapter import _rich_normalize_linebreaks
            rich_message["markdown"] = _rich_normalize_linebreaks(rich_message["markdown"])
        if media_specs:
            rich_message["media"] = [*rich_message.get("media", []), *(_rich_media_entry(m) for m in media_specs)]
        payload["rich_message"] = rich_message
        return hoist_rich_media_uploads(payload)


class _RichMediaTransient(Exception):
    """Internal: a transient/ambiguous rich-media failure (may have landed — never re-send)."""

    def __init__(self, exc: BaseException, safe_text: str):
        super().__init__(safe_text)
        self.exc = exc
        self.safe_text = safe_text
        retry_after = getattr(exc, "retry_after", None)
        self.retry_after = float(retry_after) if retry_after is not None else None
