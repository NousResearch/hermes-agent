"""Bot API 10.1 rich control prompts: MarkdownV2/HTML + inline keyboards → InputRichMessage HTML.

``TelegramAdapter`` gains this as a mixin (same pattern as ``TelegramGenerationMixin``). Scope:
convert existing control-prompt payloads (approval cards, pickers, confirm prompts) into rich
messages with native ``<tg-button-row>`` buttons; send/edit with the same permanent-vs-transient
contract as ``_try_send_rich`` (permanent → ``None`` so the caller falls back to the legacy prompt;
transient/unknown → a non-retryable ambiguous result so nothing is re-sent publicly).

Gated by ``extra.rich_controls`` — independent from ``rich_messages`` (finals opt-in).

Callbacks pass through VERBATIM: existing prefixes (``ea:``, ``mp:``, ``cl:``, …) are never
rewritten; only labels/attribute values are HTML-escaped.
"""

from __future__ import annotations

import html as _html
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import SendResult
from gateway.platforms.helpers import bounded_put
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

# Bound for the (chat_id, message_id) registry of rich control messages.
_RICH_CONTROL_REGISTRY_CAP = 512
# Bot API rich-message cap; above it fall back to the legacy prompt (lenient: HTML bytes).
_RICH_CONTROL_MAX_CHARS = 32768
# Telegram callback_data byte bounds.
_CALLBACK_DATA_MIN_BYTES, _CALLBACK_DATA_MAX_BYTES = 1, 64
_BUTTONS_PER_ROW_MAX = 8

_ALLOWED_STYLES = ("danger", "success", "primary", "link")
_DANGER_LABEL_WORDS = ("cancel", "deny", "stop", "delete", "remove", "reject", "no", "off")
_SUCCESS_LABEL_WORDS = ("approve", "confirm", "yes", "allow", "save", "ok", "accept")


def esc_attr(value: Any) -> str:
    """HTML-escape an attribute value (callback_data, style) — quotes and ampersands included."""
    return _html.escape(str(value if value is not None else ""), quote=True)


def esc_text(value: Any) -> str:
    """HTML-escape text content (labels, body text)."""
    return _html.escape(str(value if value is not None else ""), quote=False)


# --- MarkdownV2 → rich HTML ---------------------------------------------------------------
# MarkdownV2 escapes: every special char may carry a protective backslash.
_MDV2_UNESCAPE_RE = re.compile(r"\\([_*\[\]()~`>#\+\-=|{}.!\\])")
# Inside pre/code only ` and \ are escaped.
_MDV2_CODE_UNESCAPE_RE = re.compile(r"\\([`\\])")


def _mdv2_unescape(text: str) -> str:
    return _MDV2_UNESCAPE_RE.sub(r"\1", text)


def _mdv2_code_unescape(text: str) -> str:
    return _MDV2_CODE_UNESCAPE_RE.sub(r"\1", text)


def mdv2_to_html(text: str) -> str:
    """MarkdownV2 control text → rich HTML, preserving code/pre/bold/italic/underline/strike/
    spoiler/links. Escapes (``\\*``) are consumed first so markers pair only when genuinely open;
    inner content is then un-escaped group-wise. Control prompts are short and well-formed
    (produced by ``format_message``), so a single paired-marker pass suffices — no nesting."""
    if not text:
        return ""
    stash: Dict[str, str] = {}

    def _ph(html_segment: str) -> str:
        key = f"\x00RC{len(stash)}\x00"
        stash[key] = html_segment
        return key

    # 0) Stash every escape sequence (``\\x``) BEFORE marker matching; restored at the end.
    t = re.sub(r"\\(.)", lambda m: _ph(esc_text(m.group(1))), text)
    # 1) Fenced code, inline code (content may still contain escaped ` and \ — unescape inside).
    t = re.sub(r"```([\s\S]*?)```", lambda m: _ph(f"<pre>{esc_text(_mdv2_code_unescape(m.group(1)))}</pre>"), t)
    t = re.sub(r"`([^`\n]+)`", lambda m: _ph(f"<code>{esc_text(_mdv2_code_unescape(m.group(1)))}</code>"), t)
    # 2) Links: label and URL may contain placeholders (escaped chars) — keep them verbatim.
    t = re.sub(r"\[([^\]\n]+)\]\(((?:[^()\\]|\\.)+)\)",
               lambda m: _ph(f'<a href="{esc_attr(_mdv2_unescape(m.group(2)))}">'
                             f"{esc_text(_mdv2_unescape(m.group(1)))}</a>"), t)
    # 3) Paired markers; placeholders inside the groups are already-escaped literal chars.
    t = re.sub(r"\|\|([\s\S]*?)\|\|", lambda m: _ph(f"<tg-spoiler>{m.group(1)}</tg-spoiler>"), t)
    t = re.sub(r"\*\*([\s\S]*?)\*\*", lambda m: _ph(f"<b>{m.group(1)}</b>"), t)
    t = re.sub(r"__([\s\S]*?)__", lambda m: _ph(f"<u>{m.group(1)}</u>"), t)
    t = re.sub(r"\*([^*\n]+)\*", lambda m: _ph(f"<b>{m.group(1)}</b>"), t)
    t = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", lambda m: _ph(f"<i>{m.group(1)}</i>"), t)
    t = re.sub(r"~([^~\n]+)~", lambda m: _ph(f"<s>{m.group(1)}</s>"), t)
    # 4) Remaining bare text is literal (markers above consumed all pairs) — escape it.
    t = esc_text(t)
    for key in reversed(list(stash)):
        t = t.replace(key, stash[key])
    return t


def _parse_mode_kind(parse_mode: Any) -> str:
    s = str(parse_mode or "").lower()
    if "html" in s:
        return "html"
    if "markdown" in s:
        return "markdown"
    return "plain"


def _infer_style(label: str) -> Optional[str]:
    """Danger/success accents from the label; None keeps the client's default rendering."""
    lowered = label.lower()
    if any(w in lowered for w in _DANGER_LABEL_WORDS):
        return "danger"
    if any(w in lowered for w in _SUCCESS_LABEL_WORDS):
        return "success"
    return None


def markup_to_tg_button_rows(reply_markup: Any) -> List[List[Dict[str, Any]]]:
    """InlineKeyboardMarkup (or its ``to_dict()`` form) → callback-button rows. Non-callback buttons
    (url/web_app/switch_inline_query/…) are dropped: rich buttons carry exactly ONE action field;
    callback_data must be 1–64 bytes (Telegram contract). Rows that end up empty are dropped."""
    if reply_markup is None:
        return []
    if hasattr(reply_markup, "to_dict"):
        try:
            reply_markup = reply_markup.to_dict()
        except Exception:
            return []
    raw_rows = reply_markup.get("inline_keyboard") if isinstance(reply_markup, dict) else None
    if not isinstance(raw_rows, (list, tuple)):
        return []
    rows: List[List[Dict[str, Any]]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, (list, tuple)):
            continue
        row: List[Dict[str, Any]] = []
        for button in raw_row:
            if not isinstance(button, dict):
                continue
            data = button.get("callback_data")
            if not isinstance(data, str) or not data:
                continue  # url/web_app/… buttons: not convertible to a rich callback button
            if not (_CALLBACK_DATA_MIN_BYTES <= len(data.encode("utf-8")) <= _CALLBACK_DATA_MAX_BYTES):
                continue
            row.append({"text": str(button.get("text") or ""), "callback_data": data})
            if len(row) >= _BUTTONS_PER_ROW_MAX:
                break
        if row:
            rows.append(row)
    return rows


def buttons_html(rows: List[List[Dict[str, Any]]]) -> str:
    """Callback-button rows → ``<tg-button-row>`` HTML (parent contract: type=\"callback_data\")."""
    parts: List[str] = []
    for row in rows:
        buttons = []
        for button in row:
            style = _infer_style(button["text"])
            style_attr = f' style="{esc_attr(style)}"' if style in _ALLOWED_STYLES else ""
            buttons.append(
                f'<tg-button type="callback_data" data="{esc_attr(button["callback_data"])}"{style_attr}>'
                f"{esc_text(button['text'])}</tg-button>")
        if buttons:
            parts.append("<tg-button-row>" + "".join(buttons) + "</tg-button-row>")
    return "\n".join(parts)


def rich_control_html(text: str, parse_mode: Any, reply_markup: Any = None) -> str:
    """Body + buttons as one rich-HTML string. ``parse_mode`` HTML passes through (already escaped
    by its builder); MarkdownV2 converts; plain text escapes."""
    kind = _parse_mode_kind(parse_mode)
    if kind == "html":
        body = str(text or "")
    elif kind == "markdown":
        body = mdv2_to_html(text or "")
    else:
        body = esc_text(text or "")
    rows = markup_to_tg_button_rows(reply_markup)
    if not rows:
        return body
    buttons = buttons_html(rows)
    return f"{body}\n{buttons}" if body else buttons


def rich_control_payload(text: str, parse_mode: Any, reply_markup: Any = None) -> Dict[str, Any]:
    """``InputRichMessage`` for a control prompt (agreed facade contract: exactly one of
    html/markdown/blocks → we always use ``html``; buttons embedded as tg-button-row blocks)."""
    return {"html": rich_control_html(text, parse_mode, reply_markup)}


def _retry_after_of(exc: Exception) -> Optional[float]:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        m = re.search(r"retry\s+(?:in\s+)?(\d+)", str(exc).lower())
        if m:
            retry_after = float(m.group(1))
    return float(retry_after) if retry_after is not None else None


class TelegramRichControlsMixin:
    """Rich control sends/edits + the (chat_id, message_id) registry — host is ``TelegramAdapter``
    (relies on ``_bot``, ``_is_rich_fallback_error``, ``_coerce_bool_extra`` and
    ``_rich_send_disabled`` latching from the adapter)."""

    def _rich_controls_enabled(self) -> bool:
        """``extra.rich_controls`` — independent from ``rich_messages``; default off = legacy."""
        coerce = getattr(self, "_coerce_bool_extra", None)
        if callable(coerce):
            return bool(coerce("rich_controls", False))
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        value = extra.get("rich_controls", False)
        return str(value).strip().lower() in {"true", "1", "yes", "on"} if isinstance(value, str) else bool(value)

    def _rich_control_payload(self, text: str, parse_mode: Any, reply_markup: Any = None) -> Dict[str, Any]:
        """Facade hook (PrivateControls contract): InputRichMessage dict for a control payload."""
        return rich_control_payload(text, parse_mode, reply_markup)

    # --- bounded registry of sent rich control messages -----------------------------------
    def _rich_control_registry(self) -> Dict[Tuple[str, str], bool]:
        registry = getattr(self, "_rich_control_messages", None)
        if registry is None:
            registry = self._rich_control_messages = {}
        return registry

    def register_rich_control_message(self, chat_id: Any, message_id: Any) -> None:
        bounded_put(self._rich_control_registry(),
                    (str(normalize_telegram_chat_id(chat_id)), str(message_id)), True, _RICH_CONTROL_REGISTRY_CAP)

    def is_rich_control_message(self, chat_id: Any, message_id: Any) -> bool:
        return (str(normalize_telegram_chat_id(chat_id)), str(message_id)) in self._rich_control_registry()

    def forget_rich_control_message(self, chat_id: Any, message_id: Any) -> None:
        self._rich_control_registry().pop((str(normalize_telegram_chat_id(chat_id)), str(message_id)), None)

    # --- send / edit -----------------------------------------------------------------------
    def _rich_control_link_preview(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        preview = kwargs.get("link_preview_options")
        if preview is not None:
            if hasattr(preview, "to_dict"):
                try:
                    return {"link_preview_options": preview.to_dict()}
                except Exception:
                    pass
        if kwargs.get("disable_web_page_preview"):
            return {"link_preview_options": {"is_disabled": True}}
        return {}

    async def _try_send_rich_control(self, kwargs: Dict[str, Any]) -> Optional[SendResult]:
        """One ``sendRichMessage`` for a control prompt built from the LEGACY send kwargs
        (chat_id/text/parse_mode/reply_markup/thread/reply anchor). ``None`` = permanent rejection
        (caller falls back to the legacy prompt); transient/unknown → ambiguous non-retryable
        result (the request may have landed — the caller must NOT re-send). Success registers the
        message and returns the raw message-like object in ``raw_response``."""
        if not (self._rich_controls_enabled() and not getattr(self, "_rich_send_disabled", False) and self._bot):
            return None
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(kwargs.get("chat_id")),
            "rich_message": rich_control_payload(
                kwargs.get("text") or "", kwargs.get("parse_mode"), kwargs.get("reply_markup"))}
        thread_id = kwargs.get("message_thread_id")
        if thread_id is not None:
            payload["message_thread_id"] = int(thread_id)
        direct_topic_id = kwargs.get("direct_messages_topic_id")
        if direct_topic_id is not None:
            payload["direct_messages_topic_id"] = int(direct_topic_id)
        reply_to = kwargs.get("reply_to_message_id")
        if reply_to is not None:
            payload["reply_parameters"] = {"message_id": int(reply_to)}
        if kwargs.get("disable_notification"):
            payload["disable_notification"] = True
        payload.update(self._rich_control_link_preview(kwargs))
        if len(payload["rich_message"]["html"]) > _RICH_CONTROL_MAX_CHARS:
            return None  # over the rich cap → the legacy path owns chunking
        from plugins.platforms.telegram.adapter import (  # lazy: adapter imports this mixin
            _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _redact_telegram_error_text)
        try:
            msg = await _await_with_thread_deadline(
                self._bot.do_api_request("sendRichMessage", api_kwargs=payload),
                timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as exc:
            if self._is_rich_fallback_error(exc):
                return None
            # Never re-send: the request may have reached Telegram (public duplicate risk).
            return SendResult(success=False, error=_redact_telegram_error_text(exc), retryable=False,
                              retry_after=_retry_after_of(exc),
                              raw_response={"ambiguous": True, "what": "sendRichMessage", "exc": exc})
        message_id = msg.get("message_id") if isinstance(msg, dict) else getattr(msg, "message_id", None)
        if isinstance(msg, dict) and message_id is None:
            message_id = (msg.get("result") or {}).get("message_id")
        if message_id is None:
            return SendResult(success=False, error="rich control response without message_id", retryable=False,
                              raw_response={"ambiguous": True, "what": "sendRichMessage"})
        chat_id = payload["chat_id"]
        self.register_rich_control_message(chat_id, message_id)
        message_like = SimpleNamespace(message_id=message_id, chat_id=chat_id)
        return SendResult(success=True, message_id=str(message_id),
                          raw_response={"rich_control": True, "message": message_like})

    async def _edit_control_rich(
        self, chat_id: Any, message_id: Any, text: str, parse_mode: Any = None, reply_markup: Any = None,
    ) -> Optional[SendResult]:
        """Rich in-place edit of a registered control message (``editMessageText`` +
        ``rich_message``). Same None/transient contract as the send. A successful edit with no
        ``reply_markup`` is terminal → the registry entry is dropped (pagination edits keep it)."""
        if not (self._rich_controls_enabled() and self._bot):
            return None
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "message_id": int(message_id),
            "rich_message": rich_control_payload(text, parse_mode, reply_markup)}
        payload.update(self._rich_control_link_preview({}))
        from plugins.platforms.telegram.adapter import (  # lazy: same cycle-avoidance
            _TEXT_SEND_DEADLINE, _await_with_thread_deadline, _redact_telegram_error_text)
        try:
            await _await_with_thread_deadline(
                self._bot.do_api_request("editMessageText", api_kwargs=payload),
                timeout=_TEXT_SEND_DEADLINE, label="telegram-send", dump_on_blocked_loop=False)
        except Exception as exc:
            if "not modified" in str(exc).lower():
                if reply_markup is None:
                    self.forget_rich_control_message(chat_id, message_id)
                return SendResult(success=True, message_id=str(message_id))
            if self._is_rich_fallback_error(exc):
                return None
            return SendResult(success=False, error=_redact_telegram_error_text(exc), retryable=False,
                              retry_after=_retry_after_of(exc),
                              raw_response={"ambiguous": True, "what": "rich editMessageText", "exc": exc})
        if reply_markup is None:
            self.forget_rich_control_message(chat_id, message_id)
        return SendResult(success=True, message_id=str(message_id))

    def wrap_rich_control_query(self, query: Any) -> Any:
        """``query`` unchanged unless its message is a REGISTERED rich control; then a facade whose
        ``edit_message_text`` rich-edits in place. Permanent rich rejection falls back to the raw
        legacy edit (same message id, idempotent); a transient/ambiguous failure RE-RAISES the
        original exception — the rich edit may have landed, so an immediate legacy edit of the
        same message could race it ("no duplicate" contract). Everything else delegates."""
        message = getattr(query, "message", None)
        chat_id = getattr(message, "chat_id", None)
        message_id = getattr(message, "message_id", None)
        if chat_id is None or message_id is None or not self.is_rich_control_message(chat_id, message_id):
            return query
        return _RichControlQueryFacade(self, query, chat_id, message_id)


class _RichControlQueryFacade:
    """CallbackQuery facade: rich-edit recognized control messages, delegate the rest."""

    def __init__(self, adapter: TelegramRichControlsMixin, query: Any, chat_id: Any, message_id: Any):
        self._adapter = adapter
        self._query = query
        self._chat_id = chat_id
        self._message_id = message_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._query, name)

    async def edit_message_text(self, text: str = None, parse_mode: Any = None, reply_markup: Any = None, **kwargs: Any):
        result = await self._adapter._edit_control_rich(self._chat_id, self._message_id, text or "", parse_mode, reply_markup)
        if result is not None and result.success:
            return SimpleNamespace(message_id=self._message_id, chat_id=self._chat_id)
        raw = getattr(result, "raw_response", None) or {}
        if raw.get("ambiguous"):
            # Transient/ambiguous: re-raise the ORIGINAL exception — never race a legacy edit.
            raise raw.get("exc") or RuntimeError(result.error or "rich control edit failed ambiguously")
        # Permanent rejection only → the caller's legacy edit (same id, idempotent).
        return await self._query.edit_message_text(text=text, parse_mode=parse_mode, reply_markup=reply_markup, **kwargs)
