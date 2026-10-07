"""Bot API 10.3 rich control prompts: MarkdownV2/HTML + inline keyboards → InputRichMessage HTML.

``TelegramAdapter`` gains this as a mixin (same pattern as ``TelegramGenerationMixin``). Scope:
convert existing control-prompt payloads (approval cards, pickers, confirm prompts) into rich
messages with native ``<tg-button-row>`` buttons; send/edit with the same permanent-vs-transient
contract as ``_try_send_rich`` (permanent → ``None`` so the caller falls back to the legacy prompt;
transient/unknown → a non-retryable ambiguous result so nothing is re-sent publicly).

Gated by ``extra.rich_controls`` — independent from ``rich_messages`` (finals opt-in).

Buttons cover ALL RichMessageButton actions (Bot API 10.3): callback_data, url, web_app,
login_url, switch_inline_query, switch_inline_query_current_chat,
switch_inline_query_chosen_chat, copy_text, disabled. Any button that cannot be
represented (callback_game/pay/malformed) makes the whole markup unsupported — the caller
falls back to the legacy prompt; a user action is NEVER silently dropped.
Callbacks pass through VERBATIM: existing prefixes (``ea:``, ``mp:``, ``cl:``, …) are never
rewritten; only labels/attribute values are HTML-escaped. Button labels are PLAIN text from
the legacy ``InlineKeyboardButton.text`` field (no entity channel), so they are escaped
losslessly; custom-emoji/datetime rich labels are not representable through this converter
and are left to future callers building ``InputRichMessage`` blocks directly.

Row alignment (opt-in, no global flag): ``InlineKeyboardMarkup(api_kwargs={"align": …})``
applies one alignment to every row; ``api_kwargs={"row_alignments": [...]}`` gives a
per-row list (missing/invalid entries → no align attribute = client default).
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

# All RichMessageButton action kinds (Bot API 10.3). Exactly ONE per button.
_RICH_BUTTON_ACTIONS = (
    "callback_data", "url", "web_app", "login_url", "switch_inline_query",
    "switch_inline_query_current_chat", "switch_inline_query_chosen_chat", "copy_text", "disabled")
# ``<tg-button-row>`` alignment contract: left/center/right per row.
_ROW_ALIGNMENTS = ("left", "center", "right")
_SNAKE_TO_DASH = re.compile(r"_")


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
    t = re.sub(r"\\(.)", lambda m: _ph(_html.escape(m.group(1), quote=True)), text)
    # 1) Fenced code, inline code (content may still contain escaped ` and \ — unescape inside).
    t = re.sub(r"```([\s\S]*?)```", lambda m: _ph(f"<pre>{esc_text(_mdv2_code_unescape(m.group(1)))}</pre>"), t)
    t = re.sub(r"`([^`\n]+)`", lambda m: _ph(f"<code>{esc_text(_mdv2_code_unescape(m.group(1)))}</code>"), t)
    # 2) Links: label and URL may contain placeholders (escaped chars) — keep them verbatim.
    # Escape the raw marker contents before wrapping them. Placeholder tokens for escaped
    # MarkdownV2 characters contain no HTML-significant bytes and are restored below.
    t = re.sub(r"\[([^\]\n]+)\]\(((?:[^()\\]|\\.)+)\)",
               lambda m: _ph(f'<a href="{esc_attr(_mdv2_unescape(m.group(2)))}">'
                             f"{esc_text(m.group(1))}</a>"), t)
    t = re.sub(r"\|\|([\s\S]*?)\|\|", lambda m: _ph(f"<tg-spoiler>{esc_text(m.group(1))}</tg-spoiler>"), t)
    t = re.sub(r"\*\*([\s\S]*?)\*\*", lambda m: _ph(f"<b>{esc_text(m.group(1))}</b>"), t)
    t = re.sub(r"__([\s\S]*?)__", lambda m: _ph(f"<u>{esc_text(m.group(1))}</u>"), t)
    t = re.sub(r"\*([^*\n]+)\*", lambda m: _ph(f"<b>{esc_text(m.group(1))}</b>"), t)
    t = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", lambda m: _ph(f"<i>{esc_text(m.group(1))}</i>"), t)
    t = re.sub(r"~([^~\n]+)~", lambda m: _ph(f"<s>{esc_text(m.group(1))}</s>"), t)
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
    """InlineKeyboardMarkup (or its ``to_dict()`` form) → rich-button rows covering ALL
    RichMessageButton actions (Bot API 10.3): callback_data, url, web_app, login_url,
    switch_inline_query[_current_chat|_chosen_chat], copy_text, disabled. Non-representable
    buttons (callback_game, pay, no-action, malformed) make the WHOLE markup unsupported —
    callers gate on ``rich_control_markup_supported`` and fall back to the legacy prompt, so
    no user action is ever silently dropped. Rows that end up empty are dropped. Input rows
    longer than 8 buttons are SPLIT at the 8-button boundary (no silent truncation)."""
    return _normalize_markup(reply_markup)[1]


def _markup_rows_or_none(reply_markup: Any) -> Optional[Tuple[List[List[Dict[str, Any]]], List[Optional[str]]]]:
    """``to_dict()`` normalization shared by the converter and the gate. Returns
    ``(rows, alignments)`` or ``None`` when the markup itself is malformed (unknown object
    shape) → unsupported, legacy fallback. Alignment source: markup-level ``align`` (one
    value for every row) or ``row_alignments`` (per row, in ORIGINAL row order; split rows
    inherit their source row's alignment; missing/invalid entries → None = no attribute)."""
    if reply_markup is None:
        return [], []
    if hasattr(reply_markup, "to_dict"):
        try:
            reply_markup = reply_markup.to_dict()
        except Exception:
            return None
    if not isinstance(reply_markup, dict):
        return None
    raw_rows = reply_markup.get("inline_keyboard")
    if not isinstance(raw_rows, (list, tuple)):
        return None
    align = reply_markup.get("align") if reply_markup.get("align") in _ROW_ALIGNMENTS else None
    row_alignments_raw = reply_markup.get("row_alignments")
    per_row: List[Any] = list(row_alignments_raw) if isinstance(row_alignments_raw, (list, tuple)) else []
    rows: List[List[Dict[str, Any]]] = []
    alignments: List[Optional[str]] = []
    for row_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, (list, tuple)):
            return None
        normalized = [_normalize_button(b) for b in raw_row]
        if any(b is None for b in normalized):
            return None  # a button in this row is not representable → whole markup unsupported
        # Per-row entry wins; a missing entry falls back to the markup-level align; an
        # INVALID per-row entry drops the attribute (never guesses).
        row_align = per_row[row_index] if row_index < len(per_row) else align
        if row_align not in _ROW_ALIGNMENTS:
            row_align = align if (row_index >= len(per_row)) else None
        # Split at the 8-button cap: each split row keeps the source row's alignment.
        for start in range(0, len(normalized), _BUTTONS_PER_ROW_MAX):
            rows.append(normalized[start:start + _BUTTONS_PER_ROW_MAX])
            alignments.append(row_align)
    return rows, alignments


def _normalize_button(button: Any) -> Optional[Dict[str, Any]]:
    """One ``InlineKeyboardButton``-shaped dict → normalized rich-button dict, or ``None``
    when the button carries no action representable as a RichMessageButton (then the caller
    must fall back to the legacy keyboard — never drop it silently).

    Output shape: ``{"text", "action": (kind, value), "style": str|None, "disabled": bool}``
    where *value* is the raw API field value (string or sub-dict). ``disabled`` is a MARKER:
    PTB 22.8 has no ``disabled`` param, so callers may set it via button-level ``api_kwargs``
    (``InlineKeyboardButton(..., api_kwargs={"disabled": {}})`` — ``to_dict()`` flattens it)
    NEXT TO a real carrier action (callback_data/url/…); the rich renderer then emits
    ``type="disabled"`` and the legacy PTB send still carries a valid carrier.
    """
    if not isinstance(button, dict):
        return None
    # Bot API 10.3 / PTB 22.8: a button may carry only ONE action field; unknown action
    # kinds (callback_game, pay, …) are not representable as RichMessageButton.
    kind = next((k for k in _RICH_BUTTON_ACTIONS if k in button), None)
    if kind is None:
        return None
    value = button.get(kind)
    if kind == "callback_data":
        if not isinstance(value, str) or not (
                _CALLBACK_DATA_MIN_BYTES <= len(value.encode("utf-8")) <= _CALLBACK_DATA_MAX_BYTES):
            return None
    elif kind in ("url", "switch_inline_query", "switch_inline_query_current_chat"):
        if not isinstance(value, str):
            return None
    elif kind in ("web_app", "login_url", "switch_inline_query_chosen_chat", "copy_text"):
        if not isinstance(value, dict):
            return None
    elif kind == "disabled":
        if value not in ({}, None):
            return None
        value = {}
    disabled = isinstance(button.get("disabled"), dict) and kind != "disabled"
    return {
        "text": str(button.get("text") or ""),
        "action": (kind, value),
        "style": str(button["style"]) if isinstance(button.get("style"), str) and button["style"] else None,
        "disabled": disabled,
    }


def _normalize_markup(reply_markup: Any) -> Tuple[bool, List[List[Dict[str, Any]]], List[Optional[str]]]:
    """Full markup normalization: ``(supported, rows, alignments)``. ``supported`` False
    (malformed markup or a non-representable button) → callers must reject to the legacy
    path — never drop a user action."""
    normalized = _markup_rows_or_none(reply_markup)
    if normalized is None:
        return False, [], []
    return True, normalized[0], normalized[1]


def rich_control_markup_supported(reply_markup: Any) -> bool:
    """True only when every supplied button is representable as a RichMessageButton (any of
    the supported action kinds). Unknown/malformed buttons or a broken markup → False: the
    caller falls back to the legacy prompt so no user action is silently dropped."""
    if reply_markup is None:
        return True
    return _normalize_markup(reply_markup)[0]


def buttons_html(rows: List[List[Dict[str, Any]]], alignments: Optional[List[Optional[str]]] = None) -> str:
    """Rich-button rows → ``<tg-button-row>`` HTML using the official Bot API 10.3 grammar:
    ``<tg-button-row align="left|center|right">`` with ``<tg-button type="…" …>`` children
    (types: url/callback_data/web_app/login_url/switch_inline_query/
    switch_inline_query_current_chat/switch_inline_query_chosen_chat/copy_text/disabled;
    attrs: url, data, query, text, forward-text, bare request-write-access/allow-*-chats).
    Labels are plain ``InlineKeyboardButton.text`` — HTML-escaped losslessly. Explicit styles
    pass through; absent styles are INFERRED from the label; style=link applies to callback
    buttons only. *alignments* (optional, one entry per row; None = no align attribute)."""
    parts: List[str] = []
    for row_index, row in enumerate(rows):
        buttons = []
        for button in row:
            buttons.append(_button_html(button))
        if not buttons:
            continue
        align = alignments[row_index] if alignments and row_index < len(alignments) else None
        align_attr = f' align="{esc_attr(align)}"' if align in _ROW_ALIGNMENTS else ""
        parts.append(f"<tg-button-row{align_attr}>" + "".join(buttons) + "</tg-button-row>")
    return "\n".join(parts)


def _attrs_to_html(attrs: Tuple[str, ...]) -> str:
    """Attribute names → bare presence flags in HTML (snake_case → dashed)."""
    return "".join(f' {_SNAKE_TO_DASH.sub("-", name)}' for name in attrs)

def _button_html(button: Dict[str, Any]) -> str:
    """Normalized button dict → one ``<tg-button …>label</tg-button>`` per the official
    Bot API 10.3 rich-HTML grammar. Disabled-with-carrier buttons drop the carrier action
    and render as ``type="disabled"`` (inactive)."""
    kind, value = button["action"]
    label = esc_text(button["text"])
    disabled = bool(button.get("disabled"))
    if disabled:
        kind, value = "disabled", {}
    attrs = ""
    if kind == "callback_data":
        attrs = f' data="{esc_attr(value)}"'
    elif kind in ("url", "web_app"):
        attrs = f' url="{esc_attr(value.get("url") if isinstance(value, dict) else value)}"'
    elif kind == "login_url":
        attrs = f' url="{esc_attr(value.get("url") or "")}"'
        if value.get("forward_text"):
            attrs += f' forward-text="{esc_attr(value.get("forward_text"))}"'
        if value.get("request_write_access"):
            attrs += ' request-write-access'
    elif kind in ("switch_inline_query", "switch_inline_query_current_chat"):
        # May be empty: the API then inserts just the bot's username.
        attrs = f' query="{esc_attr(value)}"'
    elif kind == "switch_inline_query_chosen_chat":
        attrs = f' query="{esc_attr(value.get("query") or "")}"'
        for flag in ("allow_user_chats", "allow_bot_chats", "allow_group_chats", "allow_channel_chats"):
            if value.get(flag):
                attrs += _attrs_to_html((flag,))
    elif kind == "copy_text":
        attrs = f' text="{esc_attr(value.get("text") or "")}"'
    # style: explicit wins; inferred only when absent; link only for callback buttons.
    style = button.get("style")
    if style not in _ALLOWED_STYLES and not disabled:
        style = _infer_style(button["text"])
    if style == "link" and (disabled or kind != "callback_data"):
        style = None
    if style is not None and style in _ALLOWED_STYLES:
        attrs += f' style="{esc_attr(style)}"'
    return f'<tg-button type="{esc_attr(kind)}"{attrs}>{label}</tg-button>'


def rich_control_html(text: str, parse_mode: Any, reply_markup: Any = None) -> str:
    """Body + buttons as one rich-HTML string. ``parse_mode`` HTML passes through (already escaped
    by its builder); MarkdownV2 converts; plain text escapes. Button rows embed row alignment
    when the markup carries ``align``/``row_alignments`` api_kwargs."""
    kind = _parse_mode_kind(parse_mode)
    if kind == "html":
        body = str(text or "")
    elif kind == "markdown":
        body = mdv2_to_html(text or "")
    else:
        body = esc_text(text or "")
    _, rows, alignments = _normalize_markup(reply_markup)
    if not rows:
        return body
    buttons = buttons_html(rows, alignments)
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

    def _rich_control_markup_supported(self, reply_markup: Any) -> bool:
        return rich_control_markup_supported(reply_markup)
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
        if not self._rich_control_markup_supported(kwargs.get("reply_markup")):
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
        link_preview_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[SendResult]:
        """Rich in-place edit of a registered control message (``editMessageText`` +
        ``rich_message``). Same None/transient contract as the send. A successful edit with no
        ``reply_markup`` is terminal → the registry entry is dropped (pagination edits keep it).
        *link_preview_kwargs* carries the caller's ``link_preview_options`` /
        ``disable_web_page_preview`` edit kwargs (PTB ``edit_message_text`` accepts them)."""
        if not (self._rich_controls_enabled() and self._bot):
            return None
        payload: Dict[str, Any] = {
            "chat_id": normalize_telegram_chat_id(chat_id), "message_id": int(message_id),
            "rich_message": rich_control_payload(text, parse_mode, reply_markup)}
        payload.update(self._rich_control_link_preview(link_preview_kwargs or {}))
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

    async def edit_message_text(self, *args: Any, text: Any = None, parse_mode: Any = None,
                                reply_markup: Any = None, **kwargs: Any):
        """PTB-compatible ``CallbackQuery.edit_message_text``: positional (text, parse_mode,
        reply_markup, …) and keyword forms both work; link-preview kwargs are honored by the
        rich edit, everything else rides the legacy fallback call verbatim."""
        extra_positional: Tuple[Any, ...] = ()
        if args:
            if text is None:
                text = args[0]
            if len(args) > 1 and parse_mode is None:
                parse_mode = args[1]
            if len(args) > 2 and reply_markup is None:
                reply_markup = args[2]
            extra_positional = args[3:]
        result = await self._adapter._edit_control_rich(
            self._chat_id, self._message_id, text or "", parse_mode, reply_markup,
            link_preview_kwargs=kwargs)
        if result is not None and result.success:
            return SimpleNamespace(message_id=self._message_id, chat_id=self._chat_id)
        raw = getattr(result, "raw_response", None) or {}
        if raw.get("ambiguous"):
            # Transient/ambiguous: re-raise the ORIGINAL exception — never race a legacy edit.
            raise raw.get("exc") or RuntimeError(result.error or "rich control edit failed ambiguously")
        # Permanent rejection only → the caller's legacy edit (same id, idempotent).
        return await self._query.edit_message_text(
            *extra_positional, text=text, parse_mode=parse_mode, reply_markup=reply_markup, **kwargs)
