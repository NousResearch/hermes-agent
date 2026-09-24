"""Google Chat outbound text formatting and Cards v2 rendering.

Extracted from ``adapter.py``; the card dict shapes and key order here are the
wire format and must stay byte-identical.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict

# Invisible Unicode codepoints that render as tofu (□) in Google Chat's
# restricted font stack: ZWS/ZWNJ/ZWJ, bidi marks, word joiner, BOM and
# Variation Selectors (Chat ignores them and often shows a blank box).
# Pattern lifted from PR #14965.
_INVISIBLE_RE = re.compile(
    "["
    "\u200b"          # Zero-Width Space
    "\u200c"          # Zero-Width Non-Joiner
    "\u200d"          # Zero-Width Joiner (ZWJ)
    "\u200e\u200f"    # LTR / RTL marks
    "\u2060"          # Word Joiner
    "\ufeff"          # BOM / Zero-Width No-Break Space
    "\ufe00-\ufe0f"   # Variation Selectors 1-16 (VS1–VS16)
    "\U000e0100-\U000e01ef"  # Variation Selectors 17-256
    "]"
)


def format_message(content: str) -> str:
    """Convert standard Markdown to Google Chat's dialect.

    Chat renders only ``*bold*``, ``_italic_``, ``~strike~`` and code; ``**bold**``,
    ``# headers`` and ``[text](url)`` must be converted. Fenced and inline code
    are protected via placeholders so literal asterisks/brackets inside them
    survive; invisible tofu codepoints are stripped at the end.
    """
    if not content:
        return content
    placeholders: Dict[str, str] = {}

    def _ph(value: str) -> str:
        key = f"\x00GC{len(placeholders)}\x00"
        placeholders[key] = value
        return key

    text = content
    # Protect fenced blocks first, then inline code.
    text = re.sub(r"(```(?:[^\n]*\n)?[\s\S]*?```)", lambda m: _ph(m.group(0)), text)
    text = re.sub(r"(`[^`]+`)", lambda m: _ph(m.group(0)), text)
    # Headers (## Title) → *Title* (Chat has no header support).
    text = re.sub(r"^#{1,6}\s+(.+)$", lambda m: _ph(f"*{m.group(1).strip()}*"), text, flags=re.MULTILINE)
    # ***text*** → *_text_*, then **text** → *text*.
    text = re.sub(r"\*\*\*(.+?)\*\*\*", lambda m: _ph(f"*_{m.group(1)}_*"), text)
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: _ph(f"*{m.group(1)}*"), text)
    # [text](url) → <url|text> (Slack-style angle-bracket).
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: _ph(f"<{m.group(2)}|{m.group(1)}>"), text)
    text = _INVISIBLE_RE.sub("", text)
    # Collapse double spaces left over from stripped chars.
    text = re.sub(r"  +", " ", text)
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def _required_str(mapping: Dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    value = str(value).strip() if value is not None else ""
    if not value:
        raise ValueError(f"{context}.{key} is required")
    return value


def _copy_opt(dst: Dict[str, Any], src: Dict[str, Any], *keys: tuple[str, str]) -> Dict[str, Any]:
    """Copy truthy ``src[src_key]`` into ``dst[dst_key]`` as str, in the given order."""
    for src_key, dst_key in keys:
        if src.get(src_key):
            dst[dst_key] = str(src[src_key])
    return dst


def _required_list(mapping: Dict[str, Any], key: str, error: str) -> list:
    items = mapping.get(key) or []
    if not isinstance(items, list) or not items:
        raise ValueError(error)
    return items


def _button_to_chat(button: Dict[str, Any]) -> Dict[str, Any]:
    text = _required_str(button, "text", "button")
    action = _required_str(button, "action", "button")
    raw_params = button.get("parameters") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("button.parameters must be an object")
    parameters = [{"key": str(key), "value": str(value)} for key, value in sorted(raw_params.items())]
    return {"text": text, "onClick": {"action": {"function": action, "parameters": parameters}}}


def _text_widget(widget: Dict[str, Any]) -> Dict[str, Any]:
    return {"textParagraph": {"text": format_message(_required_str(widget, "text", "widget"))}}


def _decorated_text_widget(widget: Dict[str, Any]) -> Dict[str, Any]:
    decorated: Dict[str, Any] = {
        "text": format_message(_required_str(widget, "text", "widget")),
        "wrapText": bool(widget.get("wrap_text", True))}
    return {"decoratedText": _copy_opt(decorated, widget, ("top_label", "topLabel"), ("bottom_label", "bottomLabel"))}


def _image_widget(widget: Dict[str, Any]) -> Dict[str, Any]:
    image = {"imageUrl": _required_str(widget, "image_url", "widget")}
    return {"image": _copy_opt(image, widget, ("alt_text", "altText"))}


def _buttons_widget(widget: Dict[str, Any]) -> Dict[str, Any]:
    raw_buttons = _required_list(widget, "buttons", "button widgets require at least one button")
    return {"buttonList": {"buttons": [_button_to_chat(btn) for btn in raw_buttons]}}


def _selection_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("selection items must be objects")
    return {
        "text": _required_str(item, "text", "selection item"),
        "value": _required_str(item, "value", "selection item"),
        "selected": bool(item.get("selected", False))}


def _selection_widget(widget: Dict[str, Any]) -> Dict[str, Any]:
    name = _required_str(widget, "name", "widget")
    raw_items = _required_list(widget, "items", "selection widgets require at least one item")
    return {
        "selectionInput": {
            "name": name,
            "label": str(widget.get("label") or name),
            "type": str(widget.get("selection_type") or "CHECK_BOX"),
            "items": [_selection_item(item) for item in raw_items]}}


_WIDGET_RENDERERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "text": _text_widget,
    "text_paragraph": _text_widget,
    "decorated_text": _decorated_text_widget,
    "buttons": _buttons_widget,
    "button_list": _buttons_widget,
    "selection": _selection_widget,
    "selection_input": _selection_widget,
    "image": _image_widget,
    "divider": lambda widget: {"divider": {}},
}


def _widget_to_chat(widget: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(widget, dict):
        raise ValueError("card widgets must be objects")
    widget_type = str(widget.get("type") or "").strip()
    renderer = _WIDGET_RENDERERS.get(widget_type)
    if renderer is None:
        raise ValueError(f"unsupported widget type: {widget_type or '<missing>'}")
    return renderer(widget)


def _section_to_chat(section: Any) -> Dict[str, Any]:
    if not isinstance(section, dict):
        raise ValueError("card sections must be objects")
    widgets = _required_list(section, "widgets", "card section widgets must contain at least one widget")
    rendered: Dict[str, Any] = {"widgets": [_widget_to_chat(w) for w in widgets]}
    return _copy_opt(rendered, section, ("header", "header"))


def _header_to_chat(header: Any) -> Dict[str, Any]:
    if not isinstance(header, dict):
        raise ValueError("card.header must be an object")
    rendered: Dict[str, Any] = {"title": _required_str(header, "title", "card.header")}
    _copy_opt(rendered, header, ("subtitle", "subtitle"))
    if header.get("image_url"):
        rendered["imageUrl"] = str(header["image_url"])
        rendered["imageType"] = str(header.get("image_type") or "SQUARE")
    return _copy_opt(rendered, header, ("image_alt_text", "imageAltText"))


def card_spec_to_cards_v2(card_spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(card_spec, dict):
        raise ValueError("card must be an object")
    raw_sections = _required_list(card_spec, "sections", "card.sections must contain at least one section")
    card: Dict[str, Any] = {"sections": [_section_to_chat(s) for s in raw_sections]}
    header = card_spec.get("header")
    if header:
        card["header"] = _header_to_chat(header)
    return {"cardId": str(card_spec.get("card_id") or "hermes-card"), "card": card}
