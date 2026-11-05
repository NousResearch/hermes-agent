"""Allowlist sanitizer for Matrix ``formatted_body`` HTML (moved out of the adapter facade)."""

from __future__ import annotations

import re
from html import escape as _html_escape
from html.parser import HTMLParser


class MatrixHtmlSanitizer(HTMLParser):
    """Allowlist sanitizer for Matrix-compatible formatted HTML."""

    _ALLOWED_TAGS = {
        "a", "b", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "li", "ol",
        "p", "pre", "s", "strike", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul"}
    _VOID_TAGS = {"br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []
        self._skip_depth = 0

    @staticmethod
    def _safe_url(value: str) -> str:
        stripped = re.sub(r"[\x00-\x1f\x7f]+", "", value or "").strip()
        match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", stripped)
        scheme = match.group(1).lower() if match else ""
        if scheme and scheme not in {"http", "https", "matrix", "mailto"}:
            return ""
        return stripped

    def _safe_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        safe: list[str] = []
        for key, value in attrs:
            attr = str(key or "").lower()
            raw_value = "" if value is None else str(value)
            if tag == "a" and attr == "href":
                href = self._safe_url(raw_value)
                if href:
                    safe.append(f' href="{_html_escape(href, quote=True)}"')
            elif tag == "code" and attr == "class" and re.fullmatch(r"language-[A-Za-z0-9_+.-]{1,64}", raw_value):
                safe.append(f' class="{_html_escape(raw_value, quote=True)}"')
        return "".join(safe)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._ALLOWED_TAGS:
            self._parts.append(f"<{tag}>" if tag in self._VOID_TAGS else f"<{tag}{self._safe_attrs(tag, attrs)}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth or tag not in self._ALLOWED_TAGS or tag in self._VOID_TAGS:
            return
        self._parts.append(f"</{tag}>")

    def _emit(self, text: str) -> None:
        if not self._skip_depth:
            self._parts.append(text)

    def handle_data(self, data: str) -> None:
        self._emit(_html_escape(data))

    def handle_entityref(self, name: str) -> None:
        self._emit(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._emit(f"&#{name};")

    def get_html(self) -> str:
        return "".join(self._parts)
