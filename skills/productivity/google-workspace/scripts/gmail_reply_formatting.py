"""MIME extraction and visible quoted-history formatting for Gmail replies."""

from __future__ import annotations

import base64
import html
import re
from html.parser import HTMLParser


_BLOCK_TAGS = {
    "address", "blockquote", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ol", "p", "pre", "table", "tbody", "td", "th", "thead", "tr", "ul",
}
_SAFE_TAGS = _BLOCK_TAGS | {
    "a", "b", "br", "code", "del", "em", "hr", "i", "s", "small", "span",
    "strong", "sub", "sup", "u",
}
_DROP_CONTENT_TAGS = {"embed", "form", "head", "iframe", "object", "script", "style", "svg"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_SAFE_ATTRS = {
    "a": {"href", "title"},
    "blockquote": {"type"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan"},
}
_SAFE_STYLE_PROPERTIES = {
    "background-color", "border", "border-bottom", "border-left", "border-right",
    "border-top", "color", "font-family", "font-size", "font-style", "font-weight",
    "line-height", "margin", "margin-bottom", "margin-left", "margin-right", "margin-top",
    "padding", "padding-bottom", "padding-left", "padding-right", "padding-top",
    "text-align", "text-decoration", "white-space",
}
_UNSAFE_STYLE_RE = re.compile(r"url\s*\(|expression\s*\(|@import|javascript:", re.I)
_MESSAGE_ID_RE = re.compile(r"<[^<>\s]+>")
_PLAIN_QUOTE_MARKER_RE = re.compile(
    r"^\s*(?:>+|On .+ wrote:\s*$|-{2,}\s*(?:Original Message|Forwarded message)\s*-{2,}\s*$)",
    re.I,
)
_QUOTE_CLASSES = {"gmail_extra", "gmail_quote", "moz-cite-prefix", "protonmail_quote", "yahoo_quoted"}
_QUOTE_IDS = {"appendonsend", "divreplyfwdmsg", "isforwardcontent"}


def _is_prior_quote(tag: str, attrs: dict[str, str]) -> bool:
    classes = set(attrs.get("class", "").lower().split())
    return (
        bool(classes & _QUOTE_CLASSES)
        or attrs.get("id", "").lower() in _QUOTE_IDS
        or "data-hermes-quote" in attrs
        or (tag == "blockquote" and attrs.get("type", "").lower() == "cite")
    )


def _decode_body_data(data: str, charset: str = "utf-8") -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    try:
        return raw.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _part_headers(part: dict) -> dict[str, str]:
    return {
        header.get("name", "").lower(): header.get("value", "")
        for header in part.get("headers", [])
        if header.get("name")
    }


def _part_charset(part: dict) -> str:
    content_type = _part_headers(part).get("content-type", "")
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type, re.I)
    return match.group(1) if match else "utf-8"


def extract_reply_bodies(message: dict, fetch_attachment=None) -> tuple[str, str]:
    """Return the first non-attachment plain and HTML bodies in a Gmail payload."""
    found: dict[str, str] = {}
    message_id = message.get("id", "")

    def walk(part: dict) -> None:
        headers = _part_headers(part)
        disposition = headers.get("content-disposition", "").lower()
        if part.get("filename") or disposition.startswith("attachment"):
            return

        mime_type = part.get("mimeType", "").lower()
        if mime_type in {"text/plain", "text/html"} and mime_type not in found:
            body = part.get("body", {})
            data = body.get("data", "")
            attachment_id = body.get("attachmentId", "")
            if not data and attachment_id and fetch_attachment:
                attachment = fetch_attachment(message_id, attachment_id) or {}
                data = attachment.get("data", "")
            if data:
                found[mime_type] = _decode_body_data(data, _part_charset(part))

        for child in part.get("parts", []) or []:
            walk(child)

    walk(message.get("payload", {}))
    return found.get("text/plain", ""), found.get("text/html", "")


def _safe_style(value: str) -> str:
    if _UNSAFE_STYLE_RE.search(value):
        return ""
    declarations = []
    for declaration in value.split(";"):
        if ":" not in declaration:
            continue
        name, raw_value = declaration.split(":", 1)
        name = name.strip().lower()
        raw_value = raw_value.strip()
        if name in _SAFE_STYLE_PROPERTIES and raw_value:
            declarations.append(f"{name}: {raw_value}")
    return "; ".join(declarations)


def _safe_href(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("#"):
        return stripped
    if re.match(r"^(?:https?://|mailto:)", stripped, re.I):
        return stripped
    return ""


class _SanitizingHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.drop_depth = 0
        self.open_tags: list[tuple[str, bool]] = []
        self.stopped = False

    def _close_open_safe_tags(self) -> None:
        for tag, dropped in reversed(self.open_tags):
            if not dropped and tag in _SAFE_TAGS and tag not in _VOID_TAGS:
                self.output.append(f"</{tag}>")
        self.open_tags.clear()
        self.drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.stopped:
            return
        tag = tag.lower()
        attrs_dict = {name.lower(): value or "" for name, value in attrs}
        if _is_prior_quote(tag, attrs_dict):
            self._close_open_safe_tags()
            self.stopped = True
            return
        dropped = self.drop_depth > 0 or tag in _DROP_CONTENT_TAGS
        is_void = tag in _VOID_TAGS
        if not is_void:
            self.open_tags.append((tag, dropped))
        if dropped:
            if not is_void:
                self.drop_depth += 1
            return
        if tag not in _SAFE_TAGS:
            return

        safe_attrs = []
        for name, value in attrs:
            name = name.lower()
            value = value or ""
            if name.startswith("on"):
                continue
            if name == "style":
                value = _safe_style(value)
                if value:
                    safe_attrs.append((name, value))
                continue
            if name not in _SAFE_ATTRS.get(tag, set()):
                continue
            if name == "href":
                value = _safe_href(value)
                if not value:
                    continue
            safe_attrs.append((name, value))

        rendered_attrs = "".join(
            f' {name}="{html.escape(value, quote=True)}"' for name, value in safe_attrs
        )
        self.output.append(f"<{tag}{rendered_attrs}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.stopped:
            return
        tag = tag.lower()
        matching = next(
            (index for index in range(len(self.open_tags) - 1, -1, -1) if self.open_tags[index][0] == tag),
            None,
        )
        if matching is None:
            return
        opened_tag, dropped = self.open_tags.pop(matching)
        if dropped:
            self.drop_depth = max(0, self.drop_depth - 1)
            return
        if opened_tag in _SAFE_TAGS and opened_tag not in {"br", "hr"}:
            self.output.append(f"</{opened_tag}>")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth and not self.stopped:
            self.output.append(html.escape(data))

    def handle_entityref(self, name: str) -> None:
        if not self.drop_depth and not self.stopped:
            self.output.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.drop_depth and not self.stopped:
            self.output.append(f"&#{name};")


def sanitize_quoted_html(value: str) -> str:
    parser = _SanitizingHTMLParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.output).strip()


class _HTMLToTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.drop_depth = 0
        self.open_tags: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {name.lower(): value or "" for name, value in attrs}
        dropped = (
            self.drop_depth > 0
            or tag in _DROP_CONTENT_TAGS
            or _is_prior_quote(tag, attrs_dict)
        )
        is_void = tag in _VOID_TAGS
        if not is_void:
            self.open_tags.append((tag, dropped))
        if dropped:
            if not is_void:
                self.drop_depth += 1
            return
        if tag == "br":
            self.output.append("\n")
        elif tag == "li":
            self.output.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.output.append("\n")

    def handle_endtag(self, tag: str) -> None:
        matching = next(
            (index for index in range(len(self.open_tags) - 1, -1, -1) if self.open_tags[index][0] == tag),
            None,
        )
        if matching is None:
            return
        _, dropped = self.open_tags.pop(matching)
        if dropped:
            self.drop_depth = max(0, self.drop_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.output.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.drop_depth:
            self.output.append(data)


def html_to_text(value: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(value)
    parser.close()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in "".join(parser.output).splitlines()]
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    return "\n".join(compact).strip()


def strip_plain_quote_chain(value: str) -> str:
    lines = value.splitlines()
    kept = []
    for line in lines:
        if _PLAIN_QUOTE_MARKER_RE.match(line):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def build_references(headers: dict[str, str]) -> str:
    candidates: list[str] = []
    for name in ("references", "in-reply-to", "message-id"):
        value = headers.get(name, "")
        ids = _MESSAGE_ID_RE.findall(value)
        if not ids and value.strip():
            ids = value.split()
        for message_id in ids:
            if message_id not in candidates:
                candidates.append(message_id)
    return " ".join(candidates)


def compose_quoted_reply(
    reply_body: str,
    *,
    reply_is_html: bool,
    original_plain: str,
    original_html: str,
    original_from: str,
    original_date: str,
) -> tuple[str, bool]:
    """Compose a reply and its visible original-message quote."""
    attribution = f"On {original_date}, {original_from} wrote:" if original_date else f"{original_from} wrote:"
    use_html = reply_is_html or bool(original_html)

    if use_html:
        new_html = reply_body if reply_is_html else html.escape(reply_body).replace("\n", "<br>\n")
        quoted_html = sanitize_quoted_html(original_html)
        if not quoted_html and original_plain:
            quoted_html = html.escape(strip_plain_quote_chain(original_plain)).replace("\n", "<br>\n")
        if not quoted_html:
            return new_html, True
        quote = (
            '<div class="gmail_quote" data-hermes-quote="original">'
            f'<div class="gmail_attr">{html.escape(attribution)}</div>'
            '<blockquote style="margin:0 0 0 .8ex;border-left:1px solid #ccc;'
            f'padding-left:1ex">{quoted_html}</blockquote></div>'
        )
        return f"{new_html}<br><br>{quote}", True

    original_text = strip_plain_quote_chain(original_plain)
    if not original_text and original_html:
        original_text = html_to_text(original_html)
    if not original_text:
        return reply_body, False
    quoted = "\n".join(f"> {line}" if line else ">" for line in original_text.splitlines())
    return f"{reply_body.rstrip()}\n\n{attribution}\n{quoted}", False
