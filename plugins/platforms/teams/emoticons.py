"""Convert Teams emoticon image attachments to text before they become media.

Teams sends stickers/emoticons as tiny CDN PNGs (often 20x20). Vision models
then 400 them as undersized (xAI ``invalid_image``, <512 pixels). Identify those
payloads and replace them with unicode or ``(name)`` so real screenshots still
go through as images.
"""

from __future__ import annotations

import re
from typing import Any

_HEX_TOKEN = re.compile(r"^[0-9a-fA-F]{2,8}$")
# ``1f389_partypopper`` / ``1f1fa_1f1f8_flagus`` — hex codepoints then a short name.
_HEX_NAME_RE = re.compile(r"^(?:[0-9a-fA-F]{2,8}_)+[A-Za-z][A-Za-z0-9-]*$")
_SHORT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{2,}$")
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(
    r"""([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)
_EMOTICON_ID_IN_URL = re.compile(r"/emoticons/([^/]+)/", re.IGNORECASE)


def is_emoticon_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "/emoticons/" in lowered or "personal-expressions" in lowered


def is_emoticon_id(name: str) -> bool:
    """True for Teams itemids like ``1f389_partypopper`` or ``partypopper``."""
    if not name:
        return False
    stem, ext = _stem_ext(name)
    if ext in _IMAGE_EXTS:
        return bool(_HEX_NAME_RE.fullmatch(stem))
    return bool(_HEX_NAME_RE.fullmatch(stem) or _SHORT_NAME_RE.fullmatch(stem))


def is_emoticon_attachment(att: Any) -> bool:
    """True when this Bot Framework attachment is a Teams emoticon image, not a photo."""
    url = _str_attr(att, "content_url", "contentUrl")
    if is_emoticon_url(url):
        return True
    ctype = _str_attr(att, "content_type", "contentType").lower()
    if ctype in ("text/html", "text/plain") or ctype.startswith("application/vnd.microsoft.card"):
        return False
    name = _str_attr(att, "name", "itemid", "itemId")
    return bool(is_emoticon_id(name) and (not ctype or ctype.startswith("image/")))


def unicode_from_emoticon_id(itemid: str) -> str:
    """``1f389_partypopper`` / ``1f389`` → 🎉; unknown ids → empty string."""
    return _unicode_from_itemid(itemid)


def convert_inbound_emoticons(activity: Any) -> str:
    """Deduped unicode / ``(name)`` tokens for every emoticon on this activity."""
    tokens, _skip = extract_emoticons_from_activity(activity)
    return " ".join(tokens)


def extract_emoticons_from_activity(activity: Any) -> tuple[list[str], set[str]]:
    """Deduped text tokens and content URLs of emoticon sprites to skip downloading."""
    seen: set[str] = set()
    tokens: list[str] = []
    skip_urls: set[str] = set()

    def add(text: str, key: str) -> None:
        if not text:
            return
        k = (key or text).lower()
        if k in seen:
            return
        seen.add(k)
        tokens.append(text)

    for att in _as_list(getattr(activity, "attachments", None)):
        ctype = _str_attr(att, "content_type", "contentType").lower()
        if ctype in ("text/html", "text/plain"):
            for key, text, src in _emoticons_from_html(_html_content(att)):
                add(text, key)
                if src:
                    skip_urls.add(src)
            continue
        if not is_emoticon_attachment(att):
            continue
        url = _str_attr(att, "content_url", "contentUrl")
        if url:
            skip_urls.add(url)
        name = _str_attr(att, "name", "itemid", "itemId")
        key = _id_from_url(url) or name
        add(emoticon_to_text(itemid=name or key, alt=_str_attr(att, "alt"), url=url), key)

    for ent in _as_list(getattr(activity, "entities", None)):
        etype = _str_attr(ent, "type").lower()
        if "emoji" not in etype:
            continue
        itemid = _str_attr(ent, "itemid", "itemId", "name")
        alt = _str_attr(ent, "text", "unicode", "alt", "name")
        add(emoticon_to_text(itemid=itemid, alt=alt), itemid or alt)

    return tokens, skip_urls


def emoticon_to_text(*, itemid: str = "", alt: str = "", url: str = "") -> str:
    if _looks_like_emoji_alt(alt):
        return alt.strip()
    ident = itemid or _id_from_url(url)
    unicode_text = _unicode_from_itemid(ident)
    if unicode_text:
        return unicode_text
    short = _short_name(ident)
    return f"({short})" if short else ""


def _emoticons_from_html(html: str) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    if not html:
        return hits
    for tag in _IMG_TAG_RE.findall(html):
        attrs = {
            m.group(1).lower(): (m.group(2) or m.group(3) or m.group(4) or "")
            for m in _ATTR_RE.finditer(tag)
        }
        src = attrs.get("src", "")
        itemid = attrs.get("itemid", "")
        itemtype = attrs.get("itemtype", "")
        alt = attrs.get("alt", "")
        if not (is_emoticon_url(src) or "emoji" in itemtype.lower() or is_emoticon_id(itemid)):
            continue
        key = itemid or _id_from_url(src)
        text = emoticon_to_text(itemid=key, alt=alt, url=src)
        if text:
            hits.append((key or text, text, src))
    return hits


def _unicode_from_itemid(itemid: str) -> str:
    stem, _ext = _stem_ext(itemid)
    chars: list[str] = []
    for part in stem.replace("-", "_").split("_"):
        if not _HEX_TOKEN.fullmatch(part):
            break
        cp = int(part, 16)
        if cp > 0x10FFFF:
            break
        chars.append(chr(cp))
    return "".join(chars)


def _short_name(itemid: str) -> str:
    stem, _ext = _stem_ext(itemid)
    if not stem:
        return ""
    parts = stem.replace("-", "_").split("_")
    while parts and _HEX_TOKEN.fullmatch(parts[0]):
        parts.pop(0)
    return "_".join(parts) if parts else stem


def _id_from_url(url: str) -> str:
    match = _EMOTICON_ID_IN_URL.search(url or "")
    return match.group(1) if match else ""


def _looks_like_emoji_alt(alt: str) -> bool:
    text = (alt or "").strip()
    if not text or len(text) > 16:
        return False
    return not any(("A" <= c <= "Z") or ("a" <= c <= "z") or ("0" <= c <= "9") for c in text)


def _stem_ext(name: str) -> tuple[str, str]:
    if not name:
        return "", ""
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        dotted = f".{ext.lower()}"
        if dotted in _IMAGE_EXTS:
            return stem, dotted
    return name, ""


def _html_content(att: Any) -> str:
    content = att.get("content") if isinstance(att, dict) else getattr(att, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        body = content.get("body") or content.get("text") or ""
        return body if isinstance(body, str) else ""
    return ""


def _str_attr(obj: Any, *names: str) -> str:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []
