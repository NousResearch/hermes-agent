"""Pure parsers for inbound DingTalk ``ChatbotMessage`` payloads (no I/O, no adapter state)."""

import json
from typing import Any, List, Optional, Tuple

from gateway.platforms.base import MessageType

# DingTalk rich-text item type → runtime content type
DINGTALK_TYPE_MAPPING = {"picture": "image", "voice": "audio"}

# File extension → MIME type for DingTalk file/image messages. image/* MIMEs
# make ``extract_media`` classify msgtype='image'/'file' payloads as PHOTO.
EXT_MAP = {
    "pdf": "application/pdf", "doc": "application/msword", "xls": "application/vnd.ms-excel",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp",
    "md": "text/markdown", "txt": "text/plain", "csv": "text/csv", "zip": "application/zip", "mp4": "video/mp4",
}

# rich-text runtime type → (media_types entry, MessageType promotion when still TEXT)
_RICH_MEDIA = {"image": ("image", MessageType.PHOTO), "video": ("video", MessageType.VIDEO), "file": ("application/octet-stream", MessageType.DOCUMENT)}


def _extensions(message: Any) -> Any:
    return getattr(message, "extensions", {}) or {}


def _ext_content(message: Any) -> Optional[dict]:
    """``extensions['content']`` when it is a dict, else None."""
    content = _extensions(message).get("content", {})
    return content if isinstance(content, dict) else None


def _rich_list(message: Any) -> Optional[list]:
    """Rich-text item list from either SDK shape (``rich_text_content.rich_text_list`` or legacy ``rich_text``)."""
    rich_text = getattr(message, "rich_text_content", None) or getattr(message, "rich_text", None)
    if not rich_text:
        return None
    rich_list = getattr(rich_text, "rich_text_list", None) or rich_text
    return rich_list if isinstance(rich_list, list) else None


def _url_of(raw: Any) -> str:
    return (raw.get("url", "") or raw.get("docUrl", "")) if isinstance(raw, dict) else ""


def _card_text(message: Any) -> str:
    """msgtype='card' (钉钉文档分享卡片 / link card): title + doc URL from ``extensions['card']``."""
    extensions = _extensions(message)
    card = extensions.get("card", {})
    content = ""
    if isinstance(card, dict):
        title, raw_content = card.get("title", ""), card.get("content", "")
        doc_url = _url_of(raw_content)
        if isinstance(raw_content, str) and raw_content.strip():
            try:
                doc_url = _url_of(json.loads(raw_content.strip()))
            except (ValueError, TypeError):
                doc_url = raw_content
        content = " ".join(([f"[文档] {title}"] if title else []) + ([doc_url] if doc_url else []))
    ext_text = extensions.get("text", {})  # last-resort: raw text field from extensions
    return content or ((ext_text.get("content", "") or "").strip() if isinstance(ext_text, dict) else "")


def _interactive_card_text(message: Any) -> str:
    """msgtype='interactiveCard': ``extensions['content']`` carries title + biz_custom_action_url."""
    ext_content = _ext_content(message) or {}
    doc_url, title = ext_content.get("biz_custom_action_url", ""), ext_content.get("title", "")
    return " ".join([f"[文档卡片] {title}" if title else "[文档卡片]"] + ([doc_url] if doc_url else [])) if (doc_url or title) else ""


def _ext_field(message: Any, field: str) -> Any:
    ext_content = _ext_content(message)
    return ext_content.get(field, "") if ext_content else ""


# Fallbacks by msgtype when no plain/rich text was found (types are exclusive): audio -> DingTalk speech
# recognition text; file -> fileName; card / interactiveCard -> title + doc URL.
_EMPTY_TEXT_FALLBACKS = {
    "audio": lambda m: (_ext_field(m, "recognition") or "").strip(),
    "file": lambda m: f"[文件] {_ext_field(m, 'fileName')}" if _ext_field(m, "fileName") else "",
    "card": _card_text,
    "interactiveCard": _interactive_card_text,
}


def _rich_item_text(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or ""
    return getattr(item, "text", "") or ""


def extract_text(message: Any) -> str:
    """Extract plain text from a DingTalk chatbot message. Handles both SDK shapes: legacy ``message.text`` dict ``{"content": ...}`` and >= 0.20 ``TextContent``
    (whose ``__str__`` is ``"TextContent(content=...)"`` — always read ``.content`` first)."""
    text = getattr(message, "text", None) or ""
    content = (text.content or "" if hasattr(text, "content") else text.get("content", "") if isinstance(text, dict) else str(text)).strip()
    rich_list = _rich_list(message) if not content else None
    if rich_list is not None:
        content = " ".join(t for t in map(_rich_item_text, rich_list) if t).strip()
    fallback = _EMPTY_TEXT_FALLBACKS.get(getattr(message, "message_type", "")) if not content else None
    # Do NOT strip "@bot": the mention is routed structurally (callback ``isInAtList``), and
    # regex-stripping @handles would damage e-mails, SSH URLs and literal "@openai" references.
    return fallback(message) if fallback else content


def extract_media(message: Any) -> Tuple[MessageType, List[str], List[str]]:
    """Return ``(MessageType, [download codes/urls], [mime types])`` for a message."""
    msg_type = MessageType.TEXT
    media_urls: List[str] = []
    media_types: List[str] = []
    image_content = getattr(message, "image_content", None)
    if download_code := (getattr(image_content, "download_code", None) if image_content else None):
        media_urls.append(download_code)
        media_types.append("image")
        msg_type = MessageType.PHOTO
    for item in _rich_list(message) or ():
        if not isinstance(item, dict):
            continue
        dl_code = item.get("downloadCode") or item.get("download_code") or ""
        if not dl_code:
            continue
        item_type = item.get("type", "")
        mapped = DINGTALK_TYPE_MAPPING.get(item_type, "file")
        # "voice" items are native voice notes → STT (VOICE); "audio" file uploads stay AUDIO.
        mime, promoted = ("audio", MessageType.VOICE if item_type == "voice" else MessageType.AUDIO) if mapped == "audio" else _RICH_MEDIA[mapped]
        media_urls.append(dl_code)
        media_types.append(mime)
        if msg_type == MessageType.TEXT:
            msg_type = promoted
    msg_type_str = getattr(message, "message_type", "") or ""
    if msg_type_str == "picture" and not media_urls:
        msg_type = MessageType.PHOTO
    elif msg_type_str == "richText":
        # Only re-derive when the scan above left TEXT — resetting a VOICE/AUDIO/VIDEO/DOCUMENT
        # promotion here dropped native voice notes back to TEXT and skipped STT.
        if msg_type == MessageType.TEXT and any("image" in t for t in media_types):
            msg_type = MessageType.PHOTO
    elif msg_type_str == "audio":
        # Voice message: recognition text is already in the text. Do NOT add media_urls, or
        # run.py's transcription enrichment overwrites it with a failed STT attempt.
        if msg_type == MessageType.TEXT:
            msg_type = MessageType.VOICE
    elif msg_type_str in ("file", "image"):
        ext_content = _ext_content(message) or {}
        if dl_code := ext_content.get("downloadCode") or "":
            fname = ext_content.get("fileName", "")
            mime = EXT_MAP.get(fname.rsplit(".", 1)[-1].lower() if fname and "." in fname else "", "application/octet-stream")
            media_urls.append(dl_code)
            media_types.append(mime)
            if msg_type == MessageType.TEXT:  # image messages, and files with image MIME (a .png attachment) → PHOTO
                msg_type = MessageType.PHOTO if (msg_type_str == "image" or mime.startswith("image/")) else MessageType.DOCUMENT
    return msg_type, media_urls, media_types


def collect_download_codes(message: Any) -> List[Tuple[Any, str]]:
    """Return ``(container, key)`` pairs whose download code should be resolved to a URL."""
    codes: List[Tuple[Any, str]] = []
    img_content = getattr(message, "image_content", None)
    if img_content and getattr(img_content, "download_code", None):
        codes.append((img_content, "download_code"))
    rich_text = getattr(message, "rich_text_content", None)
    for item in (getattr(rich_text, "rich_text_list", []) or []) if rich_text else []:
        if isinstance(item, dict):
            codes.extend((item, key) for key in ("downloadCode", "pictureDownloadCode", "download_code") if item.get(key))
    if (getattr(message, "message_type", "") or "") in ("file", "image"):
        ext_content = _ext_content(message)
        if ext_content and ext_content.get("downloadCode"):
            codes.append((ext_content, "downloadCode"))
    return codes
