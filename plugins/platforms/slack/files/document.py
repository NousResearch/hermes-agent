"""Text a Slack document puts in the message: a small text-like file is inlined whole."""

import re

from gateway.platforms.base import _TEXT_INJECT_EXTENSIONS

#: Ceiling on the file text inlined into a message, shared by documents and email bodies.
TEXT_INJECT_MAX = 100 * 1024
#: Largest file cached at all (Slack's bot upload cap), shared by documents and emails.
DOCUMENT_MAX = 20 * 1024 * 1024


def document_injection(raw_bytes: bytes, display_name: str, ext: str, mimetype: str) -> str:
    """``[Content of <name>]`` for a text-like document up to :data:`TEXT_INJECT_MAX` bytes, else
    ``""``. Gated on extension/MIME, not a blind UTF-8 decode: PDF and zip headers decode too."""
    is_text = ext in _TEXT_INJECT_EXTENSIONS or (mimetype or "").startswith("text/")
    if not is_text or len(raw_bytes) > TEXT_INJECT_MAX:
        return ""
    try:
        text_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    safe_name = re.sub(r"[^\w.\- ]", "_", display_name)
    return f"[Content of {safe_name}]:\n{text_content}"
