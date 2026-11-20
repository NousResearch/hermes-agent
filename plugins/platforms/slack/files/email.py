"""Slack email files: a mail sent to a channel's email address.

Slack posts it as Slackbot (``user: USLACKBOT`` with a ``bot_id``), as a file whose ``mode`` and
``filetype`` are both ``"email"``, although the file object reference lists neither value. The
object carries the mail itself: ``from``/``to``/``cc`` as lists of ``{address, name, original}``,
``subject``, ``plain_text`` and ``email_security.has_valid_dmarc``. Its download is the HTML body
without the sender, so the agent reads the object.
"""

import unicodedata
from typing import Any, Awaitable, Callable, Optional

from agent.compression_marker import elide
from gateway.platforms.base import cache_document_from_bytes_async

from .document import DOCUMENT_MAX, TEXT_INJECT_MAX

FilesInfo = Callable[[dict[str, Any]], Awaitable[Optional[dict[str, Any]]]]

_HEADERS = (("From", "from"), ("To", "to"), ("Cc", "cc"))
#: Per-header ceiling: the subject and every recipient list are set by the sender.
_HEADER_MAX = 1000
#: Control characters (C0, C1, DEL), format characters (bidi overrides, zero-width) and Unicode
#: line/paragraph separators.
_BREAKING_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
#: A display name cannot hold an address of its own: ``Boss <boss@corp.example>, x`` would read
#: as the sender ahead of the real address.
_MAILBOX_PUNCTUATION = '<>,;"'


def is_email_file(file_obj: dict[str, Any]) -> bool:
    return file_obj.get("mode") == "email" or file_obj.get("filetype") == "email"


def _line(value: Any) -> str:
    """Mail headers are written by whoever sent the mail: one line, no brackets, no control, format
    or separator character, so a header cannot forge a line or a ``[...]`` marker, nor reorder or
    hide part of itself."""
    text = "".join(" " if c in "[]" or unicodedata.category(c) in _BREAKING_CATEGORIES else c
                   for c in str(value or ""))
    return " ".join(text.split())


def _bare(value: Any) -> str:
    """A header part that must not look like a mailbox list of its own."""
    return " ".join("".join(" " if c in _MAILBOX_PUNCTUATION else c for c in _line(value)).split())


def _mailbox(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    name, address = _bare(entry.get("name")), _bare(entry.get("address"))
    if name and address:
        return f'"{name}" <{address}>'
    return address or (f'"{name}"' if name else _bare(entry.get("original")))


def _mailboxes(entries: Any) -> str:
    boxes = (_mailbox(e) for e in (entries if isinstance(entries, list) else []))
    return elide(", ".join(b for b in boxes if b), _HEADER_MAX)


def _subject(file_obj: dict[str, Any]) -> str:
    return elide(_line(file_obj.get("subject") or file_obj.get("title") or file_obj.get("name")), _HEADER_MAX)


def _plain_text(file_obj: dict[str, Any]) -> str:
    body = file_obj.get("plain_text")
    return body if isinstance(body, str) and body.strip() else ""


def _dmarc(file_obj: dict[str, Any]) -> str:
    """Slack's own check of the sender's domain: the From line alone is whatever the sender wrote."""
    security = file_obj.get("email_security")
    valid = security.get("has_valid_dmarc") if isinstance(security, dict) else None
    return "pass" if valid is True else "fail" if valid is False else "unknown"


def email_marker(file_obj: dict[str, Any]) -> str:
    """Thread-context marker for an email file: who sent it and its subject."""
    sender = _mailboxes(file_obj.get("from"))
    return f"[email{' from ' + sender if sender else ''}: {_subject(file_obj) or '(no subject)'}]"


def render_email(file_obj: dict[str, Any]) -> str:
    """The whole mail as text: headers (From, DMARC, To, Cc, Subject), then the plain-text body."""
    fields = [(label, _mailboxes(file_obj.get(key))) for label, key in _HEADERS]
    fields.insert(1, ("DMARC", _dmarc(file_obj)))
    headers = "\n".join(f"{label}: {value}" for label, value in [*fields, ("Subject", _subject(file_obj))] if value)
    return f"{headers}\n\n{_plain_text(file_obj)}"


async def _readable(file_obj: dict[str, Any], files_info: FilesInfo) -> Optional[dict[str, Any]]:
    """The email object with its plain text, ``files_info`` filling only the fields the event lacks
    (called only when ``plain_text`` or ``from`` is missing). None for any other file, or for an
    email with no plain text anywhere: the caller then downloads it as a document."""
    if not is_email_file(file_obj):
        return None
    if not (_plain_text(file_obj) and file_obj.get("from")):
        completed = await files_info(file_obj) or {}
        file_obj = {**completed, **{k: v for k, v in file_obj.items() if v not in (None, "", [])}}
        if not _plain_text(file_obj):
            file_obj["plain_text"] = completed.get("plain_text")
    return file_obj if _plain_text(file_obj) else None


async def cache_email_file(file_obj: dict[str, Any], files_info: FilesInfo) -> Optional[tuple[str, str, str]]:
    """``(cached_path, media_type, injection)`` like a text document's, for an email file with plain
    text: the whole rendered mail (never the HTML download) is cached as ``text/plain``, so the
    gateway frames it as file content, not as the user's words. It is inlined as ``[Email: ...]``
    up to :data:`TEXT_INJECT_MAX` bytes; a longer mail is not inlined, the agent reads the file, as
    for a document. None for any other file, and for a mail over :data:`DOCUMENT_MAX`, which then
    meets the document path's own size gate."""
    file_obj = await _readable(file_obj, files_info)
    if file_obj is None:
        return None
    raw = render_email(file_obj).encode("utf-8")
    if len(raw) > DOCUMENT_MAX:
        return None
    cached_path = await cache_document_from_bytes_async(raw, "email.txt")
    if len(raw) > TEXT_INJECT_MAX:
        return cached_path, "text/plain", ""
    return cached_path, "text/plain", f"[Email: {_subject(file_obj) or '(no subject)'}]:\n{raw.decode('utf-8')}"
