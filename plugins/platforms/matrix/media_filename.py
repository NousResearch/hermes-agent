"""The filename of an inbound Matrix attachment."""

import mimetypes
from pathlib import PurePosixPath


def _single_name(text: str) -> str:
    name = text.strip()
    if not name.isprintable() or name in {".", ".."}:
        return ""
    return name


def inbound_media_filename(declared: object, body: str) -> str:
    """Return the attachment's filename, or "" when the event does not give one.

    ``declared`` is the event's ``filename`` and ``body`` is its ``body`` without any reply
    fallback. The Matrix spec makes ``body`` the filename when ``filename`` is absent, but some
    clients put a caption there instead. Such a body counts as a filename only when it is one line
    with no directory part and ends in an extension, and either the extension maps to a known MIME
    type or the body contains no spaces. The result never contains a path separator.
    """
    if str(declared or "").strip():
        return _single_name(str(declared).replace("\\", "/").rsplit("/", 1)[-1])
    name = _single_name(body)
    if not name or "/" in name or "\\" in name or not PurePosixPath(name).suffix:
        return ""
    if mimetypes.guess_type(name)[0] is None and " " in name:
        return ""
    return name
