"""Canonical in-app links for what the LitKit tools return: a review document, a LitSpace file,
a LitSpace folder.

Each builder returns one markdown link, ``[label](target)``, whose target is one of three
app-relative paths:

    /matters/{matterId}?doc={documentId}[&page={N}]
    /litspace/matters/{litspaceMatterId}/documents/{fileId}
    /litspace/matters/{litspaceMatterId}/files?path={folder}

The app verifies every such link against the viewer's access before it renders, and adds its
own origin where one is needed (Slack). The host never writes an origin: ``LITCO_INSTANCE_URL``
may be a tailnet address. Pure: no network, no env, no clock.

A bad id raises ``ValueError``; the caller omits the link. A row without a link is correct, a
row with a wrong link is not.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import quote

LINK_MAX = 400
LABEL_MAX = 160
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_BREAKS = re.compile(r"\r\n|[\r\n\v\f\x85  ]")
# The brief's three, plus the two that can open a span (code, autolink/HTML) past the link's end.
_ESCAPE = frozenset("\\[]`<")


def _id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _UUID.match(value):
        raise ValueError(f"{what} is not a LitKit id (uuid)")
    return value


def _value(text: str) -> str:
    """Percent-encode down to the unreserved set: no ``/ ? = & ( )`` or space survives raw."""
    return quote(text, safe="")


def _label(raw: Any, room: int) -> str:
    """The escaped label, clipped to ``LABEL_MAX`` source characters and to ``room`` written ones."""
    text = _BREAKS.sub(" ", str(raw if raw is not None else "")).strip()
    if not text:
        raise ValueError("a link needs a label")
    clipped = len(text) > LABEL_MAX
    if clipped:
        text = text[: LABEL_MAX - 1]
    out, used = [], 0
    for n, ch in enumerate(text):
        piece = "\\" + ch if ch in _ESCAPE else ch
        tail = 1 if (clipped or n < len(text) - 1) else 0  # room for "…" if anything is cut
        if used + len(piece) + tail > room:
            clipped = True
            break
        out.append(piece)
        used += len(piece)
    if not out:
        return ""
    return "".join(out) + ("…" if clipped else "")


def _link(label: Any, target: str) -> str:
    room = LINK_MAX - len(target) - len("[]()")
    if room < 2:  # one character and the ellipsis
        return ""
    text = _label(label, room)
    return f"[{text}]({target})" if text else ""


def document_link(matter_id: str, doc_id: str, label: str, page: Optional[int] = None) -> str:
    target = f"/matters/{_value(_id(matter_id, 'matter id'))}?doc={_value(_id(doc_id, 'document id'))}"
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        target += f"&page={page}"
    return _link(label, target)


def file_link(litspace_matter_id: str, file_id: str, label: str) -> str:
    target = (f"/litspace/matters/{_value(_id(litspace_matter_id, 'LitSpace matter id'))}"
              f"/documents/{_value(_id(file_id, 'file id'))}")
    return _link(label, target)


def folder_link(litspace_matter_id: str, folder: str, label: Optional[str] = None) -> str:
    """``folder`` is LitSpace's ``folderDisplay``: segments joined by ``/``. The vault root gives ``""``."""
    ls_id = _value(_id(litspace_matter_id, "LitSpace matter id"))
    if not isinstance(folder, str) or not folder.strip().strip("/"):
        return ""
    path = "/".join(_value(segment) for segment in folder.split("/"))
    return _link(folder if label is None else label, f"/litspace/matters/{ls_id}/files?path={path}")
