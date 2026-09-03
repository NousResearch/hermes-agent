"""D5 / M5 — server-side history media projection.

Derives per-message media refs from STORED session data (literal ``MEDIA:``
tags in persisted reply text), so a reopened transcript renders media
deterministically instead of depending on the live event replay (D1) or on
literal tag survival in the rendered markdown.

Extraction reuses the delivery pipeline's own gates —
:func:`gateway.media_events.extract_media_from_stored_text`, which wraps
``BasePlatformAdapter.extract_media`` + the delivery-path policy check — so a
projected ref is exactly what live delivery would accept, with one documented
relaxation: a vanished file still projects (as fallback metadata) instead of
disappearing from history. Each accepted path is then split by existence:

* still statable → the D1 payload shape ``{path, kind, mime, size}`` plus
  ``available: true`` (the same fields a live ``media.deliverable`` event
  carries, so the desktop registry row contract applies unchanged);
* gone           → fallback metadata ``{path, available: false, name, kind,
  mime}`` — no ``size`` (unknown), enough for the desktop's never-silent
  fallback card. A vanished file must never silently disappear from history.

Bounded by construction: extraction runs over the text of the messages in the
already-paged response only (no transcript-wide rescan), and the only
filesystem work is one ``stat`` per extracted ref (inside
``describe_media_deliverable``). The projection is derived on read — no
schema change, no migration, and identical stores produce identical output
(stable first-occurrence order), which is the "reopened sessions render
deterministically" guarantee.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from gateway.media_events import (
    describe_media_deliverable,
    extract_media_from_stored_text,
    media_kind_for_mime,
)

logger = logging.getLogger(__name__)

# Guard against pathological stored rows (a runaway tool result can be
# megabytes of base64-ish text). The MEDIA tags that matter sit in the reply
# prose; a row this large is truncated for extraction only — the returned
# message content is untouched.
_MAX_MESSAGE_TEXT_CHARS = 200_000


def _candidate_texts(message: Dict[str, Any]) -> List[str]:
    """Stored text fields that can carry deliverable tags, bounded.

    ``content`` is the persisted reply (or tool result); ``display_content``
    is the backend-projected user-visible view when the physical row also
    carries internal scaffolding. Non-string shapes (structured content
    blocks) never carried MEDIA tags — delivery extraction is defined on
    reply text.
    """
    texts: List[str] = []
    for key in ("content", "display_content"):
        value = message.get(key)
        if isinstance(value, str) and "MEDIA:" in value:
            texts.append(value[:_MAX_MESSAGE_TEXT_CHARS])
    return texts


def _fallback_row(path: str) -> Dict[str, Any]:
    """Metadata for a ref whose file is gone — no stat, no guessing a size.

    Kind/mime come from the extension table only (the same source
    ``describe_media_deliverable`` uses for real files), so a missing
    ``photo.png`` still renders as an image card with a name instead of a
    silent nothing. ``name`` is computed slash-tolerantly because stored refs
    may be Windows-style paths while this runs on POSIX.
    """
    from gateway.platforms.media_cache import mime_for_ext

    name = path.replace("\\", "/").rsplit("/", 1)[-1] or path
    dot = name.rfind(".")
    suffix = name[dot:].lower() if dot > -1 else ""
    mime = mime_for_ext(suffix)
    return {
        "path": path,
        "available": False,
        "name": name,
        "kind": media_kind_for_mime(mime),
        "mime": mime,
    }


def project_message_media(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Media ref rows for one stored message (empty when it tags nothing)."""
    paths: List[str] = []
    for text in _candidate_texts(message):
        for path in extract_media_from_stored_text(text):
            if path and path not in paths:
                paths.append(path)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        described = describe_media_deliverable(path)
        if described is not None:
            row = dict(described)
            row["available"] = True
        else:
            row = _fallback_row(path)
        rows.append(row)
    return rows


def build_media_refs_for_messages(
    messages: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Projection map ``{message_index: [media rows]}`` for a page of messages.

    Indices refer to the caller's list order (the endpoint stamps them onto
    the same rows it returns). Messages without refs are absent from the map;
    extraction failures degrade to "no refs for that message" — history
    rendering must never raise into the transcript read path.
    """
    projection: Dict[int, List[Dict[str, Any]]] = {}
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        try:
            rows = project_message_media(message)
        except Exception:
            logger.debug(
                "history media projection failed for message index %s",
                index,
                exc_info=True,
            )
            continue
        if rows:
            projection[index] = rows
    return projection
