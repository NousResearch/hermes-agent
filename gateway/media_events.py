"""Structured media-deliverable events (D1) for desktop surfaces.

When reply processing extracts media deliverables, the emitting surface also
publishes one ``media.deliverable`` event frame through the tui-gateway event
transport (``tui_gateway.server.write_json``), where it gains the standard
per-session ``seq`` stamp and bounded replay ring — a reconnecting desktop
recovers it via ``session.events.since`` exactly like tool/reasoning events.

Payload contract: ``{path, kind, mime, size, session_id, origin}``.

Routing safety: frames are written ONLY when the frame's session has a live
client transport. There is deliberately no stdio fallback — the messaging
gateway's stdout is protocol-free, and a gateway process that never imported
tui_gateway must stay byte-identical on stdout. Emission is therefore
best-effort and silent whenever no desktop client is listening; the mature
platform-adapter delivery path is the sole owner of actual file delivery.

Emission sites (see plans/media-delivery/M1.md):
- gateway/run.py turn-level auto-append (origin ``gateway``),
- gateway/run.py post-stream explicit rescan (origin ``gateway``),
- tui_gateway/server.py turn completion on the serve/desktop backend
  (origin ``serve``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEDIA_EVENT_TYPE = "media.deliverable"

# Payload ``origin`` values. ``relay`` is reserved for the relay-envelope
# milestone (D2); keeping the enum here gives every emitter one vocabulary.
ORIGIN_GATEWAY = "gateway"
ORIGIN_SERVE = "serve"
ORIGIN_RELAY = "relay"

# MIME class → event kind. Anything else (including application/octet-stream
# subclasses like zip/pdf/docx) is a ``document`` — mirroring how the desktop
# renderer treats non-av, non-image media cards.
_KIND_BY_CLASS = {
    "image": "image",
    "video": "video",
    "audio": "audio",
}


def _mime_class(mime: str) -> str:
    return (mime or "").split(";", 1)[0].strip().split("/", 1)[0].lower()


def media_kind_for_mime(mime: str) -> str:
    """Event/media kind for a mime string (the one shared vocabulary).

    AV + image classes map to their kind; everything else — including
    ``application/octet-stream`` subclasses like zip/pdf/docx — is a
    ``document``, mirroring how the desktop renderer treats non-av,
    non-image media cards. Public because the history projection (D5)
    classifies fallback rows for files whose stat failed, and it must agree
    with the live-event classification byte-for-byte.
    """
    return _KIND_BY_CLASS.get(_mime_class(mime), "document")


def describe_media_deliverable(path: str) -> Optional[Dict[str, Any]]:
    """Build the static payload fields for one deliverable file.

    Returns ``None`` when the path does not resolve to an existing regular
    file — events must describe real files. The caller supplies
    ``session_id``/``origin`` (process-local context this helper cannot know).
    """
    if not path:
        return None
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file():
        return None

    from gateway.platforms.media_cache import mime_for_ext

    mime = mime_for_ext(resolved.suffix.lower())
    kind = media_kind_for_mime(mime)
    try:
        size = resolved.stat().st_size
    except OSError:
        return None
    return {"path": str(resolved), "kind": kind, "mime": mime, "size": size}


def extract_media_from_reply(reply: str) -> List[str]:
    """Validated absolute paths of explicit ``MEDIA:`` tags in a reply text.

    Runs the delivery pipeline's own extraction and safety gates
    (``BasePlatformAdapter.extract_media`` + ``filter_media_delivery_paths``)
    so an event mirrors exactly the files delivery would accept — never a
    looser scrape. Deduplicated, order-preserving.
    """
    if not reply or "MEDIA:" not in reply:
        return []
    try:
        from gateway.platforms.base import BasePlatformAdapter

        media_files, _cleaned = BasePlatformAdapter.extract_media(reply)
        safe = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    except Exception:
        logger.debug("media-deliverable extraction failed", exc_info=True)
        return []

    paths: List[str] = []
    for media_path, _is_voice in safe:
        path = str(media_path)
        if path and path not in paths:
            paths.append(path)
    return paths


def extract_media_from_stored_text(text: str) -> List[str]:
    """Validated absolute paths of ``MEDIA:`` tags in STORED reply text.

    The D5 history-projection counterpart of :func:`extract_media_from_reply`:
    identical extraction, but the existence requirement is relaxed so a tag
    whose file has since vanished still projects (with fallback metadata)
    instead of silently disappearing from a reopened transcript. Policy gates
    are unchanged — credential/system paths and strict-mode containment are
    rejected exactly as in live delivery.

    Each extracted path is validated directly via
    :func:`gateway.platforms.base.validate_media_delivery_path` with
    ``require_exists=False`` rather than through
    ``filter_media_delivery_paths``: a vanished file is the DOMINANT real
    case in stored history (M0 evidence), and the filter's WARNING per
    dropped path would fire on every transcript reopen for files that
    merely aged out — noise, not signal. Deduplicated, order-preserving.
    """
    if not text or "MEDIA:" not in text:
        return []
    try:
        from gateway.platforms.base import (
            BasePlatformAdapter,
            validate_media_delivery_path,
        )

        media_files, _cleaned = BasePlatformAdapter.extract_media(text)
    except Exception:
        logger.debug("stored-media extraction failed", exc_info=True)
        return []

    paths: List[str] = []
    for media_path, _is_voice in media_files:
        safe_path = validate_media_delivery_path(
            str(media_path), require_exists=False
        )
        if safe_path and safe_path not in paths:
            paths.append(safe_path)
    return paths


def build_media_deliverable_payloads(
    paths: List[str],
    session_id: str,
    origin: str,
) -> List[Dict[str, Any]]:
    """Full payloads for statable files; callers emit them on their own
    event boundary (e.g. the serve backend's ``_emit``)."""
    payloads: List[Dict[str, Any]] = []
    for path in paths or []:
        described = describe_media_deliverable(path)
        if described is None:
            continue
        payload = dict(described)
        payload["session_id"] = session_id
        payload["origin"] = origin
        payloads.append(payload)
    return payloads


def emit_media_deliverable(
    session_id: str,
    path: str,
    *,
    origin: str,
) -> bool:
    """Emit one ``media.deliverable`` frame for a real file, best-effort.

    Returns True when the frame was handed to a live session transport.
    Silent no-op when the session has no transport, tui_gateway is not
    loaded, or the write fails — never raises into the reply path.
    """
    if not session_id or not path:
        return False

    try:
        described = describe_media_deliverable(path)
        if described is None:
            return False

        from tui_gateway import server as tui_server

        session = (tui_server._sessions or {}).get(session_id)
        transport = (session or {}).get("transport")
        if transport is None:
            # No desktop client listening on this session. No stdio fallback:
            # see module docstring for the routing-safety rationale.
            return False

        payload = dict(described)
        payload["session_id"] = session_id
        payload["origin"] = origin

        frame = {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": MEDIA_EVENT_TYPE,
                "session_id": session_id,
                "payload": payload,
            },
        }
        # write_json routes event frames carrying a session id to that
        # session's transport (precedence 1) and stamps the replay seq —
        # the stdio fallback is unreachable because we verified the
        # transport above. Its bool return mirrors the transport write.
        ok = bool(tui_server.write_json(frame))
        if not ok:
            logger.debug(
                "media.deliverable transport write rejected (session=%s)", session_id
            )
        return ok
    except Exception:
        logger.debug("media.deliverable emission failed", exc_info=True)
        return False


def emit_media_deliverables(
    session_id: str,
    paths: List[str],
    *,
    origin: str,
) -> int:
    """Emit one frame per path; returns the number actually emitted."""
    emitted = 0
    for path in paths or []:
        if emit_media_deliverable(session_id, path, origin=origin):
            emitted += 1
    return emitted
