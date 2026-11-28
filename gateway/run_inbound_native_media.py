"""Native inbound audio/PDF routing and outbound part materialization for the messaging gateway."""
from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import mimetypes
import os
from typing import Any

from agent.image_routing import (
    decide_audio_input_mode,
    decide_pdf_input_mode,
    supports_audio_in_user_turns,
    supports_pdf_in_user_turns,
)
from hermes_cli.config import load_config
from tools.media_io import (
    MAX_NATIVE_MEDIA_RAW_BYTES,
    is_valid_audio_bytes,
    is_valid_pdf_bytes,
    normalize_audio_mime,
    peek_file_header_nofollow,
    read_file_bounded_nofollow,
)

logger = logging.getLogger(__name__)


def _is_pdf_candidate(path: str, mtype: str) -> bool:
    """True when an inbound attachment's MIME type (ignoring ``;params``) or extension indicates a PDF."""
    base_mime = str(mtype or "").split(";", 1)[0].strip().lower()
    return base_mime == "application/pdf" or os.path.splitext(path)[1].lower() == ".pdf"


def has_native_media_candidates(event: Any, audio_file_paths: list[str]) -> bool:
    """True when *event* carries any candidate audio file or PDF eligible for native routing."""
    if audio_file_paths:
        return True
    event_urls = list(getattr(event, "media_urls", None) or ())
    if not event_urls:
        return False
    event_types = list(getattr(event, "media_types", None) or ())
    return any(
        _is_pdf_candidate(path, event_types[i] if i < len(event_types) else "")
        for i, path in enumerate(event_urls)
    )


def _resolve_runtime_readonly(runner: Any, *, source: Any, session_key: str) -> tuple[str, dict[str, Any]]:
    """Invoke ``runner._resolve_session_agent_runtime`` without mutating runner-wide fallback notice slots."""
    fn = runner._resolve_session_agent_runtime
    try:
        params = inspect.signature(fn).parameters
        accepts_record_notice = (
            "record_notice" in params
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        )
    except (TypeError, ValueError):
        accepts_record_notice = False
    if accepts_record_notice:
        return fn(source=source, session_key=session_key, record_notice=False)
    return fn(source=source, session_key=session_key)


def prepare_inbound_native_audio_and_pdfs(
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    audio_file_paths: list[str],
) -> tuple[list[str], set[str], list[dict[str, str]]]:
    """Identify eligible inbound audio files and PDFs for native attachment without mutating session state.

    Returns ``(remaining_audio_file_paths, native_pdf_paths, queued_attachments)``. Never touches ``MessageType.VOICE`` STT.
    """
    if not has_native_media_candidates(event, audio_file_paths):
        return audio_file_paths, set(), []
    try:
        turn_model, runtime_kwargs = _resolve_runtime_readonly(
            runner, source=source, session_key=session_key,
        )
        rt = runtime_kwargs or {}
        provider = str(rt.get("provider") or "").strip()
        base_url = str(rt.get("base_url") or "").strip()
        requested_provider = str(rt.get("requested_provider") or provider).strip()
        api_mode = str(rt.get("api_mode") or "").strip()
        cfg = load_config()
        audio_native = (
            bool(audio_file_paths)
            and supports_audio_in_user_turns(provider, turn_model, cfg, base_url=base_url, api_mode=api_mode)
            and decide_audio_input_mode(
                provider, turn_model, cfg, requested_provider=requested_provider, allow_network=False, base_url=base_url,
            ) == "native"
        )
        pdf_native = (
            bool(getattr(event, "media_urls", None))
            and supports_pdf_in_user_turns(provider, turn_model, cfg, base_url=base_url, api_mode=api_mode)
            and decide_pdf_input_mode(
                provider, turn_model, cfg, requested_provider=requested_provider, allow_network=False, base_url=base_url,
            ) == "native"
        )
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("gateway native audio/PDF capability probe failed: %s", exc)
        return audio_file_paths, set(), []

    if not (audio_native or pdf_native):
        return audio_file_paths, set(), []

    queued: list[dict[str, str]] = []
    remaining_audio: list[str] = []
    seen_audio: set[str] = set()
    names = runner._inbound_original_names(event)
    event_urls = list(getattr(event, "media_urls", None) or ())
    event_types = list(getattr(event, "media_types", None) or ())
    mime_by_path = {
        p: str(event_types[i])
        for i, p in enumerate(event_urls)
        if i < len(event_types) and event_types[i]
    }

    for path in audio_file_paths:
        if audio_native:
            if path in seen_audio:
                continue
            head = peek_file_header_nofollow(path, max_file_bytes=MAX_NATIVE_MEDIA_RAW_BYTES)
            if head is not None and is_valid_audio_bytes(head):
                declared = mime_by_path.get(path, "")
                raw_mime = declared if declared.lower().startswith("audio/") else (mimetypes.guess_type(path)[0] or "audio/mpeg")
                mime = normalize_audio_mime(raw_mime)
                if mime.startswith("audio/"):
                    display, _ = runner._inbound_attachment_display_name(path, names.get(path, ""))
                    queued.append({"kind": "audio", "path": path, "mime_type": mime, "filename": display})
                    seen_audio.add(path)
                    continue
        remaining_audio.append(path)

    native_pdf_paths: set[str] = set()
    if pdf_native and event_urls:
        for i, path in enumerate(event_urls):
            if path in native_pdf_paths:
                continue
            mtype = event_types[i] if i < len(event_types) else ""
            if not _is_pdf_candidate(path, mtype):
                continue
            head = peek_file_header_nofollow(path, max_file_bytes=MAX_NATIVE_MEDIA_RAW_BYTES)
            if head is None or not is_valid_pdf_bytes(head):
                continue
            display, _ = runner._inbound_attachment_display_name(path, names.get(path, ""))
            queued.append({"kind": "document", "path": path, "mime_type": "application/pdf", "filename": display})
            native_pdf_paths.add(path)

    return remaining_audio, native_pdf_paths, queued


async def prepare_inbound_native_media_async(
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    audio_file_paths: list[str],
) -> tuple[list[str], set[str], list[dict[str, str]]]:
    """Event-loop wrapper: skip thread-pool dispatch when *event* has no audio/PDF candidates."""
    if not has_native_media_candidates(event, audio_file_paths):
        return audio_file_paths, set(), []
    return await asyncio.to_thread(
        prepare_inbound_native_audio_and_pdfs, runner, event, source, session_key, audio_file_paths,
    )


def route_inbound_native_audio_and_pdfs(
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    audio_file_paths: list[str],
) -> tuple[list[str], set[str]]:
    """Queue eligible inbound audio files and PDFs for native attachment.

    Returns ``(remaining_audio_file_paths, native_pdf_paths)``. Never touches ``MessageType.VOICE`` STT.
    """
    remaining_audio, native_pdf_paths, queued = prepare_inbound_native_audio_and_pdfs(
        runner, event, source, session_key, audio_file_paths,
    )
    if queued:
        runner._session_state(session_key).persistent.native_media_attachments = queued
    return remaining_audio, native_pdf_paths


def materialize_native_media_parts_with_skipped(
    items: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build outbound ``audio``/``document`` content parts from queued gateway records, returning ``(parts, skipped)``."""
    parts: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in items:
        raw_path = str(item.get("path") or "")
        kind = str(item.get("kind") or "")
        mime_type = str(item.get("mime_type") or "")
        filename = str(item.get("filename") or os.path.basename(raw_path))
        if not raw_path or kind not in ("audio", "document"):
            skipped.append(item)
            continue
        raw = read_file_bounded_nofollow(raw_path, max_bytes=MAX_NATIVE_MEDIA_RAW_BYTES)
        if raw is None:
            skipped.append(item)
            continue
        if kind == "audio" and not is_valid_audio_bytes(raw):
            skipped.append(item)
            continue
        if kind == "document" and mime_type == "application/pdf" and not is_valid_pdf_bytes(raw):
            skipped.append(item)
            continue
        encoded = base64.b64encode(raw).decode("ascii")
        if kind == "audio":
            parts.append({"type": "audio", "mime_type": normalize_audio_mime(mime_type) or "audio/mpeg", "data": encoded})
        else:
            parts.append({
                "type": "document",
                "mime_type": mime_type or "application/pdf",
                "data": encoded,
                "filename": filename,
            })
    return parts, skipped


def materialize_native_media_parts(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build outbound ``audio``/``document`` content parts from queued gateway attachment records."""
    parts, _ = materialize_native_media_parts_with_skipped(items)
    return parts
