"""Shared attachment normalization helpers for gateway message events."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


@dataclass
class MessageAttachment:
    """Normalized attachment representation used across gateway platforms."""

    kind: str
    mime_type: str = ""
    local_path: str = ""
    remote_url: str = ""
    analysis_ref: str = ""
    filename: str = ""
    is_animated: bool = False
    platform_meta: dict[str, Any] = field(default_factory=dict)


def infer_attachment_kind(*, mime_type: str = "", path: str = "", filename: str = "") -> str:
    normalized = str(mime_type or "").strip().lower()
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith(("audio/", "voice/")):
        return "audio"
    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith(("application/", "text/")):
        return "document"

    suffix = _ref_suffix(filename or path)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}:
        return "image"
    if suffix in {".ogg", ".mp3", ".wav", ".m4a", ".flac", ".aac"}:
        return "audio"
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return "video"
    if suffix:
        return "document"
    return "other"


def attachments_from_legacy_media(
    media_urls: Iterable[str],
    media_sources: Iterable[str],
    media_types: Iterable[str],
) -> List[MessageAttachment]:
    urls = list(media_urls or [])
    sources = list(media_sources or [])
    types = list(media_types or [])
    attachments: List[MessageAttachment] = []
    for i, local_path in enumerate(urls):
        remote_url = str(sources[i] if i < len(sources) else "").strip()
        mime_type = str(types[i] if i < len(types) else "").strip()
        analysis_ref = remote_url or str(local_path or "").strip()
        attachments.append(
            MessageAttachment(
                kind=infer_attachment_kind(
                    mime_type=mime_type,
                    path=str(local_path or ""),
                    filename=_basename(local_path),
                ),
                mime_type=mime_type,
                local_path=str(local_path or "").strip(),
                remote_url=remote_url,
                analysis_ref=analysis_ref,
                filename=_basename(local_path),
                is_animated=_is_animated_media_ref(mime_type=mime_type, ref=analysis_ref or local_path),
            )
        )
    return attachments


def legacy_media_from_attachments(
    attachments: Iterable[MessageAttachment],
) -> Tuple[List[str], List[str], List[str]]:
    media_urls: List[str] = []
    media_sources: List[str] = []
    media_types: List[str] = []
    for attachment in attachments or []:
        media_urls.append(str(attachment.local_path or attachment.analysis_ref or "").strip())
        media_sources.append(str(attachment.remote_url or attachment.analysis_ref or "").strip())
        media_types.append(str(attachment.mime_type or "").strip())
    return media_urls, media_sources, media_types


def _basename(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    try:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            return Path(parsed.path or "").name
    except Exception:
        pass
    return Path(value).name


def _ref_suffix(ref: str) -> str:
    value = str(ref or "").strip()
    if not value:
        return ""
    try:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            value = parsed.path
    except Exception:
        pass
    return Path(value).suffix.lower()


def _is_animated_media_ref(*, mime_type: str = "", ref: str = "") -> bool:
    normalized = str(mime_type or "").strip().lower()
    return normalized == "image/gif" or _ref_suffix(ref) == ".gif"
