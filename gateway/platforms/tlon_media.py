"""Tlon media and blob helpers.

Tlon posts can carry two attachment shapes:
- rich Story image blocks inside ``content``
- a serialized ``blob`` field for files, voice memos, and videos

This module parses those shapes, formats lightweight text annotations for the
agent, and downloads safe HTTP(S) blobs into Hermes' existing media caches.
"""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from gateway.platforms.base import (
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
)

MAX_BLOB_DOWNLOAD_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class TlonBlobEntry:
    """A supported entry parsed from a Tlon post ``blob`` field."""

    type: str
    file_uri: Optional[str] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None
    duration: Optional[float] = None
    transcription: Optional[str] = None


@dataclass(frozen=True)
class TlonDownloadedAttachment:
    """A local attachment downloaded from Tlon media."""

    path: str
    content_type: str


def extract_image_blocks(content: Any) -> List[Tuple[str, str]]:
    """Return ``(url, alt)`` pairs from Story image blocks."""
    if not isinstance(content, list):
        return []

    images: List[Tuple[str, str]] = []
    for verse in content:
        if not isinstance(verse, dict):
            continue
        image = (verse.get("block") or {}).get("image")
        if not isinstance(image, dict):
            continue
        src = image.get("src")
        if isinstance(src, str) and src:
            alt = image.get("alt")
            images.append((src, alt if isinstance(alt, str) else ""))
    return images


def parse_blob_data(blob: Optional[str]) -> List[TlonBlobEntry]:
    """Parse Tlon blob JSON, keeping only supported attachment entries."""
    if not blob:
        return []
    try:
        parsed = json.loads(blob)
    except (TypeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    entries: List[TlonBlobEntry] = []
    for raw in parsed:
        if not isinstance(raw, dict):
            continue
        entry_type = raw.get("type")
        if entry_type not in {"file", "voicememo", "video"}:
            continue

        file_uri = raw.get("fileUri")
        name = raw.get("name")
        mime_type = raw.get("mimeType")
        size = raw.get("size")
        duration = raw.get("duration")
        transcription = raw.get("transcription")

        entries.append(
            TlonBlobEntry(
                type=entry_type,
                file_uri=file_uri if isinstance(file_uri, str) else None,
                name=name if isinstance(name, str) else None,
                mime_type=mime_type if isinstance(mime_type, str) else None,
                size=size if isinstance(size, int) and size >= 0 else None,
                duration=(
                    float(duration)
                    if isinstance(duration, (int, float)) and duration >= 0
                    else None
                ),
                transcription=(
                    transcription if isinstance(transcription, str) else None
                ),
            )
        )
    return entries


def format_blob_annotations(entries: Iterable[TlonBlobEntry]) -> str:
    """Format blob metadata as plain text for the agent."""
    lines: List[str] = []
    for entry in entries:
        uri = f" {entry.file_uri}" if entry.file_uri else ""
        if entry.type == "file":
            name = entry.name or "file"
            mime = entry.mime_type or "unknown"
            size = _format_size(entry.size)
            lines.append(f"[file: {name} ({mime}, {size})]{uri}")
        elif entry.type == "voicememo":
            duration = f"{round(entry.duration)}s" if entry.duration else "unknown duration"
            lines.append(f"[voice memo: {duration}]{uri}")
            if entry.transcription:
                lines.append(f'Transcription: "{entry.transcription}"')
        elif entry.type == "video":
            name = entry.name or "video"
            mime = entry.mime_type or "video"
            size = _format_size(entry.size)
            lines.append(f"[video: {name} ({mime}, {size})]{uri}")
    return "\n".join(lines)


async def download_story_images(
    content: Any,
    *,
    max_bytes: int = MAX_BLOB_DOWNLOAD_BYTES,
) -> List[TlonDownloadedAttachment]:
    """Download Story image blocks into Hermes' image cache."""
    attachments: List[TlonDownloadedAttachment] = []
    for url, _alt in extract_image_blocks(content):
        downloaded = await download_media_url(url, max_bytes=max_bytes)
        if downloaded:
            attachments.append(downloaded)
    return attachments


async def download_blob_attachments(
    entries: Iterable[TlonBlobEntry],
    *,
    max_bytes: int = MAX_BLOB_DOWNLOAD_BYTES,
) -> Tuple[List[TlonDownloadedAttachment], List[str]]:
    """Download supported blob attachments and return ``(files, notices)``."""
    attachments: List[TlonDownloadedAttachment] = []
    notices: List[str] = []

    for entry in entries:
        if not entry.file_uri:
            continue
        if entry.size is not None and entry.size > max_bytes:
            notices.append(_too_large_notice(entry, entry.size, max_bytes))
            continue

        downloaded = await download_media_url(
            entry.file_uri,
            filename_hint=entry.name,
            fallback_content_type=entry.mime_type,
            max_bytes=max_bytes,
        )
        if downloaded:
            attachments.append(downloaded)
    return attachments, notices


async def download_media_url(
    url: str,
    *,
    filename_hint: Optional[str] = None,
    fallback_content_type: Optional[str] = None,
    max_bytes: int = MAX_BLOB_DOWNLOAD_BYTES,
) -> Optional[TlonDownloadedAttachment]:
    """Download a safe HTTP(S) URL into the right Hermes cache."""
    if not _is_safe_http_url(url):
        return None

    try:
        import aiohttp
    except ImportError:
        return None

    try:
        timeout = aiohttp.ClientTimeout(total=45)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    return None

                content_type = (
                    resp.headers.get("content-type")
                    or fallback_content_type
                    or "application/octet-stream"
                )
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    return None

                chunks: List[bytes] = []
                total = 0
                async for chunk in resp.content.iter_chunked(1024 * 64):
                    total += len(chunk)
                    if total > max_bytes:
                        return None
                    chunks.append(chunk)
    except Exception:
        return None

    data = b"".join(chunks)
    media_type = content_type.split(";", 1)[0].strip().lower()
    ext = _extension_for(media_type, url)
    filename = _safe_filename(filename_hint, ext)

    try:
        if media_type.startswith("image/"):
            path = cache_image_from_bytes(data, ext=ext)
        elif media_type.startswith("audio/"):
            path = cache_audio_from_bytes(data, ext=ext)
        else:
            path = cache_document_from_bytes(data, filename)
    except Exception:
        return None

    return TlonDownloadedAttachment(path=path, content_type=media_type)


def message_type_for_media(content_type: str):
    """Return the Hermes MessageType best matching a content type."""
    from gateway.platforms.base import MessageType

    if content_type.startswith("image/"):
        return MessageType.PHOTO
    if content_type.startswith("audio/"):
        return MessageType.VOICE
    if content_type.startswith("video/"):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


def combined_message_type(content_types: List[str]):
    """Return a single MessageType for a list of downloaded content types."""
    from gateway.platforms.base import MessageType

    if not content_types:
        return MessageType.TEXT
    if all(ct.startswith("image/") for ct in content_types):
        return MessageType.PHOTO
    if all(ct.startswith("audio/") for ct in content_types):
        return MessageType.VOICE
    if all(ct.startswith("video/") for ct in content_types):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


def _is_safe_http_url(url: str) -> bool:
    """Conservative SSRF guard for attachment downloads."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    if os.getenv("TLON_ALLOW_PRIVATE_MEDIA_URLS", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return True

    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for info in infos:
        sockaddr = info[4]
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def _extension_for(content_type: str, url: str) -> str:
    ext = mimetypes.guess_extension(content_type) or ""
    if ext == ".jpe":
        ext = ".jpg"
    if ext:
        return ext
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else ".bin"


def _safe_filename(filename_hint: Optional[str], ext: str) -> str:
    if filename_hint:
        name = Path(filename_hint).name.replace("\x00", "").strip()
        if name and name not in {".", ".."}:
            return name
    return f"tlon-attachment{ext}"


def _format_size(size: Optional[int]) -> str:
    if size is None:
        return "unknown size"
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _too_large_notice(entry: TlonBlobEntry, size: int, max_bytes: int) -> str:
    label = entry.name or ("voice memo" if entry.type == "voicememo" else "blob")
    return (
        f"[blob not downloaded: {label} is {_format_size(size)}, "
        f"over the {_format_size(max_bytes)} limit]"
    )
