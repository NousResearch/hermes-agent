from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import BinaryIO

_MAGIC = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG\r\n\x1a\n",), "image/webp": (b"RIFF",)}


def validate_upload(content_type: str, head: bytes) -> str:
    mime = (content_type or "").split(";", 1)[0].lower()
    if mime not in _MAGIC or not any(head.startswith(sig) for sig in _MAGIC[mime]):
        raise ValueError("invalid_media: MIME and file signature do not match")
    if mime == "image/webp" and len(head) >= 12 and head[8:12] != b"WEBP":
        raise ValueError("invalid_media: invalid WebP signature")
    return mime


def stream_copy(source: BinaryIO, target_dir: Path, content_type: str, *, max_bytes: int = 500 * 1024 * 1024) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    part = target_dir / f".{secrets.token_hex(12)}.part"
    digest = hashlib.sha256(); size = 0; head = b""
    try:
        with part.open("wb") as out:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk: break
                if not head: head = chunk[:64]
                size += len(chunk)
                if size > max_bytes: raise ValueError("media_too_large")
                digest.update(chunk); out.write(chunk)
        mime = validate_upload(content_type, head)
        filename = digest.hexdigest()
        destination = target_dir / filename
        part.replace(destination)
        return {"filename": filename, "sha256": digest.hexdigest(), "bytes": size, "mimeType": mime, "path": str(destination)}
    except Exception:
        part.unlink(missing_ok=True)
        raise
