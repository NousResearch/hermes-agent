"""Private media cache file writing shared by the base adapter's media helpers."""

import os
import re
import uuid
from pathlib import Path


def write_cache_file(cache_dir: Path, prefix: str, ext: str, data: bytes,
                     filename: str | None = None) -> str:
    """Write data under a unique cache name, retaining a safe inbound stem when supplied.

    The file is written owner-only (0600) — inbound media is private user
    content and the cache dir is hardened to 0700 by ``_secure_media_cache_dir``;
    one directory should not carry two file-mode conventions. Best-effort on
    platforms where POSIX mode bits are advisory (Windows): falls back to a
    plain write rather than failing the media write.
    """
    stem = Path(filename if isinstance(filename, str) else "").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")[:64]
    suffix = f"_{stem}" if stem else ""
    filepath = cache_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}{ext}"
    try:
        fd = os.open(str(filepath), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    except OSError:
        filepath.write_bytes(data)
        return str(filepath)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return str(filepath)
