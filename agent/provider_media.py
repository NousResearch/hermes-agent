"""``$HERMES_HOME/cache/<kind>/`` materialisation helpers for the image/video
generation provider ABCs.

Several backends return *ephemeral* delivery URLs that expire before a downstream
consumer (Telegram ``send_photo``, browser fetch) can resolve them, so providers
materialise the bytes locally at tool-completion time. Filenames are
``<prefix>_<YYYYMMDD_HHMMSS>_<uuid8>.<ext>``.
"""

from __future__ import annotations

import base64
import datetime
import uuid
from pathlib import Path
from typing import Dict, Tuple


def cache_dir(kind: str) -> Path:
    """Return ``$HERMES_HOME/cache/<kind>/``, creating parents as needed."""
    from hermes_constants import get_hermes_home
    path = get_hermes_home() / "cache" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(kind: str, prefix: str, extension: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    return cache_dir(kind) / f"{prefix}_{ts}_{short}.{extension}"


def save_bytes(kind: str, raw: bytes, *, prefix: str, extension: str) -> Path:
    """Write raw bytes to the cache and return the absolute path."""
    path = cache_path(kind, prefix, extension)
    path.write_bytes(raw)
    return path


def save_b64(kind: str, b64_data: str, *, prefix: str, extension: str) -> Path:
    """Decode base64 data into the cache and return the absolute path."""
    return save_bytes(kind, base64.b64decode(b64_data), prefix=prefix, extension=extension)


def save_url(
    kind: str, url: str, *, prefix: str, timeout: float, max_bytes: int, chunk_size: int,
    content_types: Dict[str, str], url_extensions: Tuple[str, ...], default_extension: str,
    label: str, empty_error: str,
) -> Path:
    """Stream-download *url* into the cache with a size cap.

    The extension comes from the response ``Content-Type`` (an explicit table —
    never inherit a type pointing at HTML/JSON from a degenerate response), then
    the URL suffix (some CDNs return ``application/octet-stream``), then
    *default_extension*. Raises on any network / HTTP / oversize / empty error so
    callers can fall back to the bare URL; a partial file is never left behind.
    """
    import requests
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    extension = content_types.get(content_type)
    if extension is None:
        url_path = url.split("?", 1)[0].lower()
        extension = next(
            ("jpg" if ext == "jpeg" else ext for ext in url_extensions if url_path.endswith(f".{ext}")),
            default_extension,
        )
    path = cache_path(kind, prefix, extension)
    bytes_written = 0
    with path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                fh.close()
                _unlink_quiet(path)
                raise ValueError(
                    f"{label} at {url} exceeds {max_bytes // (1024 * 1024)}MB cap; refusing to cache."
                )
            fh.write(chunk)

    if bytes_written == 0:
        _unlink_quiet(path)
        raise ValueError(empty_error.format(url=url))

    return path


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
