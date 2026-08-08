"""Document cache helpers for gateway platform adapters.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition
campaign, wave 1 — shard s2, cluster c3, 12 move votes). Functions moved
verbatim; ``base.py`` re-exports them. ``_resolve_cache_dir`` and
``_cleanup_cache_dir`` stay in ``base.py`` (the image/audio/video cache
helpers that remain there still use them) and are imported here at the
bottom of this module (cycle break).
"""

import uuid
from pathlib import Path

def get_document_cache_dir() -> Path:
    """Return the document cache directory, creating it if it doesn't exist."""
    d = _resolve_cache_dir("DOCUMENT_CACHE_DIR", "cache/documents", "document_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """
    Save raw document bytes to the cache and return the absolute file path.

    The cached filename preserves the original human-readable name with a
    unique prefix: ``doc_{uuid12}_{original_filename}``.

    Args:
        data: Raw document bytes.
        filename: Original filename (e.g. "report.pdf").

    Returns:
        Absolute path to the cached document file as a string.

    Raises:
        ValueError: If the sanitized path escapes the cache directory.
    """
    cache_dir = get_document_cache_dir()
    # Sanitize: strip directory components, null bytes, and control characters
    safe_name = Path(filename).name if filename else "document"
    safe_name = safe_name.replace("\x00", "").strip()
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "document"
    cached_name = f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    filepath = cache_dir / cached_name
    # Final safety check: ensure path stays inside cache dir
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    filepath.write_bytes(data)
    return str(filepath)


def cleanup_document_cache(max_age_hours: int = 24) -> int:
    """
    Delete cached documents older than *max_age_hours*.

    Returns the number of files removed.
    """
    return _cleanup_cache_dir(get_document_cache_dir(), max_age_hours)


from gateway.platforms.base import (  # noqa: E402
    _cleanup_cache_dir,
    _resolve_cache_dir,
)
