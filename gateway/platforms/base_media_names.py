"""Filename helpers for the inbound-media cache: keep a hostile name from becoming a path.

Both inputs are sender-controlled (a remote ``Content-Type`` suffix, a remote document name), so
each helper returns something that is safe to join onto a cache directory and keeps the file
identifiable. Kept out of ``base.py`` so the cache path logic there does not grow.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MAX_DOCUMENT_NAME = 80


def safe_cache_extension(ext: str) -> str:
    """Last alphanumeric suffix of ``ext`` (max 10 chars), dot-prefixed; ``""`` when there is none."""
    tail = str(ext).strip().rsplit(".", 1)[-1]
    alnum = "".join(ch for ch in tail if ch.isalnum())[:10]
    return f".{alnum}" if alnum else ""


def capped_document_name(filename: str) -> str:
    """``filename`` reduced to one path component, bounded in length, and openable on Windows."""
    name = (Path(filename).name if filename else "document").replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        name = "document"
    if len(name) > MAX_DOCUMENT_NAME:
        stem, dot, suffix = name.rpartition(".")
        if not dot or len(suffix) > 10:
            stem, suffix = name, ""
        else:
            suffix = f".{suffix}"
        digest = hashlib.sha256(name.encode("utf-8", "replace")).hexdigest()[:12]
        name = f"{stem[:60]}_{digest}{suffix}"
    if os.name == "nt":  # ``:`` turns the name into an alternate data stream open() cannot read
        name = name.replace(":", "_")
    return name
