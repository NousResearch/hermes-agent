"""Hashing utilities."""

from __future__ import annotations

import hashlib


def sha256_text(value: str) -> str:
    """Hash normalized text."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
