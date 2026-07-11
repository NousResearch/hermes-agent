"""Shared path-safe naming for session-derived artifacts."""

from __future__ import annotations

import hashlib
import re


def safe_session_filename_component(session_id: str) -> str:
    """Return a stable single filename component for an untrusted session ID."""
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")
    sanitized = sanitized[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    digest = hashlib.sha256(
        raw.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]
    return f"{sanitized}_{digest}"
