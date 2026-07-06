"""Error classification helpers for Proton Pass failures."""

from __future__ import annotations

from agent.secret_sources.base import ErrorKind


def classify_protonpass_error(message: str) -> ErrorKind:
    """Map a redacted Proton Pass error string onto the shared taxonomy."""
    lowered = message.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return ErrorKind.TIMEOUT
    if "binary not available" in lowered or "failed to invoke" in lowered:
        return ErrorKind.BINARY_MISSING
    if any(tok in lowered for tok in ("invalid", "expired", "auth", "login")):
        return ErrorKind.AUTH_FAILED
    if any(tok in lowered for tok in ("malformed", "not valid", "pass://")):
        return ErrorKind.REF_INVALID
    if "empty" in lowered:
        return ErrorKind.EMPTY_VALUE
    if any(tok in lowered for tok in ("network", "connection", "download", "dns")):
        return ErrorKind.NETWORK
    return ErrorKind.INTERNAL
