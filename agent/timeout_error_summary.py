"""Phase-specific one-liners for transport-timeout API errors.

A timeout that survives the retry loop otherwise surfaces as the SDK's bare
text: ``Request timed out.`` (openai ``APITimeoutError``) or ``timed out`` (a
raw httpx ``ReadTimeout`` mid-stream). Neither says whether a connection was
ever made, nor which setting to raise. The openai SDK wraps EVERY httpx
timeout, connect included, in ``APITimeoutError``, so the phase is read from
the most specific timeout in the explicit ``__cause__`` chain. ``__context__``
is not followed: an HTTP error raised while a timeout was being handled is a
different failure and keeps its own summary.
"""

from __future__ import annotations

from typing import Optional

# No TCP/TLS connection was established: raising the read timeout will not help.
_CONNECT_TIMEOUT_TYPES = frozenset({"ConnectTimeout", "ConnectTimeoutError"})
# The request was sent (or a stream was open) but no bytes arrived in time.
# ``APITimeoutError`` without a transport cause also lands here.
_READ_TIMEOUT_TYPES = frozenset({
    "ReadTimeout", "TimeoutException", "ReadTimeoutError", "APITimeoutError",
})
# Waiting for a pooled connection or while uploading the request: the provider
# never got to answer, so neither hint applies and the existing summary stays.
_OTHER_TIMEOUT_TYPES = frozenset({"PoolTimeout", "WriteTimeout"})


def _timeout_phase(error: BaseException) -> Optional[str]:
    """``"connect"``, ``"read"``, or ``None`` when no timeout is in the chain."""
    phase = None
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in _CONNECT_TIMEOUT_TYPES:
            return "connect"
        if name in _OTHER_TIMEOUT_TYPES:
            return None
        if name not in _READ_TIMEOUT_TYPES:
            break
        phase = "read"
        current = current.__cause__
    return phase


def summarize_timeout_error(error: BaseException, provider: Optional[str] = None) -> Optional[str]:
    """Timeout summary naming the phase and the setting to change, or ``None``
    for any other error (the caller's existing summary handling then applies).

    Both lines keep the "timed out" wording: cron failure notices and the error
    classifier re-read this summary as text and match on it.
    """
    phase = _timeout_phase(error)
    if phase == "connect":
        return ("Connection to the provider timed out before it was established "
                "(check the base_url and network path)")
    if phase == "read":
        key = provider.strip() if isinstance(provider, str) and provider.strip() else "<provider>"
        return ("Request timed out waiting for the provider to respond (if the model needs "
                f"longer, raise providers.{key}.request_timeout_seconds in config.yaml)")
    return None
