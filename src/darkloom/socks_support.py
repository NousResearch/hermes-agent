"""Fail-closed validation for the HTTPX SOCKS transport."""

import httpx


SOCKS_SUPPORT_ERROR = (
    "SOCKS transport unavailable; install darkloom with its declared "
    "httpx[socks] dependency. Direct fallback is disabled."
)


class SocksSupportError(RuntimeError):
    """Raised when the installed HTTP stack cannot construct SOCKS transports."""


def require_socks_support(proxy_url: str = "socks5://127.0.0.1:9050") -> None:
    """Construct the sync and async SOCKS transports without opening a socket.

    HTTPX resolves and imports its SOCKS backend while constructing these
    transports.  No request is issued, so this is safe to run before Tor starts.
    Any backend or version error is deliberately replaced with one stable,
    actionable exception rather than permitting a direct connection.
    """
    sync_transport = None
    try:
        sync_transport = httpx.HTTPTransport(proxy=proxy_url)
        httpx.AsyncHTTPTransport(proxy=proxy_url)
    except Exception:
        raise SocksSupportError(SOCKS_SUPPORT_ERROR) from None
    finally:
        if sync_transport is not None:
            sync_transport.close()
