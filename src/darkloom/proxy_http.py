"""Fail-closed HTTP helpers for requests that must use a local Tor proxy."""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from darkloom.privacy import get_logger

import httpx

from darkloom.policy import NetworkChannel, authorize

logger = get_logger(__name__)

DEFAULT_PROXY = "socks5://127.0.0.1:9050"
_TRUE_VALUES = frozenset(("1", "true", "yes"))
_FALSE_VALUES = frozenset(("0", "false", "no"))


class TorUnavailableError(RuntimeError):
    """Raised before a request when its required Tor transport is unavailable."""


def _require_tor_enabled() -> None:
    value = os.environ.get("TOR_ENABLED")
    if value is None or not value.strip():
        raise TorUnavailableError("Tor is required, but TOR_ENABLED is unset or empty")
    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        raise TorUnavailableError("Tor is required, but TOR_ENABLED explicitly disables it")
    if normalized not in _TRUE_VALUES:
        raise TorUnavailableError(f"invalid TOR_ENABLED value: {value!r}")


def _get_proxy_url() -> str:
    """Get a fresh authenticated SOCKS5 proxy URL for request-scoped circuit isolation.

    Each call generates a unique credential, ensuring Tor's IsolateSOCKSAuth
    creates a separate circuit for each request. This prevents cross-request
    circuit correlation.
    """
    import uuid
    base = os.environ.get("TOR_PROXY", DEFAULT_PROXY)
    # Only generate credentials for valid socks5:// URLs
    if base.startswith("socks5://"):
        from urllib.parse import urlparse
        parsed = urlparse(base)
        # Don't generate credentials for malformed URLs — let the validator reject them
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            return base
        hostport = parsed.netloc or "127.0.0.1:9050"
        credential = uuid.uuid4().hex[:12]
        return f"socks5://{credential}@{hostport}"
    return base




def _validated_proxy_address(proxy_url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(proxy_url)
        port = parsed.port
    except ValueError as exc:
        raise TorUnavailableError(f"malformed TOR_PROXY: {exc}") from exc

    if parsed.scheme.lower() != "socks5":
        raise TorUnavailableError("TOR_PROXY must use the socks5 scheme")
    if parsed.password is not None:
        raise TorUnavailableError("password authentication in TOR_PROXY is not supported")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise TorUnavailableError("TOR_PROXY must not contain a path, query, or fragment")
    if parsed.hostname is None or port is None:
        raise TorUnavailableError("TOR_PROXY must include a host and port")

    hostname = parsed.hostname
    if hostname == "localhost":
        pass
    else:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError as exc:
            raise TorUnavailableError("TOR_PROXY must target a literal loopback address") from exc
        if not is_loopback:
            raise TorUnavailableError("TOR_PROXY must target a loopback address")
    return hostname, port


def _verify_socks_proxy(host: str, port: int, timeout: float) -> None:
    """Perform an unauthenticated SOCKS5 greeting without sending target traffic."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(b"\x05\x01\x00")
            reply = connection.recv(2)
    except OSError as exc:
        raise TorUnavailableError(f"Tor SOCKS proxy is unreachable: {exc}") from exc
    if reply != b"\x05\x00":
        raise TorUnavailableError("TOR_PROXY did not complete a SOCKS5 handshake")


def _get_transport(*, timeout: float = 30.0) -> httpx.HTTPTransport:
    """Return a verified Tor transport, or raise without constructing a direct one."""
    _require_tor_enabled()
    proxy_url = _get_proxy_url()
    host, port = _validated_proxy_address(proxy_url)
    _verify_socks_proxy(host, port, timeout)
    return httpx.HTTPTransport(proxy=proxy_url)


def _result(response: httpx.Response) -> dict[str, Any]:
    response.raise_for_status()
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "text": response.text,
        "url": str(response.url),
    }


def tor_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> dict[str, Any]:
    """Make an HTTP request through a validated, reachable local Tor SOCKS proxy."""
    if "use_tor" in kwargs:
        raise TypeError("tor_request does not accept use_tor; Tor is always required")
    transport = _get_transport(timeout=timeout)
    with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as client:
        return _result(client.request(method, url, **kwargs))


def tor_get(url: str, timeout: float = 30.0, **kwargs: Any) -> dict[str, Any]:
    """Make an HTTP GET through Tor; this API has no direct fallback."""
    return tor_request("GET", url, timeout=timeout, **kwargs)


def tor_post(url: str, timeout: float = 30.0, **kwargs: Any) -> dict[str, Any]:
    """Make an HTTP POST through Tor; this API has no direct fallback."""
    return tor_request("POST", url, timeout=timeout, **kwargs)


@dataclass(frozen=True)
class DirectConnectionPolicy:
    """Explicit capability required to opt out of anonymous routing."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a non-empty reason is required for direct HTTP access")


def explicitly_direct_request(
    policy: DirectConnectionPolicy,
    method: str,
    url: str,
    timeout: float = 30.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Make an intentionally non-anonymous request under an explicit policy."""
    if not isinstance(policy, DirectConnectionPolicy):
        raise TypeError("an explicit DirectConnectionPolicy is required")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        return _result(client.request(method, url, **kwargs))


def check_tor_connection(timeout: float = 30.0) -> dict[str, Any]:
    """Verify both SOCKS availability and Tor Project's routing verdict."""
    result: dict[str, Any] = {
        "tor_available": False,
        "using_tor": False,
        "exit_ip": None,
        "error": None,
    }
    try:
        data = tor_get("https://check.torproject.org/", timeout=timeout)
        result["tor_available"] = True
        text = data["text"]
        result["using_tor"] = "Congratulations" in text and "Tor" in text
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inject_tor_env(socks_proxy_url: str = DEFAULT_PROXY) -> None:
    """Enable Tor for this process and configure its local SOCKS proxy."""
    os.environ["TOR_PROXY"] = socks_proxy_url
    os.environ["TOR_ENABLED"] = "1"
    logger.info("Tor environment injected: TOR_ENABLED=1, TOR_PROXY=%s", socks_proxy_url)


def clear_tor_env() -> None:
    """Remove Tor environment variables."""
    os.environ.pop("TOR_PROXY", None)
    os.environ.pop("TOR_ENABLED", None)
    logger.info("Tor environment cleared")
