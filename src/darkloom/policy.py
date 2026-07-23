"""Central fail-closed network policy for Hermes integrations.

Every operation which can create a socket or launch a network-capable child must
obtain a decision here *before* performing that operation.  Strict mode is a
default-deny policy: only the explicitly enumerated Tor bootstrap/control paths
and proxy-aware TCP clients are allowed.
"""

from __future__ import annotations

import os
from enum import Enum
from urllib.parse import urlparse


class NetworkPolicyError(PermissionError):
    """Raised before an operation would violate the active network policy."""


class NetworkChannel(str, Enum):
    HTTP = "http"
    MCP = "mcp"
    GATEWAY = "gateway"
    PLATFORM = "platform"
    BROWSER = "browser"
    WEB_TOOL = "web_tool"
    LLM = "llm"
    SUBPROCESS = "subprocess"
    RAW_SOCKET = "raw_socket"
    UDP_VOICE = "udp_voice"
    SMTP = "smtp"
    IMAP = "imap"
    IRC = "irc"
    TOR_BOOTSTRAP = "tor_bootstrap"
    TOR_CONTROL = "tor_control"


_PROXY_REQUIRED = {
    NetworkChannel.HTTP, NetworkChannel.MCP, NetworkChannel.GATEWAY,
    NetworkChannel.PLATFORM, NetworkChannel.BROWSER, NetworkChannel.WEB_TOOL,
    NetworkChannel.LLM, NetworkChannel.SUBPROCESS, NetworkChannel.RAW_SOCKET,
}
_UNSUPPORTED = {
    NetworkChannel.UDP_VOICE, NetworkChannel.SMTP, NetworkChannel.IMAP,
    NetworkChannel.IRC,
}
_EXPLICIT_DIRECT = {NetworkChannel.TOR_BOOTSTRAP, NetworkChannel.TOR_CONTROL}
_TRUE = {"1", "true", "yes", "on"}


def is_strict_mode() -> bool:
    return os.environ.get("TOR_STRICT_MODE", "").strip().lower() in _TRUE


def enable_strict_mode() -> None:
    """Activate the process-wide, fail-closed policy."""
    os.environ["TOR_STRICT_MODE"] = "1"


def configured_proxy() -> str | None:
    for key in ("TOR_PROXY", "ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def _valid_proxy(proxy_url: str | None) -> bool:
    if not proxy_url:
        return False
    parsed = urlparse(proxy_url)
    return parsed.scheme.lower() in {"socks5", "socks5h", "http", "https"} and bool(parsed.hostname)


def authorize(
    channel: NetworkChannel | str,
    *,
    proxy_url: str | None = None,
    proxy_aware: bool = True,
    local_only: bool = False,
) -> None:
    """Authorize a network operation, raising before I/O when it is unsafe.

    Outside strict mode this is intentionally non-invasive.  In strict mode an
    unknown channel is denied, unsupported protocols are always denied, and all
    ordinary clients must be proxy-aware with a configured proxy.  ``local_only``
    is not a bypass; callers must use the explicit ``tor_control`` capability.
    """
    if not is_strict_mode():
        return
    try:
        selected = NetworkChannel(channel)
    except ValueError as exc:
        raise NetworkPolicyError(f"strict mode denies unknown network channel: {channel}") from exc
    if selected in _UNSUPPORTED:
        raise NetworkPolicyError(f"strict mode denies unsupported channel: {selected.value}")
    if selected in _EXPLICIT_DIRECT:
        return
    if selected not in _PROXY_REQUIRED:
        raise NetworkPolicyError(f"strict mode has no allow rule for: {selected.value}")
    if not proxy_aware:
        raise NetworkPolicyError(f"strict mode denies non-proxy-aware {selected.value}")
    proxy = proxy_url or configured_proxy()
    if not _valid_proxy(proxy):
        raise NetworkPolicyError(f"strict mode requires a valid proxy for {selected.value}")


def authorize_subprocess(*, proxy_aware: bool, proxy_url: str | None = None) -> None:
    """Authorize a child launch before ``Popen``/``run`` is called."""
    authorize(NetworkChannel.SUBPROCESS, proxy_aware=proxy_aware, proxy_url=proxy_url)


def authorize_raw_socket(channel: NetworkChannel | str = NetworkChannel.RAW_SOCKET) -> None:
    """Authorize a raw socket before ``socket.socket`` is called."""
    authorize(channel, proxy_aware=False)
