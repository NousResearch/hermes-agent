"""Proxy resolution and NO_PROXY matching for gateway platform adapters.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition
campaign, wave 1 — shard s1, cluster c6, 36 move votes). Every function is
moved verbatim; ``base.py`` re-exports them so ``from
gateway.platforms.base import ...`` call sites are unchanged. The
module-level ``logger`` reuses the historical ``gateway.platforms.base``
logger name so log records are byte-identical.
"""

import ipaddress
import logging
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit

from utils import normalize_proxy_url

logger = logging.getLogger("gateway.platforms.base")

def _detect_macos_system_proxy() -> str | None:
    """Read the macOS system HTTP(S) proxy via ``scutil --proxy``.

    Returns an ``http://host:port`` URL string if an HTTP or HTTPS proxy is
    enabled, otherwise *None*.  Falls back silently on non-macOS or on any
    subprocess error.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["scutil", "--proxy"], timeout=3, text=True, encoding='utf-8', errors='replace', stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    props: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if " : " in line:
            key, _, val = line.partition(" : ")
            props[key.strip()] = val.strip()

    # Prefer HTTPS, fall back to HTTP
    for enable_key, host_key, port_key in (
        ("HTTPSEnable", "HTTPSProxy", "HTTPSPort"),
        ("HTTPEnable", "HTTPProxy", "HTTPPort"),
    ):
        if props.get(enable_key) == "1":
            host = props.get(host_key)
            port = props.get(port_key)
            if host and port:
                return f"http://{host}:{port}"
    return None


def _split_host_port(value: str) -> tuple[str, int | None]:
    raw = str(value or "").strip()
    if not raw:
        return "", None
    if "://" in raw:
        parsed = urlsplit(raw)
        return (parsed.hostname or "").lower().rstrip("."), parsed.port
    if raw.startswith("[") and "]" in raw:
        host, _, rest = raw[1:].partition("]")
        port = None
        if rest.startswith(":") and rest[1:].isdigit():
            port = int(rest[1:])
        return host.lower().rstrip("."), port
    if raw.count(":") == 1:
        host, _, maybe_port = raw.rpartition(":")
        if maybe_port.isdigit():
            return host.lower().rstrip("."), int(maybe_port)
    return raw.lower().strip("[]").rstrip("."), None


def _no_proxy_entries() -> list[str]:
    entries: list[str] = []
    for key in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(key, "")
        entries.extend(part.strip() for part in raw.split(",") if part.strip())
    return entries


def _no_proxy_entry_matches(entry: str, host: str, port: int | None = None) -> bool:
    token = str(entry or "").strip().lower()
    if not token:
        return False
    if token == "*":
        return True

    token_host, token_port = _split_host_port(token)
    if token_port is not None and port is not None and token_port != port:
        return False
    if token_port is not None and port is None:
        return False
    if not token_host:
        return False

    try:
        network = ipaddress.ip_network(token_host, strict=False)
        try:
            return ipaddress.ip_address(host) in network
        except ValueError:
            return False
    except ValueError:
        pass

    try:
        token_ip = ipaddress.ip_address(token_host)
        try:
            return ipaddress.ip_address(host) == token_ip
        except ValueError:
            return False
    except ValueError:
        pass

    if token_host.startswith("*."):
        suffix = token_host[1:]
        return host.endswith(suffix)
    if token_host.startswith("."):
        return host == token_host[1:] or host.endswith(token_host)
    return host == token_host or host.endswith(f".{token_host}")


def should_bypass_proxy(target_hosts: str | list[str] | tuple[str, ...] | set[str] | None) -> bool:
    """Return True when NO_PROXY/no_proxy matches at least one target host.

    Supports exact hosts, domain suffixes, wildcard suffixes, IP literals,
    CIDR ranges, optional host:port entries, and ``*``.
    """
    entries = _no_proxy_entries()
    if not entries or not target_hosts:
        return False
    if isinstance(target_hosts, str):
        candidates = [target_hosts]
    else:
        candidates = list(target_hosts)
    for candidate in candidates:
        host, port = _split_host_port(str(candidate))
        if not host:
            continue
        if any(_no_proxy_entry_matches(entry, host, port) for entry in entries):
            return True
    return False


def resolve_proxy_url(
    platform_env_var: str | None = None,
    *,
    target_hosts: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> str | None:
    """Return a proxy URL from env vars, or macOS system proxy.

    Check order:
      0. *platform_env_var* (e.g. ``DISCORD_PROXY``) — highest priority
      1. HTTPS_PROXY / HTTP_PROXY / ALL_PROXY (and lowercase variants)
      2. macOS system proxy via ``scutil --proxy`` (auto-detect)

    Returns *None* if no proxy is found, or if NO_PROXY/no_proxy matches one
    of ``target_hosts``.
    """
    if platform_env_var:
        value = (os.environ.get(platform_env_var) or "").strip()
        if value:
            if should_bypass_proxy(target_hosts):
                return None
            return normalize_proxy_url(value)
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = (os.environ.get(key) or "").strip()
        if value:
            if should_bypass_proxy(target_hosts):
                return None
            return normalize_proxy_url(value)
    detected = normalize_proxy_url(_detect_macos_system_proxy())
    if detected and should_bypass_proxy(target_hosts):
        return None
    return detected


def proxy_kwargs_for_bot(proxy_url: str | None) -> dict:
    """Build kwargs for ``commands.Bot()`` / ``discord.Client()`` with proxy.

    Returns:
      - SOCKS URL  → ``{"connector": ProxyConnector(..., rdns=True)}``
      - HTTP URL   → ``{"proxy": url}``
      - *None*     → ``{}``

    ``rdns=True`` forces remote DNS resolution through the proxy — required
    by many SOCKS implementations (Shadowrocket, Clash) and essential for
    bypassing DNS pollution behind the GFW.
    """
    if not proxy_url:
        return {}
    if proxy_url.lower().startswith("socks"):
        try:
            from aiohttp_socks import ProxyConnector

            connector = ProxyConnector.from_url(proxy_url, rdns=True)
            return {"connector": connector}
        except ImportError:
            logger.warning(
                "aiohttp_socks not installed — SOCKS proxy %s ignored. "
                "Run: pip install aiohttp-socks",
                proxy_url,
            )
            return {}
    return {"proxy": proxy_url}


def proxy_kwargs_for_aiohttp(proxy_url: str | None) -> tuple[dict, dict]:
    """Build kwargs for standalone ``aiohttp.ClientSession`` with proxy.

    Returns ``(session_kwargs, request_kwargs)`` where:
      - With aiohttp-socks → ``({"connector": ProxyConnector(...)}, {})``
        for *all* proxy schemes (SOCKS **and** HTTP/HTTPS).
      - HTTP without aiohttp-socks → ``({}, {"proxy": url})``.
      - None → ``({}, {})``.

    Prefer the connector path: it works transparently with libraries
    (like mautrix) that call ``session.request()`` without forwarding
    per-request ``proxy=`` kwargs.

    Usage::

        sess_kw, req_kw = proxy_kwargs_for_aiohttp(proxy_url)
        async with aiohttp.ClientSession(**sess_kw) as session:
            async with session.get(url, **req_kw) as resp:
                ...
    """
    if not proxy_url:
        return {}, {}
    try:
        from aiohttp_socks import ProxyConnector

        connector = ProxyConnector.from_url(proxy_url, rdns=True)
        return {"connector": connector}, {}
    except ImportError:
        if proxy_url.lower().startswith("socks"):
            logger.warning(
                "aiohttp_socks not installed — SOCKS proxy %s ignored. "
                "Run: pip install aiohttp-socks",
                proxy_url,
            )
            return {}, {}
        return {}, {"proxy": proxy_url}


def is_host_excluded_by_no_proxy(hostname: str, no_proxy_value: str | None = None) -> bool:
    """Return True when ``hostname`` matches a ``NO_PROXY`` entry.

    Supports comma- or whitespace-separated entries with optional leading dots
    and ``*.`` wildcards, which match both the apex domain and subdomains.
    """
    raw = no_proxy_value
    if raw is None:
        raw = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""

    raw = raw.strip()
    if not raw:
        return False

    lower_hostname = hostname.lower()
    for entry in re.split(r"[\s,]+", raw):
        normalized = entry.strip().lower()
        if not normalized:
            continue
        if normalized == "*":
            return True

        if normalized.startswith("*."):
            normalized = normalized[2:]
        elif normalized.startswith("."):
            normalized = normalized[1:]

        if lower_hostname == normalized or lower_hostname.endswith(f".{normalized}"):
            return True

    return False
