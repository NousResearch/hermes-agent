"""Network accessibility and proxy helpers for gateway platform adapters."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import socket as _socket
import subprocess
import sys
from typing import Callable

from agent.proxy_bypass import first_proxy_env_value, should_bypass_proxy as _should_bypass_proxy
from utils import normalize_proxy_url

logger = logging.getLogger(__name__)


def is_network_accessible(host: str) -> bool:
    """True if *host* would expose the server beyond loopback.

    IPv4-mapped loopback addresses such as ``::ffff:127.0.0.1`` are treated as
    loopback. Hostnames are resolved; DNS failure fails closed as public-facing.
    """
    with contextlib.suppress(ValueError):
        addr = ipaddress.ip_address(host)
        mapped = getattr(addr, "ipv4_mapped", None)
        return not (addr.is_loopback or (mapped and mapped.is_loopback))
    try:
        resolved = _socket.getaddrinfo(host, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM)
        return any(not ipaddress.ip_address(sockaddr[0]).is_loopback for *_, sockaddr in resolved)
    except (_socket.gaierror, OSError):
        return True


def detect_macos_system_proxy() -> str | None:
    """Return the macOS system HTTP(S) proxy URL when one is enabled."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["scutil", "--proxy"],
            timeout=3,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    props = {
        key.strip(): val.strip()
        for key, sep, val in (line.strip().partition(" : ") for line in out.splitlines())
        if sep
    }
    for enable_key, host_key, port_key in (
        ("HTTPSEnable", "HTTPSProxy", "HTTPSPort"),
        ("HTTPEnable", "HTTPProxy", "HTTPPort"),
    ):
        if props.get(enable_key) == "1" and props.get(host_key) and props.get(port_key):
            return f"http://{props[host_key]}:{props[port_key]}"
    return None


def should_bypass_proxy(target_hosts: str | list[str] | tuple[str, ...] | set[str] | None) -> bool:
    """True when NO_PROXY/no_proxy matches at least one target host."""
    return _should_bypass_proxy(target_hosts)


def _config_section(name: str) -> dict:
    """Read-only ``config.yaml`` section ``name``; ``{}`` when unavailable."""
    try:
        from hermes_cli.config import load_config_readonly as _load_config

        cfg = _load_config()
    except Exception:
        return {}
    section = cfg.get(name) if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def gateway_trust_env() -> bool:
    """Whether gateway HTTP clients should honor inherited proxy/cert env vars."""
    value = _config_section("gateway").get("trust_env", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value) if value is not None else True


def resolve_proxy_url(
    platform_env_var: str | None = None,
    *,
    target_hosts: str | list[str] | tuple[str, ...] | set[str] | None = None,
    configured: str | None = None,
    trust_env_fn: Callable[[], bool] = gateway_trust_env,
    system_proxy_fn: Callable[[], str | None] = detect_macos_system_proxy,
) -> str | None:
    """Resolve a proxy URL for an adapter, honoring profile-scoped platform vars."""
    from gateway.platforms._shared import get_scoped_secret as _get_scoped_proxy_var

    value = (_get_scoped_proxy_var(platform_env_var, "") or "").strip() if platform_env_var else ""
    if not value:
        value = str(configured or "").strip()
    if not value:
        if not trust_env_fn():
            return None
        value = first_proxy_env_value()
    proxy = normalize_proxy_url(value or system_proxy_fn())
    return None if proxy and should_bypass_proxy(target_hosts) else proxy


def aiohttp_socks_connector(proxy_url: str):
    """Return an ``aiohttp_socks.ProxyConnector`` for ``proxy_url`` when available."""
    try:
        from aiohttp_socks import ProxyConnector

        return ProxyConnector.from_url(proxy_url, rdns=True)
    except ImportError:
        if proxy_url.lower().startswith("socks"):
            logger.warning(
                "aiohttp_socks not installed — SOCKS proxy %s ignored. Run: pip install aiohttp-socks",
                proxy_url,
            )
        return None


def proxy_kwargs_for_bot(proxy_url: str | None) -> dict:
    """Kwargs for ``commands.Bot()`` / ``discord.Client()``."""
    if not proxy_url:
        return {}
    if proxy_url.lower().startswith("socks"):
        connector = aiohttp_socks_connector(proxy_url)
        return {"connector": connector} if connector is not None else {}
    return {"proxy": proxy_url}


def proxy_kwargs_for_aiohttp(proxy_url: str | None) -> tuple[dict, dict]:
    """Return ``(session_kwargs, request_kwargs)`` for ``aiohttp.ClientSession``."""
    if not proxy_url:
        return {}, {}
    connector = aiohttp_socks_connector(proxy_url)
    if connector is not None:
        return {"connector": connector}, {}
    return ({}, {}) if proxy_url.lower().startswith("socks") else ({}, {"proxy": proxy_url})


def is_host_excluded_by_no_proxy(hostname: str, no_proxy_value: str | None = None) -> bool:
    """Return True when ``hostname`` matches a ``NO_PROXY`` entry."""
    return _should_bypass_proxy(hostname, no_proxy_value=no_proxy_value)
