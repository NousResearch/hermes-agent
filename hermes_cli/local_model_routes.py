"""Classify model endpoints that must pass local-GPU admission."""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any
from urllib.request import getproxies, proxy_bypass
from urllib.parse import urlsplit

_LOCAL_ROUTE_DNS_TIMEOUT_SECONDS = 0.35
_LOCAL_ROUTE_DNS_SLOTS = threading.BoundedSemaphore(2)
_LOCAL_PROVIDER_NAMES = ("ollama", "llama.cpp", "llamacpp", "lmstudio", "mlx", "vllm")
_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".lan", ".internal", ".home.arpa")
_SHARED_ADDRESS_SPACE = ipaddress.IPv4Network("100.64.0.0/10")


def _address_is_local(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (address.is_private or address.is_loopback or address.is_link_local
            or (isinstance(address, ipaddress.IPv4Address) and address in _SHARED_ADDRESS_SPACE))


def _host_resolves_to_local_address(host: str) -> bool:
    """Bound DNS work and deny on resolver failure, timeout, or an unknown answer."""
    if not _LOCAL_ROUTE_DNS_SLOTS.acquire(blocking=False):
        return True
    finished = threading.Event()
    outcome: dict[str, Any] = {}

    def resolve() -> None:
        try:
            outcome["addresses"] = {
                item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
                if item and len(item) > 4 and item[4]
            }
        except OSError as exc:
            outcome["error"] = exc
        finally:
            _LOCAL_ROUTE_DNS_SLOTS.release()
            finished.set()

    from agent.memory_provider import spawn_context_thread

    spawn_context_thread(resolve, name="local-route-dns-check", daemon=True).start()
    if not finished.wait(_LOCAL_ROUTE_DNS_TIMEOUT_SECONDS):
        return True
    addresses = outcome.get("addresses") or ()
    if outcome.get("error") or not addresses:
        return True
    for value in addresses:
        try:
            if _address_is_local(ipaddress.ip_address(value.split("%", 1)[0])):
                return True
        except ValueError:
            return True
    return False


def _is_known_official_provider_host(provider: str, host: str) -> bool:
    """Skip DNS for a provider's registered endpoint; all custom FQDNs are resolved."""
    try:
        from hermes_cli.providers import get_provider

        definition = get_provider(provider, allow_network=False)
        official_host = urlsplit(str(getattr(definition, "base_url", "") or "")).hostname
    except (ImportError, KeyError, TypeError, ValueError):
        return False
    if not official_host:
        return False
    official_host = official_host.lower().rstrip(".")
    host = host.lower().rstrip(".")
    return host == official_host or host.endswith("." + official_host)


def _configured_proxy_is_local(endpoint_scheme: str, endpoint_host: str) -> bool:
    """Treat a local HTTP client proxy as a local model route unless NO_PROXY bypasses it."""
    try:
        if proxy_bypass(endpoint_host):
            return False
        proxies = getproxies()
    except (OSError, ValueError):
        return True
    proxy = proxies.get(endpoint_scheme) or proxies.get("all")
    if not proxy:
        return False
    try:
        parsed_proxy = urlsplit(proxy if "://" in proxy else f"//{proxy}")
        proxy_host = (parsed_proxy.hostname or "").lower().rstrip(".")
    except ValueError:
        return True
    if not proxy_host:
        return True
    if proxy_host == "localhost" or proxy_host.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    try:
        return _address_is_local(ipaddress.ip_address(proxy_host))
    except ValueError:
        return "." not in proxy_host or _host_resolves_to_local_address(proxy_host)


def is_local_llm_route(provider: Any, base_url: Any) -> bool:
    """Identify local/private model endpoints conservatively for fail-closed admission."""
    provider_name = str(provider or "").strip().lower()
    if any(name in provider_name for name in _LOCAL_PROVIDER_NAMES):
        return True
    try:
        parsed_endpoint = urlsplit(str(base_url or ""))
        host = (parsed_endpoint.hostname or "").lower().rstrip(".")
    except ValueError:
        return True
    if not host:
        return False
    if host == "localhost" or host.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    if parsed_endpoint.scheme in {"http", "https"} and _configured_proxy_is_local(
        parsed_endpoint.scheme, host,
    ):
        return True
    try:
        return _address_is_local(ipaddress.ip_address(host))
    except ValueError:
        if "." not in host:
            return True
        if _is_known_official_provider_host(provider_name, host):
            return False
        return _host_resolves_to_local_address(host)
