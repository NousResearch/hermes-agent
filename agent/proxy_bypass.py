"""NO_PROXY matching shared by the LLM transport (``agent/process_bootstrap.py``) and the
gateway platform adapters (``gateway/platforms/base.py``).

One matcher so "is this host in NO_PROXY" has one answer everywhere: exact hosts, domain
suffixes (``example.com``, ``.example.com``, ``*.example.com``), IP literals, CIDR ranges,
optional ``host:port`` entries and ``*``. The stdlib ``proxy_bypass_environment`` understands
none of the CIDR / ``*.`` forms, which is why the LLM path used to route ``10.x`` endpoints
through the corporate proxy while Telegram/Discord bypassed it.

Leaf module: stdlib only, importable during early boot.
"""

from __future__ import annotations

import ipaddress
import os
import re
from urllib.parse import urlsplit

PROXY_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


def first_proxy_env_value() -> str:
    """First non-empty HTTPS_PROXY / HTTP_PROXY / ALL_PROXY value (any case), or ''."""
    return next((v for k in PROXY_ENV_KEYS if (v := (os.environ.get(k) or "").strip())), "")


def split_host_port(value: str) -> tuple[str, int | None]:
    """``(host, port)`` from a URL (scheme optional: ``//host/path``), ``[v6]:port``,
    ``host:port`` or bare host; host lowercased."""
    raw = str(value or "").strip()
    if not raw:
        return "", None
    if "://" in raw or raw.startswith("//"):
        parsed = urlsplit(raw)
        host, port = parsed.hostname or "", parsed.port
    elif raw.startswith("[") and "]" in raw:
        host, _, rest = raw[1:].partition("]")
        port = int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
    elif raw.count(":") == 1 and raw.rpartition(":")[2].isdigit():
        host, _, port_s = raw.rpartition(":")
        port = int(port_s)
    else:
        host, port = raw.strip("[]"), None
    return host.lower().rstrip("."), port


def no_proxy_entries(no_proxy_value: str | None = None) -> list[str]:
    """Comma/whitespace-separated NO_PROXY entries; from the environment (both casings) when
    ``no_proxy_value`` is None."""
    if no_proxy_value is None:
        no_proxy_value = ",".join(os.environ.get(key, "") for key in ("NO_PROXY", "no_proxy"))
    return [part for part in re.split(r"[\s,]+", no_proxy_value.strip()) if part]


def _ip_or_none(value: str, parse=ipaddress.ip_address):
    """``parse(value)`` or None on ``ValueError`` (``parse`` is ip_address / ip_network)."""
    try:
        return parse(value)
    except ValueError:
        return None


def no_proxy_entry_matches(entry: str, host: str, port: int | None = None) -> bool:
    token = str(entry or "").strip().lower()
    if not token:
        return False
    if token == "*":
        return True
    token_host, token_port = split_host_port(token)
    if not token_host or (token_port is not None and (port is None or token_port != port)):
        return False
    host_ip = _ip_or_none(host)
    network = _ip_or_none(token_host, lambda v: ipaddress.ip_network(v, strict=False))
    if network is not None:  # CIDR or bare IP literal (a /32 / /128 network)
        return host_ip is not None and host_ip in network
    # ``*.example.com`` and ``.example.com`` both mean apex + subdomains (curl/requests
    # convention, and what is_host_excluded_by_no_proxy promised the Slack adapter).
    suffix = token_host.removeprefix("*").removeprefix(".")
    return host == suffix or host.endswith(f".{suffix}")


def should_bypass_proxy(
    target_hosts: str | list[str] | tuple[str, ...] | set[str] | None, *, no_proxy_value: str | None = None,
) -> bool:
    """True when NO_PROXY (the environment, or ``no_proxy_value``) matches at least one target
    host (a URL, ``host:port`` or bare host)."""
    entries = no_proxy_entries(no_proxy_value)
    if not entries or not target_hosts:
        return False
    candidates = [target_hosts] if isinstance(target_hosts, str) else list(target_hosts)
    return any(
        host and any(no_proxy_entry_matches(entry, host, port) for entry in entries)
        for host, port in map(split_host_port, map(str, candidates)))
