"""Network helpers for the Desktop setup flow: public-IP detection and DNS A
record verification. Pure logic with injectable seams so tests never touch the
network or shell."""
from __future__ import annotations

import ipaddress
import logging
import re
import shutil
import socket
import subprocess
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Ordered public-IP echo services; the first valid IPv4 answer wins.
IP_ECHO_SERVICES = ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com")
#: Public resolvers consulted for A-record verification (besides the system one).
VERIFY_RESOLVERS = ("1.1.1.1", "8.8.8.8")

_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _valid_ipv4(text: str) -> Optional[str]:
    try:
        addr = ipaddress.IPv4Address(text.strip())
        return str(addr)
    except (ipaddress.AddressValueError, ValueError):
        return None


def public_ip(*, fetch: Optional[Callable[[str], str]] = None, timeout: float = 6.0) -> Optional[str]:
    """This machine's public IPv4 as seen from the internet, or None.

    ``fetch(url) -> body`` is injectable for tests; the default uses urllib with
    a short timeout per service. A VPS behind NAT may see a different address
    than its interface — that is exactly the address a DNS A record must name.
    """
    if fetch is None:
        import urllib.request

        def _default_fetch(url: str) -> str:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read(256).decode("utf-8", "replace")

        fetch = _default_fetch

    for url in IP_ECHO_SERVICES:
        try:
            candidate = _valid_ipv4(fetch(url) or "")
            if candidate:
                return candidate
        except Exception as exc:  # health: allow BLE001 -- any probe failure falls through to the next service
            logger.debug("public-IP probe %s failed: %s", url, exc)
    return None


def _dig_a(host: str, resolver: Optional[str], *, run: Optional[Callable] = None) -> list[str]:
    """A records via ``dig +short`` (optionally at a specific resolver)."""
    if shutil.which("dig") is None:
        return []
    cmd = ["dig", "+short", "+time=3", "+tries=1"]
    if resolver:
        cmd.append(f"@{resolver}")
    cmd += [host, "A"]
    runner = run or subprocess.run
    try:
        result = runner(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("dig %s failed: %s", host, exc)
        return []
    answers = []
    for line in (result.stdout or "").splitlines():
        line = line.strip().rstrip(".")
        if _IPV4_RE.match(line) and _valid_ipv4(line):
            answers.append(line)
    return answers


def _system_a(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_INET)
        return sorted({str(info[4][0]) for info in infos})
    except OSError:
        return []


def resolve_a(host: str, *, resolvers: tuple = VERIFY_RESOLVERS,
              run: Optional[Callable] = None) -> dict[str, list[str]]:
    """A records per source: each public resolver plus the system resolver.
    Returns ``{source: [ips]}``; sources with no answer are omitted."""
    seen: dict[str, list[str]] = {}
    for resolver in resolvers:
        answers = _dig_a(host, resolver, run=run)
        if answers:
            seen[resolver] = answers
    system = _system_a(host)
    if system:
        seen["system"] = system
    return seen


def verify_a_record(host: str, expected_ip: str, *, resolvers: tuple = VERIFY_RESOLVERS,
                    run: Optional[Callable] = None) -> tuple[bool, dict[str, list[str]]]:
    """True when ANY consulted resolver returns the expected IPv4 for *host*.

    One matching answer is enough: propagation is cache expiry, and the user's
    own resolver may lag a public one. The per-source map is returned so the
    wizard can show what each resolver currently sees.
    """
    expected = _valid_ipv4(expected_ip)
    if expected is None:
        return False, {}
    seen = resolve_a(host, resolvers=resolvers, run=run)
    ok = any(expected in answers for answers in seen.values())
    return ok, seen


def valid_hostname(name: str) -> bool:
    """A plain DNS hostname (no scheme, no wildcard, no trailing dot, labels
    1-63 chars, total <= 253). ``.local`` names are accepted here; the flow
    decides which mode they belong to."""
    name = (name or "").strip().lower()
    if not name or len(name) > 253 or name.endswith("."):
        return False
    if "://" in name or "/" in name or "*" in name:
        return False
    labels = name.split(".")
    if len(labels) < 2:
        return False
    return all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)


def is_local_name(name: str) -> bool:
    """mDNS single-label ``<name>.local`` (RFC 6762 names are one label + .local)."""
    name = (name or "").strip().lower()
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.local", name))