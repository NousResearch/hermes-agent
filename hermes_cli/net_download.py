"""Network-fetch helpers with proxy detection and mirror fallback.

Hermes occasionally needs to fetch small files from the internet (the
cua-driver install script, release tarballs, model metadata, …). On many
machines those fetches sit behind a corporate/NAT proxy, and on others the
official endpoints (raw.githubusercontent.com, objects.githubusercontent.com)
are slow or blocked — China-mainland networks being the common case.

This module gives every such fetch the same resilient strategy:

1. **Explicit proxy** — the caller's environment (``HTTPS_PROXY`` /
   ``HTTP_PROXY``) wins when set.
2. **System proxy** — on macOS, fall back to the system-configured HTTP(S)
   proxy (Surge, Clash, …) read via ``scutil --proxy``, so a proxy that the
   user turned on in System Settings actually reaches the child curl without
   them having to export anything.
3. **Direct** — no proxy detected: try the official URL as-is.
4. **Mirror fallback** — if the official URL fails (DNS/TLS/HTTP error),
   retry through community GitHub mirrors (``ghfast.top``,
   ``gh-proxy.com``) that prefix-wrap the original URL. Mirrors are only
   tried after the official path fails, so users with working direct access
   are never redirected.
5. **DNS-pollution fallback** — on some networks (China mainland being the
   common case) the system DNS returns poisoned A records for hosts such as
   ``huggingface.co`` / ``hf-mirror.com``, so the official URL fails with a
   DNS/connect error even though the site itself is reachable. When a fetch
   fails that way, we re-resolve the host through the DNSPod DNS-over-HTTPS
   endpoint (``doh.pub``, reachable directly from CN) and retry the **same
   official URL** with ``curl --resolve <host>:<port>:<ip>``. This only swaps
   the resolved IP — the URL, TLS hostname check and content source stay
   official, so it is supply-chain-safe for executed content too (unlike
   mirrors, which are third-party and stay opt-in/disabled for executed
   content).

Design notes:

* No ``shell=True`` anywhere; every subprocess call is an argv list.
* Proxy env is injected into the *child* env, never mutated on the parent.
* Mirrors are opt-in by default via ``allow_mirrors`` — callers that
  download content which will later be executed (install scripts, pinned
  binaries) must never let a third-party mirror supply it unless they
  explicitly opt in; the default keeps mirrors off.

Content-class contract
^^^^^^^^^^^^^^^^^^^^^^

Every fetch declares the *class* of the content it transports via
``content_class``. The class decides whether mirrors may ever be tried —
**independently of what the caller happens to pass** — so a future call site
cannot accidentally re-enable mirrors for executed content:

* ``content_class="executed"`` (default) — bytes that will be executed or
  imported (shell scripts, binaries, wheels). Mirrors are **always disabled**;
  passing ``allow_mirrors=True`` logs a warning and is ignored.
* ``content_class="data"`` — non-executed payloads (model weights, metadata).
  Mirrors remain opt-in via ``allow_mirrors=True``; the default stays off.

* All functions are pure-ish and take an explicit ``env`` so tests can
  inject a fake environment and a fake ``curl`` via ``curl_cmd``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Community GitHub mirrors used as fallbacks. Each is a prefix wrapper:
# https://ghfast.top/https://github.com/…  (also works for
# https://raw.githubusercontent.com/… and release-assets URLs).
_DEFAULT_MIRRORS: Tuple[str, ...] = (
    "https://ghfast.top/",
    "https://gh-proxy.com/",
)

_PROXY_ENV_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")

# DNS-over-HTTPS endpoint used by the DNS-pollution fallback. DNSPod is
# reachable directly from mainland China (unlike Cloudflare/Google DoH),
# so it works even when every public resolver is returning poisoned
# answers for the target host.
_DOH_ENDPOINT = "https://doh.pub/dns-query"
_DOH_ACCEPT_HEADER = "accept: application/dns-json"

# Hosts whose A records are commonly poisoned on CN networks. A failed
# fetch for one of these always triggers the DoH fallback, without
# needing to pattern-match the error text.
_POLLUTED_HOSTS: Tuple[str, ...] = ("huggingface.co", "hf-mirror.com")

# Substrings in curl's stderr that indicate a DNS/connect-layer failure —
# the only failure classes the DoH fallback can plausibly fix. HTTP-level
# errors (404, 403, …) never trigger it, so a real server error is not
# masked by an extra resolution round-trip.
_DNS_ERROR_MARKERS: Tuple[str, ...] = (
    "could not resolve host",
    "failed to connect",
    "connection refused",
    "no route to host",
    "name or service not known",
    "temporary failure in name resolution",
)

# Upper bound on IPs tried during one DoH fallback.
_MAX_DOH_IPS = 4


def _env_value(env: Dict[str, str], key: str) -> str:
    return env.get(key, "").strip()


def explicit_proxy(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Return the caller's explicit proxy URL, or ``None``.

    ``HTTPS_PROXY`` (case-insensitive) wins over ``HTTP_PROXY``; either is
    accepted because the endpoints we fetch are HTTPS anyway and many users
    only set one. ``ALL_PROXY`` is the last resort.
    """
    env = env if env is not None else dict(os.environ)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = _env_value(env, key)
        if value:
            return value
    return None


def _macos_system_proxy() -> Optional[str]:
    """Read the macOS system HTTP(S) proxy via ``scutil --proxy``.

    Returns the HTTPS proxy (or HTTP proxy as a fallback) as
    ``http://host:port``. Returns ``None`` when the proxy is disabled or
    ``scutil`` is unavailable (non-macOS, container, missing tool).
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["scutil", "--proxy"],
            capture_output=True, text=True, timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    # Collect key → value pairs first, then assemble host+port. The key
    # order in `scutil --proxy` output is not guaranteed, so a port may
    # precede its host line.
    values: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        values[key.strip()] = raw.strip()
    if values.get("HTTPSEnable") == "1" and values.get("HTTPSProxy"):
        return _normalize_proxy_url(f"{values['HTTPSProxy']}:{values.get('HTTPSPort', '')}" if values.get("HTTPSPort") else values["HTTPSProxy"])
    if values.get("HTTPEnable") == "1" and values.get("HTTPProxy"):
        return _normalize_proxy_url(f"{values['HTTPProxy']}:{values.get('HTTPPort', '')}" if values.get("HTTPPort") else values["HTTPProxy"])
    return None


def _normalize_proxy_url(raw: str) -> str:
    """Prefix ``http://`` when a proxy host is bare (``127.0.0.1:6152``)."""
    raw = raw.strip()
    if not raw:
        return raw
    if re.match(r"^https?://", raw, re.IGNORECASE):
        return raw
    return f"http://{raw}"


def detect_proxy(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Resolve the effective proxy for a fetch.

    Priority: explicit env var → macOS system proxy → ``None``.
    """
    env = env if env is not None else dict(os.environ)
    explicit = explicit_proxy(env)
    if explicit:
        return explicit
    return _macos_system_proxy()


def proxy_env_for(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return a copy of ``env`` with the detected proxy injected.

    Never mutates the caller's dict. When no proxy is found the copy is
    returned unchanged (so child processes still inherit whatever the
    parent shell exported).
    """
    base = dict(env) if env is not None else dict(os.environ)
    proxy = detect_proxy(base)
    if proxy:
        base.setdefault("HTTPS_PROXY", proxy)
        base.setdefault("HTTP_PROXY", proxy)
    return base


def mirror_candidates(url: str, mirrors: Sequence[str] = _DEFAULT_MIRRORS) -> List[str]:
    """Return mirror-wrapped copies of a GitHub URL.

    ``ghfast.top``/``gh-proxy.com`` prefix-wrap any github.com /
    raw.githubusercontent.com URL. Non-GitHub URLs return ``[]``.
    """
    if not re.match(r"^https://(github\.com|raw\.githubusercontent\.com|objects\.githubusercontent\.com|release-assets\.githubusercontent\.com)/", url):
        return []
    return [f"{mirror.rstrip('/')}/{url}" for mirror in mirrors]


def _host_from_url(url: str) -> Optional[str]:
    """Return the host of an http(s) URL, or ``None`` when unparsable."""
    match = re.match(r"^https?://([^/:]+)", url)
    return match.group(1) if match else None


def _port_from_url(url: str) -> str:
    """Return the port to use in ``--resolve`` for ``url``."""
    match = re.match(r"^https?://[^/:]+:(\d+)", url)
    if match:
        return match.group(1)
    return "443" if url.startswith("https") else "80"


def _is_pollution_candidate(host: str) -> bool:
    """True for hosts whose DNS answers are commonly poisoned on CN networks."""
    return host in _POLLUTED_HOSTS


def _looks_like_dns_or_connect_error(detail: str) -> bool:
    """True when a curl failure looks like a DNS/connect-layer problem.

    The DoH fallback can only fix resolution/connectivity failures; an
    HTTP-level error (404/403) is a real server response and must not be
    retried through a different resolver.
    """
    low = detail.lower()
    return any(marker in low for marker in _DNS_ERROR_MARKERS)


def _should_try_doh_fallback(url: str, detail: str) -> bool:
    """Decide whether a failed fetch is worth a DoH re-resolution.

    Known-polluted hosts always qualify; other hosts only when the error
    text indicates a DNS/connect-layer failure.
    """
    host = _host_from_url(url)
    if not host:
        return False
    if _is_pollution_candidate(host):
        return True
    return _looks_like_dns_or_connect_error(detail)


def _is_valid_ipv4(data: str) -> bool:
    """True only for syntactically *and* numerically valid IPv4 addresses.

    A pure format regex (``\\d{1,3}(\\.\\d{1,3}){3}``) would accept
    ``999.999.999.999``; the numeric range check rejects out-of-range
    octets so a poisoned/malformed answer can never be used for ``--resolve``.
    """
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", data):
        return False
    return all(0 <= int(part) <= 255 for part in data.split("."))


def resolve_dns_doh(
    host: str,
    *,
    timeout: int = 5,
    env: Optional[Dict[str, str]] = None,
    curl_cmd: Optional[str] = None,
) -> List[str]:
    """Resolve ``host`` A records via DNSPod DNS-over-HTTPS.

    Returns a list of IPv4 addresses (``[]`` on any failure — curl missing,
    non-zero exit, timeout, unparsable JSON, no A records). The query is
    intentionally sent **without** proxy injection: ``doh.pub`` is reachable
    directly from mainland China, and a proxy rule must never be able to
    intercept the resolution that exists to bypass poisoned answers.
    """
    curl = curl_cmd or shutil.which("curl")
    if not curl or not host:
        return []
    args = [
        curl, "-fsSL",
        "--connect-timeout", "5",
        "--max-time", str(timeout),
        "-H", _DOH_ACCEPT_HEADER,
        f"{_DOH_ENDPOINT}?name={host}&type=A",
    ]
    child_env = dict(env) if env is not None else dict(os.environ)
    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout + 5,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError:
        return []
    ips: List[str] = []
    for answer in payload.get("Answer", []) or []:
        data = answer.get("data", "") if isinstance(answer, dict) else ""
        # Only valid IPv4 A records; ignore AAAA / CNAME / malformed
        # (including out-of-range octets like 999.x.x.x).
        if answer.get("type") == 1 and _is_valid_ipv4(data):
            ips.append(data)
    return ips


def _retry_with_doh_resolve(
    url: str,
    dest: str,
    *,
    timeout: int,
    env: Dict[str, str],
    curl: str,
    extra_args: Sequence[str],
) -> Tuple[bool, str]:
    """Retry ``url`` through ``--resolve`` with DoH-obtained real IPs.

    Returns ``(True, '')`` when one of the IPs connects and the download
    lands; ``(False, '')`` when nothing could be attempted or every IP
    failed. The caller keeps its original failure detail when we return
    ``False`` so a failed fallback never hides the root cause.
    """
    host = _host_from_url(url)
    if not host:
        return False, ""
    port = _port_from_url(url)
    ips = resolve_dns_doh(host, timeout=5, env=env, curl_cmd=curl)
    if not ips:
        return False, ""
    for ip in ips[:_MAX_DOH_IPS]:
        args = [
            curl, "-fsSL",
            "--connect-timeout", "10",
            "--max-time", str(timeout),
            "--resolve", f"{host}:{port}:{ip}",
            "-o", dest,
            url,
        ]
        if extra_args:
            args = args[:1] + list(extra_args) + args[1:]
        try:
            result = subprocess.run(
                args,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout + 10,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(
                "curl_download: DNS-pollution fallback succeeded for %s via %s", host, ip
            )
            return True, ""
    return False, ""


def curl_download(
    url: str,
    dest: str,
    *,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
    curl_cmd: Optional[str] = None,
    extra_args: Sequence[str] = (),
    dns_fallback: bool = True,
) -> Tuple[bool, str]:
    """Download ``url`` to ``dest`` with curl.

    Returns ``(ok, detail)`` where ``detail`` is the error text on failure.
    Proxy is resolved from ``env`` (explicit env var or macOS system proxy)
    and injected into the child environment — a proxy the user enabled in
    System Settings reaches the fetch without any manual export.

    When ``dns_fallback`` is enabled (default) and the direct attempt fails
    with a DNS/connect-layer error (or targets a known DNS-polluted host),
    the host is re-resolved through DNSPod DoH and the same official URL is
    retried with ``--resolve``. The fallback never changes the URL or the
    TLS identity — it only swaps the resolved IP — so the fetched bytes are
    still from the official server.
    """
    curl = curl_cmd or shutil.which("curl")
    if not curl:
        return False, "curl not found on PATH"
    child_env = proxy_env_for(env)
    # --connect-timeout fails fast when the host is unreachable (blocked
    # DNS/TLS), so a down official endpoint doesn't burn the full
    # --max-time before the mirror fallback kicks in.
    args = [curl, "-fsSL", "--connect-timeout", "10", "--max-time", str(timeout), "-o", dest, url]
    if extra_args:
        args = args[:1] + list(extra_args) + args[1:]
    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout + 10,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if dns_fallback and _should_try_doh_fallback(url, detail):
            ok, _ = _retry_with_doh_resolve(
                url, dest, timeout=timeout, env=child_env,
                curl=curl, extra_args=extra_args,
            )
            if ok:
                return True, ""
            # DoH resolution or every IP failed — keep the original error
            # so the root cause is never masked by the fallback.
        return False, detail[:300] or f"curl exit {result.returncode}"
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        return False, "empty or missing output file"
    return True, ""


def fetch_with_fallback(
    url: str,
    dest: str,
    *,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
    content_class: str = "executed",
    allow_mirrors: Optional[bool] = None,
    curl_cmd: Optional[str] = None,
    dns_fallback: bool = True,
) -> Tuple[bool, str]:
    """Download with the proxy-aware official-then-mirror strategy.

    ``content_class`` is the security contract (see module docstring):
    ``"executed"`` (default) permanently disables mirrors — a third-party
    mirror must never supply bytes that will be executed or imported, so an
    accidental ``allow_mirrors=True`` is warned about and ignored. ``"data"``
    keeps mirrors strictly opt-in via ``allow_mirrors=True``.

    Tries the official URL first (with proxy), then a DNS-pollution
    ``--resolve`` retry when the failure looks like poisoned DNS (both
    inside :func:`curl_download`), then each mirror candidate in order
    (only when the content class + allow_mirrors permit). Returns
    ``(True, '')`` on success or ``(False, detail)`` with a summary of every
    attempt.
    """
    if content_class == "executed" and allow_mirrors:
        logger.warning(
            "fetch_with_fallback: allow_mirrors=True ignored for content_class='executed' "
            "(supply-chain safety: mirrors must never supply executed content)"
        )
        allow_mirrors = False
    if allow_mirrors is None:
        allow_mirrors = False

    attempts: List[Tuple[str, str]] = []
    ok, detail = curl_download(url, dest, timeout=timeout, env=env, curl_cmd=curl_cmd,
                               dns_fallback=dns_fallback)
    if ok:
        return True, ""
    attempts.append((url, detail))

    if allow_mirrors:
        for mirror_url in mirror_candidates(url):
            ok, detail = curl_download(mirror_url, dest, timeout=timeout, env=env, curl_cmd=curl_cmd,
                                       dns_fallback=dns_fallback)
            if ok:
                logger.info("fetch_with_fallback: official URL failed; mirror succeeded: %s", mirror_url)
                return True, ""
            attempts.append((mirror_url, detail))
    summary = "; ".join(f"{u}: {d}" for u, d in attempts)
    return False, summary
