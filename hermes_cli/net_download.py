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

Design notes:

* No ``shell=True`` anywhere; every subprocess call is an argv list.
* Proxy env is injected into the *child* env, never mutated on the parent.
* Mirrors are opt-in by default via ``allow_mirrors`` — installers that
  must not be redirected (e.g. anything checksum-pinned) can disable them.
* All functions are pure-ish and take an explicit ``env`` so tests can
  inject a fake environment and a fake ``curl`` via ``curl_cmd``.
"""

from __future__ import annotations

import logging
import os
import platform
import re
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


def curl_download(
    url: str,
    dest: str,
    *,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
    curl_cmd: Optional[str] = None,
    extra_args: Sequence[str] = (),
) -> Tuple[bool, str]:
    """Download ``url`` to ``dest`` with curl.

    Returns ``(ok, detail)`` where ``detail`` is the error text on failure.
    Proxy is resolved from ``env`` (explicit env var or macOS system proxy)
    and injected into the child environment — a proxy the user enabled in
    System Settings reaches the fetch without any manual export.
    """
    import shutil

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
    allow_mirrors: bool = True,
    curl_cmd: Optional[str] = None,
) -> Tuple[bool, str]:
    """Download with the proxy-aware official-then-mirror strategy.

    Tries the official URL first (with proxy), then each mirror candidate
    in order. Returns ``(True, '')`` on success or ``(False, detail)`` with
    a summary of every attempt.
    """
    attempts: List[Tuple[str, str]] = []
    ok, detail = curl_download(url, dest, timeout=timeout, env=env, curl_cmd=curl_cmd)
    if ok:
        return True, ""
    attempts.append((url, detail))

    if allow_mirrors:
        for mirror_url in mirror_candidates(url):
            ok, detail = curl_download(mirror_url, dest, timeout=timeout, env=env, curl_cmd=curl_cmd)
            if ok:
                logger.info("fetch_with_fallback: official URL failed; mirror succeeded: %s", mirror_url)
                return True, ""
            attempts.append((mirror_url, detail))
    summary = "; ".join(f"{u}: {d}" for u, d in attempts)
    return False, summary
