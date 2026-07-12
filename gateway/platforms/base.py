"""
Base platform adapter interface.

All platform adapters (Telegram, Discord, WhatsApp, Weixin, and more) inherit from this
and implement the required methods.
"""

import asyncio
import inspect
import ipaddress
import logging
import os
import random
import re
import socket as _socket
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from urllib.parse import urlsplit

from utils import normalize_proxy_url

logger = logging.getLogger(__name__)

# Audio file extensions Hermes recognizes for native audio delivery.
# Kept in sync with tools/send_message_tool.py and cron/scheduler.py via
# should_send_media_as_audio() below.
_AUDIO_EXTS = frozenset({'.ogg', '.opus', '.mp3', '.wav', '.m4a', '.flac'})
# Telegram's Bot API sendAudio only accepts MP3 / M4A. Other audio
# formats either need to go through sendVoice (Opus/OGG) or must be
# delivered as a regular document.
_TELEGRAM_AUDIO_ATTACHMENT_EXTS = frozenset({'.mp3', '.m4a'})
_TELEGRAM_VOICE_EXTS = frozenset({'.ogg', '.opus'})
_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS = 30.0


def _platform_name(platform) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Build platform-aware thread metadata for adapter sends.

    Most platforms route threaded sends with a generic ``thread_id`` metadata
    value. Telegram private-chat topics created through Hermes' DM-topic helper
    are exposed in updates as ``message_thread_id`` plus a reply anchor. Live
    user-message replies route with ``message_thread_id`` + ``reply_to_message_id``;
    synthetic/resumed sends that have no reply anchor fall back to Telegram's
    ``direct_messages_topic_id`` when the Bot API supports it.
    """
    thread_id = getattr(source, "thread_id", None)
    if thread_id is None:
        return None
    metadata = {"thread_id": thread_id}
    if _platform_name(getattr(source, "platform", None)) == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        tid = str(thread_id)
        if tid and tid not in {"", "1"}:
            metadata["direct_messages_topic_id"] = tid
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    return metadata


def _mark_notify_metadata(metadata: dict | None) -> dict:
    """Clone metadata and mark a user-visible reply as notify-worthy."""
    notify_metadata = dict(metadata) if metadata else {}
    notify_metadata["notify"] = True
    return notify_metadata


def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics.

    Telegram forum/supergroup topics should be routed by topic metadata, not by
    replying to the triggering message. Hermes-created Telegram private-chat
    topic lanes prefer replying to the triggering user message so the answer
    stays attached to the active lane; synthetic/resumed sends fall back to
    ``direct_messages_topic_id`` metadata when no message id is available.
    """
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None))
    thread_id = getattr(source, "thread_id", None)
    if platform == "telegram" and thread_id and getattr(source, "chat_type", None) == "dm":
        # Reply to the triggering user message. Replying to Telegram's earlier
        # topic seed/anchor can render the bot response outside the active lane.
        return getattr(event, "message_id", None) or getattr(event, "reply_to_message_id", None)
    if platform == "telegram" and thread_id:
        return None
    if platform == "feishu" and thread_id and getattr(event, "reply_to_message_id", None):
        return getattr(event, "reply_to_message_id", None)
    return getattr(event, "message_id", None)


def should_send_media_as_audio(platform, ext: str, is_voice: bool = False) -> bool:
    """Return True when a media file should use the platform's audio sender.

    Other platforms: every recognized audio extension routes through the
    audio sender.

    Telegram: the Bot API only accepts MP3/M4A for sendAudio and
    Opus/OGG for sendVoice. Opus/OGG is only routed as audio when the
    caller flagged ``is_voice=True`` (so we don't turn a regular audio
    attachment into a voice bubble just because the file happens to be
    Opus). Everything else falls through to document delivery by
    returning ``False``.
    """
    normalized_ext = (ext or "").lower()
    if normalized_ext not in _AUDIO_EXTS:
        return False
    if _platform_name(platform) == "telegram":
        if normalized_ext in _TELEGRAM_VOICE_EXTS:
            return is_voice
        return normalized_ext in _TELEGRAM_AUDIO_ATTACHMENT_EXTS
    return True


def utf16_len(s: str) -> int:
    """Count UTF-16 code units in *s*.

    Telegram's message-length limit (4 096) is measured in UTF-16 code units,
    **not** Unicode code-points.  Characters outside the Basic Multilingual
    Plane (emoji like 😀, CJK Extension B, musical symbols, …) are encoded as
    surrogate pairs and therefore consume **two** UTF-16 code units each, even
    though Python's ``len()`` counts them as one.

    Ported from nearai/ironclaw#2304 which discovered the same discrepancy in
    Rust's ``chars().count()``.
    """
    return len(s.encode("utf-16-le")) // 2


def _prefix_within_utf16_limit(s: str, limit: int) -> str:
    """Return the longest prefix of *s* whose UTF-16 length ≤ *limit*.

    Unlike a plain ``s[:limit]``, this respects surrogate-pair boundaries so
    we never slice a multi-code-unit character in half.
    """
    if utf16_len(s) <= limit:
        return s
    # Binary search for the longest safe prefix
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if utf16_len(s[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]


def _custom_unit_to_cp(s: str, budget: int, len_fn) -> int:
    """Return the largest codepoint offset *n* such that ``len_fn(s[:n]) <= budget``.

    Used by :meth:`BasePlatformAdapter.truncate_message` when *len_fn* measures
    length in units different from Python codepoints (e.g. UTF-16 code units).
    Falls back to binary search which is O(log n) calls to *len_fn*.
    """
    if len_fn(s) <= budget:
        return len(s)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def is_network_accessible(host: str) -> bool:
    """Return True if *host* would expose the server beyond loopback.

    Loopback addresses (127.0.0.1, ::1, IPv4-mapped ::ffff:127.0.0.1)
    are local-only.  Unspecified addresses (0.0.0.0, ::) bind all
    interfaces.  Hostnames are resolved; DNS failure fails closed.
    """
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback:
            return False
        # ::ffff:127.0.0.1 — Python reports is_loopback=False for mapped
        # addresses, so check the underlying IPv4 explicitly.
        if getattr(addr, "ipv4_mapped", None) and addr.ipv4_mapped.is_loopback:
            return False
        return True
    except ValueError:
        # when host variable is a hostname, we should try to resolve below
        pass

    try:
        resolved = _socket.getaddrinfo(
            host, None, _socket.AF_UNSPEC, _socket.SOCK_STREAM,
        )
        # if the hostname resolves into at least one non-loopback address,
        # then we consider it to be network accessible
        for _family, _type, _proto, _canonname, sockaddr in resolved:
            addr = ipaddress.ip_address(sockaddr[0])
            if not addr.is_loopback:
                return True
        return False
    except (_socket.gaierror, OSError):
        return True


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
            ["scutil", "--proxy"], timeout=3, text=True, stderr=subprocess.DEVNULL,
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


import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple, Union
from enum import Enum

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_default_hermes_root, get_hermes_dir, get_hermes_home


GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE = (
    "Secure secret entry is not supported over messaging. "
    "Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually."
)


def safe_url_for_log(url: str, max_len: int = 80) -> str:
    """Return a URL string safe for logs (no query/fragment/userinfo)."""
    if max_len <= 0:
        return ""

    if url is None:
        return ""

    raw = str(url)
    if not raw:
        return ""

    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw[:max_len]

    if parsed.scheme and parsed.netloc:
        # Strip potential embedded credentials (user:pass@host).
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        base = f"{parsed.scheme}://{netloc}"
        path = parsed.path or ""
        if path and path != "/":
            basename = path.rsplit("/", 1)[-1]
            safe = f"{base}/.../{basename}" if basename else f"{base}/..."
        else:
            safe = base
    else:
        safe = raw

    if len(safe) <= max_len:
        return safe
    if max_len <= 3:
        return "." * max_len
    return f"{safe[:max_len - 3]}..."


async def _ssrf_redirect_guard(response):
    """Re-validate each redirect target to prevent redirect-based SSRF.

    Without this, an attacker can host a public URL that 302-redirects to
    http://169.254.169.254/ and bypass the pre-flight is_safe_url() check.

    Must be async because httpx.AsyncClient awaits response event hooks.
    """
    from tools.url_safety import is_safe_url, redirect_target_from_response
    redirect_url = redirect_target_from_response(response)
    if redirect_url and not is_safe_url(redirect_url):
        raise ValueError(
            f"Blocked redirect to private/internal address: {safe_url_for_log(redirect_url)}"
        )


# ---------------------------------------------------------------------------
# Image cache utilities
#
# When users send images on messaging platforms, we download them to a local
# cache directory so they can be analyzed by the vision tool (which accepts
# local file paths). This avoids issues with ephemeral platform URLs
# (e.g. Telegram file URLs expire after ~1 hour).
# ---------------------------------------------------------------------------

# Import-time default. Tests monkeypatch this; the get_*_cache_dir() getters
# re-resolve per call so the active profile override is honored.
IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")


def _resolve_cache_dir(constant_name: str, new_subpath: str, old_name: str) -> Path:
    """Resolve fresh via get_hermes_dir (active profile), unless a test has
    monkeypatched the constant away from its import-time default."""
    fresh = get_hermes_dir(new_subpath, old_name)
    current = globals().get(constant_name)
    default = _CACHE_DIR_IMPORT_DEFAULTS.get(constant_name)
    if current is not None and default is not None and current != default:
        return Path(current)
    return fresh

# ---------------------------------------------------------------------------
# Inbound media size cap (#13145)
#
# Inbound image / audio / video payloads are buffered fully into process
# memory before being written to the cache directory. With no cap, a single
# large upload (Discord Nitro allows 500 MB) — or a remote URL in an inbound
# message payload pointing at an arbitrarily large file — can spike RAM and
# OOM-kill the gateway. The ``cache_*_from_bytes`` helpers (the shared funnel
# every platform reaches eventually) and the ``cache_*_from_url`` downloaders
# enforce this cap, so the protection holds regardless of which platform
# adapter or code path produced the bytes.
#
# Configurable via ``gateway.max_inbound_media_bytes`` in config.yaml.
# ``0`` disables the cap. Default 128 MiB — generous enough for ordinary
# photos/voice notes/short clips while still bounding a hostile upload.
# ---------------------------------------------------------------------------
DEFAULT_INBOUND_MEDIA_MAX_BYTES = 128 * 1024 * 1024


def get_inbound_media_max_bytes() -> int:
    """Return the max inbound image/audio/video bytes allowed in memory.

    Reads ``gateway.max_inbound_media_bytes`` from config.yaml. ``0`` (or a
    negative / unparseable value) disables the cap. Non-fatal if config is
    unreadable — falls back to the default.
    """
    try:
        from hermes_cli.config import load_config as _load_config
        cfg = _load_config()
    except Exception:
        return DEFAULT_INBOUND_MEDIA_MAX_BYTES
    gw = cfg.get("gateway", {}) if isinstance(cfg, dict) else {}
    if not isinstance(gw, dict) or "max_inbound_media_bytes" not in gw:
        return DEFAULT_INBOUND_MEDIA_MAX_BYTES
    try:
        return int(gw["max_inbound_media_bytes"])
    except (TypeError, ValueError):
        return DEFAULT_INBOUND_MEDIA_MAX_BYTES


def validate_inbound_media_size(
    size: int,
    *,
    media_type: str = "media",
    max_bytes: Optional[int] = None,
) -> None:
    """Raise ``ValueError`` if an inbound media payload exceeds the cap.

    A ``max_bytes`` of ``0`` (or the configured cap resolving to ``0``)
    disables the check entirely. Passing ``max_bytes`` lets callers resolve
    the limit once and reuse it across an incremental read.
    """
    limit = get_inbound_media_max_bytes() if max_bytes is None else max_bytes
    if limit and size > limit:
        raise ValueError(
            f"Inbound {media_type} payload is too large "
            f"({size} bytes > {limit} bytes)"
        )


async def _read_httpx_body_with_limit(response, *, media_type: str) -> bytes:
    """Read an httpx streaming response body without exceeding the media cap.

    Rejects early on an oversized ``Content-Length`` header, then re-checks
    the running total as chunks arrive so a lying/absent header can't smuggle
    an unbounded body past the cap.
    """
    max_bytes = get_inbound_media_max_bytes()
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            logger.debug(
                "Ignoring invalid Content-Length for inbound %s: %r",
                media_type, content_length,
            )
        else:
            validate_inbound_media_size(
                declared_size, media_type=media_type, max_bytes=max_bytes,
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        validate_inbound_media_size(total, media_type=media_type, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def get_image_cache_dir() -> Path:
    """Return the image cache directory, creating it if it doesn't exist."""
    d = _resolve_cache_dir("IMAGE_CACHE_DIR", "cache/images", "image_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _looks_like_image(data: bytes) -> bool:
    """Return True if *data* starts with a known image magic-byte sequence."""
    if len(data) < 4:
        return False
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return True
    if data[:2] == b"BM":
        return True
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return True
    return False


def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """
    Save raw image bytes to the cache and return the absolute file path.

    Args:
        data: Raw image bytes.
        ext:  File extension including the dot (e.g. ".jpg", ".png").

    Returns:
        Absolute path to the cached image file as a string.

    Raises:
        ValueError: If *data* does not look like a valid image (e.g. an HTML
            error page returned by the upstream server).
    """
    validate_inbound_media_size(len(data), media_type="image")
    if not _looks_like_image(data):
        snippet = data[:80].decode("utf-8", errors="replace")
        raise ValueError(
            f"Refusing to cache non-image data as {ext} "
            f"(starts with: {snippet!r})"
        )
    cache_dir = get_image_cache_dir()
    filename = f"img_{uuid.uuid4().hex[:12]}{ext}"
    filepath = cache_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


async def cache_image_from_url(url: str, ext: str = ".jpg", retries: int = 2) -> str:
    """
    Download an image from a URL and save it to the local cache.

    Retries on transient failures (timeouts, 429, 5xx) with exponential
    backoff so a single slow CDN response doesn't lose the media.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".jpg", ".png").
        retries: Number of retry attempts on transient failures.

    Returns:
        Absolute path to the cached image file as a string.

    Raises:
        ValueError: If the URL targets a private/internal network (SSRF protection).
    """
    from tools.url_safety import is_safe_url
    if not is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL (SSRF protection): {safe_url_for_log(url)}")

    import httpx
    _log = logging.getLogger(__name__)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        event_hooks={"response": [_ssrf_redirect_guard]},
    ) as client:
        for attempt in range(retries + 1):
            try:
                async with client.stream(
                    "GET",
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                ) as response:
                    response.raise_for_status()
                    content = await _read_httpx_body_with_limit(
                        response, media_type="image",
                    )
                return cache_image_from_bytes(content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    _log.debug(
                        "Media cache retry %d/%d for %s (%.1fs): %s",
                        attempt + 1,
                        retries,
                        safe_url_for_log(url),
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise


def cleanup_image_cache(max_age_hours: int = 24) -> int:
    """
    Delete cached images older than *max_age_hours*.

    Returns the number of files removed.
    """
    import time

    cache_dir = get_image_cache_dir()
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ---------------------------------------------------------------------------
# Audio cache utilities
#
# Same pattern as image cache -- voice messages from platforms are downloaded
# here so the STT tool (OpenAI Whisper) can transcribe them from local files.
# ---------------------------------------------------------------------------

AUDIO_CACHE_DIR = get_hermes_dir("cache/audio", "audio_cache")


def get_audio_cache_dir() -> Path:
    """Return the audio cache directory, creating it if it doesn't exist."""
    d = _resolve_cache_dir("AUDIO_CACHE_DIR", "cache/audio", "audio_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """
    Save raw audio bytes to the cache and return the absolute file path.

    Args:
        data: Raw audio bytes.
        ext:  File extension including the dot (e.g. ".ogg", ".mp3").

    Returns:
        Absolute path to the cached audio file as a string.
    """
    validate_inbound_media_size(len(data), media_type="audio")
    cache_dir = get_audio_cache_dir()
    filename = f"audio_{uuid.uuid4().hex[:12]}{ext}"
    filepath = cache_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


async def cache_audio_from_url(url: str, ext: str = ".ogg", retries: int = 2) -> str:
    """
    Download an audio file from a URL and save it to the local cache.

    Retries on transient failures (timeouts, 429, 5xx) with exponential
    backoff so a single slow CDN response doesn't lose the media.

    Args:
        url: The HTTP/HTTPS URL to download from.
        ext: File extension including the dot (e.g. ".ogg", ".mp3").
        retries: Number of retry attempts on transient failures.

    Returns:
        Absolute path to the cached audio file as a string.

    Raises:
        ValueError: If the URL targets a private/internal network (SSRF protection).
    """
    from tools.url_safety import is_safe_url
    if not is_safe_url(url):
        raise ValueError(f"Blocked unsafe URL (SSRF protection): {safe_url_for_log(url)}")

    import httpx
    _log = logging.getLogger(__name__)

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        event_hooks={"response": [_ssrf_redirect_guard]},
    ) as client:
        for attempt in range(retries + 1):
            try:
                async with client.stream(
                    "GET",
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "audio/*,*/*;q=0.8",
                    },
                ) as response:
                    response.raise_for_status()
                    content = await _read_httpx_body_with_limit(
                        response, media_type="audio",
                    )
                return cache_audio_from_bytes(content, ext)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                    raise
                if attempt < retries:
                    wait = 1.5 * (attempt + 1)
                    _log.debug(
                        "Audio cache retry %d/%d for %s (%.1fs): %s",
                        attempt + 1,
                        retries,
                        safe_url_for_log(url),
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise


# ---------------------------------------------------------------------------
# Video cache utilities
#
# Same pattern as image/audio cache -- videos from platforms are downloaded
# here so the agent can reference them by local file path.
# ---------------------------------------------------------------------------

VIDEO_CACHE_DIR = get_hermes_dir("cache/videos", "video_cache")

SUPPORTED_VIDEO_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


def get_video_cache_dir() -> Path:
    """Return the video cache directory, creating it if it doesn't exist."""
    d = _resolve_cache_dir("VIDEO_CACHE_DIR", "cache/videos", "video_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_video_from_bytes(data: bytes, ext: str = ".mp4") -> str:
    """Save raw video bytes to the cache and return the absolute file path."""
    validate_inbound_media_size(len(data), media_type="video")
    cache_dir = get_video_cache_dir()
    filename = f"video_{uuid.uuid4().hex[:12]}{ext}"
    filepath = cache_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


# ---------------------------------------------------------------------------
# Document cache utilities
#
# Same pattern as image/audio cache -- documents from platforms are downloaded
# here so the agent can reference them by local file path.
# ---------------------------------------------------------------------------

DOCUMENT_CACHE_DIR = get_hermes_dir("cache/documents", "document_cache")
SCREENSHOT_CACHE_DIR = get_hermes_dir("cache/screenshots", "browser_screenshots")

# Import-time defaults; _resolve_cache_dir compares against these to tell a
# test monkeypatch from an unmodified constant.
_CACHE_DIR_IMPORT_DEFAULTS = {
    "IMAGE_CACHE_DIR": IMAGE_CACHE_DIR,
    "AUDIO_CACHE_DIR": AUDIO_CACHE_DIR,
    "VIDEO_CACHE_DIR": VIDEO_CACHE_DIR,
    "DOCUMENT_CACHE_DIR": DOCUMENT_CACHE_DIR,
    "SCREENSHOT_CACHE_DIR": SCREENSHOT_CACHE_DIR,
}

_HERMES_HOME = get_hermes_home()
_HERMES_ROOT = get_default_hermes_root()
MEDIA_DELIVERY_ALLOW_DIRS_ENV = "HERMES_MEDIA_ALLOW_DIRS"
MEDIA_DELIVERY_TRUST_RECENT_ENV = "HERMES_MEDIA_TRUST_RECENT_FILES"
MEDIA_DELIVERY_TRUST_RECENT_SECONDS_ENV = "HERMES_MEDIA_TRUST_RECENT_SECONDS"
# Strict mode toggles the original allowlist+recency path-validation behavior.
# Off by default — symmetric with inbound (we accept any document type the
# user uploads), and with the denylist still blocking obvious credential /
# system paths. Operators running public-facing gateways where prompt
# injection from one user could exfiltrate the host's secrets to that same
# user should set this to true.
MEDIA_DELIVERY_STRICT_ENV = "HERMES_MEDIA_DELIVERY_STRICT"
MEDIA_DELIVERY_SAFE_ROOTS = (
    IMAGE_CACHE_DIR,
    AUDIO_CACHE_DIR,
    VIDEO_CACHE_DIR,
    DOCUMENT_CACHE_DIR,
    SCREENSHOT_CACHE_DIR,
    _HERMES_HOME / "image_cache",
    _HERMES_HOME / "audio_cache",
    _HERMES_HOME / "video_cache",
    _HERMES_HOME / "document_cache",
    _HERMES_HOME / "browser_screenshots",
    # Canonical cache layout — listed alongside the legacy *_cache dirs so
    # generated artifacts deliver on installs that have both (#31733).
    _HERMES_HOME / "cache" / "images",
    _HERMES_HOME / "cache" / "audio",
    _HERMES_HOME / "cache" / "videos",
    _HERMES_HOME / "cache" / "documents",
    _HERMES_HOME / "cache" / "screenshots",
)

# Default recency window for trusting freshly-produced files (seconds).
# The agent's actual work generally completes well inside 10 minutes; legitimate
# build artifacts (PDFs from pandoc, plots from matplotlib, etc.) almost always
# land seconds before delivery. Old system files (/etc/passwd, ~/.ssh/id_rsa,
# stray credentials) have mtimes measured in days or months — well outside this
# window — so prompt-injection paths pointing at pre-existing host files are
# still rejected.
_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS = 600

# Hard denylist applied even when a path would otherwise pass recency trust.
# These prefixes hold credentials, system state, or process introspection that
# should never be uploaded as a gateway attachment, regardless of how new the
# file looks. The cache-dir allowlist still beats this — an operator-configured
# allowed root can intentionally live under one of these prefixes (rare, but
# their choice).
_MEDIA_DELIVERY_DENIED_PREFIXES = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/root",
    "/boot",
    "/var/log",
    "/var/lib",
    "/var/run",
)

# Within $HOME we additionally deny common credential / config directories.
# Resolved at check time against the live $HOME so containers and alt-home
# setups work correctly.
_MEDIA_DELIVERY_DENIED_HOME_SUBPATHS = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
    ".config",
    ".azure",
    ".gcloud",
    "Library/Keychains",  # macOS
)


# Canonical cache subdirectories that hold deliverable artifacts. Used both
# for the top-level safe roots above and to enumerate per-profile cache roots
# at check time (see _media_delivery_allowed_roots).
_MEDIA_DELIVERY_CACHE_SUBDIRS = (
    "images",
    "audio",
    "videos",
    "documents",
    "screenshots",
)


def _profile_cache_roots() -> List[Path]:
    """Return per-profile canonical cache roots under the shared Hermes root.

    Profile gateways write generated artifacts to
    ``<root>/profiles/<name>/cache/{images,audio,...}``. The static safe-roots
    list only covers the *active* HERMES_HOME's cache, so a gateway running at
    the root (e.g. ``HERMES_HOME=/opt/data``) while the model emits a
    profile-scoped path silently fails delivery. Enumerated dynamically at
    check time so profiles created after startup are covered, and so the
    resolved profile path is allowlisted *before* the ``/root`` system denylist
    is consulted (which otherwise wins when HERMES_HOME is symlinked under a
    denied prefix and $HOME is not that prefix). See issue #31733.
    """
    roots: List[Path] = []
    profiles_dir = _HERMES_ROOT / "profiles"
    try:
        profile_dirs = [p for p in profiles_dir.iterdir() if p.is_dir()]
    except OSError:
        return roots
    for profile_dir in profile_dirs:
        for subdir in _MEDIA_DELIVERY_CACHE_SUBDIRS:
            roots.append(profile_dir / "cache" / subdir)
    return roots


def _media_delivery_allowed_roots() -> List[Path]:
    """Return roots from which model-emitted local media may be delivered."""
    roots = [Path(root) for root in MEDIA_DELIVERY_SAFE_ROOTS]
    roots.extend(_profile_cache_roots())
    extra_roots = os.environ.get(MEDIA_DELIVERY_ALLOW_DIRS_ENV, "")
    for chunk in extra_roots.split(os.pathsep):
        for raw_root in chunk.split(","):
            raw_root = raw_root.strip()
            if not raw_root:
                continue
            root = Path(os.path.expanduser(raw_root))
            if root.is_absolute():
                roots.append(root)
    return roots


def _media_delivery_recency_seconds() -> float:
    """Return the recency window for trusting freshly-produced files.

    0 disables recency-based trust entirely (pure-allowlist mode).
    """
    raw = os.environ.get(MEDIA_DELIVERY_TRUST_RECENT_ENV, "1").strip().lower()
    if raw in ("0", "false", "no", "off", ""):
        return 0.0
    try:
        custom = os.environ.get(MEDIA_DELIVERY_TRUST_RECENT_SECONDS_ENV, "").strip()
        if custom:
            seconds = float(custom)
            return max(0.0, seconds)
    except (TypeError, ValueError):
        pass
    return float(_MEDIA_DELIVERY_TRUST_RECENT_DEFAULT_SECONDS)


def _media_delivery_strict_mode() -> bool:
    """Return True when path validation should require allowlist/recency match.

    Off by default. In non-strict mode, ``validate_media_delivery_path``
    accepts any existing regular file that isn't under the credential /
    system-path denylist — restoring the pre-#29523 behavior for the
    single-user case. Strict mode preserves the original
    allowlist+recency-window logic for operators running public-facing
    gateways where prompt injection from one user shouldn't be able to
    exfiltrate the host's secrets to that same user.
    """
    raw = os.environ.get(MEDIA_DELIVERY_STRICT_ENV, "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _media_delivery_denied_paths() -> List[Path]:
    """Return absolute denylist paths under which delivery is never allowed."""
    denied = [Path(p) for p in _MEDIA_DELIVERY_DENIED_PREFIXES]
    home = Path(os.path.expanduser("~"))
    for sub in _MEDIA_DELIVERY_DENIED_HOME_SUBPATHS:
        denied.append(home / sub)
    # The active Hermes profile and shared Hermes root both contain control
    # files and credentials. Only cache subdirectories under them are
    # explicitly allowlisted above (matched BEFORE this denylist in
    # validate_media_delivery_path, so generated media still delivers).
    #
    # These are the per-file credential / secret stores that live at the
    # HERMES_HOME root. The set mirrors the canonical read guard in
    # agent/file_safety.py (get_read_block_error / build_write_denied_*) so the
    # delivery (read/exfil) side can't trail the write side: a credential the
    # agent is forbidden to write or read must also never be auto-attached to a
    # chat reply. Enumerated explicitly per-file rather than denying the whole
    # tree, so skills/, logs/, and ad-hoc agent-written files under ~/.hermes
    # stay deliverable (see #32090, #34425).
    _ROOT_CREDENTIAL_FILES = (
        ".env",
        "auth.json",
        "auth.lock",
        "credentials",
        "config.yaml",
        # Anthropic PKCE / OAuth refresh credential store.
        ".anthropic_oauth.json",
        # Google Workspace skill: auto-refreshing OAuth token (mtime bumps
        # every turn, which defeated the strict-mode recency window) plus the
        # pending-exchange session/verifier file.
        "google_token.json",
        "google_oauth_pending.json",
        os.path.join("auth", "google_oauth.json"),
        # Webhook subscription HMAC secrets.
        "webhook_subscriptions.json",
        # Bitwarden Secrets Manager plaintext disk cache.
        os.path.join("cache", "bws_cache.json"),
    )
    # Directory trees whose every child is credential material.
    #
    # mcp-tokens/ holds live MCP OAuth access tokens (<server>.json) and
    # dynamically-registered client credentials (<server>.client.json); see
    # tools/mcp_oauth.py. Same credential class as auth.json/credentials/.
    # The write side already denies it (file_tools _check_sensitive_path);
    # this pairs the media-delivery (exfil) side so a prompt-injection MEDIA
    # tag can't deliver a live bearer token as a native attachment.
    # (session/kanban SQLite stores are handled by #41071 — kept out here.)
    _ROOT_CREDENTIAL_DIRS = (
        "pairing",
        "mcp-tokens",
    )
    for hermes_root in (_HERMES_HOME, _HERMES_ROOT):
        for rel in _ROOT_CREDENTIAL_FILES:
            denied.append(hermes_root / rel)
        for rel in _ROOT_CREDENTIAL_DIRS:
            denied.append(hermes_root / rel)
    return denied


def _path_under_denied_prefix(resolved: Path) -> bool:
    """Return True if ``resolved`` lives under a deny-listed system path.

    One narrow exception: when a denied prefix IS the running user's own home,
    the home itself is not treated as denied. ``/root`` is on the system-path
    denylist so that a non-root gateway can't deliver another user's home, but
    on a root-run gateway ``$HOME=/root`` and the operator's own deliverables
    (``/root/work/proposal.docx``) live directly under it. The credential
    sub-directories inside home (``~/.ssh``, ``~/.aws``, ...) and Hermes
    secrets (``~/.hermes/.env``, ``auth.json``) are *separate, more-specific*
    denied paths, so they stay blocked regardless of this exception — it can
    only un-block a plain file sitting in the running user's home tree, never a
    credential location or another user's home.
    """
    try:
        home = Path(os.path.expanduser("~")).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        home = None
    for denied in _media_delivery_denied_paths():
        try:
            resolved_denied = denied.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if not (_path_is_within(resolved, resolved_denied) or resolved == resolved_denied):
            continue
        # Allow the running user's own home tree; its credential sub-dirs are
        # caught by their own (more-specific) denylist entries above.
        if home is not None and resolved_denied == home:
            continue
        return True
    return False


def _file_is_recently_produced(resolved: Path, window_seconds: float) -> bool:
    """Return True if the file's mtime is within ``window_seconds`` of now.

    Used as a session-scoped trust signal: agents almost always produce
    delivery artifacts within seconds of asking to send them, while
    prompt-injection paths pointing at pre-existing host files (/etc/passwd,
    ~/.ssh/id_rsa) have mtimes measured in days or months.
    """
    if window_seconds <= 0:
        return False
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        return False
    return (time.time() - mtime) <= window_seconds


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_media_delivery_path(path: str) -> Optional[str]:
    """Return a safe absolute file path for native media delivery, else None.

    Default mode (single-user / private gateway): accept any existing regular
    file that isn't under the credential / system-path denylist
    (``_MEDIA_DELIVERY_DENIED_PREFIXES`` + ``~/.ssh``, ``~/.aws``, etc.).
    This matches the symmetry of inbound delivery — Telegram/Discord/Slack
    will hand the agent any file the user uploads, and the agent can hand
    back any file that isn't a credential.

    Strict mode (opt-in via ``gateway.strict`` in ``config.yaml`` or
    ``HERMES_MEDIA_DELIVERY_STRICT=1``): the file MUST live under a
    Hermes-managed cache, under an operator-allowlisted root
    (``HERMES_MEDIA_ALLOW_DIRS``), or be freshly produced inside the
    configured recency window. Suitable for public-facing bots where
    prompt injection from one user shouldn't be able to exfiltrate the
    host's secrets to that same user.

    Symlinks are resolved before any containment / denylist check.
    """
    if not path:
        return None

    candidate = str(path).strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "`\"'":
        candidate = candidate[1:-1].strip()
    candidate = candidate.lstrip("`\"'").rstrip("`\"',.;:)}]")
    if not candidate:
        return None

    try:
        expanded = Path(os.path.expanduser(candidate))
    except (OSError, RuntimeError, ValueError):
        # expanduser raises ValueError("embedded null byte") for a ~\x00 path.
        return None
    if not expanded.is_absolute():
        return None

    try:
        resolved = expanded.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    if not resolved.is_file():
        return None

    # Cache / operator allowlist is always honored — these are unconditionally
    # trusted regardless of mode.
    for root in _media_delivery_allowed_roots():
        try:
            resolved_root = root.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        if _path_is_within(resolved, resolved_root):
            return str(resolved)

    # Non-strict mode (default): accept anything not on the denylist.
    # The denylist still blocks /etc, /proc, ~/.ssh, ~/.aws, and the
    # credential/secret stores under the Hermes root (~/.hermes/.env,
    # auth.json, .anthropic_oauth.json, google_token.json, pairing/, ...) —
    # so the obvious prompt-injection / credential-exfil sites
    # (``MEDIA:/etc/passwd``, ``MEDIA:~/.ssh/id_rsa``,
    # ``MEDIA:~/.hermes/google_token.json``) remain rejected.
    if not _media_delivery_strict_mode():
        if _path_under_denied_prefix(resolved):
            return None
        return str(resolved)

    # Strict mode: fall back to recency-based trust for freshly-produced
    # files (e.g. ``pandoc -o /tmp/report.pdf`` or
    # ``write_file("/home/user/report.pdf", ...)``). System paths and
    # credential locations remain blocked even when "recent" — see
    # ``_MEDIA_DELIVERY_DENIED_PREFIXES`` for the denylist.
    window = _media_delivery_recency_seconds()
    if window > 0 and not _path_under_denied_prefix(resolved):
        if _file_is_recently_produced(resolved, window):
            return str(resolved)

    return None


# Neutralise control chars and the Unicode line separators (NEL, LS, PS) that
# str.splitlines() / log aggregators treat as breaks, so a model-emitted path
# can't forge a second log line. Truncated to keep records bounded.
_LOG_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")


def _log_safe_path(path: str) -> str:
    """Return a single-line, length-bounded path for log output."""
    return _LOG_UNSAFE_CHARS.sub("?", str(path))[:200]


SUPPORTED_DOCUMENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".log": "text/plain",
    ".json": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".zip": "application/zip",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ts": "text/plain",
    ".py": "text/plain",
    ".sh": "text/plain",
}


# ---------------------------------------------------------------------------
# Text-injection extension allowlist
#
# Files whose contents are safe to inline into the prompt (UTF-8 text) when
# small enough. This is intentionally an extension/MIME gate, NOT a blind
# UTF-8 decode: binary formats like PDF/zip/docx can begin with decodable
# ASCII headers and must never be inlined. Any uploaded file is still cached
# and surfaced to the agent regardless of whether it lands in this set —
# this only controls inline-vs-path-pointer for the prompt.
# ---------------------------------------------------------------------------

_TEXT_INJECT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".log",
    ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".env", ".properties",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat",
    ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".java", ".kt",
    ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r", ".jl",
    ".swift", ".m", ".scala", ".clj", ".ex", ".exs", ".erl",
    ".sql", ".graphql", ".proto", ".tf", ".hcl",
    ".dockerfile", ".makefile", ".cmake", ".gradle",
    ".rst", ".tex", ".srt", ".vtt", ".diff", ".patch",
}


# ---------------------------------------------------------------------------
# Image document types
#
# Image extensions that platforms may deliver as "documents" rather than
# native photo attachments (Telegram users uploading via the file picker,
# clients that wrap stickers/screenshots as files, etc.). When we see one
# of these, we route the bytes through the image cache and the normal
# vision/photo handling path instead of rejecting them as unsupported
# documents.
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_DOCUMENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ---------------------------------------------------------------------------
# Media-delivery extension allowlist — SINGLE SOURCE OF TRUTH
#
# Both extractors that turn response text into native attachments derive their
# extension set from this tuple:
#   * ``extract_media()``       — explicit ``MEDIA:<path>`` tags
#   * ``extract_local_files()`` — bare absolute/home paths the agent mentions
#
# Historically these two carried independently-maintained extension lists.
# ``extract_media`` had a narrow list (no .md/.json/.yaml/.xml/.html/...) while
# ``extract_local_files`` had a broad one. Combined with the unconditional
# ``MEDIA:\\s*\\S+`` cleanup at the dispatch sites, that mismatch created a
# silent black hole: a ``MEDIA:/report.md`` tag failed the narrow extract_media
# match, got stripped from the body by the loose cleanup regex, and was then
# invisible to extract_local_files — the file was never delivered (issue
# #34517). Keeping one list eliminates the drift; building the cleanup regexes
# from the same set means a tag is only stripped when its extension is one we
# can actually deliver, so an unknown-extension path survives in the body
# instead of vanishing.
#
# Covers images (inline), video (inline where supported), audio (voice/audio),
# documents/spreadsheets/presentations (send_document), archives, and rendered
# web output. The dispatch partition (image vs video vs document) lives in
# ``gateway/run.py``.
# ---------------------------------------------------------------------------

MEDIA_DELIVERY_EXTS: Tuple[str, ...] = (
    # Images (embed inline)
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",
    # Video (embed inline where supported)
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    # Audio (delivered as voice/audio where supported)
    ".mp3", ".wav", ".ogg", ".opus", ".m4a", ".flac",
    # Documents (uploaded as file attachments)
    ".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md", ".epub",
    # Spreadsheets / data
    ".xlsx", ".xls", ".ods", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
    # Presentations
    ".pptx", ".ppt", ".odp", ".key",
    # Archives
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".apk", ".ipa",
    # Web / rendered output
    ".html", ".htm",
)

# Regex alternation fragment of bare extensions (no leading dot), e.g.
# ``png|jpe?g|...``. ``jpe?g`` collapses jpg/jpeg into one branch. Sorted
# longest-first so the alternation never matches a shorter ext as a prefix of
# a longer one (e.g. ``.tar`` before ``.tar.gz`` components).
_MEDIA_EXT_ALTERNATION = "|".join(
    sorted((e.lstrip(".") for e in MEDIA_DELIVERY_EXTS), key=len, reverse=True)
)

# Anchored ``MEDIA:<path>`` cleanup pattern. Unlike the old loose
# ``MEDIA:\\s*\\S+``, this only strips a tag whose path ends in a known
# deliverable extension (optionally quoted/backticked). A ``MEDIA:`` tag with
# an unknown extension is left in the text so it can still be picked up by the
# bare-path detector (extract_local_files) downstream rather than silently
# deleted. Shared by the non-streaming dispatch path and the streaming
# consumer so both behave identically.
# Path anchors: ``~/`` (Unix home-relative), ``/`` (Unix absolute),
# ``X:\\`` or ``X:/`` (Windows drive-letter absolute — #34632).
MEDIA_TAG_CLEANUP_RE = re.compile(
    r'''[`"']?MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])\S+(?:[^\S\n]+\S+)*?\.(?:''' + _MEDIA_EXT_ALTERNATION + r'''))'''
    r'''(?=[\s`"',;:)\]}]|$)[`"']?''',
    re.IGNORECASE,
)

# Extension-less absolute paths (e.g. Caddyfile, Dockerfile, Makefile) are
# intentionally excluded from MEDIA_TAG_CLEANUP_RE — they are validated and
# delivered via MEDIA_EXTENSIONLESS_TAG_RE so prompt-injection paths that do
# not exist on disk are left visible instead of silently dropped. Paths with
# an unknown but present extension (e.g. .weirdext) stay on the #34517
# bare-path fallback and are not handled here.
MEDIA_EXTENSIONLESS_TAG_RE = re.compile(
    r'''[`"']?MEDIA:\s*'''
    r'''(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|'''
    r'''(?:~/|/|[A-Za-z]:[/\\])[^\s\n`"']+)'''
    r'''[`"']?\s*''',
    re.IGNORECASE,
)


def _normalize_media_tag_path(raw: str) -> str:
    path = str(raw or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
        path = path[1:-1].strip()
    return path.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _path_lacks_deliverable_extension(path: str) -> bool:
    """True only when the basename has no extension (Caddyfile, Makefile, …)."""
    return not Path(path).suffix


def _strip_media_tag_directives(text: str) -> str:
    """Remove MEDIA: tags and [[audio_as_voice]] / [[as_document]] markers."""
    if (
        "MEDIA:" not in text
        and "[[audio_as_voice]]" not in text
        and "[[as_document]]" not in text
    ):
        return text
    cleaned = text.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")

    def _strip_extensionless(match: re.Match) -> str:
        path = _normalize_media_tag_path(match.group("path"))
        if not path or not _path_lacks_deliverable_extension(path):
            return match.group(0)
        return "" if validate_media_delivery_path(path) else match.group(0)

    cleaned = MEDIA_TAG_CLEANUP_RE.sub("", cleaned)
    return MEDIA_EXTENSIONLESS_TAG_RE.sub(_strip_extensionless, cleaned)


def get_document_cache_dir() -> Path:
    """Return the document cache directory, creating it if it doesn't exist."""
    d = _resolve_cache_dir("DOCUMENT_CACHE_DIR", "cache/documents", "document_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """
    Save raw document bytes to the cache and return the absolute file path.

    The cached filename preserves the original human-readable name with a
    unique prefix: ``doc_{uuid12}_{original_filename}``.

    Args:
        data: Raw document bytes.
        filename: Original filename (e.g. "report.pdf").

    Returns:
        Absolute path to the cached document file as a string.

    Raises:
        ValueError: If the sanitized path escapes the cache directory.
    """
    cache_dir = get_document_cache_dir()
    # Sanitize: strip directory components, null bytes, and control characters
    safe_name = Path(filename).name if filename else "document"
    safe_name = safe_name.replace("\x00", "").strip()
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "document"
    cached_name = f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    filepath = cache_dir / cached_name
    # Final safety check: ensure path stays inside cache dir
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    filepath.write_bytes(data)
    return str(filepath)


def cleanup_document_cache(max_age_hours: int = 24) -> int:
    """
    Delete cached documents older than *max_age_hours*.

    Returns the number of files removed.
    """
    import time

    cache_dir = get_document_cache_dir()
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ---------------------------------------------------------------------------
# Unified media caching
#
# One entry point for "I have raw attachment bytes from a platform — cache them
# and tell me what I got." Classifies by extension/MIME against the shared
# registries above, routes to the right cache_*_from_bytes helper, and returns
# a small result the caller can store and/or describe in a transcript. Used by
# both the addressed-message path and the observed-group-context path, on any
# platform — not Telegram-specific.
# ---------------------------------------------------------------------------

@dataclass
class CachedMedia:
    """Result of caching one attachment's bytes."""

    path: str                 # absolute cache path, agent-visible (sandbox-translated)
    media_type: str           # MIME type recorded on the MessageEvent
    kind: str                 # "image" | "video" | "audio" | "document"
    display_name: str         # human-readable name for transcript notes

    def context_note(self) -> str:
        """One-line transcript annotation pointing the agent at the file."""
        return f"[{self.kind} '{self.display_name}' saved at: {self.path}]"


def _resolve_media_ext(filename: str, mime_type: str) -> str:
    """Best-effort file extension from filename, then MIME fallback."""
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext:
            return ext
    mime = (mime_type or "").lower()
    if not mime:
        return ""
    for table in (
        SUPPORTED_IMAGE_DOCUMENT_TYPES,
        SUPPORTED_VIDEO_TYPES,
        SUPPORTED_DOCUMENT_TYPES,
    ):
        for ext, m in table.items():
            if m == mime:
                return ext
    return ""


def cache_media_bytes(
    data: bytes,
    *,
    filename: str = "",
    mime_type: str = "",
    default_kind: Optional[str] = None,
) -> Optional[CachedMedia]:
    """Classify and cache raw attachment bytes; return a CachedMedia or None.

    ``default_kind`` ("image"/"video"/"audio"/"document") biases classification
    when the extension/MIME are ambiguous — e.g. a Telegram native photo whose
    file has no usable name. Any non-image/video/audio file is cached as a
    document and surfaced to the agent (arbitrary types get
    ``application/octet-stream``); only images that fail validation
    (``cache_image_from_bytes`` raises ValueError) return None.
    """
    from tools.credential_files import to_agent_visible_cache_path

    ext = _resolve_media_ext(filename, mime_type)
    mime = (mime_type or "").lower()
    display = re.sub(r"[^\w.\- ]", "_", filename) if filename else (ext.lstrip(".") or "file")

    is_image = (
        mime.startswith("image/")
        or ext in SUPPORTED_IMAGE_DOCUMENT_TYPES
        or default_kind == "image"
    )
    is_video = mime.startswith("video/") or ext in SUPPORTED_VIDEO_TYPES or default_kind == "video"
    is_audio = mime.startswith("audio/") or default_kind == "audio"

    if is_image:
        img_ext = ext if ext in SUPPORTED_IMAGE_DOCUMENT_TYPES else ".jpg"
        try:
            path = cache_image_from_bytes(data, ext=img_ext)
        except ValueError:
            return None
        out_mime = mime if mime.startswith("image/") else SUPPORTED_IMAGE_DOCUMENT_TYPES.get(img_ext, "image/jpeg")
        return CachedMedia(to_agent_visible_cache_path(path), out_mime, "image", display)

    if is_video:
        vid_ext = ext if ext in SUPPORTED_VIDEO_TYPES else ".mp4"
        path = cache_video_from_bytes(data, ext=vid_ext)
        return CachedMedia(to_agent_visible_cache_path(path), SUPPORTED_VIDEO_TYPES.get(vid_ext, "video/mp4"), "video", display)

    if is_audio:
        aud_ext = ext if ext in {".ogg", ".mp3", ".wav", ".m4a", ".opus", ".flac"} else ".ogg"
        path = cache_audio_from_bytes(data, ext=aud_ext)
        out_mime = mime if mime.startswith("audio/") else f"audio/{aud_ext.lstrip('.')}"
        return CachedMedia(to_agent_visible_cache_path(path), out_mime, "audio", display)

    # Any other file type is cached and surfaced to the agent as a local path
    # so it can be inspected with terminal / read_file / etc. Authorization to
    # talk to the agent is the gate that matters — once a user is allowed to
    # message it, the file-extension allowlist must not silently drop their
    # uploads. Known extensions keep their precise MIME; everything else is
    # tagged application/octet-stream (or the caller-supplied MIME) so the
    # agent knows it's an arbitrary file and reaches for terminal tools.
    fallback_name = filename or (f"document{ext}" if ext else "document.bin")
    path = cache_document_from_bytes(data, fallback_name)
    if ext in SUPPORTED_DOCUMENT_TYPES:
        out_mime = SUPPORTED_DOCUMENT_TYPES[ext]
    else:
        out_mime = mime if mime else "application/octet-stream"
    return CachedMedia(to_agent_visible_cache_path(path), out_mime, "document", display or fallback_name)


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    LOCATION = "location"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    COMMAND = "command"  # /command style


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class MessageEvent:
    """
    Incoming message from a platform.
    
    Normalized representation that all adapters produce.
    """
    # Message content
    text: str
    message_type: MessageType = MessageType.TEXT
    
    # Source information
    source: SessionSource = None
    
    # Original platform data
    raw_message: Any = None
    message_id: Optional[str] = None

    # Platform-specific update identifier.  For Telegram this is the
    # ``update_id`` from the PTB Update wrapper; other platforms currently
    # ignore it.  Used by ``/restart`` to record the triggering update so the
    # new gateway can advance the Telegram offset past it and avoid processing
    # the same ``/restart`` twice if PTB's graceful-shutdown ACK times out
    # ("Error while calling `get_updates` one more time to mark all fetched
    # updates" in gateway.log).
    platform_update_id: Optional[int] = None
    
    # Media attachments
    # media_urls: local file paths (for vision tool access)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    
    # Reply context
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None  # Text of the replied-to message (for context injection)
    reply_to_author_id: Optional[str] = None
    reply_to_author_name: Optional[str] = None
    reply_to_is_own_message: bool = False  # True when the user replied to this bot/assistant's message
    
    # Auto-loaded skill(s) for topic/channel bindings (e.g., Telegram DM Topics,
    # Discord channel_skill_bindings).  A single name or ordered list.
    auto_skill: Optional[str | list[str]] = None

    # Per-channel ephemeral system prompt (e.g. Discord channel_prompts).
    # Applied at API call time and never persisted to transcript history.
    channel_prompt: Optional[str] = None

    # Channel context recovered by history backfill (e.g. messages between
    # bot turns that were missed due to require_mention).  Kept separate
    # from ``text`` so the sender-prefix logic in run.py can operate on the
    # trigger message alone, then prepend this context afterward.
    channel_context: Optional[str] = None
    
    # Internal flag — set for synthetic events (e.g. background process
    # completion notifications) that must bypass user authorization checks.
    internal: bool = False

    # Free-form per-event metadata.  Adapters may set platform-specific
    # signals here (e.g. WhatsApp sets ``whatsapp_from_owner=True`` when
    # the bridge is configured to forward owner-typed messages).  Plugins
    # consume via ``event.metadata.get(...)`` and must not rely on any
    # particular key existing.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_command(self) -> bool:
        """Check if this is a command message (e.g., /new, /reset)."""
        return self.text.startswith("/")
    
    def get_command(self) -> Optional[str]:
        """Extract command name if this is a command message."""
        if not self.is_command():
            return None
        # Split on space and get first word, strip the /
        parts = self.text.split(maxsplit=1)
        raw = parts[0][1:].lower() if parts else None
        if raw and "@" in raw:
            raw = raw.split("@", 1)[0]
        # Reject file paths: valid command names never contain /
        if raw and "/" in raw:
            return None
        return raw
    
    def get_command_args(self) -> str:
        """Get the arguments after a command."""
        if not self.is_command():
            return self.text
        parts = self.text.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        # iOS auto-corrects -- to — (em dash) and - to – (en dash)
        args = args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
        return args


@dataclass
class TextDebounceState:
    event: MessageEvent
    task: asyncio.Task | None
    first_ts: float
    last_ts: float


_PLAINTEXT_GATEWAY_RESTART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+hermes[.!?\s]*$", re.IGNORECASE),
)


def coerce_plaintext_gateway_command(event: "MessageEvent") -> None:
    """Rewrite a tiny set of DM plaintext admin phrases into slash commands.

    This keeps high-impact operational phrases like ``restart gateway`` out of
    the LLM/tool path, where they can trigger a self-restart from inside the
    currently running agent and leave the gateway stuck in ``draining`` while it
    waits for that same agent to finish.

    Scope is intentionally narrow: DM text messages only, exact restart-style
    phrases only. Group chats keep natural-language semantics.
    """
    try:
        if event is None or event.message_type != MessageType.TEXT:
            return
        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return
        source = getattr(event, "source", None)
        if getattr(source, "chat_type", None) != "dm":
            return
        for pattern in _PLAINTEXT_GATEWAY_RESTART_PATTERNS:
            if pattern.match(text):
                event.text = "/restart"
                return
    except Exception:
        return


class FinalDeliveryState(str, Enum):
    """Explicit tri-state delivery contract.

    ``NOT_HANDLED``
        Zero content visible; safe to fall back.
    ``FULLY_DELIVERED``
        All content has been delivered.
    ``PARTIALLY_DELIVERED``
        Some content is already visible; must NOT re-send the full payload.
    """
    NOT_HANDLED = "not_handled"
    FULLY_DELIVERED = "fully_delivered"
    PARTIALLY_DELIVERED = "partially_delivered"


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None
    # Adapter-specific metadata.  Cross-layer contracts that affect delivery
    # semantics must be documented at the producer and consumer sites.  Current
    # known contract: Telegram edit overflow partials set
    # raw_response["partial_overflow"] with delivered_chunks, total_chunks,
    # last_message_id, delivered_prefix, and continuation_message_ids so the
    # stream consumer can send the missing tail instead of marking a clipped
    # response complete.
    retryable: bool = False  # True for transient connection errors — base will retry automatically
    # Server-requested retry delay in seconds (e.g. Telegram FloodWait retry_after).
    # When present, _send_with_retry() honors this instead of its default backoff.
    retry_after: Optional[float] = None
    # When the adapter had to split an oversized payload across multiple
    # platform messages (e.g. Telegram edit_message overflow split-and-deliver),
    # ``message_id`` is the LAST visible message id (so subsequent edits target
    # the most recent chunk) and these are the additional message ids that
    # made up the full payload, in send order.  Empty tuple for the common
    # single-message case.
    continuation_message_ids: tuple = ()
    # Machine-readable failure category (set only when ``success`` is False).
    # ``error`` stays the human-readable detail string; ``error_kind`` lets
    # consumers branch deterministically instead of substring-matching the raw
    # provider message.  One of the values in :data:`SEND_ERROR_KINDS` or
    # ``None`` (unset / not classified).  Producers should set this via
    # :func:`classify_send_error`.
    error_kind: Optional[str] = None
    # Explicit tri-state delivery contract (D1).  When set, this overrides the
    # legacy success/failure binary for downstream fallback decisions.
    # PARTIALLY_DELIVERED must always pair with success=False (enforced in
    # __post_init__).  Defaults to None for backward compatibility — legacy
    # callers that don't set it are handled by effective_delivery_state().
    delivery_state: Optional[FinalDeliveryState] = None

    def __post_init__(self) -> None:
        if self.delivery_state is None:
            return
        if self.delivery_state is FinalDeliveryState.FULLY_DELIVERED and not self.success:
            raise ValueError(
                "SendResult with FULLY_DELIVERED delivery_state must have success=True."
            )
        if self.delivery_state is FinalDeliveryState.NOT_HANDLED and self.success:
            raise ValueError(
                "SendResult with NOT_HANDLED delivery_state must have success=False."
            )
        if self.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED and self.success:
            raise ValueError(
                "SendResult with PARTIALLY_DELIVERED delivery_state must have "
                "success=False to avoid reporting partial delivery as success."
            )


def effective_delivery_state(result: SendResult) -> FinalDeliveryState:
    """Return the effective delivery state, inferring from legacy fields when
    ``delivery_state`` is not explicitly set."""
    if result.delivery_state is not None:
        return result.delivery_state
    return (
        FinalDeliveryState.FULLY_DELIVERED
        if result.success
        else FinalDeliveryState.NOT_HANDLED
    )


# Machine-readable send-failure categories.  Kept platform-neutral so every
# adapter can populate ``SendResult.error_kind`` from the same vocabulary and
# the gateway can decide — once, in one place — whether a failure is worth
# surfacing to the user.
#
#   too_long      content exceeded the platform's per-message size cap; the
#                 adapter typically recovers via continuation/split, so this is
#                 informational rather than a hard failure.
#   bad_format    the platform rejected the message markup/entities (parse
#                 error); a plain-text retry is the actionable fix.
#   forbidden     the bot is blocked, kicked, or lacks permission to post to the
#                 target — the bot CANNOT reach the user, so there is nowhere to
#                 surface a notice.
#   not_found     the target chat/thread/message no longer exists.
#   rate_limited  the platform throttled the send (flood control).
#   transient     a connection-level failure that is safe to retry.
#   unknown       classification did not match any known shape.
SEND_ERROR_KINDS = frozenset(
    {
        "too_long",
        "bad_format",
        "forbidden",
        "not_found",
        "rate_limited",
        "transient",
        "unknown",
    }
)

# ``not_found`` substrings split by blast radius.  A *chat-level* not_found means
# the chat/user/group itself is gone, so the whole target is dead.  A
# *thread/topic/message-level* not_found (a deleted forum topic, an edited-away
# message) leaves the parent chat reachable — it must NOT mark the whole chat
# dead.  ``classify_send_error`` collapses both into ``"not_found"``;
# ``is_chat_level_not_found`` recovers the distinction for the dead-target path.
# See gateway.dead_targets.
_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS = ("chat not found",)
_SUBCHAT_NOT_FOUND_SUBSTRINGS = (
    "message to edit not found",
    "message to reply not found",
    "thread not found",
    "topic_deleted",
    "message_id_invalid",
)


def _error_blob(exc: Optional[BaseException] = None, error_text: str = "") -> str:
    """Build the lowercased text blob both send-error classifiers match against.

    Single source of truth so ``classify_send_error`` and
    ``is_chat_level_not_found`` can never drift (e.g. one including the
    exception class name and the other not) and silently disagree on the same
    failure.  Includes ``str(exc)`` (when non-empty) and the exception's class
    name, plus any explicit ``error_text``.
    """
    parts = []
    if error_text:
        parts.append(error_text)
    if exc is not None:
        exc_str = str(exc)
        if exc_str:
            parts.append(exc_str)
        parts.append(exc.__class__.__name__)
    return " ".join(parts).lower()


def classify_send_error(exc: Optional[BaseException], error_text: str = "") -> str:
    """Map a send exception / error string to a :data:`SEND_ERROR_KINDS` value.

    Platform-neutral: matches on the lowercased text of ``exc`` (and/or the
    explicit ``error_text``) against the substrings the major messaging APIs
    use.  Conservative — anything unrecognized returns ``"unknown"`` so callers
    never mistake an unclassified failure for a benign one.
    """
    blob = _error_blob(exc, error_text)
    if not blob.strip():
        return "unknown"
    if "message_too_long" in blob or "too long" in blob or "message is too long" in blob:
        return "too_long"
    if (
        "can't parse entities" in blob
        or "cant parse entities" in blob
        or "can't find end" in blob
        or "unsupported start tag" in blob
        or ("entity" in blob and "parse" in blob)
        or ("bad request" in blob and "entit" in blob)
    ):
        return "bad_format"
    if (
        "forbidden" in blob
        or "bot was blocked" in blob
        or "blocked by the user" in blob
        or "user is deactivated" in blob
        or "not enough rights" in blob
        or "have no rights" in blob
        or "not a member" in blob
    ):
        return "forbidden"
    if any(s in blob for s in _CHAT_LEVEL_NOT_FOUND_SUBSTRINGS) or any(
        s in blob for s in _SUBCHAT_NOT_FOUND_SUBSTRINGS
    ):
        return "not_found"
    if (
        "flood" in blob
        or "too many requests" in blob
        or "retry after" in blob
        or "rate limit" in blob
    ):
        return "rate_limited"
    for pat in _RETRYABLE_ERROR_PATTERNS:
        if pat in blob:
            return "transient"
    if "connecttimeout" in blob:
        return "transient"
    return "unknown"


def is_chat_level_not_found(exc: Optional[BaseException] = None, error_text: str = "") -> bool:
    """Whether a ``not_found`` failure means the *whole chat* is gone.

    :func:`classify_send_error` collapses chat-level and thread/topic/message-level
    not_found into the single ``"not_found"`` kind.  Only the chat-level case (the
    chat/user/group no longer exists) should mark a delivery target dead; a deleted
    forum topic or an edited-away message leaves the parent chat reachable.  When
    both a chat-level and a sub-chat marker are present, the sub-chat reading wins
    (conservative: never kill a chat that may still be reachable).

    Argument order mirrors :func:`classify_send_error` (``exc`` first) and both
    share :func:`_error_blob`, so the two classifiers cannot disagree on the same
    failure.
    """
    blob = _error_blob(exc, error_text)
    if any(s in blob for s in _SUBCHAT_NOT_FOUND_SUBSTRINGS):
        return False
    return any(s in blob for s in _CHAT_LEVEL_NOT_FOUND_SUBSTRINGS)


class EphemeralReply(str):
    """System-notice reply that auto-deletes after a TTL.

    Slash-command handlers in ``gateway/run.py`` can return this wrapper
    instead of a plain string to request that the reply message be deleted
    after ``ttl_seconds`` on platforms that support ``delete_message``.

    Subclassing ``str`` keeps the wrapper transparent to anything that
    treats handler return values as text (existing tests use ``in`` /
    ``startswith`` / equality; the ``_process_message_background`` pipeline
    extracts attachments from the string content).  ``isinstance(r,
    EphemeralReply)`` still distinguishes ephemeral replies from plain
    strings so the send path can schedule deletion.

    Platforms that don't override :meth:`BasePlatformAdapter.delete_message`
    silently ignore the TTL — the message is sent normally and left in
    place.  When ``ttl_seconds`` is ``None``, the pipeline uses the
    configured ``display.ephemeral_system_ttl`` default.  A default of ``0``
    disables auto-deletion globally, preserving prior behavior.
    """

    ttl_seconds: Optional[int]

    def __new__(cls, text: str, ttl_seconds: Optional[int] = None):
        instance = super().__new__(cls, text)
        instance.ttl_seconds = ttl_seconds
        return instance

    @property
    def text(self) -> str:
        """Return the underlying text.

        Provided for call sites that want an explicit string conversion,
        though ``str(reply)`` and using ``reply`` directly where a string
        is expected both work identically.
        """
        return str.__str__(self)


def merge_pending_message_event(
    pending_messages: Dict[str, MessageEvent],
    session_key: str,
    event: MessageEvent,
    *,
    merge_text: bool = False,
) -> None:
    """Store or merge a pending event for a session.

    Photo bursts/albums often arrive as multiple near-simultaneous PHOTO
    events. Merge those into the existing queued event so the next turn sees
    the whole burst.

    When ``merge_text`` is enabled, rapid follow-up TEXT events are appended
    instead of replacing the pending turn. This is used for Telegram bursty
    follow-ups so a multi-part user thought is not silently truncated to only
    the last queued fragment.
    """
    existing = pending_messages.get(session_key)
    if existing:
        existing_is_photo = getattr(existing, "message_type", None) == MessageType.PHOTO
        incoming_is_photo = event.message_type == MessageType.PHOTO
        existing_has_media = bool(existing.media_urls)
        incoming_has_media = bool(event.media_urls)

        if existing_is_photo and incoming_is_photo:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
            return

        if existing_has_media or incoming_has_media:
            if incoming_has_media:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
            if event.text:
                if existing.text:
                    existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
                else:
                    existing.text = event.text
            if existing_is_photo or incoming_is_photo:
                existing.message_type = MessageType.PHOTO
            elif (
                getattr(existing, "message_type", None) == MessageType.TEXT
                and event.message_type != MessageType.TEXT
            ):
                existing.message_type = event.message_type
            return

        if (
            merge_text
            and getattr(existing, "message_type", None) == MessageType.TEXT
            and event.message_type == MessageType.TEXT
        ):
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            return

    pending_messages[session_key] = event


# Error substrings that indicate a transient *connection* failure worth retrying.
# "timeout" / "timed out" / "readtimeout" / "writetimeout" are intentionally
# excluded: a read/write timeout on a non-idempotent call (e.g. send_message)
# means the request may have reached the server — retrying risks duplicate
# delivery.  "connecttimeout" is safe because the connection was never
# established.  Platforms that know a timeout is safe to retry should set
# SendResult.retryable = True explicitly.
_RETRYABLE_ERROR_PATTERNS = (
    "connecterror",
    "connectionerror",
    "connectionreset",
    "connectionrefused",
    "connecttimeout",
    "network",
    "broken pipe",
    "remotedisconnected",
    "eoferror",
)


# Type for message handlers.  Handlers may return a plain string (normal
# reply), an ``EphemeralReply`` to opt the reply into auto-deletion, or
# ``None`` when the response was already delivered (e.g. via streaming).
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[Union[str, "EphemeralReply"]]]]


def resolve_channel_prompt(
    config_extra: dict,
    channel_id: str,
    parent_id: str | None = None,
) -> str | None:
    """Resolve a per-channel ephemeral prompt from platform config.

    Looks up ``channel_prompts`` in the adapter's ``config.extra`` dict.
    Prefers an exact match on *channel_id*; falls back to *parent_id*
    (useful for forum threads / child channels inheriting a parent prompt).

    Returns the prompt string, or None if no match is found.  Blank/whitespace-
    only prompts are treated as absent.
    """
    prompts = config_extra.get("channel_prompts") or {}
    if not isinstance(prompts, dict):
        return None

    for key in (channel_id, parent_id):
        if not key:
            continue
        prompt = prompts.get(key)
        if prompt is None:
            continue
        prompt = str(prompt).strip()
        if prompt:
            return prompt
    return None


def resolve_channel_skills(
    config_extra: dict,
    channel_id: str,
    parent_id: str | None = None,
) -> list[str] | None:
    """Resolve auto-loaded skill(s) for a channel/thread from platform config.

    Looks up ``channel_skill_bindings`` in the adapter's ``config.extra`` dict.

    Config format::

        channel_skill_bindings:
          - id: "C0123"          # Slack channel ID or Discord channel/forum ID
            skills: ["skill-a", "skill-b"]
          - id: "D0ABCDE"
            skill: "solo-skill"  # single string also accepted

    Prefers an exact match on *channel_id*; falls back to *parent_id*
    (useful for forum threads / Slack threads inheriting the parent channel's
    binding).

    Returns a deduplicated list of skill names (order preserved), or None if
    no match is found.
    """
    bindings = config_extra.get("channel_skill_bindings") or []
    if not isinstance(bindings, list) or not bindings:
        return None
    ids_to_check: set[str] = set()
    if channel_id:
        ids_to_check.add(str(channel_id))
    if parent_id:
        ids_to_check.add(str(parent_id))
    if not ids_to_check:
        return None
    for entry in bindings:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", ""))
        if entry_id in ids_to_check:
            skills = entry.get("skills") or entry.get("skill")
            if isinstance(skills, str):
                s = skills.strip()
                return [s] if s else None
            if isinstance(skills, list) and skills:
                seen: list[str] = []
                for name in skills:
                    if not isinstance(name, str):
                        continue
                    nm = name.strip()
                    if nm and nm not in seen:
                        seen.append(nm)
                return seen or None
    return None


def _strip_media_directives(text: str) -> str:
    """Strip internal delivery directives ([[audio_as_voice]], [[as_document]],
    MEDIA:<path>) so they never render as visible text.

    Backstop only: run ``extract_media`` first. MEDIA cleanup uses the shared
    ``MEDIA_TAG_CLEANUP_RE`` (only tags whose path has a known deliverable
    extension are removed; an unknown-extension tag is intentionally left so the
    bare-path detector downstream can still pick it up, per #34517). Validated
    extension-less tags (e.g. ``MEDIA:/output/Caddyfile``) are also removed.
    [[...]] is exact.
    """
    if not text:
        return text
    return _strip_media_tag_directives(text)


class BasePlatformAdapter(ABC):
    """
    Base class for platform adapters.
    
    Subclasses implement platform-specific logic for:
    - Connecting and authenticating
    - Receiving messages
    - Sending messages/responses
    - Handling media
    """

    # Whether this platform renders triple-backtick fenced code blocks (i.e.
    # ``format_message`` translates/preserves markdown fences into a real code
    # block).  Capability flag for markdown-aware presentation choices.
    # Default False (plain-text platforms); markdown-rendering adapters set True.
    # Tool-progress uses this to render a terminal command as a bare fenced code
    # block (no language tag — Slack mrkdwn would print the tag as a literal
    # first code line).  Plain-text platforms fall back to the short truncated
    # preview (see gateway/run.py progress_callback).
    supports_code_blocks: bool = False

    # Whether this adapter can deliver an ASYNC notification back to the agent
    # AFTER a turn ends — i.e. wake a fresh turn to surface a background
    # process completion (terminal notify_on_complete / watch_patterns) or a
    # detached subagent result (delegate_task background=True).
    #
    # True for adapters that hold a persistent outbound channel (Telegram,
    # Discord, Slack, ... — they have a real ``send()`` and the gateway runs
    # the watcher/drain loops). False for stateless request/response adapters
    # (the API server): every route closes its channel when the turn ends, so
    # there is nowhere to push a later completion. The gateway propagates this
    # into the ``HERMES_SESSION_ASYNC_DELIVERY`` contextvar at session-bind
    # time; tools read it via ``async_delivery_supported()`` and refuse to make
    # a delivery promise they can't keep. A new stateless adapter only needs to
    # set this to False to stay correct-by-default.
    supports_async_delivery: bool = True

    # Whether this adapter's ``send()`` splits long content into multiple
    # messages via ``truncate_message()``.  When True, the delivery router
    # (gateway/delivery.py) skips gateway-level truncation and lets the
    # adapter chunk natively — preserving full output on platforms that
    # support multi-message delivery (Discord, Telegram, …).  Default False
    # (conservative); adapters verified to chunk in ``send()`` set True.
    splits_long_messages: bool = False

    # The command prefix users can always TYPE on this platform to reach
    # Hermes commands.  Default "/" (most platforms deliver "/approve" etc.
    # as plain message text).  Platforms where typing a leading "/" is
    # intercepted or restricted by the client (Slack blocks native slash
    # commands inside threads; Matrix clients reserve "/" for client-local
    # commands) ship a "!" alias rewrite in their adapter and set this to
    # "!" so user-facing instruction text ("Reply `!approve` ...") tells
    # users the form that actually works everywhere.  Capability flag —
    # shared prompt builders read it via getattr(adapter,
    # "typed_command_prefix", "/"); no per-platform branching at call sites.
    typed_command_prefix: str = "/"

    # Whether this adapter supports the ``in_channel`` continuable-cron surface
    # (``platforms.<p>.extra.cron_continuable_surface: in_channel``): a
    # continuable cron job delivered FLAT into a channel (no dedicated thread),
    # with the user's plain channel reply continuing the job in-context via the
    # shared-channel session.  Only coherent on a platform that has BOTH a
    # flat-reply outbound gate AND a whole-channel inbound session bucket keyed
    # ``(platform, chat_id, None)`` — today that is Slack (``reply_in_thread:
    # false``).  Default False: an unsupported platform fails SAFE, treating
    # ``in_channel`` as ``thread`` (a threaded continuation ≈ today's
    # behaviour), never a dropped continuation.  Read generically by the cron
    # scheduler via ``getattr(adapter, "supports_inchannel_continuable",
    # False)`` — no per-platform branching at the call site (the key stays a
    # generic seam; Slack is merely the first consumer).
    supports_inchannel_continuable: bool = False

    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
        # Optional hook (e.g. Telegram DM topic recovery) that rewrites
        # ``event.source.thread_id`` before session keying. Returns the
        # corrected thread_id or None to leave the source untouched.
        self._topic_recovery_fn: Optional[Callable[[Any], Optional[str]]] = None
        self._running = False
        self._fatal_error_code: Optional[str] = None
        self._fatal_error_message: Optional[str] = None
        self._fatal_error_retryable = True
        self._fatal_error_handler: Optional[Callable[["BasePlatformAdapter"], Awaitable[None] | None]] = None
        
        # Track active message handlers per session for interrupt support.
        # _active_sessions stores the per-session interrupt Event; _session_tasks
        # maps session → the specific Task currently processing it so that
        # session-terminating commands (/stop, /new, /reset) can cancel the
        # right task and release the adapter-level guard deterministically.
        # Without the owner-task map, an old task's finally block could delete
        # a newer task's guard, leaving stale busy state.
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        self._session_tasks: Dict[str, asyncio.Task] = {}
        # Legacy busy_text_mode env var; when unset the runner syncs the
        # resolved value (driven by busy_input_mode) onto the adapter after
        # construction (gateway/run.py). Default to "interrupt" so a stray
        # pre-sync read matches the single-knob default rather than silently
        # queueing.
        self._busy_text_mode: str = (
            os.environ.get("HERMES_GATEWAY_BUSY_TEXT_MODE", "interrupt").strip().lower()
            or "interrupt"
        )
        self._busy_text_debounce_seconds: float = _float_env(
            "HERMES_GATEWAY_BUSY_TEXT_DEBOUNCE_SECONDS", 0.35
        )
        self._busy_text_hard_cap_seconds: float = _float_env(
            "HERMES_GATEWAY_BUSY_TEXT_HARD_CAP_SECONDS", 1.0
        )
        self._text_debounce: dict[str, TextDebounceState] = {}
        # Background message-processing tasks spawned by handle_message().
        # Gateway shutdown cancels these so an old gateway instance doesn't keep
        # working on a task after --replace or manual restarts.
        self._background_tasks: set[asyncio.Task] = set()
        # One-shot callbacks to fire after the main response is delivered.
        # Keyed by session_key. Values are either a bare callback (legacy) or
        # a ``(generation, callback)`` tuple so GatewayRunner can make deferred
        # deliveries generation-aware and avoid stale runs clearing callbacks
        # registered by a fresher run for the same session.
        self._post_delivery_callbacks: Dict[str, Any] = {}
        self._expected_cancelled_tasks: set[asyncio.Task] = set()
        self._busy_session_handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]] = None
        # Optional authorization check, registered by GatewayRunner. Used by
        # adapters that fetch external context (e.g. Slack thread history) to
        # mark senders not on the allowlist as unverified in LLM context,
        # mitigating indirect prompt injection from third parties in a shared
        # thread/channel.
        self._authorization_check: Optional[Callable[[str, Optional[str], Optional[str]], bool]] = None
        # Auto-TTS on voice input: ``_auto_tts_default`` is the global default
        # (``voice.auto_tts`` in config.yaml, pushed by GatewayRunner on connect).
        # Per-chat overrides live in two sets populated from ``_voice_mode``:
        #   - ``_auto_tts_enabled_chats``: chat explicitly opted in via ``/voice on``
        #     or ``/voice tts`` (mode is ``voice_only`` or ``all``). Fires even when
        #     the global default is False.
        #   - ``_auto_tts_disabled_chats``: chat explicitly opted out via
        #     ``/voice off`` (mode is ``off``). Suppresses auto-TTS even when the
        #     global default is True.
        # The gate in _process_message() is:
        #   fire if chat in _auto_tts_enabled_chats
        #     OR (_auto_tts_default and chat not in _auto_tts_disabled_chats)
        self._auto_tts_default: bool = False
        self._auto_tts_enabled_chats: set = set()
        self._auto_tts_disabled_chats: set = set()
        # Chats where typing indicator is paused (e.g. during approval waits).
        # _keep_typing skips send_typing when the chat_id is in this set.
        self._typing_paused: set = set()

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        """Return the length function for measuring message size on this platform.

        Override in adapters whose platform counts characters differently from
        Python ``len`` (e.g. Telegram counts UTF-16 code units).
        """
        return len

    @property
    def enforces_own_access_policy(self) -> bool:
        """Whether this adapter gates inbound access before dispatch.

        Some adapters (WeCom, Weixin, Yuanbao, QQBot, WhatsApp) implement a
        documented config-driven access surface — ``dm_policy`` / ``group_policy`` /
        ``allow_from`` / ``group_allow_from`` in ``PlatformConfig.extra`` — and
        enforce it at intake: a message is dropped inside the adapter and never
        reaches the gateway unless it already passed that policy.

        The gateway's env-based allowlist check runs *after* the adapter. When
        no env allowlist is configured, the gateway consults this flag so it can
        honor a config-only ``dm_policy: allowlist`` / ``allow_from`` (which the
        adapter already enforced) instead of double-denying it. Crucially, the
        flag alone is NOT "already authorized": these adapters default
        ``dm_policy`` / ``group_policy`` to ``"open"``, which forwards every
        sender, so the gateway trusts the adapter only when its effective policy
        for the chat type is an actual ``"allowlist"`` restriction — never for
        ``"open"`` (that would be the network-exposed fail-open SECURITY.md §2.6
        forbids). Open access still requires an explicit
        ``{PLATFORM}_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOW_ALL_USERS`` opt-in.

        Adapters that own their access policy override this to return ``True``.
        Adapters that delegate access control to the gateway leave it ``False``
        (the default).
        """
        return False

    @property
    def authorization_is_upstream(self) -> bool:
        """Whether inbound on this adapter was already authorized UPSTREAM.

        Distinct from ``enforces_own_access_policy``: that flag describes an
        adapter that enforces a LOCAL, config-driven access surface
        (``dm_policy: allowlist`` / ``allow_from``) the gateway can mirror. This
        flag describes an adapter whose authorization is performed by a TRUSTED
        UPSTREAM over an authenticated transport — there is no local policy to
        consult, and the env allowlist (``{PLATFORM}_ALLOWED_USERS``) does not
        apply because the sender identity isn't a platform account the operator
        configures here.

        The relay adapter is the sole user: it fronts the Team Gateway
        connector over a per-instance-authenticated WebSocket, and the connector
        performs owner-only author-binding resolution BEFORE delivering — a
        message only reaches this gateway because the connector resolved it to
        THIS instance's bound user (``user_instance_binding``). The author id is
        read off the event the connector observed, never gateway-asserted. So an
        inbound relay event carries an authorization decision already made by a
        trusted, authenticated upstream; default-denying it (no env allowlist ⇒
        deny) is incorrect.

        This is NOT a fail-open: it is authorization DELEGATED to a trusted
        upstream that authenticated the transport (the relay WS secret) and
        enforced owner-only binding, as opposed to authorization being ABSENT.
        It only takes effect for an adapter that explicitly overrides this to
        ``True``; every network-exposed direct adapter leaves it ``False`` and
        the env-allowlist default-deny continues to apply unchanged.
        """
        return False

    def supports_draft_streaming(
        self,
        chat_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Whether this adapter supports native streaming-draft updates.

        Telegram Bot API 9.5 introduced ``sendMessageDraft``, which renders an
        animated streaming preview as the bot calls it repeatedly with the
        same ``draft_id`` and growing text.  Adapters that implement
        ``send_draft`` should return True here for the chat types where the
        platform supports it (Telegram restricts drafts to private DMs).

        Default implementation returns False.  Stream consumers fall back to
        the edit-based path (``send`` + ``edit_message``) when this returns
        False or when ``send_draft`` raises.
        """
        return False

    def prefers_fresh_final_streaming(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Whether the stream consumer should finalize a streamed reply by
        sending a *fresh* final message (and deleting the preview) instead of
        final-editing the preview.

        Some adapters can send richer final messages than their current edit
        implementation supports. Telegram is the motivating case: Hermes sends
        final replies through ``sendRichMessage`` but still finalizes streamed
        previews through its existing MarkdownV2 edit path until Bot API 10.1's
        ``rich_message`` edit parameter is wired directly. Such adapters
        override this to ask the consumer to re-deliver the completed answer as
        a new rich message and best-effort delete the stale preview, so the
        final rendering matches the rich send path.

        Default implementation returns False — legacy platforms keep the
        edit-in-place finalization path.
        """
        return False

    def streaming_overflow_limit(self) -> Optional[int]:
        """Max single-message length (in this adapter's ``message_len_fn``
        units) the stream consumer may accumulate before it splits, when the
        adapter can deliver a larger message than its legacy per-message limit.

        Telegram Bot API 10.1 Rich Messages accept up to 32,768 chars in a
        single ``sendRichMessage`` / ``sendRichMessageDraft``, far above the
        4,096 MarkdownV2 limit.  Adapters with such a richer send/draft path
        override this so the consumer doesn't fragment a reply that fits one
        rich message; the live edit preview is still bound by the platform's
        edit limit, but the finalized reply (and DM draft preview) is delivered
        whole.

        Return ``None`` (default) to use ``MAX_MESSAGE_LENGTH``.
        """
        return None

    async def send_draft(
        self,
        chat_id: str,
        draft_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send or update an animated streaming-draft preview.

        Reuse the same ``draft_id`` (any non-zero int) across consecutive
        calls within a single response so the platform animates the preview
        rather than re-creating it.  Different responses must use different
        ``draft_id`` values within the same chat to avoid animating over a
        prior bubble.

        Drafts have no message_id and cannot be edited, replied to, or
        deleted via normal message APIs.  When the response finishes, the
        caller delivers the final answer as a regular ``send`` and the
        draft preview clears naturally on the client.

        Default implementation raises NotImplementedError; adapters that
        also return True from :meth:`supports_draft_streaming` must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement send_draft"
        )

    # ── Structured stream-event rendering ────────────────────────────────
    #
    # These methods let an adapter decide *how* to present each structured
    # streaming event (see gateway/stream_events.py).  The default
    # implementations reproduce the historical behavior exactly: assistant
    # text/commentary/segment events delegate to the stream consumer, and
    # tool events render the same "emoji tool_name: preview" chrome the
    # gateway has always produced.  Adapters override these to be more native
    # to their platform (e.g. Telegram streaming a MarkdownV2 ```bash``` block
    # as a draft; iMessage eating tool chrome it cannot format).
    #
    # The contract is presentation-only: nothing rendered here is persisted to
    # conversation history.  History is owned by the agent; what an adapter
    # chooses to "eat" must never change the bytes the agent stored.

    def render_message_event(self, event: Any, sink: Any) -> None:
        """Render a MessageChunk / MessageStop / Commentary onto the sink.

        Default: map onto the stream consumer's existing primitives, preserving
        today's behavior 1:1.  ``sink`` is a GatewayStreamConsumer.
        """
        from gateway.stream_events import MessageChunk, MessageStop, Commentary

        if isinstance(event, MessageChunk):
            if event.text:
                sink.on_delta(event.text)
        elif isinstance(event, MessageStop):
            # An intermediate stop (text → tool → text) is a segment break;
            # the terminal stop is signalled by the gateway via finish(),
            # not here, so we only break segments on non-final stops.
            if not event.final:
                sink.on_segment_break()
        elif isinstance(event, Commentary):
            if event.text:
                sink.on_commentary(event.text)

    def format_tool_event(self, event: Any, *, mode: str = "all",
                          preview_max_len: int = 40) -> Optional[str]:
        """Return the rendered chrome for a ToolCallChunk, or None to eat it.

        Reproduces the gateway's historical tool-progress formatting: an emoji
        for the tool, the tool name, and a short argument preview (or the full
        args dict in ``verbose`` mode).  Adapters that cannot render tool chrome
        (no message editing, plain-text only) should override to return None so
        the event is dropped rather than spamming separate bubbles.

        ``mode`` is the resolved tool-progress mode ("all" / "new" / "verbose");
        ``preview_max_len`` mirrors the ``tool_preview_length`` config (0 means
        "no cap" in verbose mode).
        """
        from gateway.stream_events import ToolCallChunk
        if not isinstance(event, ToolCallChunk):
            return None

        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(event.tool_name, default="⚙️")

        if mode == "verbose":
            if event.args:
                import json
                args_str = json.dumps(event.args, ensure_ascii=False, default=str)
                if preview_max_len > 0 and len(args_str) > preview_max_len:
                    args_str = args_str[:preview_max_len - 3] + "..."
                return f"{emoji} {event.tool_name}({list(event.args.keys())})\n{args_str}"
            if event.preview:
                return f"{emoji} {event.tool_name}: \"{event.preview}\""
            return f"{emoji} {event.tool_name}..."

        # "all" / "new": short preview, capped (default 40 to keep gateway
        # progress bubbles compact — they persist as permanent messages).
        preview = event.preview
        if preview:
            cap = preview_max_len if preview_max_len > 0 else 40
            if len(preview) > cap:
                preview = preview[:cap - 3] + "..."
            return f"{emoji} {event.tool_name}: \"{preview}\""
        return f"{emoji} {event.tool_name}..."

    @property
    def has_fatal_error(self) -> bool:
        return self._fatal_error_message is not None

    @property
    def fatal_error_message(self) -> Optional[str]:
        return self._fatal_error_message

    @property
    def fatal_error_code(self) -> Optional[str]:
        return self._fatal_error_code

    @property
    def fatal_error_retryable(self) -> bool:
        return self._fatal_error_retryable

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        """Whether auto-TTS on voice input should fire for ``chat_id``.

        Decision layers (Issue #16007):
          1. Explicit ``/voice on`` or ``/voice tts`` → always fire (even if
             ``voice.auto_tts`` is False).
          2. Explicit ``/voice off`` → never fire.
          3. Fall back to the global ``voice.auto_tts`` config default.
        """
        if chat_id in self._auto_tts_enabled_chats:
            return True
        if chat_id in self._auto_tts_disabled_chats:
            return False
        return bool(self._auto_tts_default)

    def set_fatal_error_handler(self, handler: Callable[["BasePlatformAdapter"], Awaitable[None] | None]) -> None:
        self._fatal_error_handler = handler

    def _mark_connected(self) -> None:
        self._running = True
        self._fatal_error_code = None
        self._fatal_error_message = None
        self._fatal_error_retryable = True
        self._write_runtime_status_safe("connected", platform_state="connected", error_code=None, error_message=None)

    def _mark_disconnected(self) -> None:
        self._running = False
        if self.has_fatal_error:
            return
        self._write_runtime_status_safe("disconnected", platform_state="disconnected", error_code=None, error_message=None)

    def _set_fatal_error(self, code: str, message: str, *, retryable: bool) -> None:
        self._running = False
        self._fatal_error_code = code
        self._fatal_error_message = message
        self._fatal_error_retryable = retryable
        self._write_runtime_status_safe("fatal", platform_state="fatal", error_code=code, error_message=message)

    def _write_runtime_status_safe(self, context: str, **kwargs) -> None:
        """Write runtime status; log first failure per context at warning, rest at debug.

        Status writes can fail on permissions, ENOSPC, missing status dir, etc.
        A persistently failing status dir used to be silent (``except: pass``).
        Logging every failure would spam the log on reconnect loops, so this
        surfaces the first failure per (platform, context) at warning level and
        downgrades subsequent failures to debug.
        """
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(platform=self.platform.value, **kwargs)
        except Exception as exc:
            # Use getattr so object.__new__(...) test harnesses that skip __init__
            # don't blow up on attribute access.
            logged = getattr(self, "_status_write_logged", None)
            if logged is None:
                logged = set()
                try:
                    self._status_write_logged = logged
                except Exception:
                    pass
            key = (self.platform.value, context)
            if key not in logged:
                logger.warning(
                    "Failed to write runtime status (%s) for %s: %s (further failures at debug level)",
                    context, self.platform.value, exc,
                )
                logged.add(key)
            else:
                logger.debug("Failed to write runtime status (%s) for %s: %s", context, self.platform.value, exc)

    async def _notify_fatal_error(self) -> None:
        handler = self._fatal_error_handler
        if not handler:
            return
        result = handler(self)
        if asyncio.iscoroutine(result):
            await result

    def _acquire_platform_lock(self, scope: str, identity: str, resource_desc: str) -> bool:
        """Acquire a scoped lock for this adapter. Returns True on success."""
        from gateway.status import acquire_scoped_lock
        self._platform_lock_scope = scope
        self._platform_lock_identity = identity
        acquired, existing = acquire_scoped_lock(
            scope, identity, metadata={'platform': self.platform.value}
        )
        if acquired:
            return True
        owner_pid = existing.get('pid') if isinstance(existing, dict) else None
        message = (
            f'{resource_desc} already in use'
            + (f' (PID {owner_pid})' if owner_pid else '')
            + '. Stop the other gateway first.'
        )
        logger.error('[%s] %s', self.name, message)
        self._set_fatal_error(f'{scope}_lock', message, retryable=True)
        return False

    def _release_platform_lock(self) -> None:
        """Release the scoped lock acquired by _acquire_platform_lock."""
        identity = getattr(self, '_platform_lock_identity', None)
        if not identity:
            return
        from gateway.status import release_scoped_lock
        release_scoped_lock(self._platform_lock_scope, identity)
        self._platform_lock_identity = None

    @property
    def name(self) -> str:
        """Human-readable name for this adapter."""
        return self.platform.value.title()
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is currently connected."""
        return self._running
    
    def set_message_handler(self, handler: MessageHandler) -> None:
        """
        Set the handler for incoming messages.
        
        The handler receives a MessageEvent and should return
        an optional response string.
        """
        self._message_handler = handler

    def set_topic_recovery_fn(
        self,
        fn: Optional[Callable[[Any], Optional[str]]],
    ) -> None:
        """Install a thread_id-recovery hook (Telegram DM topic mode).

        The hook is called with ``event.source`` before session keying;
        a non-None return value replaces ``source.thread_id``. Pass
        ``None`` to clear the hook.
        """
        # Guard against subclasses that initialize via ``object.__new__`` in
        # tests and never run ``BasePlatformAdapter.__init__``.
        self._topic_recovery_fn = fn  # type: ignore[attr-defined]

    def _apply_topic_recovery(self, event: MessageEvent) -> None:
        """Rewrite ``event.source.thread_id`` in place if the hook returns one."""
        recover = getattr(self, "_topic_recovery_fn", None)
        if recover is None:
            return
        source = getattr(event, "source", None)
        if source is None:
            return
        try:
            recovered = recover(source)
        except Exception:
            logger.debug("topic recovery hook failed", exc_info=True)
            return
        if recovered is None or str(recovered) == str(source.thread_id or ""):
            return
        try:
            event.source = dataclasses.replace(source, thread_id=str(recovered))
        except Exception:
            logger.debug("topic recovery rewrite failed", exc_info=True)

    def set_busy_session_handler(self, handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]]) -> None:
        """Set an optional handler for messages arriving during active sessions."""
        self._busy_session_handler = handler

    def set_authorization_check(
        self,
        callback: Optional[Callable[[str, Optional[str], Optional[str]], bool]],
    ) -> None:
        """Register a platform-bound authorization check.

        The callback signature is ``(user_id, chat_type, chat_id) -> bool``.
        It is used by adapters that pull external context (e.g. Slack thread
        replies via ``conversations.replies``) to flag messages from senders
        that are not on the configured allowlist, so the LLM can treat them
        as unverified background reference rather than authoritative input.
        """
        self._authorization_check = callback

    def _is_sender_authorized(
        self,
        user_id: Optional[str],
        chat_type: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Optional[bool]:
        """Return whether ``user_id`` is on the allowlist, if a check is configured.

        Returns ``True``/``False`` when an authorization check has been
        registered via :meth:`set_authorization_check`. Returns ``None``
        when no check is registered (caller should treat as "trust unknown"
        and preserve legacy behaviour).
        """
        if not user_id or self._authorization_check is None:
            return None
        try:
            return bool(self._authorization_check(user_id, chat_type, chat_id))
        except Exception:
            logger.warning(
                "[%s] Authorization check raised for user %s; treating as unknown",
                self.name, user_id, exc_info=True,
            )
            return None
    
    def set_session_store(self, session_store: Any) -> None:
        """
        Set the session store for checking active sessions.
        
        Used by adapters that need to check if a thread/conversation
        has an active session before processing messages (e.g., Slack
        thread replies without explicit mentions).
        """
        self._session_store = session_store
    
    @abstractmethod
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """
        Connect to the platform and start receiving messages.

        Args:
            is_reconnect: False on a cold first boot (the gateway is
                starting this platform for the first time); True when the
                reconnect watcher is re-establishing a platform that was
                previously running and dropped after an outage. Adapters
                that buffer a server-side update queue (e.g. Telegram's Bot
                API) should preserve that queue when ``is_reconnect`` is
                True so messages sent during the outage are delivered rather
                than silently discarded. Adapters with no such queue may
                ignore the flag.

        Returns True if connection was successful.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""
        pass
    
    @abstractmethod
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """
        Send a message to a chat.
        
        Args:
            chat_id: The chat/channel ID to send to
            content: Message content (may be markdown)
            reply_to: Optional message ID to reply to
            metadata: Additional platform-specific options
        
        Returns:
            SendResult with success status and message ID
        """
        pass

    # Default: the adapter treats ``finalize=True`` on edit_message as a
    # no-op and is happy to have the stream consumer skip redundant final
    # edits.  Subclasses that *require* an explicit finalize call to close
    # out the message lifecycle (e.g. rich card / AI assistant surfaces
    # such as DingTalk AI Cards) override this to True (class attribute or
    # property) so the stream consumer knows not to short-circuit.
    REQUIRES_EDIT_FINALIZE: bool = False

    async def create_handoff_thread(
        self,
        parent_chat_id: str,
        name: str,
    ) -> Optional[str]:
        """Create a fresh thread under ``parent_chat_id`` for a session handoff.

        Used by the gateway's handoff watcher when transferring a CLI
        session to a thread-capable platform — the new thread isolates the
        handed-off conversation from any pre-existing chat in the home
        channel and gives users a clean per-handoff scrollback.

        Returns the new thread/topic id (as a string) on success, or
        ``None`` if the platform doesn't support threading or the
        attempt failed (permissions, topics-mode off, etc.). When ``None``
        is returned the watcher falls back to using ``parent_chat_id``
        directly.

        Default implementation returns ``None`` — adapters that support
        threads override this. See:
          - Telegram: forum topics in groups, DM topics with bot API 9.4+
          - Discord:  text-channel threads (1440-min auto-archive)
          - Slack:    seed-message thread anchoring
        """
        return None


    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """
        Edit a previously sent message. Optional — platforms that don't
        support editing return success=False and callers fall back to
        sending a new message.

        ``finalize`` signals that this is the last edit in a streaming
        sequence.  Most platforms (Telegram, Slack, Discord, Matrix,
        etc.) treat it as a no-op because their edit APIs have no notion
        of message lifecycle state — an edit is an edit.  Platforms that
        render streaming updates with a distinct "in progress" state and
        require explicit closure (e.g. rich card / AI assistant surfaces
        such as DingTalk AI Cards) use it to finalize the message and
        transition the UI out of the streaming indicator — those should
        also set ``REQUIRES_EDIT_FINALIZE = True`` so callers route a
        final edit through even when content is unchanged.  Callers
        should set ``finalize=True`` on the final edit of a streamed
        response (typically when ``got_done`` fires in the stream
        consumer) and leave it ``False`` on intermediate edits.
        """
        return SendResult(success=False, error="Not supported")

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> bool:
        """
        Delete a previously sent message.  Optional — platforms that don't
        support deletion return ``False`` and callers fall back to leaving
        the message in place.

        Used by the stream consumer's fresh-final cleanup path (see
        openclaw/openclaw#72038) to remove long-lived preview messages
        after sending the completed reply as a fresh message so the
        platform's visible timestamp reflects completion time.

        Returns ``True`` on successful deletion, ``False`` otherwise.
        Subclasses should override for platforms with a deletion API
        (e.g. Telegram ``deleteMessage``).
        """
        return False

    def _get_ephemeral_system_ttl_default(self) -> int:
        """Read ``display.ephemeral_system_ttl`` from config.

        Returns the TTL in seconds to use when an :class:`EphemeralReply`
        does not specify one explicitly.  ``0`` (the default) disables
        auto-deletion.  Non-fatal if config is unreadable.
        """
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            return 0
        try:
            cfg = _load_config()
        except Exception:
            return 0
        display = cfg.get("display", {}) if isinstance(cfg, dict) else {}
        if not isinstance(display, dict):
            return 0
        raw = display.get("ephemeral_system_ttl", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _schedule_ephemeral_delete(
        self,
        chat_id: str,
        message_id: str,
        ttl_seconds: int,
    ) -> None:
        """Spawn a detached task that deletes ``message_id`` after ``ttl_seconds``.

        Best-effort — failures (gateway restart, permission denied, message
        too old for Telegram's 48h window) are swallowed at debug level.
        Does not block the caller.
        """

        async def _run_delete() -> None:
            try:
                await asyncio.sleep(max(1, int(ttl_seconds)))
                await self.delete_message(chat_id=chat_id, message_id=message_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(
                    "[%s] Ephemeral delete failed for %s/%s: %s",
                    self.name, chat_id, message_id, e,
                )

        coro = _run_delete()
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop (e.g. unit tests that never reach the async
            # path).  Close the coroutine cleanly so Python doesn't warn
            # about it never being awaited, then drop silently.
            coro.close()

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a three-option slash-command confirmation prompt.

        Used by the gateway's generic slash-confirm primitive (see
        ``GatewayRunner._request_slash_confirm``) for commands that have a
        non-destructive but expensive side effect the user should explicitly
        acknowledge — the current caller is ``/reload-mcp``, which
        invalidates the provider prompt cache.

        Platforms with inline-button support (Telegram, Discord, Slack,
        Matrix, Feishu) should override this to render three buttons:
        Approve Once / Always Approve / Cancel.  Button callbacks MUST be
        routed back through the gateway by calling
        ``GatewayRunner._resolve_slash_confirm(confirm_id, choice)`` where
        ``choice`` is ``"once"`` / ``"always"`` / ``"cancel"``.

        Platforms without button UIs leave this as the default and fall
        through to the gateway's text fallback (which sends ``message`` as
        plain text and intercepts the next ``/approve`` / ``/always`` /
        ``/cancel`` reply).

        ``confirm_id`` is a short string generated by the gateway; the
        adapter stores it alongside any platform-specific state needed to
        route the callback (e.g. Telegram's ``_approval_state`` dict).
        """
        return SendResult(success=False, error="Not supported")

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a clarify prompt to the user.

        Two render modes:

          * **Multiple choice** (``choices`` is a non-empty list) — adapters
            that override this should render inline buttons (one per choice
            plus a final "Other" / free-text option).  Button callbacks
            MUST resolve via
            ``tools.clarify_gateway.resolve_gateway_clarify(clarify_id, response)``
            with the chosen string.  Picking the "Other" button calls
            ``mark_awaiting_text(clarify_id)`` so the next message in the
            session is captured as the response.

          * **Open-ended** (``choices`` is None or empty) — render the
            question as a plain text message; the next user message in the
            session is captured by the gateway's text-intercept and
            resolves the clarify automatically (see
            ``GatewayRunner._maybe_intercept_clarify_text``).

        The default implementation falls back to a numbered text list,
        which works on every platform — the user replies with a number
        ("2") or with the literal choice text, and the gateway intercepts
        and resolves.  For the text fallback path, the default calls
        ``mark_awaiting_text()`` so that the gateway text-intercept
        (:meth:`GatewayRunner._maybe_intercept_clarify_text`) catches the
        user's reply instead of timing out.
        Adapters with native button UIs (Telegram, Discord) SHOULD
        override this for a richer UX.
        """
        if choices:
            lines = [f"❓ {question}", ""]
            for i, choice in enumerate(choices, start=1):
                lines.append(f"  {i}. {choice}")
            lines.append("")
            lines.append("Reply with the number, the option text, or your own answer.")
            text = "\n".join(lines)
            # Text fallback: enable text-capture so the gateway intercept
            # picks up the user's typed reply (e.g. "2" or choice text).
            from tools.clarify_gateway import mark_awaiting_text
            mark_awaiting_text(clarify_id)
        else:
            text = f"❓ {question}"
        return await self.send(
            chat_id=chat_id,
            content=text,
            metadata=metadata,
        )

    async def send_private_notice(
        self,
        chat_id: str,
        user_id: Optional[str],
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a notice privately when the platform supports it.

        The default implementation falls back to a normal send so callers can
        use one code path across platforms.
        """
        return await self.send(
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """
        Send a typing indicator.
        
        Override in subclasses if the platform supports it.
        metadata: optional dict with platform-specific context (e.g. thread_id for Slack).
        """
        pass

    async def stop_typing(self, chat_id: str) -> None:
        """Stop a persistent typing indicator (if the platform uses one).

        Override in subclasses that start background typing loops.
        Default is a no-op for platforms with one-shot typing indicators.
        """
        pass

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images.

        Accepts ``http(s)://``, ``file://`` URIs in the first tuple
        element.

        Default implementation sends each item individually,
        routing animated GIFs through ``send_animation`` and local
        files through ``send_image_file``.

        Override in subclasses to bundle into a single native API call
        (e.g. Signal's multi-attachment RPC)
        """
        from urllib.parse import unquote as _unquote

        for image_url, alt_text in images:
            if human_delay > 0:
                await asyncio.sleep(human_delay)
            try:
                logger.info(
                    "[%s] Sending image: %s (alt=%s)",
                    self.name,
                    safe_url_for_log(image_url),
                    alt_text[:30] if alt_text else "",
                )
                if image_url.startswith("file://"):
                    img_result = await self.send_image_file(
                        chat_id=chat_id,
                        image_path=_unquote(image_url[7:]),
                        caption=alt_text if alt_text else None,
                        metadata=metadata,
                    )
                elif self._is_animation_url(image_url):
                    img_result = await self.send_animation(
                        chat_id=chat_id,
                        animation_url=image_url,
                        caption=alt_text if alt_text else None,
                        metadata=metadata,
                    )
                else:
                    img_result = await self.send_image(
                        chat_id=chat_id,
                        image_url=image_url,
                        caption=alt_text if alt_text else None,
                        metadata=metadata,
                    )
                if not img_result.success:
                    logger.error("[%s] Failed to send image: %s", self.name, img_result.error)
            except Exception as img_err:
                logger.error("[%s] Error sending image: %s", self.name, img_err, exc_info=True)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send an image natively via the platform API.
        
        Override in subclasses to send images as proper attachments
        instead of plain-text URLs. Default falls back to sending the
        URL as a text message.
        """
        # Fallback: send URL as text (subclasses override for native images)
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)
    
    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send an animated GIF natively via the platform API.
        
        Override in subclasses to send GIFs as proper animations
        (e.g., Telegram send_animation) so they auto-play inline.
        Default falls back to send_image.
        """
        return await self.send_image(chat_id=chat_id, image_url=animation_url, caption=caption, reply_to=reply_to, metadata=metadata)
    
    @staticmethod
    def _is_animation_url(url: str) -> bool:
        """Check if a URL points to an animated GIF (vs a static image)."""
        lower = url.lower().split('?')[0]  # Strip query params
        return lower.endswith('.gif')

    @staticmethod
    def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Extract image URLs from markdown and HTML image tags in a response.
        
        Finds patterns like:
        - ![alt text](https://example.com/image.png)
        - <img src="https://example.com/image.png">
        - <img src="https://example.com/image.png"></img>
        
        Args:
            content: The response text to scan.
        
        Returns:
            Tuple of (list of (url, alt_text) pairs, cleaned content with image tags removed).
        """
        images = []
        cleaned = content
        
        # Match markdown images: ![alt](url)
        md_pattern = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
        for match in re.finditer(md_pattern, content):
            alt_text = match.group(1)
            url = match.group(2)
            # Only extract URLs that look like actual images
            if any(url.lower().endswith(ext) or ext in url.lower() for ext in
                   ['.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery']):
                images.append((url, alt_text))
        
        # Match HTML img tags: <img src="url"> or <img src="url"></img> or <img src="url"/>
        html_pattern = r'<img\s+src=["\']?(https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
        for match in re.finditer(html_pattern, content):
            url = match.group(1)
            images.append((url, ""))
        
        # Remove only the matched image tags from content (not all markdown images)
        if images:
            extracted_urls = {url for url, _ in images}
            def _remove_if_extracted(match):
                url = match.group(2) if match.lastindex >= 2 else match.group(1)
                return '' if url in extracted_urls else match.group(0)
            cleaned = re.sub(md_pattern, _remove_if_extracted, cleaned)
            cleaned = re.sub(html_pattern, _remove_if_extracted, cleaned)
            # Clean up leftover blank lines
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return images, cleaned
    
    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send an audio file as a native voice message via the platform API.

        Override in subclasses to send audio as voice bubbles (Telegram)
        or file attachments (Discord). Default falls back to a friendly
        notice — never echo the local audio_path into chat, since it is a
        host filesystem path that would leak the Hermes home layout.
        """
        # audio_path is intentionally NOT included in the chat text — it is a
        # host-local path that leaks filesystem layout. The path is logged for
        # operator diagnostics instead.
        logger.warning(
            "[%s] send_voice fallback: native audio send unavailable for %s",
            self.name, audio_path,
        )
        text = "⚠️ Couldn't deliver the audio attachment."
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    def prepare_tts_text(self, text: str) -> str:
        """Prepare text for TTS. Override to filter tool output, code, etc.

        Default strips markdown formatting and truncates to 4000 chars.
        """
        return re.sub(r'[*_`#\[\]()]', '', text)[:4000].strip()

    async def play_tts(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        """
        Play auto-TTS audio for voice replies.

        Override in subclasses for invisible playback (e.g. Web UI).
        Default falls back to send_voice (shows audio player).
        """
        return await self.send_voice(chat_id=chat_id, audio_path=audio_path, **kwargs)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a video natively via the platform API.

        Override in subclasses to send videos as inline playable media.
        Default falls back to a friendly notice — never echo the local
        video_path into chat, since it is a host filesystem path that
        would leak the Hermes home layout.
        """
        # See send_voice for the rationale: do not echo host paths into chat.
        logger.warning(
            "[%s] send_video fallback: native video send unavailable for %s",
            self.name, video_path,
        )
        text = "⚠️ Couldn't deliver the video attachment."
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a document/file natively via the platform API.

        Override in subclasses to send files as downloadable attachments.
        Default falls back to a friendly notice — never echo the local
        file_path into chat, since it is a host filesystem path that
        would leak the Hermes home layout.
        """
        # See send_voice for the rationale: do not echo host paths into chat.
        logger.warning(
            "[%s] send_document fallback: native file send unavailable for %s",
            self.name, file_path,
        )
        # file_name is supplied by callers and represents the user-facing
        # filename (already non-sensitive — it is what the agent named the
        # output). Only show it when the caller passed one explicitly.
        if file_name:
            text = f"⚠️ Couldn't deliver the file attachment ({file_name})."
        else:
            text = "⚠️ Couldn't deliver the file attachment."
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """
        Send a local image file natively via the platform API.

        Unlike send_image() which takes a URL, this takes a local file path.
        Override in subclasses for native photo attachments. Default falls
        back to a friendly notice — never echo the local image_path into
        chat, since it is a host filesystem path that would leak the
        Hermes home layout.
        """
        # See send_voice for the rationale: do not echo host paths into chat.
        logger.warning(
            "[%s] send_image_file fallback: native image send unavailable for %s",
            self.name, image_path,
        )
        text = "⚠️ Couldn't deliver the image attachment."
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    @staticmethod
    def validate_media_delivery_path(path: str) -> Optional[str]:
        """Return a resolved path if it is safe for native attachment upload."""
        return validate_media_delivery_path(path)

    @staticmethod
    def filter_media_delivery_paths(media_files) -> List[Tuple[str, bool]]:
        """Drop unsafe MEDIA paths and normalize accepted paths."""
        safe_media: List[Tuple[str, bool]] = []
        for media_path, is_voice in media_files or []:
            raw = str(media_path)
            safe_path = validate_media_delivery_path(raw)
            if safe_path:
                safe_media.append((safe_path, bool(is_voice)))
            else:
                logger.warning("Skipping unsafe MEDIA directive path: %s", _log_safe_path(raw))
        return safe_media

    @staticmethod
    def filter_local_delivery_paths(file_paths) -> List[str]:
        """Drop unsafe bare local file paths and normalize accepted paths."""
        safe_paths: List[str] = []
        for file_path in file_paths or []:
            raw = str(file_path)
            safe_path = validate_media_delivery_path(raw)
            if safe_path:
                safe_paths.append(safe_path)
            else:
                logger.warning("Skipping unsafe local file path: %s", _log_safe_path(raw))
        return safe_paths


    @staticmethod
    def _mask_protected_spans(content: str) -> str:
        """Replace content inside fenced code blocks, inline code spans,
        and blockquotes with spaces to prevent MEDIA: false positives.

        Preserves character count so regex match offsets stay valid.
        Skips masking backtick-quoted paths in MEDIA: tags (e.g.
        ``MEDIA:`/path/to/file.png` ``) to avoid breaking path extraction.
        """
        chars = list(content)
        n = len(chars)

        # Build list of (start, end) spans to mask
        spans: list = []

        # Fenced code blocks: ```...```
        for m in re.finditer(r'```[^\n]*\n.*?```', content, re.DOTALL):
            spans.append((m.start(), m.end()))

        # Inline code: `...` but NOT backtick-quoted paths in MEDIA: tags
        for m in re.finditer(r'`[^`\n]+`', content):
            start = m.start()
            # Check if this is a backtick-quoted path after MEDIA:
            prefix = content[max(0, start - 20):start]
            if re.search(r'MEDIA:\s*$', prefix):
                continue  # This is a MEDIA path quote, not inline code
            spans.append((start, m.end()))

        # Blockquote lines: > at line start
        for m in re.finditer(r'^>.*$', content, re.MULTILINE):
            spans.append((m.start(), m.end()))

        # Apply masking
        for start, end in spans:
            for i in range(start, end):
                if chars[i] != '\n':
                    chars[i] = ' '

        return ''.join(chars)


    @staticmethod
    def _mask_json_string_media(content: str) -> str:
        """Blank out ``MEDIA:<bare-path>`` occurrences that sit inside a JSON
        string *value* so they are never delivered as real attachments.

        Serialized tool results frequently embed a previous reply's text, e.g.::

            {"result": "MEDIA:/Users/x/.hermes/media/generated/stale.png"}

        Here the ``MEDIA:`` is part of stored text, not an outbound directive,
        but the bare-path branch of ``MEDIA_TAG_CLEANUP_RE`` would still match it
        and re-deliver a stale file. (Regression report #34375.)

        The discriminator is precise so legitimate tags are untouched:

        * Only spans opened by a JSON value-context quote (``:``, ``,``, ``{`` or
          ``[`` immediately before the ``"``) are considered.
        * Within such a span, only a ``MEDIA:`` followed by a **bare** path
          (``/``, ``~/`` or ``X:\\``) is masked. A ``MEDIA:"..."`` quoted-path
          tag — a real LLM output format the extractor supports — is not bare and
          is left alone.
        * Tags at line start, after prose whitespace, or indented are outside any
          JSON value span and are never affected.

        Offsets are preserved (matched chars replaced with spaces, newlines kept)
        so downstream match positions stay valid.
        """
        if '"' not in content or "MEDIA:" not in content:
            return content
        chars = list(content)
        # JSON value-context string: a quote preceded by : , { or [ (optional ws),
        # capturing the (escape-aware) string body up to the closing quote.
        for m in re.finditer(r'(?<=[:,{\[])\s*"((?:[^"\\\n]|\\.)*)"', content):
            seg = m.group(1)
            if re.search(r'MEDIA:\s*(?:~/|/|[A-Za-z]:[/\\])', seg):
                for i in range(m.start(1), m.end(1)):
                    if chars[i] != '\n':
                        chars[i] = ' '
        return ''.join(chars)

    @staticmethod
    def extract_media(content: str) -> Tuple[List[Tuple[str, bool]], str]:
        """
        Extract MEDIA:<path> tags and [[audio_as_voice]] directives from response text.

        The TTS tool returns responses like:
            [[audio_as_voice]]
            MEDIA:/path/to/audio.ogg

        Skills that produce large/lossless images (e.g. info-graph, where a
        rendered JPG is 1-2 MB but Telegram's sendPhoto recompresses to
        ~200 KB at 1280px) can use ``[[as_document]]`` to request unmodified
        delivery via sendDocument instead of sendPhoto/sendMediaGroup. The
        directive is detected at the dispatch sites (which have access to the
        original response); this method just strips it so it never leaks into
        user-visible text. Per-file granularity is intentionally not exposed —
        when an agent emits ``[[as_document]]`` once, every image path in the
        same response is delivered as a document, mirroring the all-or-nothing
        scope of ``[[audio_as_voice]]``.

        Args:
            content: The response text to scan.

        Returns:
            Tuple of (list of (path, is_voice) pairs, cleaned content with tags removed).
        """
        media = []
        cleaned = content

        # Check for [[audio_as_voice]] directive
        has_voice_tag = "[[audio_as_voice]]" in content
        cleaned = cleaned.replace("[[audio_as_voice]]", "")
        # Strip [[as_document]] directive — callers inspect the original
        # ``content`` for it (so they can still react to it); here we just
        # keep it out of the user-visible cleaned text.
        cleaned = cleaned.replace("[[as_document]]", "")
        
        # Extract MEDIA:<path> tags, allowing optional whitespace after the colon
        # and quoted/backticked paths for LLM-formatted outputs. The extension
        # set is the shared MEDIA_DELIVERY_EXTS source of truth (built once into
        # MEDIA_TAG_CLEANUP_RE) so it can never drift from extract_local_files.
        media_pattern = MEDIA_TAG_CLEANUP_RE
        # Mask example/stored MEDIA: paths before scanning so they are never
        # delivered as real attachments:
        #  - code blocks / inline code / blockquotes hold prose examples (#35695)
        #  - serialized JSON string values hold stored tool-result text (#34375)
        # Both maskers are offset-preserving (chars -> spaces) so match offsets
        # stay valid; chaining them masks the union of both protected regions.
        scan_content = BasePlatformAdapter._mask_protected_spans(content)
        scan_content = BasePlatformAdapter._mask_json_string_media(scan_content)
        for match in media_pattern.finditer(scan_content):
            path = _normalize_media_tag_path(match.group("path"))
            if path:
                try:
                    media.append((os.path.expanduser(path), has_voice_tag))
                except (OSError, RuntimeError, ValueError):
                    # Skip a crafted ~\x00 path rather than aborting extraction
                    # and dropping every other attachment in the response.
                    continue

        seen_paths = {p for p, _ in media}
        for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(scan_content):
            path = _normalize_media_tag_path(match.group("path"))
            if not path or not _path_lacks_deliverable_extension(path):
                continue
            safe = validate_media_delivery_path(path)
            if safe and safe not in seen_paths:
                media.append((safe, has_voice_tag))
                seen_paths.add(safe)

        # Remove the delivered MEDIA tags from the user-visible text. Mask a
        # length-equal copy of ``cleaned`` (same union of protected regions) to
        # *locate* the real tag spans, then delete exactly those spans from the
        # *unmasked* ``cleaned``. Masking is only a locator — protected spans
        # (code blocks, quotes, JSON-embedded MEDIA: text) must survive verbatim
        # in the delivered text, not be blanked to whitespace. Masking
        # ``cleaned`` (not ``content``) keeps offsets valid after the
        # [[audio_as_voice]] / [[as_document]] directives are removed.
        if media:
            masked_cleaned = BasePlatformAdapter._mask_protected_spans(cleaned)
            masked_cleaned = BasePlatformAdapter._mask_json_string_media(masked_cleaned)
            spans = [m.span() for m in media_pattern.finditer(masked_cleaned)]
            for match in MEDIA_EXTENSIONLESS_TAG_RE.finditer(masked_cleaned):
                path = _normalize_media_tag_path(match.group("path"))
                if not path or not _path_lacks_deliverable_extension(path):
                    continue
                if validate_media_delivery_path(path):
                    spans.append(match.span())
            if spans:
                chars = list(cleaned)
                for start, end in sorted(spans, reverse=True):
                    del chars[start:end]
                cleaned = "".join(chars)
                cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return media, cleaned

    @staticmethod
    def strip_media_directives_for_display(text: str) -> str:
        """Strip MEDIA: directives from streamed/display text.

        Known-extension tags are removed unconditionally (same as
        ``MEDIA_TAG_CLEANUP_RE``). Extension-less tags are removed only when
        ``validate_media_delivery_path`` accepts the path so undeliverable
        paths stay visible for debugging.
        """
        if (
            "MEDIA:" not in text
            and "[[audio_as_voice]]" not in text
            and "[[as_document]]" not in text
        ):
            return text
        cleaned = _strip_media_tag_directives(text)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.rstrip()

    @staticmethod
    def extract_local_files(content: str) -> Tuple[List[str], str]:
        """
        Detect bare local file paths in response text for native delivery.

        Matches absolute paths (/...) and tilde paths (~/) ending in common
        image, video, audio, or document extensions.  Validates each
        candidate with ``os.path.isfile()`` to avoid false positives from
        URLs or non-existent paths.

        The extension list is broader than just images/video so the agent
        can produce arbitrary artifacts (charts, PDFs, spreadsheets, code
        archives, CSVs) and have them ship to the user as native uploads
        without needing an explicit ``MEDIA:`` tag.  Image / video
        extensions still embed inline where the platform supports it;
        document extensions route through ``send_document``.  The dispatch
        partition lives in ``gateway/run.py``.

        Paths inside fenced code blocks (``` ... ```) and inline code
        (`...`) are ignored so that code samples are never mutilated.

        Returns:
            Tuple of (list of expanded file paths, cleaned text with the
            raw path strings removed).
        """
        _LOCAL_MEDIA_EXTS = MEDIA_DELIVERY_EXTS
        ext_part = '|'.join(e.lstrip('.') for e in _LOCAL_MEDIA_EXTS)

        # (?<![/:\w.]) prevents matching inside URLs (e.g. https://…/img.png)
        #             and relative paths (./foo.png)
        # (?:~/|/)    anchors to absolute or home-relative Unix paths
        # (?:[A-Za-z]:[/\\]) anchors to Windows drive-letter paths (#34632)
        path_re = re.compile(
            r'(?<![/:\w.])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.(?:' + ext_part + r')\b',
            re.IGNORECASE,
        )

        # Build spans covered by fenced code blocks and inline code
        code_spans: list = []
        for m in re.finditer(r'```[^\n]*\n.*?```', content, re.DOTALL):
            code_spans.append((m.start(), m.end()))
        for m in re.finditer(r'`[^`\n]+`', content):
            code_spans.append((m.start(), m.end()))

        def _in_code(pos: int) -> bool:
            return any(s <= pos < e for s, e in code_spans)

        found: list = []  # (raw_match_text, expanded_path)
        for match in path_re.finditer(content):
            if _in_code(match.start()):
                continue
            raw = match.group(0)
            expanded = os.path.expanduser(raw)
            if os.path.isfile(expanded):
                found.append((raw, expanded))
            else:
                # The reply mentions a deliverable-looking path that does not
                # exist on disk, so it is silently dropped from native delivery.
                # This is the most common reason a promised file never arrives
                # (the model said "here's your file" but never wrote it, or
                # referenced the wrong path). Log it so the gap is visible in
                # gateway.log rather than vanishing without a trace.
                logger.info(
                    "Skipping bare file path in reply (no file on disk): %s",
                    _log_safe_path(raw),
                )

        # Deduplicate by expanded path, preserving discovery order
        seen: set = set()
        unique: list = []
        for raw, expanded in found:
            if expanded not in seen:
                seen.add(expanded)
                unique.append((raw, expanded))

        paths = [expanded for _, expanded in unique]

        cleaned = content
        if unique:
            for raw, _exp in unique:
                cleaned = cleaned.replace(raw, '')
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return paths, cleaned

    async def _keep_typing(
        self,
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """
        Continuously send typing indicator until cancelled.
        
        Telegram/Discord typing status expires after ~5 seconds, so we refresh every 2
        to recover quickly after progress messages interrupt it.
        
        Skips send_typing when the chat is in ``_typing_paused`` (e.g. while
        the agent is waiting for dangerous-command approval).  This is critical
        for Slack's Assistant API where ``assistant_threads_setStatus`` disables
        the compose box — pausing lets the user type ``/approve`` or ``/deny``.

        Each ``send_typing`` call is bounded by a ~1.5s timeout so a slow
        network round-trip can't stall the refresh cadence.  Telegram- and
        Discord-side typing expire after ~5s; if any individual send_typing
        takes longer than the refresh interval, the bubble would die and
        stay dead until that call returns.  Abandoning the slow call lets
        the next tick fire a fresh send_typing on schedule — as long as
        one of them succeeds within the 5s platform-side window, the bubble
        stays visible across provider stalls / upstream API timeouts.
        """
        # Bound each send_typing round-trip so the refresh cadence isn't
        # gated on network health.  Must stay below ``interval`` so a slow
        # call gets abandoned before the next scheduled tick.
        _send_typing_timeout = max(0.25, min(1.5, interval - 0.25))
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                if chat_id not in self._typing_paused:
                    try:
                        await asyncio.wait_for(
                            self.send_typing(chat_id, metadata=metadata),
                            timeout=_send_typing_timeout,
                        )
                    except asyncio.TimeoutError:
                        # Slow network — abandon this tick, keep the loop
                        # on schedule so the next send_typing fires fresh.
                        pass
                    except asyncio.CancelledError:
                        raise
                    except Exception as typing_err:
                        logger.debug(
                            "[%s] send_typing error (non-fatal): %s",
                            self.name, typing_err,
                        )
                if stop_event is None:
                    await asyncio.sleep(interval)
                    continue
                loop = asyncio.get_running_loop()
                deadline = loop.time() + interval
                while not stop_event.is_set():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    # Poll instead of wait_for(stop_event.wait()).  Cancelling
                    # wait_for while it owns the inner Event.wait task can leave
                    # shutdown paths stuck awaiting the typing task on Python
                    # 3.11/pytest-asyncio; sleep cancellation is immediate.
                    await asyncio.sleep(min(0.25, remaining))
                if stop_event.is_set():
                    return
        except asyncio.CancelledError:
            pass  # Normal cancellation when handler completes
        finally:
            # Ensure the underlying platform typing loop is stopped.
            # _keep_typing may have called send_typing() after an outer
            # stop_typing() cleared the task dict, recreating the loop.
            # Cancelling _keep_typing alone won't clean that up.
            if hasattr(self, "stop_typing"):
                try:
                    await self.stop_typing(chat_id)
                except Exception:
                    pass
            self._typing_paused.discard(chat_id)

    async def _stop_typing_refresh(
        self,
        chat_id: str,
        typing_task: asyncio.Task | None = None,
        *,
        timeout: float = 0.5,
        stop_attempts: int = 2,
    ) -> None:
        """Stop the refresh task and platform typing state as one operation."""
        self._typing_paused.add(chat_id)
        try:
            if typing_task is not None and not typing_task.done():
                typing_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(typing_task), timeout=timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # The task is cancelled; don't let a slow adapter-specific
                    # cleanup block response delivery or shutdown.
                    pass
            if not hasattr(self, "stop_typing"):
                return
            attempts = max(1, stop_attempts)
            for attempt in range(attempts):
                try:
                    await self.stop_typing(chat_id)
                except Exception:
                    pass
                if attempt < attempts - 1:
                    await asyncio.sleep(0)
        finally:
            self._typing_paused.discard(chat_id)

    def pause_typing_for_chat(self, chat_id: str) -> None:
        """Pause typing indicator for a chat (e.g. during approval waits).

        Thread-safe (CPython GIL) — can be called from the sync agent thread
        while ``_keep_typing`` runs on the async event loop.
        """
        self._typing_paused.add(chat_id)

    def resume_typing_for_chat(self, chat_id: str) -> None:
        """Resume typing indicator for a chat after approval resolves."""
        self._typing_paused.discard(chat_id)

    async def interrupt_session_activity(self, session_key: str, chat_id: str) -> None:
        """Signal the active session loop to stop and clear typing immediately."""
        if session_key:
            interrupt_event = self._active_sessions.get(session_key)
            if interrupt_event is not None:
                interrupt_event.set()
        try:
            await self.stop_typing(chat_id)
        except Exception:
            pass

    def register_post_delivery_callback(
        self,
        session_key: str,
        callback: Callable,
        *,
        generation: int | None = None,
    ) -> None:
        """Register a deferred callback to fire after the main response.

        ``generation`` lets callers tie the callback to a specific gateway run
        generation so stale runs cannot clear callbacks owned by a fresher run.

        If a callback for the same ``session_key`` (and generation, when set)
        is already registered, the new callback is chained — both fire, in
        registration order, with per-callback exception isolation. This lets
        independent features (background-review release + temporary-bubble
        cleanup) coexist without clobbering each other. Stale-generation
        callers never overwrite a fresher generation's slot.
        """
        if not session_key or not callable(callback):
            return

        existing = self._post_delivery_callbacks.get(session_key)
        if existing is not None:
            if isinstance(existing, tuple) and len(existing) == 2:
                existing_gen, existing_cb = existing
            else:
                existing_gen, existing_cb = None, existing
            # Stale-generation registrations never overwrite a fresher slot.
            if (
                existing_gen is not None
                and generation is not None
                and int(generation) < int(existing_gen)
            ):
                return
            # Same-or-newer generation: chain with the existing callback so
            # both fire in registration order.
            if callable(existing_cb) and (
                existing_gen is None
                or generation is None
                or int(existing_gen) == int(generation)
            ):
                _prev = existing_cb
                _new = callback

                async def _chained() -> None:
                    # Both _prev and _new may be sync or async. The chained
                    # wrapper itself must be async because the outer invoker
                    # (``_handle_message`` etc.) awaits awaitable callbacks; a
                    # sync wrapper here would call ``_prev()`` / ``_new()`` and
                    # silently drop any returned coroutine, breaking chained
                    # async post-delivery hooks (e.g. ``/goal`` continuations).
                    for _cb in (_prev, _new):
                        try:
                            _result = _cb()
                            if inspect.isawaitable(_result):
                                await _result
                        except Exception:
                            logger.debug(
                                "Post-delivery callback failed", exc_info=True
                            )

                callback = _chained

        if generation is None:
            self._post_delivery_callbacks[session_key] = callback
        else:
            self._post_delivery_callbacks[session_key] = (int(generation), callback)

    def pop_post_delivery_callback(
        self,
        session_key: str,
        *,
        generation: int | None = None,
    ) -> Callable | None:
        """Pop a deferred callback, optionally requiring generation ownership."""
        if not session_key:
            return None
        entry = self._post_delivery_callbacks.get(session_key)
        if entry is None:
            return None
        if isinstance(entry, tuple) and len(entry) == 2:
            entry_generation, callback = entry
            if generation is not None and int(entry_generation) != int(generation):
                return None
            self._post_delivery_callbacks.pop(session_key, None)
            return callback if callable(callback) else None
        if generation is not None:
            return None
        self._post_delivery_callbacks.pop(session_key, None)
        return entry if callable(entry) else None

    # ── Processing lifecycle hooks ──────────────────────────────────────────
    # Subclasses override these to react to message processing events
    # (e.g. Discord adds 👀/✅/❌ reactions).

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Hook called when background processing begins."""

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Hook called when background processing completes."""

    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        try:
            await hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s hook failed: %s", self.name, hook_name, e)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        """Return True if the error string looks like a transient network failure."""
        if not error:
            return False
        lowered = error.lower()
        return any(pat in lowered for pat in _RETRYABLE_ERROR_PATTERNS)

    @staticmethod
    def _is_timeout_error(error: Optional[str]) -> bool:
        """Return True if the error string indicates a read/write timeout.

        Timeout errors are NOT retryable and should NOT trigger plain-text
        fallback — the request may have already been delivered.
        """
        if not error:
            return False
        lowered = error.lower()
        return "timed out" in lowered or "readtimeout" in lowered or "writetimeout" in lowered

    def _unwrap_ephemeral(self, response: Any) -> Tuple[Optional[str], int]:
        """Unwrap a handler response into (text, ttl_seconds).

        Accepts a plain string, ``None``, or an :class:`EphemeralReply`.
        Returns ``(text, ttl)`` where ``ttl > 0`` means the caller should
        schedule a deletion via :meth:`_schedule_ephemeral_delete` after
        the send succeeds.  ``ttl`` is forced to 0 when the adapter
        doesn't override :meth:`delete_message` so non-supporting
        platforms silently degrade to normal sends.
        """
        if isinstance(response, EphemeralReply):
            ttl = response.ttl_seconds
            if ttl is None:
                try:
                    ttl = int(self._get_ephemeral_system_ttl_default())
                except Exception:
                    ttl = 0
            if ttl and ttl > 0 and type(self).delete_message is BasePlatformAdapter.delete_message:
                ttl = 0
            return response.text, int(ttl or 0)
        return response, 0

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ) -> "SendResult":
        """
        Send a message with automatic retry for transient network errors.

        On permanent failures (e.g. formatting / permission errors) falls back
        to a plain-text version before giving up. If all attempts fail due to
        network errors, sends the user a brief delivery-failure notice so they
        know to retry rather than waiting indefinitely.
        """

        result = await self.send(
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=metadata,
        )

        if result.success:
            return result

        # If the adapter explicitly reports partial delivery, do NOT retry
        # or fall back — some content is already visible in the chat.
        if effective_delivery_state(result) is FinalDeliveryState.PARTIALLY_DELIVERED:
            return result

        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)

        # Timeout errors are not safe to retry (message may have been
        # delivered) and not formatting errors — return the failure as-is.
        if not is_network and self._is_timeout_error(error_str):
            return result

        if is_network:
            # Retry with exponential backoff for transient errors.
            # Honor server-requested retry_after (e.g. Telegram FloodWait)
            # when present — it is authoritative over our backoff schedule.
            server_retry_after = result.retry_after
            for attempt in range(1, max_retries + 1):
                if server_retry_after is not None:
                    delay = server_retry_after + random.uniform(0, 1)
                    server_retry_after = None  # only honor once per send
                else:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "[%s] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                    self.name, attempt, max_retries, delay, error_str,
                )
                await asyncio.sleep(delay)
                result = await self.send(
                    chat_id=chat_id,
                    content=content,
                    reply_to=reply_to,
                    metadata=metadata,
                )
                if result.success:
                    logger.info("[%s] Send succeeded on retry %d", self.name, attempt)
                    return result
                # Partial delivery: do not retry or fall back.
                if effective_delivery_state(result) is FinalDeliveryState.PARTIALLY_DELIVERED:
                    return result
                error_str = result.error or ""
                if result.retry_after is not None:
                    server_retry_after = result.retry_after
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break  # error switched to non-transient — fall through to plain-text fallback
            else:
                # All retries exhausted (loop completed without break) — notify user
                logger.error("[%s] Failed to deliver response after %d retries: %s", self.name, max_retries, error_str)
                notice = (
                    "\u26a0\ufe0f Message delivery failed after multiple attempts. "
                    "Please try again \u2014 your request was processed but the response could not be sent."
                )
                try:
                    await self.send(chat_id=chat_id, content=notice, reply_to=reply_to, metadata=metadata)
                except Exception as notify_err:
                    logger.debug("[%s] Could not send delivery-failure notice: %s", self.name, notify_err)
                return result

        # Non-network / post-retry formatting failure: try plain text as fallback
        logger.warning("[%s] Send failed: %s — trying plain-text fallback", self.name, error_str)
        fallback_result = await self.send(
            chat_id=chat_id,
            content=f"(Response formatting failed, plain text:)\n\n{content[:3500]}",
            reply_to=reply_to,
            metadata=metadata,
        )
        if not fallback_result.success:
            logger.error("[%s] Fallback send also failed: %s", self.name, fallback_result.error)
        return fallback_result

    @staticmethod
    def _merge_caption(existing_text: Optional[str], new_text: str) -> str:
        """Merge a new caption into existing text, avoiding duplicates.

        Uses line-by-line exact match (not substring) to prevent false positives
        where a shorter caption is silently dropped because it appears as a
        substring of a longer one (e.g. "Meeting" inside "Meeting agenda").
        Whitespace is normalised for comparison.
        """
        if not existing_text:
            return new_text
        existing_captions = [c.strip() for c in existing_text.split("\n\n")]
        if new_text.strip() not in existing_captions:
            return f"{existing_text}\n\n{new_text}".strip()
        return existing_text

    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        store = getattr(self, "_text_debounce", None)
        if store is None:
            store = {}
            self._text_debounce = store
        return store

    def _is_queue_text_debounce_candidate(self, event: MessageEvent) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT
            and not getattr(event, "internal", False)
            and not event.is_command()
            and bool((event.text or "").strip())
        )
        if result:
            logger.debug(
                "[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                self.name,
                getattr(event, "session_key", "?"),
                len(event.text or ""),
            )
        return result

    def _can_merge_text_debounce_events(self, existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
            if sender:
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(source, "chat_id", None):
                return (platform, "dm", str(source.chat_id))
            return None

        existing_sender = _identity(existing)
        incoming_sender = _identity(event)
        return existing_sender is not None and existing_sender == incoming_sender

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        now = time.monotonic()
        window_deadline = state.last_ts + self._busy_text_debounce_seconds
        hard_cap_deadline = state.first_ts + self._busy_text_hard_cap_seconds
        return max(0.0, min(window_deadline, hard_cap_deadline) - now)

    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)

        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution in shared sessions. The current
            # buffer becomes the next pending turn; the new sender starts a
            # fresh debounce burst when the pending slot allows it.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    merge_pending_message_event(
                        self._pending_messages,
                        session_key,
                        event,
                        merge_text=True,
                    )
                return

        now = time.monotonic()
        if state is None:
            state = TextDebounceState(
                event=event,
                task=None,
                first_ts=now,
                last_ts=now,
            )
            store[session_key] = state
        else:
            if event.text:
                state.event.text = (
                    f"{state.event.text}\n{event.text}"
                    if state.event.text
                    else event.text
                )
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now

        if state.task is not None and not state.task.done():
            state.task.cancel()

        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))

    async def _flush_text_debounce(self, session_key: str, delay: float) -> None:
        """Timer task that flushes the debounced text buffer."""
        try:
            await asyncio.sleep(delay)
            await self._flush_text_debounce_now(session_key)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            state = self._text_debounce_store().get(session_key)
            if state is not None and state.task is current:
                state.task = None

    async def _flush_text_debounce_now(self, session_key: str) -> bool:
        """Force-flush one debounced busy-text burst into the pending slot."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False

        current = asyncio.current_task()
        if state.task is not None and state.task is not current and not state.task.done():
            state.task.cancel()
        state.task = None

        existing_pending = self._pending_messages.get(session_key)
        if (
            existing_pending is not None
            and not self._can_merge_text_debounce_events(existing_pending, state.event)
        ):
            return False

        state = store.pop(session_key, None)
        if state is None:
            return False
        merge_pending_message_event(
            self._pending_messages,
            session_key,
            state.event,
            merge_text=True,
        )
        return True

    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None and state.task is not None and not state.task.done():
            state.task.cancel()

    # ------------------------------------------------------------------
    # Session task + guard ownership helpers
    # ------------------------------------------------------------------
    # These were introduced together with the _session_tasks owner map to
    # make session lifecycle reconciliation deterministic across (a) the
    # normal completion path, (b) /stop/ /new/ /reset bypass commands,
    # and (c) stale-lock self-heal on the next inbound message.

    def _release_session_guard(
        self,
        session_key: str,
        *,
        guard: Optional[asyncio.Event] = None,
    ) -> None:
        """Release the adapter-level guard for a session.

        When ``guard`` is provided, only release the entry if it still points
        at that exact Event.  This lets reset-like commands swap in a temporary
        guard while the old processing task unwinds, without having the old
        task's cleanup accidentally clear the replacement guard.
        """
        current_guard = self._active_sessions.get(session_key)
        if current_guard is None:
            return
        if guard is not None and current_guard is not guard:
            return
        del self._active_sessions[session_key]

    def _session_task_is_stale(self, session_key: str) -> bool:
        """Return True if the owner task for ``session_key`` is done/cancelled.

        A lock is "stale" when the adapter still has ``_active_sessions[key]``
        AND a known owner task in ``_session_tasks`` that has already exited.
        When there is no owner task at all, that usually means the guard was
        installed by some path other than handle_message() (tests sometimes
        install guards directly) — don't treat that as stale.  The on-entry
        self-heal only needs to handle the production split-brain case where
        an owner task was recorded, then exited without clearing its guard.
        """
        task = self._session_tasks.get(session_key)
        if task is None:
            return False
        done = getattr(task, "done", None)
        return bool(done and done())

    def _heal_stale_session_lock(self, session_key: str) -> bool:
        """Clear a stale session lock if the owner task is already gone.

        Returns True if a stale lock was healed.  Returns False if there is
        no lock, or the owner task is still alive (the normal busy case).

        This is the on-entry safety net sidbin's issue #11016 analysis calls
        for: without it, a split-brain — adapter still thinks the session is
        active, but nothing is actually processing — traps the chat in
        infinite "Interrupting current task..." until the gateway is
        restarted.
        """
        if session_key not in self._active_sessions:
            return False
        if not self._session_task_is_stale(session_key):
            return False
        logger.warning(
            "[%s] Healing stale session lock for %s (owner task is done/absent)",
            self.name,
            session_key,
        )
        self._active_sessions.pop(session_key, None)
        self._pending_messages.pop(session_key, None)
        self._session_tasks.pop(session_key, None)
        self._discard_text_debounce(session_key)
        return True

    def _start_session_processing(
        self,
        event: MessageEvent,
        session_key: str,
        *,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> bool:
        """Spawn a background processing task under the given session guard.

        Returns True on success.  If the runtime stubs ``create_task`` with a
        non-Task sentinel (some tests do this), the guard is rolled back and
        False is returned so the caller isn't left holding a half-installed
        session lock.
        """
        guard = interrupt_event or asyncio.Event()
        self._active_sessions[session_key] = guard

        task = asyncio.create_task(self._process_message_background(event, session_key))
        self._session_tasks[session_key] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            # Tests stub create_task() with lightweight sentinels that are not
            # hashable and do not support lifecycle callbacks.
            self._session_tasks.pop(session_key, None)
            self._release_session_guard(session_key, guard=guard)
            return False
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)
        return True

    async def cancel_session_processing(
        self,
        session_key: str,
        *,
        release_guard: bool = True,
        discard_pending: bool = True,
    ) -> None:
        """Cancel in-flight processing for a single session.

        ``release_guard=False`` keeps the adapter-level session guard in place
        so reset-like commands can finish atomically before follow-up messages
        are allowed to start a fresh background task.

        Bounded by a 5s timeout so a wedged finally block in the cancelled
        task (typing-task cleanup, on_processing_complete hook, etc.) can't
        stall the calling dispatch coroutine — particularly under pytest-
        asyncio where the event loop's cancellation-propagation semantics
        differ subtly from a bare ``asyncio.run`` harness.
        """
        task = self._session_tasks.pop(session_key, None)
        if task is not None and not task.done():
            logger.debug(
                "[%s] Cancelling active processing for session %s",
                self.name,
                session_key,
            )
            self._expected_cancelled_tasks.add(task)
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Cancelled task for %s did not exit within 5s; "
                    "unblocking dispatch and letting the task unwind in the background",
                    self.name, session_key,
                )
            except Exception:
                logger.debug(
                    "[%s] Session cancellation raised while unwinding %s",
                    self.name,
                    session_key,
                    exc_info=True,
                )
        if discard_pending:
            self._pending_messages.pop(session_key, None)
            self._discard_text_debounce(session_key)
        if release_guard:
            self._release_session_guard(session_key)

    async def _drain_pending_after_session_command(
        self,
        session_key: str,
        command_guard: asyncio.Event,
    ) -> None:
        """Resume the latest queued follow-up once a session command completes.

        Called at the tail of /stop, /new, and /reset dispatch.  Releases the
        command-scoped guard, then — if a follow-up message landed while the
        command was running — spawns a fresh processing task for it.
        """
        await self._flush_text_debounce_now(session_key)
        pending_event = self._pending_messages.pop(session_key, None)
        self._release_session_guard(session_key, guard=command_guard)
        if pending_event is None:
            return
        self._start_session_processing(pending_event, session_key)

    async def _dispatch_active_session_command(
        self,
        event: MessageEvent,
        session_key: str,
        cmd: str,
    ) -> None:
        """Dispatch a reset-like bypass command while preserving guard ordering.

        /stop, /new, and /reset must:
          1. Keep the session guard installed while the runner processes the
             command (so a racing follow-up message stays queued, not
             dispatched as a second parallel run).
          2. Cancel the old in-flight adapter task only AFTER the runner has
             finished handling the command (so the runner sees consistent
             state and its response is sent in order).
          3. Release the command-scoped guard and drain the latest queued
             follow-up exactly once, after 1 and 2 complete.
        """
        logger.debug(
            "[%s] Command '/%s' bypassing active-session guard for %s",
            self.name,
            cmd,
            session_key,
        )

        current_guard = self._active_sessions.get(session_key)
        command_guard = asyncio.Event()
        self._active_sessions[session_key] = command_guard
        thread_meta = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))

        try:
            response = await self._message_handler(event)
            _text, _eph_ttl = self._unwrap_ephemeral(response)
            # Send the response BEFORE cancelling the old task so the send
            # cannot be affected by task-cancellation side effects (race
            # condition fix — issue #18912).  Previously the send happened
            # after cancel_session_processing, which could silently drop the
            # "/new" confirmation when an agent was actively running.
            if _text:
                logger.info(
                    "[%s] Sending command '/%s' response (%d chars) to %s",
                    self.name,
                    cmd,
                    len(_text),
                    event.source.chat_id,
                )
                _r = await self._send_with_retry(
                    chat_id=event.source.chat_id,
                    content=_text,
                    reply_to=_reply_anchor_for_event(event),
                    metadata=_mark_notify_metadata(thread_meta),
                )
                if _eph_ttl > 0 and _r.success and _r.message_id:
                    self._schedule_ephemeral_delete(
                        chat_id=event.source.chat_id,
                        message_id=_r.message_id,
                        ttl_seconds=_eph_ttl,
                    )
            # Old adapter task (if any) is cancelled AFTER the response has
            # been sent — keeps ordering deterministic and avoids the race.
            await self.cancel_session_processing(
                session_key,
                release_guard=False,
                discard_pending=False,
            )
        except Exception:
            # On failure, restore the original guard if one still exists so
            # we don't leave the session in a half-reset state.
            if self._active_sessions.get(session_key) is command_guard:
                if session_key in self._session_tasks and current_guard is not None:
                    self._active_sessions[session_key] = current_guard
                else:
                    self._release_session_guard(session_key, guard=command_guard)
            raise

        await self._drain_pending_after_session_command(session_key, command_guard)

    async def handle_message(self, event: MessageEvent) -> None:
        """
        Process an incoming message.
        
        This method returns quickly by spawning background tasks.
        This allows new messages to be processed even while an agent is running,
        enabling interruption support.
        """
        if not self._message_handler:
            return

        coerce_plaintext_gateway_command(event)

        # Rewrite ``event.source.thread_id`` via the installed recovery hook
        # (Telegram DM topic mode) so the session key, guard checks, and
        # downstream delivery all agree on the same lane.
        # Offloaded: the sync hook must not block the loop.
        await asyncio.to_thread(self._apply_topic_recovery, event)

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

        # On-entry self-heal: if the adapter still has an _active_sessions
        # entry for this key but the owner task has already exited (done or
        # cancelled), the lock is stale.  Clear it and fall through to
        # normal dispatch so the user isn't trapped behind a dead guard —
        # this is the split-brain tail described in issue #11016.
        if session_key in self._active_sessions:
            self._heal_stale_session_lock(session_key)

        # Check if there's already an active handler for this session
        if session_key in self._active_sessions:
            # Certain commands must bypass the active-session guard and be
            # dispatched directly to the gateway runner.  Without this, they
            # are queued as pending messages and either:
            #   - leak into the conversation as user text (/stop, /new), or
            #   - deadlock (/approve, /deny — agent is blocked on Event.wait)
            #
            # Dispatch inline: call the message handler directly and send the
            # response.  Do NOT use _process_message_background — it manages
            # session lifecycle and its cleanup races with the running task
            # (see PR #4926).
            cmd = event.get_command()
            from hermes_cli.commands import should_bypass_active_session

            if should_bypass_active_session(cmd):
                # /stop, /new, /reset must cancel the in-flight adapter task
                # and preserve ordering of queued follow-ups.  Route those
                # through the dedicated handoff path that serializes
                # cancellation + runner response + pending drain.
                if cmd in {"stop", "new", "reset"}:
                    self._discard_text_debounce(session_key)
                    try:
                        await self._dispatch_active_session_command(event, session_key, cmd)
                    except Exception as e:
                        logger.error(
                            "[%s] Command '/%s' dispatch failed: %s",
                            self.name, cmd, e, exc_info=True,
                        )
                    return

                # Other bypass commands (/approve, /deny, /status,
                # /background, /restart) just need direct dispatch — they
                # don't cancel the running task.
                logger.debug(
                    "[%s] Command '/%s' bypassing active-session guard for %s",
                    self.name, cmd, session_key,
                )
                try:
                    _thread_meta = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
                    response = await self._message_handler(event)
                    _text, _eph_ttl = self._unwrap_ephemeral(response)
                    if _text:
                        _r = await self._send_with_retry(
                            chat_id=event.source.chat_id,
                            content=_text,
                            reply_to=_reply_anchor_for_event(event),
                            metadata=_mark_notify_metadata(_thread_meta),
                        )
                        if _eph_ttl > 0 and _r.success and _r.message_id:
                            self._schedule_ephemeral_delete(
                                chat_id=event.source.chat_id,
                                message_id=_r.message_id,
                                ttl_seconds=_eph_ttl,
                            )
                except Exception as e:
                    logger.error("[%s] Command '/%s' dispatch failed: %s", self.name, cmd, e, exc_info=True)
                return

            # Clarify reply bypass: if the agent is blocked on a
            # clarify_tool call, the next non-command message in this
            # session MUST reach the runner so typed numeric choices,
            # exact choices, and free-form "Other" answers can resolve the
            # clarify-intercept and unblock the agent.
            #
            # Without this bypass: the message gets queued in
            # _pending_messages as a follow-up turn instead of reaching the
            # clarify resolver, leaving the agent blocked and discarding the
            # user's answer.
            # Same shape as the /approve deadlock fix (PR #4926) — both
            # cases are "agent thread blocked on Event.wait, message must
            # reach the resolver before being treated as a new turn."
            if not cmd:
                try:
                    from tools import clarify_gateway as _clarify_mod
                    _has_text_clarify = (
                        _clarify_mod.get_pending_for_session(
                            session_key,
                            include_choice_prompts=True,
                        ) is not None
                    )
                except Exception:
                    _has_text_clarify = False

                if _has_text_clarify:
                    logger.debug(
                        "[%s] Routing message to clarify text-intercept for %s",
                        self.name, session_key,
                    )
                    try:
                        _thread_meta = _thread_metadata_for_source(
                            event.source, _reply_anchor_for_event(event)
                        )
                        response = await self._message_handler(event)
                        _text, _eph_ttl = self._unwrap_ephemeral(response)
                        if _text:
                            _r = await self._send_with_retry(
                                chat_id=event.source.chat_id,
                                content=_text,
                                reply_to=_reply_anchor_for_event(event),
                                metadata=_mark_notify_metadata(_thread_meta),
                            )
                            if _eph_ttl > 0 and _r.success and _r.message_id:
                                self._schedule_ephemeral_delete(
                                    chat_id=event.source.chat_id,
                                    message_id=_r.message_id,
                                    ttl_seconds=_eph_ttl,
                                )
                    except Exception as e:
                        logger.error(
                            "[%s] Clarify text-intercept dispatch failed: %s",
                            self.name, e, exc_info=True,
                        )
                    return

            if self._busy_session_handler is not None:
                try:
                    if await self._busy_session_handler(event, session_key):
                        return
                except Exception as e:
                    logger.error("[%s] Busy-session handler failed: %s", self.name, e, exc_info=True)

            # Special case: photo bursts/albums frequently arrive as multiple near-
            # simultaneous messages. Queue them without interrupting the active run,
            # then process them immediately after the current task finishes.
            if event.message_type == MessageType.PHOTO:
                logger.debug("[%s] Queuing photo follow-up for session %s without interrupt", self.name, session_key)
                merge_pending_message_event(self._pending_messages, session_key, event)
                return  # Don't interrupt now - will run after current task completes

            if self._is_queue_text_debounce_candidate(event):
                logger.debug(
                    "[%s] New text message while session %s is active — "
                    "debouncing follow-up (busy_text_mode=queue, window=%.2fs)",
                    self.name,
                    session_key,
                    self._busy_text_debounce_seconds,
                )
                await self._queue_text_debounce(session_key, event)
            else:
                logger.debug(
                    "[%s] New message while session %s is active — queuing follow-up "
                    "(no interrupt, will cascade after current turn)",
                    self.name,
                    session_key,
                )
                merge_pending_message_event(
                    self._pending_messages,
                    session_key,
                    event,
                    merge_text=event.message_type == MessageType.TEXT,
                )
            return  # Don't process now - will be handled after current task finishes
        
        # Mark session as active BEFORE spawning background task to close
        # the race window where a second message arriving before the task
        # starts would also pass the _active_sessions check and spawn a
        # duplicate task.  (grammY sequentialize / aiogram EventIsolation
        # pattern — set the guard synchronously, not inside the task.)
        # _start_session_processing installs the guard AND the owner-task
        # mapping atomically so stale-lock detection works.
        self._start_session_processing(event, session_key)
    
    @staticmethod
    def _get_human_delay() -> float:
        """
        Return a random delay in seconds for human-like response pacing.

        Reads from env vars:
          HERMES_HUMAN_DELAY_MODE: "off" (default) | "natural" | "custom"
          HERMES_HUMAN_DELAY_MIN_MS: minimum delay in ms (default 800, custom mode)
          HERMES_HUMAN_DELAY_MAX_MS: maximum delay in ms (default 2500, custom mode)
        """
        mode = os.getenv("HERMES_HUMAN_DELAY_MODE", "off").lower()
        if mode == "off":
            return 0.0
        if mode == "natural":
            min_ms, max_ms = 800, 2500
            return random.uniform(min_ms / 1000.0, max_ms / 1000.0)
        # custom mode — tolerate malformed env vars instead of crashing.
        try:
            min_ms = int(os.getenv("HERMES_HUMAN_DELAY_MIN_MS", "800"))
        except (TypeError, ValueError):
            min_ms = 800
        try:
            max_ms = int(os.getenv("HERMES_HUMAN_DELAY_MAX_MS", "2500"))
        except (TypeError, ValueError):
            max_ms = 2500
        return random.uniform(min_ms / 1000.0, max_ms / 1000.0)

    async def try_send_final_rich_response(
        self,
        *,
        chat_id: str,
        original_response: str,
        text_content: str,
        images: List[Tuple[str, str]],
        media_files: List[Tuple[str, bool]],
        local_files: List[str],
        force_document_attachments: bool,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        is_ephemeral_response: bool = False,
    ) -> Optional[SendResult]:
        """Optionally send a platform-native rich final response.

        Subclasses can override this to combine already-extracted text and
        media into one richer platform message. Return ``None`` to let the base
        legacy text/media delivery continue unchanged.
        """
        return None

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Background task that actually processes the message."""
        # Track delivery outcomes for the processing-complete hook
        delivery_attempted = False
        delivery_succeeded = False

        def _record_delivery(result):
            nonlocal delivery_attempted, delivery_succeeded
            if result is None:
                return
            delivery_attempted = True
            if getattr(result, "success", False):
                delivery_succeeded = True

        # Reuse the interrupt event set by handle_message() (which marks
        # the session active before spawning this task to prevent races).
        # Fall back to a new Event only if the entry was removed externally.
        interrupt_event = self._active_sessions.get(session_key) or asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        
        # Start continuous typing indicator (refreshes every 2 seconds).
        # Gated per-platform: when typing_indicator=False the refresh loop is
        # never spawned, so no "typing…" / "is thinking…" status is shown.
        # typing_task stays None; _stop_typing_refresh already no-ops on None.
        _thread_metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
        typing_task: Optional[asyncio.Task] = None
        if getattr(self.config, "typing_indicator", True):
            _keep_typing_kwargs: Dict[str, Any] = {"metadata": _thread_metadata}
            try:
                _keep_typing_sig = inspect.signature(self._keep_typing)
            except (TypeError, ValueError):
                _keep_typing_sig = None
            if _keep_typing_sig is None or "stop_event" in _keep_typing_sig.parameters:
                _keep_typing_kwargs["stop_event"] = interrupt_event
            typing_task = asyncio.create_task(
                self._keep_typing(
                    event.source.chat_id,
                    **_keep_typing_kwargs,
                )
            )

        async def _stop_typing_task() -> None:
            await self._stop_typing_refresh(
                event.source.chat_id,
                typing_task,
            )
        
        try:
            await self._run_processing_hook("on_processing_start", event)

            # Call the handler (this can take a while with tool calls)
            response = await self._message_handler(event)
            is_ephemeral_response = isinstance(response, EphemeralReply)

            # Slash-command handlers may return an EphemeralReply sentinel to
            # request that their reply message auto-delete after a TTL (used
            # for system notices like "✨ New session started!" that the user
            # doesn't need to keep in the thread).  Unwrap here so all the
            # downstream extract_media / text-processing logic sees a plain
            # string, and remember the TTL + platform capability so the
            # post-send block can schedule the deletion.
            response, _ephemeral_ttl = self._unwrap_ephemeral(response)

            # Send response if any.  A None/empty response is normal when
            # streaming already delivered the text (already_sent=True) or
            # when the message was queued behind an active agent.  Log at
            # DEBUG to avoid noisy warnings for expected behavior.
            #
            # Suppress stale response when the session was interrupted by a
            # new message that hasn't been consumed yet.  The pending message
            # is processed by the pending-message handler below (#8221/#2483).
            if (
                response
                and interrupt_event.is_set()
                and session_key in self._pending_messages
            ):
                logger.info(
                    "[%s] Suppressing stale response for interrupted session %s",
                    self.name,
                    session_key,
                )
                response = None
            if not response:
                logger.debug("[%s] Handler returned empty/None response for %s", self.name, event.source.chat_id)
            if response:
                # Capture [[as_document]] before extract_media strips it, so the
                # dispatch partition below can route image-extension files
                # through send_document instead of send_multiple_images. Used
                # by skills that produce large/lossless images (e.g. info-graph)
                # where Telegram's sendPhoto recompression destroys legibility.
                force_document_attachments = "[[as_document]]" in response

                # Pre-extract snapshot for the #29346 recovery/invariant below.
                _response_pre_extract = response

                # Extract MEDIA:<path> tags (from TTS tool) before other processing
                media_files, response = self.extract_media(response)
                media_files = self.filter_media_delivery_paths(media_files)

                # Extract image URLs and send them as native platform attachments
                images, text_content = self.extract_images(response)
                # Strip any remaining internal directives from message body (fixes #1561).
                # _strip_media_directives shares MEDIA_TAG_CLEANUP_RE, so a MEDIA: tag
                # with an unknown extension is intentionally left in the body for
                # extract_local_files below to pick up rather than silently dropped (#34517).
                text_content = _strip_media_directives(text_content).strip()
                if images:
                    logger.info("[%s] extract_images found %d image(s) in response (%d chars)", self.name, len(images), len(response))

                local_files = []
                if not is_ephemeral_response:
                    # Auto-detect bare local file paths for native media delivery
                    # (helps small models that don't use MEDIA: syntax). Skip
                    # system/command notices so config paths stay visible text
                    # instead of becoming native uploads.
                    local_files, text_content = self.extract_local_files(text_content)
                    local_files = self.filter_local_delivery_paths(local_files)
                    if local_files:
                        logger.info("[%s] extract_local_files found %d file(s) in response", self.name, len(local_files))

                # A2 (#29346): extraction can reduce a non-empty response to
                # empty text with no attachment, and the `if text_content` guard
                # below then drops it silently. Recover on every platform (#33842
                # was Discord-only); the guard avoids duplicating an attachment.
                if not (text_content or images or local_files or media_files):
                    # Recover from the post-extract_media `response`, not the raw
                    # snapshot: extract_media already stripped MEDIA (incl. spaced
                    # paths) with its full grammar, so no fragment can leak.
                    _recovered = _strip_media_directives(response).strip()
                    if _recovered:
                        logger.warning(
                            "[%s] response_delivery_recovered: extract pipeline "
                            "reduced a non-empty response (%d chars) to empty with "
                            "no attachment; delivering recovered original to %s",
                            self.name, len(_response_pre_extract), event.source.chat_id,
                        )
                        text_content = _recovered

                # Final user-visible content (text, TTS, media, files) gets
                # the existing notify=True marker. Clone once so typing/status
                # metadata stays unmarked and progress bubbles remain
                # thread-strict.
                _final_thread_metadata = _mark_notify_metadata(_thread_metadata)
                # The final-response marker is currently Feishu-only: this PR
                # wires a concrete Feishu card renderer while leaving every
                # other platform's final-message metadata unchanged. Command
                # and ephemeral replies are system notices, not assistant
                # completions, so keep them on Feishu's legacy text path.
                if (
                    self.platform == Platform.FEISHU
                    and event.message_type != MessageType.COMMAND
                    and not is_ephemeral_response
                ):
                    _final_thread_metadata["hermes_final_response"] = True

                # Auto-TTS: if voice message, generate audio FIRST (before sending text)
                # Gated via ``_should_auto_tts_for_chat``: fires when the chat has
                # an explicit ``/voice on|tts`` opt-in OR when ``voice.auto_tts`` is
                # True globally and no ``/voice off`` has been issued.
                _tts_path = None
                if (self._should_auto_tts_for_chat(event.source.chat_id)
                        and event.message_type == MessageType.VOICE
                        and text_content
                        and not media_files):
                    try:
                        from tools.tts_tool import text_to_speech_tool, check_tts_requirements
                        if check_tts_requirements():
                            import json as _json
                            speech_text = self.prepare_tts_text(text_content)
                            if not speech_text:
                                raise ValueError("Empty text after markdown cleanup")
                            tts_result_str = await asyncio.to_thread(
                                text_to_speech_tool, text=speech_text
                            )
                            tts_data = _json.loads(tts_result_str)
                            _tts_path = tts_data.get("file_path")
                    except Exception as tts_err:
                        logger.warning("[%s] Auto-TTS failed: %s", self.name, tts_err)

                # Play TTS audio before text (voice-first experience)
                _tts_caption_delivered = False
                if _tts_path and Path(_tts_path).exists():
                    try:
                        telegram_tts_caption = None
                        if (
                            self.platform == Platform.TELEGRAM
                            and text_content
                            and text_content[:1024] == text_content
                        ):
                            telegram_tts_caption = text_content
                        tts_result = await self.play_tts(
                            chat_id=event.source.chat_id,
                            audio_path=_tts_path,
                            caption=telegram_tts_caption,
                            metadata=_final_thread_metadata,
                        )
                        _tts_caption_delivered = bool(
                            telegram_tts_caption and getattr(tts_result, "success", False)
                        )
                    finally:
                        try:
                            os.remove(_tts_path)
                        except OSError:
                            pass

                _rich_final_response_delivered = False
                if not _tts_caption_delivered:
                    try:
                        _rich_result = await self.try_send_final_rich_response(
                            chat_id=event.source.chat_id,
                            original_response=_response_pre_extract,
                            text_content=text_content,
                            images=list(images),
                            media_files=list(media_files),
                            local_files=list(local_files),
                            force_document_attachments=force_document_attachments,
                            reply_to=_reply_anchor_for_event(event),
                            metadata=_final_thread_metadata,
                            is_ephemeral_response=bool(_ephemeral_ttl and _ephemeral_ttl > 0),
                        )
                    except Exception:
                        logger.warning(
                            "[%s] Final rich response hook failed; falling back to legacy delivery",
                            self.name,
                            exc_info=True,
                        )
                        _rich_result = None
                    if _rich_result is not None:
                        _rich_state = effective_delivery_state(_rich_result)
                        if _rich_state in {
                            FinalDeliveryState.FULLY_DELIVERED,
                            FinalDeliveryState.PARTIALLY_DELIVERED,
                        }:
                            _record_delivery(_rich_result)
                            _rich_final_response_delivered = True

                # Send the text portion
                if text_content and not _tts_caption_delivered and not _rich_final_response_delivered:
                    logger.info("[%s] Sending response (%d chars) to %s", self.name, len(text_content), event.source.chat_id)
                    _reply_anchor = _reply_anchor_for_event(event)
                    result = await self._send_with_retry(
                        chat_id=event.source.chat_id,
                        content=text_content,
                        reply_to=_reply_anchor,
                        metadata=_final_thread_metadata,
                    )
                    _record_delivery(result)

                    # Schedule auto-deletion of system-notice replies.
                    # Detached so the handler returns immediately; errors
                    # (permission denied, message too old) are swallowed.
                    if (
                        _ephemeral_ttl
                        and _ephemeral_ttl > 0
                        and result.success
                        and result.message_id
                    ):
                        self._schedule_ephemeral_delete(
                            chat_id=event.source.chat_id,
                            message_id=result.message_id,
                            ttl_seconds=_ephemeral_ttl,
                        )

                # Human-like pacing delay between text and media
                human_delay = self._get_human_delay()

                # Send extracted images as native attachments
                if images and not _rich_final_response_delivered:
                    logger.info("[%s] Extracted %d image(s) to send as attachments", self.name, len(images))
                    try:
                        await self.send_multiple_images(
                            chat_id=event.source.chat_id,
                            images=images,
                            metadata=_final_thread_metadata,
                            human_delay=human_delay,
                        )
                    except Exception as batch_err:
                        logger.warning("[%s] Error batching images: %s", self.name, batch_err, exc_info=True)


                if _rich_final_response_delivered:
                    media_files = []
                    local_files = []

                # Send extracted media files — route by file type
                _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
                _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

                # Partition images out of media_files + local_files so they
                # can be sent as a single batch (Signal RPC). When
                # ``[[as_document]]`` was set on the original response, image
                # files skip the photo path and route to send_document below
                # so they're delivered with original bytes (no Telegram
                # sendPhoto recompression).
                from urllib.parse import quote as _quote
                _image_paths: list = []
                _non_image_media: list = []
                for media_path, is_voice in media_files:
                    _ext = Path(media_path).suffix.lower()
                    if (_ext in _IMAGE_EXTS
                            and not is_voice
                            and not force_document_attachments):
                        _image_paths.append(media_path)
                    else:
                        _non_image_media.append((media_path, is_voice))
                _non_image_local: list = []
                for file_path in local_files:
                    if (Path(file_path).suffix.lower() in _IMAGE_EXTS
                            and not force_document_attachments):
                        _image_paths.append(file_path)
                    else:
                        _non_image_local.append(file_path)

                if _image_paths:
                    try:
                        _batch = [(f"file://{_quote(p)}", "") for p in _image_paths]
                        await self.send_multiple_images(
                            chat_id=event.source.chat_id,
                            images=_batch,
                            metadata=_final_thread_metadata,
                            human_delay=human_delay,
                        )
                    except Exception as batch_err:
                        logger.warning("[%s] Error batching images: %s", self.name, batch_err, exc_info=True)

                for media_path, is_voice in _non_image_media:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        ext = Path(media_path).suffix.lower()
                        if should_send_media_as_audio(self.platform, ext, is_voice=is_voice):
                            media_result = await self.send_voice(
                                chat_id=event.source.chat_id,
                                audio_path=media_path,
                                metadata=_final_thread_metadata,
                            )
                        elif ext in _VIDEO_EXTS:
                            media_result = await self.send_video(
                                chat_id=event.source.chat_id,
                                video_path=media_path,
                                metadata=_final_thread_metadata,
                            )
                        else:
                            media_result = await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=media_path,
                                metadata=_final_thread_metadata,
                            )

                        if not media_result.success:
                            logger.warning("[%s] Failed to send media (%s): %s", self.name, ext, media_result.error)
                    except Exception as media_err:
                        logger.warning("[%s] Error sending media: %s", self.name, media_err)

                # Send auto-detected local non-image files as native attachments
                for file_path in _non_image_local:
                    if human_delay > 0:
                        await asyncio.sleep(human_delay)
                    try:
                        ext = Path(file_path).suffix.lower()
                        if ext in _VIDEO_EXTS:
                            await self.send_video(
                                chat_id=event.source.chat_id,
                                video_path=file_path,
                                metadata=_final_thread_metadata,
                            )
                        else:
                            await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=file_path,
                                metadata=_final_thread_metadata,
                            )
                    except Exception as file_err:
                        logger.error("[%s] Error sending local file %s: %s", self.name, file_path, file_err)

                # A3 (#29346): if a non-empty response produced nothing
                # deliverable, fail loudly rather than dropping it in silence.
                _anything_delivered = (
                    delivery_attempted or _tts_caption_delivered
                    or images or local_files or media_files
                )
                if not _anything_delivered and _response_pre_extract.strip():
                    logger.error(
                        "[%s] response_delivery_dropped: non-empty response "
                        "(%d chars) produced no delivered message or attachment "
                        "for %s (empty after extract, recovery yielded nothing).",
                        self.name, len(_response_pre_extract), event.source.chat_id,
                    )

            # Determine overall success for the processing hook
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            await self._run_processing_hook(
                "on_processing_complete",
                event,
                ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE,
            )

            # The active drain owns debounce state. If a queue-mode timer has
            # not fired yet, force-flush into _pending_messages here and let
            # this task hand off the follow-up.
            await self._flush_text_debounce_now(session_key)

            # Check if there's a pending message that was queued during our processing
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                logger.debug("[%s] Processing queued follow-up message", self.name)
                # Keep the _active_sessions entry live across the turn chain
                # and only CLEAR the interrupt Event — do NOT delete the entry.
                # If we deleted here, a concurrent inbound message arriving
                # during the awaits below would pass the Level-1 guard, spawn
                # its own _process_message_background, and run simultaneously
                # with the recursive drain below.  Two agents on one
                # session_key = duplicate responses, duplicate tool calls.
                # Clearing the Event keeps the guard live so follow-ups take
                # the busy-handler path as intended.
                _active = self._active_sessions.get(session_key)
                if _active is not None:
                    _active.clear()
                await _stop_typing_task()
                # Spawn a fresh task for the pending message instead of
                # recursing.  Issue #17758: `await
                # self._process_message_background(...)` here grew the
                # call stack one frame per chained follow-up, and under
                # sustained pending-queue activity the C stack would
                # exhaust at ~2000 frames and SIGSEGV the process.
                # Mirror the late-arrival drain pattern below: hand off
                # to a new task and return so this frame can unwind.
                drain_task = asyncio.create_task(
                    self._process_message_background(pending_event, session_key)
                )
                # Hand ownership of the session to the drain task so
                # stale-lock detection keeps working while it runs.
                self._session_tasks[session_key] = drain_task
                try:
                    self._background_tasks.add(drain_task)
                    drain_task.add_done_callback(self._background_tasks.discard)
                except TypeError:
                    # Tests stub create_task() with non-hashable sentinels; tolerate.
                    pass
                return  # Drain task owns the session now.
                
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            outcome = ProcessingOutcome.CANCELLED
            if current_task is None or current_task not in self._expected_cancelled_tasks:
                outcome = ProcessingOutcome.FAILURE
            await self._run_processing_hook("on_processing_complete", event, outcome)
            raise
        except Exception as e:
            await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)
            # Send the error to the user so they aren't left with radio silence
            try:
                error_type = type(e).__name__
                error_detail = str(e)[:300] if str(e) else "no details available"
                _thread_metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
                await self.send(
                    chat_id=event.source.chat_id,
                    content=(
                        f"Sorry, I encountered an error ({error_type}).\n"
                        f"{error_detail}\n"
                        "Try again or use /reset to start a fresh session."
                    ),
                    metadata=_thread_metadata,
                )
            except Exception as notify_err:
                logger.error(
                    "[%s] Failed to send error notification to user: %s",
                    self.name, notify_err, exc_info=True,
                )  # Last resort — don't let error reporting crash the handler
        finally:
            # Stop typing before any deferred callback work.  Post-delivery
            # callbacks may perform platform I/O; a stuck callback must not
            # leave the typing refresh task running indefinitely.
            await _stop_typing_task()
            # Fire any one-shot post-delivery callback registered for this
            # session (e.g. deferred background-review notifications).
            #
            # Snapshot the callback generation HERE (after the agent has run),
            # not at the top of this task.  _hermes_run_generation is set on
            # the interrupt event by GatewayRunner._bind_adapter_run_generation
            # during _handle_message_with_agent — which happens DURING the
            # self._message_handler(event) await above.  Snapshotting earlier
            # always captured None, which bypassed the generation-ownership
            # check in pop_post_delivery_callback and let stale runs fire a
            # fresher run's callbacks.
            _callback_generation = getattr(
                interrupt_event,
                "_hermes_run_generation",
                None,
            )
            if hasattr(self, "pop_post_delivery_callback"):
                _post_cb = self.pop_post_delivery_callback(
                    session_key,
                    generation=_callback_generation,
                )
            else:
                _post_cb = getattr(self, "_post_delivery_callbacks", {}).pop(session_key, None)
            if callable(_post_cb):
                try:
                    _post_result = _post_cb()
                    if inspect.isawaitable(_post_result):
                        await asyncio.wait_for(
                            _post_result,
                            timeout=_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS,
                        )
                except (asyncio.TimeoutError, Exception):
                    pass
            # Some adapters keep platform-level typing tasks.  If callback
            # work or a late refresh recreated one, make one final bounded stop
            # before releasing the session guard.
            await self._stop_typing_refresh(
                event.source.chat_id,
                None,
                stop_attempts=1,
            )
            # Final drain/release boundary: force-flush any timer that missed
            # the in-band drain before deciding whether the guard can clear.
            await self._flush_text_debounce_now(session_key)
            # Late-arrival drain: a message may have arrived during the
            # cleanup awaits above (typing_task cancel, stop_typing).  Such
            # messages passed the Level-1 guard (entry still live, Event
            # possibly set) and landed in _pending_messages via the
            # busy-handler path.  Without this block, we would delete the
            # active-session entry and the queued message would be silently
            # dropped (user never gets a reply).
            late_pending = self._pending_messages.pop(session_key, None)
            if late_pending is not None:
                current_task = asyncio.current_task()
                existing_task = self._session_tasks.get(session_key)
                if (
                    existing_task is not None
                    and existing_task is not current_task
                ):
                    # The in-band drain (or an earlier late-arrival drain)
                    # already spawned a follow-up task that owns this
                    # session.  Re-queue the late-arrival event so that
                    # task picks it up — avoids spawning two concurrent
                    # _process_message_background tasks for the same key
                    # (#17758 follow-up: prevents the create_task path
                    # from racing with itself across the in-band/finally
                    # boundary).
                    self._pending_messages[session_key] = late_pending
                else:
                    logger.debug(
                        "[%s] Late-arrival pending message during cleanup — spawning drain task",
                        self.name,
                    )
                    _active = self._active_sessions.get(session_key)
                    if _active is not None:
                        _active.clear()
                    drain_task = asyncio.create_task(
                        self._process_message_background(late_pending, session_key)
                    )
                    # Hand ownership of the session to the drain task so stale-lock
                    # detection keeps working while it runs.
                    self._session_tasks[session_key] = drain_task
                    try:
                        self._background_tasks.add(drain_task)
                        drain_task.add_done_callback(self._background_tasks.discard)
                    except TypeError:
                        # Tests stub create_task() with non-hashable sentinels; tolerate.
                        pass
                # Leave _active_sessions[session_key] populated — the drain
                # task's own lifecycle will clean it up.
            else:
                # Clean up session tracking.  Guard-match both deletes so a
                # reset-like command that already swapped in its own
                # command_guard (and cancelled us) can't be accidentally
                # cleared by our unwind.  The command owns the session now.
                #
                # The owner-check also covers the in-band drain handoff
                # above: when we spawned a drain_task and transferred
                # ownership via ``_session_tasks[session_key] = drain_task``,
                # ``_session_tasks.get(session_key) is current_task`` is
                # False, so we leave _active_sessions populated.  Without
                # this guard, the drain task picks up the same
                # interrupt_event in its own _process_message_background
                # entry, _release_session_guard's guard-match succeeds,
                # and we'd delete the entry while the drain task is still
                # running — letting a concurrent inbound message pass
                # the Level-1 guard and spawn a second handler for the
                # same session.
                current_task = asyncio.current_task()
                if current_task is not None and self._session_tasks.get(session_key) is current_task:
                    self._cleanup_finished_session_task(session_key, interrupt_event)
    
    def _cleanup_finished_session_task(
        self, session_key: str, interrupt_event: Optional[asyncio.Event]
    ) -> None:
        """Release the session guard for a finished owner task, then drop its
        ``_session_tasks`` entry ONLY if the guard was actually released.

        Release-then-conditional-delete is the #48300 fix: when a concurrent
        path (reset/new command, drain handoff) swapped ``_active_sessions[key]``
        to a different guard, ``_release_session_guard`` skips on the guard
        mismatch and the lock stays installed. If we deleted ``_session_tasks``
        unconditionally (the old order), ``_session_task_is_stale`` would later
        see no owner task and report "not stale", so the orphaned guard would
        never be healed — a permanent session deadlock. Keeping the done-task
        entry when the guard survives lets the on-entry self-heal detect the
        stale lock and clear it on the next inbound message.
        """
        self._release_session_guard(session_key, guard=interrupt_event)
        if session_key not in self._active_sessions:
            self._session_tasks.pop(session_key, None)
    
    async def cancel_background_tasks(self) -> None:
        """Cancel any in-flight background message-processing tasks.

        Used during gateway shutdown/replacement so active sessions from the old
        process do not keep running after adapters are being torn down.

        Each cancelled task is awaited with a 5s bound so a wedged finally
        (typing-task cleanup, on_processing_complete hook) can't stall the
        whole shutdown path.  Stragglers are released from our tracking and
        allowed to finish unwinding on their own.
        """
        # Loop until no new tasks appear.  Without this, a message
        # arriving during the `await asyncio.gather` below would spawn
        # a fresh _process_message_background task (added to
        # self._background_tasks at line ~1668 via handle_message),
        # and the _background_tasks.clear() at the end of this method
        # would drop the reference — the task runs untracked against a
        # disconnecting adapter, logs send-failures, and may linger
        # until it completes on its own.  Retrying the drain until the
        # task set stabilizes closes the window.
        MAX_DRAIN_ROUNDS = 5
        for _ in range(MAX_DRAIN_ROUNDS):
            tasks = [task for task in self._background_tasks if not task.done()]
            if not tasks:
                break
            for task in tasks:
                self._expected_cancelled_tasks.add(task)
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(asyncio.shield(t) for t in tasks),
                        return_exceptions=True,
                    ),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] %d background task(s) did not exit within 5s; "
                    "releasing tracking and letting them unwind in the background",
                    self.name, len([t for t in tasks if not t.done()]),
                )
                break
            # Loop: late-arrival tasks spawned during the gather above
            # will be in self._background_tasks now.  Re-check.
        self._background_tasks.clear()
        self._expected_cancelled_tasks.clear()
        self._session_tasks.clear()
        self._pending_messages.clear()
        self._active_sessions.clear()
        for state in list(self._text_debounce_store().values()):
            if state.task is not None and not state.task.done():
                state.task.cancel()
        self._text_debounce_store().clear()

    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()
    
    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        return self._pending_messages.pop(session_key, None)
    
    def build_source(
        self,
        chat_id: str,
        chat_name: Optional[str] = None,
        chat_type: str = "dm",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        thread_id: Optional[str] = None,
        chat_topic: Optional[str] = None,
        user_id_alt: Optional[str] = None,
        chat_id_alt: Optional[str] = None,
        is_bot: bool = False,
        guild_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        role_authorized: bool = False,
        auto_thread_created: bool = False,
        auto_thread_initial_name: Optional[str] = None,
    ) -> SessionSource:
        """Helper to build a SessionSource for this platform."""
        # Normalize empty topic to None
        if chat_topic is not None and not chat_topic.strip():
            chat_topic = None
        return SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
            chat_topic=chat_topic.strip() if chat_topic else None,
            user_id_alt=user_id_alt,
            chat_id_alt=chat_id_alt,
            is_bot=is_bot,
            guild_id=str(guild_id) if guild_id else None,
            parent_chat_id=str(parent_chat_id) if parent_chat_id else None,
            message_id=str(message_id) if message_id else None,
            role_authorized=role_authorized,
            auto_thread_created=auto_thread_created,
            auto_thread_initial_name=auto_thread_initial_name,
        )
    
    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """
        Get information about a chat/channel.
        
        Returns dict with at least:
        - name: Chat name
        - type: "dm", "group", "channel"
        """
        pass
    
    def format_message(self, content: str) -> str:
        """
        Format a message for this platform.
        
        Override in subclasses to handle platform-specific formatting
        (e.g., Telegram MarkdownV2, Discord markdown).
        
        Default implementation returns content as-is.
        """
        return content
    
    @staticmethod
    def truncate_message(
        content: str,
        max_length: int = 4096,
        len_fn: Optional["Callable[[str], int]"] = None,
    ) -> List[str]:
        """
        Split a long message into chunks, preserving code block boundaries.

        When a split falls inside a triple-backtick code block, the fence is
        closed at the end of the current chunk and reopened (with the original
        language tag) at the start of the next chunk.  Multi-chunk responses
        receive indicators like ``(1/3)``.

        Args:
            content: The full message content
            max_length: Maximum length per chunk (platform-specific)
            len_fn: Optional length function for measuring string length.
                     Defaults to ``len`` (Unicode code-points).  Pass
                     ``utf16_len`` for platforms that measure message
                     length in UTF-16 code units (e.g. Telegram).

        Returns:
            List of message chunks
        """
        _len = len_fn or len
        if _len(content) <= max_length:
            return [content]

        INDICATOR_RESERVE = 10   # room for " (XX/XX)"
        FENCE_CLOSE = "\n```"

        chunks: List[str] = []
        remaining = content
        # When the previous chunk ended mid-code-block, this holds the
        # language tag (possibly "") so we can reopen the fence.
        carry_lang: Optional[str] = None

        while remaining:
            # If we're continuing a code block from the previous chunk,
            # prepend a new opening fence with the same language tag.
            prefix = f"```{carry_lang}\n" if carry_lang is not None else ""

            # How much body text we can fit after accounting for the prefix,
            # a potential closing fence, and the chunk indicator.
            headroom = max_length - INDICATOR_RESERVE - _len(prefix) - _len(FENCE_CLOSE)
            if headroom < 1:
                headroom = max_length // 2

            # Everything remaining fits in one final chunk
            if _len(prefix) + _len(remaining) <= max_length - INDICATOR_RESERVE:
                chunks.append(prefix + remaining)
                break

            # Find a natural split point (prefer newlines, then spaces).
            # When _len != len (e.g. utf16_len for Telegram), headroom is
            # measured in the custom unit.  We need codepoint-based slice
            # positions that stay within the custom-unit budget.
            #
            # _safe_slice_pos() maps a custom-unit budget to the largest
            # codepoint offset whose custom length ≤ budget.
            if _len is not len:
                # Map headroom (custom units) → codepoint slice length
                _cp_limit = _custom_unit_to_cp(remaining, headroom, _len)
            else:
                _cp_limit = headroom
            region = remaining[:_cp_limit]
            split_at = region.rfind("\n")
            if split_at < _cp_limit // 2:
                split_at = region.rfind(" ")
            if split_at < 1:
                split_at = _cp_limit

            # Avoid splitting inside an inline code span (`...`).
            # If the text before split_at has an odd number of unescaped
            # backticks, the split falls inside inline code — the resulting
            # chunk would have an unpaired backtick and any special characters
            # (like parentheses) inside the broken span would be unescaped,
            # causing MarkdownV2 parse errors on Telegram.
            candidate = remaining[:split_at]
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                # Find the last unescaped backtick and split before it
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    # Try to find a space or newline just before the backtick
                    safe_split = candidate.rfind(" ", 0, last_bt)
                    nl_split = candidate.rfind("\n", 0, last_bt)
                    safe_split = max(safe_split, nl_split)
                    if safe_split > _cp_limit // 4:
                        split_at = safe_split

            chunk_body = remaining[:split_at]
            remaining = remaining[split_at:].lstrip()

            full_chunk = prefix + chunk_body

            # Walk only the chunk_body (not the prefix we prepended) to
            # determine whether we end inside an open code block.
            in_code = carry_lang is not None
            lang = carry_lang or ""
            for line in chunk_body.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if in_code:
                        in_code = False
                        lang = ""
                    else:
                        in_code = True
                        tag = stripped[3:].strip()
                        lang = tag.split()[0] if tag else ""

            if in_code:
                # Close the orphaned fence so the chunk is valid on its own
                full_chunk += FENCE_CLOSE
                carry_lang = lang
            else:
                carry_lang = None

            chunks.append(full_chunk)

        # Append chunk indicators when the response spans multiple messages
        if len(chunks) > 1:
            total = len(chunks)
            chunks = [
                f"{chunk} ({i + 1}/{total})" for i, chunk in enumerate(chunks)
            ]

        return chunks


def iter_media_tag_paths(content: str) -> List[str]:
    """Return deliverable MEDIA:<path> paths using the canonical media parser.

    This delegates to ``BasePlatformAdapter.extract_media`` so card-rendering
    decisions share the same quoted-path, extension, and protected-span
    behavior as native attachment delivery.
    """
    media, _cleaned = BasePlatformAdapter.extract_media(content or "")
    return [path for path, _is_voice in media]
