"""
Base platform adapter interface.

All platform adapters (Telegram, Discord, WhatsApp) inherit from this
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


def _platform_name(platform) -> str:
    """Normalize a Platform enum / raw string into a lowercase name."""
    value = getattr(platform, "value", platform)
    return str(value or "").lower()


def _thread_metadata_for_source(source, reply_to_message_id: str | None = None) -> dict | None:
    """Build platform-aware thread metadata for adapter sends.

    Most platforms route threaded sends with a generic ``thread_id`` metadata
    value. Telegram private-chat topics created through Hermes' DM-topic helper
    are exposed in updates as ``message_thread_id`` plus a reply anchor, but
    outbound sends only render in the correct Telegram lane when the adapter
    supplies both ``message_thread_id`` and ``reply_to_message_id``. Mark those
    lanes so the Telegram adapter can avoid the known-bad partial routes.
    """
    thread_id = getattr(source, "thread_id", None)
    if thread_id is None:
        return None
    metadata = {"thread_id": thread_id}
    if _platform_name(getattr(source, "platform", None)) == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        anchor = reply_to_message_id or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    return metadata


def _reply_anchor_for_event(event) -> str | None:
    """Return reply_to id for platforms that need reply semantics.

    Telegram forum/supergroup topics should be routed by topic metadata, not by
    replying to the triggering message. Hermes-created Telegram private-chat
    topic lanes are different: Bot API sends reject their ``message_thread_id``
    and do not route with ``direct_messages_topic_id``. Those lanes only remain
    visible when sent with both the private topic thread id and a reply to the
    triggering user message.
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


from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable, Tuple, Union
from enum import Enum

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key
from hermes_constants import get_hermes_dir


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
    if response.is_redirect and response.next_request:
        redirect_url = str(response.next_request.url)
        from tools.url_safety import is_safe_url
        if not is_safe_url(redirect_url):
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

# Default location: {HERMES_HOME}/cache/images/ (legacy: image_cache/)
IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")


def get_image_cache_dir() -> Path:
    """Return the image cache directory, creating it if it doesn't exist."""
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_CACHE_DIR


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
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "image/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                return cache_image_from_bytes(response.content, ext)
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
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_CACHE_DIR


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """
    Save raw audio bytes to the cache and return the absolute file path.

    Args:
        data: Raw audio bytes.
        ext:  File extension including the dot (e.g. ".ogg", ".mp3").

    Returns:
        Absolute path to the cached audio file as a string.
    """
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
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)",
                        "Accept": "audio/*,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                return cache_audio_from_bytes(response.content, ext)
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
    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEO_CACHE_DIR


def cache_video_from_bytes(data: bytes, ext: str = ".mp4") -> str:
    """Save raw video bytes to the cache and return the absolute file path."""
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
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def get_document_cache_dir() -> Path:
    """Return the document cache directory, creating it if it doesn't exist."""
    DOCUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DOCUMENT_CACHE_DIR


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
    
    # Auto-loaded skill(s) for topic/channel bindings (e.g., Telegram DM Topics,
    # Discord channel_skill_bindings).  A single name or ordered list.
    auto_skill: Optional[str | list[str]] = None

    # Per-channel ephemeral system prompt (e.g. Discord channel_prompts).
    # Applied at API call time and never persisted to transcript history.
    channel_prompt: Optional[str] = None
    
    # Internal flag — set for synthetic events (e.g. background process
    # completion notifications) that must bypass user authorization checks.
    internal: bool = False

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


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None
    retryable: bool = False  # True for transient connection errors — base will retry automatically
    # When the adapter had to split an oversized payload across multiple
    # platform messages (e.g. Telegram edit_message overflow split-and-deliver),
    # ``message_id`` is the LAST visible message id (so subsequent edits target
    # the most recent chunk) and these are the additional message ids that
    # made up the full payload, in send order.  Empty tuple for the common
    # single-message case.
    continuation_message_ids: tuple = ()


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


class BasePlatformAdapter(ABC):
    """
    Base class for platform adapters.
    
    Subclasses implement platform-specific logic for:
    - Connecting and authenticating
    - Receiving messages
    - Sending messages/responses
    - Handling media
    """
    
    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
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
        self._set_fatal_error(f'{scope}_lock', message, retryable=False)
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

    def set_busy_session_handler(self, handler: Optional[Callable[[MessageEvent, str], Awaitable[bool]]]) -> None:
        """Set an optional handler for messages arriving during active sessions."""
        self._busy_session_handler = handler
    
    def set_session_store(self, session_store: Any) -> None:
        """
        Set the session store for checking active sessions.
        
        Used by adapters that need to check if a thread/conversation
        has an active session before processing messages (e.g., Slack
        thread replies without explicit mentions).
        """
        self._session_store = session_store
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the platform and start receiving messages.
        
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
        and resolves.  Adapters with native button UIs (Telegram, Discord)
        SHOULD override this for a richer UX.
        """
        if choices:
            lines = [f"❓ {question}", ""]
            for i, choice in enumerate(choices, start=1):
                lines.append(f"  {i}. {choice}")
            lines.append("")
            lines.append("Reply with the number, the option text, or your own answer.")
            text = "\n".join(lines)
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
        or file attachments (Discord). Default falls back to sending the
        file path as text.
        """
        text = f"🔊 Audio: {audio_path}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

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
        Default falls back to sending the file path as text.
        """
        text = f"🎬 Video: {video_path}"
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
        Default falls back to sending the file path as text.
        """
        text = f"📎 File: {file_path}"
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
        Override in subclasses for native photo attachments.
        Default falls back to sending the file path as text.
        """
        text = f"🖼️ Image: {image_path}"
        if caption:
            text = f"{caption}\n{text}"
        return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

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
        # and quoted/backticked paths for LLM-formatted outputs.
        media_pattern = re.compile(
            r'''[`"']?MEDIA:\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|(?:~/|/)\S+(?:[^\S\n]+\S+)*?\.(?:png|jpe?g|gif|webp|mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|txt|csv|apk|ipa)(?=[\s`"',;:)\]}]|$)|\S+)[`"']?'''
        )
        for match in media_pattern.finditer(content):
            path = match.group("path").strip()
            if len(path) >= 2 and path[0] == path[-1] and path[0] in "`\"'":
                path = path[1:-1].strip()
            path = path.lstrip("`\"'").rstrip("`\"',.;:)}]")
            if path:
                media.append((os.path.expanduser(path), has_voice_tag))

        # Remove MEDIA tags from content (including surrounding quote/backtick wrappers)
        if media:
            cleaned = media_pattern.sub('', cleaned)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return media, cleaned

    @staticmethod
    def extract_local_files(content: str) -> Tuple[List[str], str]:
        """
        Detect bare local file paths in response text for native media delivery.

        Matches absolute paths (/...) and tilde paths (~/) ending in common
        image or video extensions.  Validates each candidate with
        ``os.path.isfile()`` to avoid false positives from URLs or
        non-existent paths.

        Paths inside fenced code blocks (``` ... ```) and inline code
        (`...`) are ignored so that code samples are never mutilated.

        Returns:
            Tuple of (list of expanded file paths, cleaned text with the
            raw path strings removed).
        """
        _LOCAL_MEDIA_EXTS = (
            '.png', '.jpg', '.jpeg', '.gif', '.webp',
            '.mp4', '.mov', '.avi', '.mkv', '.webm',
        )
        ext_part = '|'.join(e.lstrip('.') for e in _LOCAL_MEDIA_EXTS)

        # (?<![/:\w.]) prevents matching inside URLs (e.g. https://…/img.png)
        #             and relative paths (./foo.png)
        # (?:~/|/)    anchors to absolute or home-relative paths
        path_re = re.compile(
            r'(?<![/:\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:' + ext_part + r')\b',
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

                def _chained() -> None:
                    try:
                        _prev()
                    except Exception:
                        logger.debug("Post-delivery callback failed", exc_info=True)
                    try:
                        _new()
                    except Exception:
                        logger.debug("Post-delivery callback failed", exc_info=True)

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

        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)

        # Timeout errors are not safe to retry (message may have been
        # delivered) and not formatting errors — return the failure as-is.
        if not is_network and self._is_timeout_error(error_str):
            return result

        if is_network:
            # Retry with exponential backoff for transient errors
            for attempt in range(1, max_retries + 1):
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
                error_str = result.error or ""
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
                    metadata=thread_meta,
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
                            metadata=_thread_meta,
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

            # Clarify text-capture bypass: if the agent is blocked on a
            # clarify_tool call awaiting a free-form text response (open-
            # ended clarify, or user picked "Other"), the next non-command
            # message in this session MUST reach the runner so the
            # clarify-intercept can resolve it and unblock the agent.
            #
            # Without this bypass: the message gets queued in
            # _pending_messages AND triggers an interrupt, killing the
            # agent run mid-clarify and discarding the user's answer.
            # Same shape as the /approve deadlock fix (PR #4926) — both
            # cases are "agent thread blocked on Event.wait, message must
            # reach the resolver before being treated as a new turn."
            if not cmd:
                try:
                    from tools import clarify_gateway as _clarify_mod
                    _has_text_clarify = (
                        _clarify_mod.get_pending_for_session(session_key) is not None
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
                                metadata=_thread_meta,
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

            # Default behavior for non-photo follow-ups: interrupt the running agent
            logger.debug("[%s] New message while session %s is active — triggering interrupt", self.name, session_key)
            self._pending_messages[session_key] = event
            # Signal the interrupt (the processing task checks this)
            self._active_sessions[session_key].set()
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
        
        # Start continuous typing indicator (refreshes every 2 seconds)
        _thread_metadata = _thread_metadata_for_source(event.source, _reply_anchor_for_event(event))
        _keep_typing_kwargs = {"metadata": _thread_metadata}
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
            typing_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(typing_task), timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                # Cancellation cleanup must not block adapter shutdown.  The
                # typing task is already cancelled; if the parent task is also
                # cancelling, let this message-processing task unwind now.
                pass
        
        try:
            await self._run_processing_hook("on_processing_start", event)

            # Call the handler (this can take a while with tool calls)
            response = await self._message_handler(event)

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

                # Extract MEDIA:<path> tags (from TTS tool) before other processing
                media_files, response = self.extract_media(response)

                # Extract image URLs and send them as native platform attachments
                images, text_content = self.extract_images(response)
                # Strip any remaining internal directives from message body (fixes #1561)
                text_content = text_content.replace("[[audio_as_voice]]", "").strip()
                text_content = text_content.replace("[[as_document]]", "").strip()
                text_content = re.sub(r"MEDIA:\s*\S+", "", text_content).strip()
                if images:
                    logger.info("[%s] extract_images found %d image(s) in response (%d chars)", self.name, len(images), len(response))

                # Auto-detect bare local file paths for native media delivery
                # (helps small models that don't use MEDIA: syntax)
                local_files, text_content = self.extract_local_files(text_content)
                if local_files:
                    logger.info("[%s] extract_local_files found %d file(s) in response", self.name, len(local_files))
                
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
                            speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
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
                if _tts_path and Path(_tts_path).exists():
                    try:
                        await self.play_tts(
                            chat_id=event.source.chat_id,
                            audio_path=_tts_path,
                            metadata=_thread_metadata,
                        )
                    finally:
                        try:
                            os.remove(_tts_path)
                        except OSError:
                            pass

                # Send the text portion
                if text_content:
                    logger.info("[%s] Sending response (%d chars) to %s", self.name, len(text_content), event.source.chat_id)
                    _reply_anchor = _reply_anchor_for_event(event)
                    # Mark final response messages for notification delivery.
                    # Platform adapters that support per-message notification
                    # control (e.g. Telegram's disable_notification) use this
                    # flag to override silent-mode and ensure the final
                    # response triggers a push notification.
                    # Clone to avoid mutating the metadata shared with the
                    # typing-indicator task (which must remain unmarked).
                    if _thread_metadata is not None:
                        _thread_metadata = dict(_thread_metadata)
                        _thread_metadata["notify"] = True
                    else:
                        _thread_metadata = {"notify": True}
                    result = await self._send_with_retry(
                        chat_id=event.source.chat_id,
                        content=text_content,
                        reply_to=_reply_anchor,
                        metadata=_thread_metadata,
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
                if images:
                    logger.info("[%s] Extracted %d image(s) to send as attachments", self.name, len(images))
                    try:
                        await self.send_multiple_images(
                            chat_id=event.source.chat_id,
                            images=images,
                            metadata=_thread_metadata,
                            human_delay=human_delay,
                        )
                    except Exception as batch_err:
                        logger.warning("[%s] Error batching images: %s", self.name, batch_err, exc_info=True)


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
                            metadata=_thread_metadata,
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
                                metadata=_thread_metadata,
                            )
                        elif ext in _VIDEO_EXTS:
                            media_result = await self.send_video(
                                chat_id=event.source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            media_result = await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
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
                                metadata=_thread_metadata,
                            )
                        else:
                            await self.send_document(
                                chat_id=event.source.chat_id,
                                file_path=file_path,
                                metadata=_thread_metadata,
                            )
                    except Exception as file_err:
                        logger.error("[%s] Error sending local file %s: %s", self.name, file_path, file_err)

            # Determine overall success for the processing hook
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            await self._run_processing_hook(
                "on_processing_complete",
                event,
                ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE,
            )

            # Check if there's a pending message that was queued during our processing
            if session_key in self._pending_messages:
                pending_event = self._pending_messages.pop(session_key)
                logger.debug("[%s] Processing queued message from interrupt", self.name)
                # Keep the _active_sessions entry live across the turn chain
                # and only CLEAR the interrupt Event — do NOT delete the entry.
                # If we deleted here, a concurrent inbound message arriving
                # during the awaits below would pass the Level-1 guard, spawn
                # its own _process_message_background, and run simultaneously
                # with the recursive drain below.  Two agents on one
                # session_key = duplicate responses, duplicate tool calls.
                # Clearing the Event keeps the guard live so follow-ups take
                # the busy-handler path (queue + interrupt) as intended.
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
            except Exception:
                pass  # Last resort — don't let error reporting crash the handler
        finally:
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
                        await _post_result
                except Exception:
                    pass
            # Stop typing indicator
            await _stop_typing_task()
            # Also cancel any platform-level persistent typing tasks (e.g. Discord)
            # that may have been recreated by _keep_typing after the last stop_typing()
            try:
                if hasattr(self, "stop_typing"):
                    await self.stop_typing(event.source.chat_id)
            except Exception:
                pass
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
                    del self._session_tasks[session_key]
                    self._release_session_guard(session_key, guard=interrupt_event)
    
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
