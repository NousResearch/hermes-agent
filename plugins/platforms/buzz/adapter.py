"""Buzz platform adapter (Block's Nostr-based human+agent community relay).

Outbound and polling go through the ``buzz`` CLI ("JSON in, JSON out", never a shell);
inbound prefers a NIP-42-authenticated WebSocket subscription with a CLI poll fallback.
Config lives in ``gateway.platforms.buzz.extra`` (relay_url, channels, home_channel,
poll_interval, cli_path, credentials_file, allowed_users, reply_in_thread, reaction_only_users)
or the matching ``BUZZ_*`` env vars (env overrides config). The only secret is
BUZZ_PRIVATE_KEY (nsec or hex): it reaches the CLI via the subprocess env and is never logged.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

# Profile-scoped read (adapter startup, Slack pattern #59739): a scoped read honors the profile's own
# secret; only an UNSCOPED read under multiplex (default-profile startup loop) falls back to the process
# env, which is that profile's own value.
from agent.secret_scope import (
    UnscopedSecretError as _UnscopedSecretError, current_secret_scope as _current_secret_scope,
    get_secret as _scoped_get_secret, is_multiplex_active as _is_multiplex_active,
)
from gateway.platforms._shared import profile_scoped as _profile_scoped


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read: an active scope is authoritative (a miss is ``default``, never an env
    borrow). Unscoped adds one rung over ``_shared``: the startup gate runs before any scope exists, so
    externally managed secrets are consulted via a one-shot profile-scope build.

    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and connects *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash
    startup/reconnect (#70652 class); there ``os.environ`` is that profile's own value, so fall back to it.
    Same pattern as ``whatsapp_common._get_wsecret`` and the WeCom/IRC/ntfy plugin adapters.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    Secondary profiles construct their adapters under a profile secret scope -- the scope is authoritative
    and a scoped miss returns ``default`` (no cross-profile borrow from ``os.environ``, which may hold
    another profile's value). The DEFAULT profile's adapter constructs and sends *unscoped* under
    multiplexing, where a bare ``get_secret`` would raise ``UnscopedSecretError`` and crash this path; there
    ``os.environ`` is that profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    """
    try:
        val = _scoped_get_secret(name, None)
    except _UnscopedSecretError:
        # DEFAULT profile's adapter constructs/connects outside any _profile_runtime_scope under
        # multiplexing; os.environ is that profile's own value there. Same pattern as Slack SLACK_APP_TOKEN
        # (#59739) and the Matrix recovery key. A *scoped* miss still returns the default (no cross-profile
        # borrow).
        val = os.getenv(name)
    if val is None and _current_secret_scope() is None:
        val = _unscoped_profile_secrets().get(name)
    return val if val is not None else default


_UNSCOPED_PROFILE_SECRETS: Optional[Dict[str, str]] = None


def _unscoped_profile_secrets() -> Dict[str, str]:
    """Process-cached profile secret mapping (external resolvers are slow); failures degrade to {}."""
    global _UNSCOPED_PROFILE_SECRETS
    if _UNSCOPED_PROFILE_SECRETS is None:
        try:
            from agent.secret_scope import build_profile_secret_scope
            from hermes_constants import get_hermes_home
            _UNSCOPED_PROFILE_SECRETS = dict(build_profile_secret_scope(get_hermes_home()))
        except Exception:
            logger.warning(
                "Buzz requirement probe could not build the profile secret "
                "scope; Bitwarden-managed credentials will not be visible "
                "to the startup gate (#95216)",
                exc_info=True,
            )
            _UNSCOPED_PROFILE_SECRETS = {}
    return _UNSCOPED_PROFILE_SECRETS


def _scoped_platform_setting(env_name, extra, key):
    """Raw non-secret setting; in a secondary profile scope ``os.environ`` is the DEFAULT profile's, so ``extra`` wins.

    Inside a secondary profile scope ``os.environ`` holds the DEFAULT profile's YAML-to-env bridge output
    (#98738), so the profile's ``PlatformConfig.extra`` is authoritative and env is not consulted: a missing
    key yields ``None`` and callers fail closed to their default instead of silently borrowing the default
    profile's relay, channels, or allowlist. Everywhere else — single-profile gateways, the default profile
    under multiplexing — the legacy ``os.getenv`` read is returned unchanged, so env-over-config precedence
    is preserved.
    """
    return (extra or {}).get(key) if _profile_scoped() else os.getenv(env_name)


logger = logging.getLogger(__name__)

from gateway.platforms.base import (
    BasePlatformAdapter, CachedMedia, SendResult, MessageEvent, MessageType, cache_media_bytes_async,
)
from gateway.config import Platform


_CHAT_KIND = 9  # ``messages get`` also returns housekeeping kinds, never dispatched
# Chat + forum post/comment; stream kinds wait for confirmed semantics. ``_is_direct_message_event``
# stays kind-9-only so a p-tagged forum post can't be reclassified as a DM and bypass mention gating.
# Kinds that carry agent-relevant conversation content and are dispatched (#90309): chat messages (9) plus
# the Buzz forum kinds — 45001 is a forum post (thread root) and 45003 a comment reply on it. Block's own
# ACP harness documents this set (``buzz-acp --kinds 9,46010,40007,45001, 45002,45003``); the stream kinds
# (46010/40007/45002) are left out until their dispatch semantics are confirmed.
_DISPATCH_KINDS = frozenset({_CHAT_KIND, 45001, 45003})
_UNRESOLVED_MENTION_ERROR_RE = re.compile(r"mention '@(?P<name>[^']+)' does not match a current channel member")
_BUZZ_PRESENTATION_MENTION_SEPARATOR = "\u200b"
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


def _escape_unresolved_presentation_mention(content: str, error: str) -> Optional[str]:
    """Make a CLI-rejected ``@name`` presentation-only via an invisible separator after the ``@`` (Buzz
    p-tags whitespace-prefixed @tokens at publish, so prose like ``@session:...`` fails preflight)."""
    match = _UNRESOLVED_MENTION_ERROR_RE.search(error or "")
    name = match.group("name") if match else ""
    if not name:
        return None
    token = re.compile(rf"(?<!\S)@{re.escape(name)}(?=$|[^A-Za-z0-9._-])", re.IGNORECASE)
    escaped, count = token.subn(lambda m: "@" + _BUZZ_PRESENTATION_MENTION_SEPARATOR + m.group(0)[1:], content)
    return escaped if count else None


_FETCH_LIMIT = 50  # events per poll / seed call
_SEEN_CAP = 500  # per-channel de-dupe set bound (events)
_CURSOR_STATE_SUBDIR = "buzz"  # per-channel cursors survive a restart under HERMES_HOME
_CURSOR_STATE_FILENAME = "channel-cursors.json"
_DM_DISCOVERY_EVERY = 5  # re-run DM discovery every N poll sweeps
_DEFAULT_POLL_INTERVAL = 4.0
_MIN_POLL_INTERVAL = 1.0
_CLI_TIMEOUT = 30.0
# Mention-resolution caches: member lists are hit on every publish containing "@"; names must not outlive a rename.
_MEMBER_CACHE_TTL = 60.0
_PROFILE_NAME_TTL = 300.0
# Inbound attachments download only after the gates pass and must match their declared NIP-94 size + SHA-256.
_MAX_INBOUND_ATTACHMENTS = 4
_MAX_INBOUND_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ATTACHMENT_DOWNLOAD_TIMEOUT = 30.0
_MAX_ATTACHMENT_FILENAME_BYTES = 120


def _safe_attachment_filename(value: str) -> str:
    """Return a basename that is safe for cache files and agent context."""
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if ord(c) >= 32 and c != "\x7f").strip()
    if name in {"", ".", ".."}:
        return "attachment.bin"
    suffix = Path(name).suffix if len(Path(name).suffix.encode("utf-8")) <= 20 else ""
    stem = name[:-len(suffix)] if suffix else name
    byte_budget = _MAX_ATTACHMENT_FILENAME_BYTES - len(suffix.encode("utf-8"))
    safe_stem = stem.encode("utf-8")[:byte_budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{safe_stem or 'attachment'}{suffix}"


def _attachment_origin(value: str) -> Optional[tuple[str, int]]:
    """Normalize a configured host/URL to an exact HTTPS-equivalent origin."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlsplit(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port or 443
    except ValueError:
        return None
    if (parsed.scheme and parsed.scheme not in {"https", "wss"}) or not host:
        return None
    return host, port

# WebSocket transport (NIP-42 authenticated Nostr subscription).
_WS_AUTH_TIMEOUT = 20.0
# Last-resort read bound: an unsurfaced relay-side close (CLOSE_WAIT) would leave us "connected" with inbound stopped.
# The library keepalive (ping_interval/ping_timeout below) should catch a dead relay first, but a relay-side
# close the transport never surfaces (observed as a CLOSE_WAIT socket with the loop parked on recv, #98097)
# leaves the gateway "connected" while inbound stops; this timeout forces the normal reconnect path instead.
_WS_READ_IDLE_TIMEOUT = 300.0
_WS_MAX_MESSAGE_BYTES = 2_000_000
_WS_MEMBERSHIP_KIND = 44100  # Buzz channel-membership event — live DM discovery
_WS_MEMBERSHIP_SUB_ID = "hermes-buzz-membership"
# Credentials JSON fallback when BUZZ_PRIVATE_KEY is not set; module-level so tests can point it at a tmpdir.
_DEFAULT_CREDENTIALS_DIR = Path("~/.config/buzz").expanduser()
# Buzz-hosted media is private to the community: same-relay URLs must be authenticated + localised for vision.
_MEDIA_URL_PATTERN = r"https?://[^\s<>\[\]()]+/media/[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?(?:\?[^\s<>\[\]()]*)?"
_MARKDOWN_MEDIA_RE = re.compile(
    rf"!\[(?P<alt>[^\]]*)\]\(\s*(?P<url>{_MEDIA_URL_PATTERN})(?:\s+[\"'][^\"']*[\"'])?\s*\)", re.IGNORECASE
)
_BARE_MEDIA_RE = re.compile(_MEDIA_URL_PATTERN, re.IGNORECASE)
_MEDIA_PATH_RE = re.compile(r"^/media/(?P<sha>[0-9a-f]{64})(?P<ext>\.[a-z0-9]{1,10})?/?$", re.IGNORECASE)


def _effective_port(parsed) -> Optional[int]:
    try:
        if parsed.port is not None:
            return parsed.port
    except ValueError:
        return None
    return {"https": 443, "wss": 443, "http": 80, "ws": 80}.get(parsed.scheme)


def _is_relay_media_url(url: str, relay_url: str) -> bool:
    """Return whether *url* is a Buzz media object on the configured relay."""
    candidate, relay = urlsplit(url), urlsplit(relay_url)
    return bool(
        candidate.scheme in ("http", "https") and candidate.hostname and relay.hostname
        and candidate.hostname.lower() == relay.hostname.lower()
        and _effective_port(candidate) == _effective_port(relay) and _MEDIA_PATH_RE.fullmatch(candidate.path)
    )


def _find_relay_media_refs(text: str, relay_url: str) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """Find same-relay media URLs and their safe text replacements."""
    urls: List[str] = []
    replacements: List[Tuple[int, int, str]] = []
    markdown_spans: List[Tuple[int, int]] = []
    for match in _MARKDOWN_MEDIA_RE.finditer(text):
        url = match.group("url")
        if not _is_relay_media_url(url, relay_url):
            continue
        markdown_spans.append(match.span())
        replacements.append((*match.span(), match.group("alt").strip()))
        if url not in urls:
            urls.append(url)
    for match in _BARE_MEDIA_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in markdown_spans):
            continue
        url = match.group(0)
        if not _is_relay_media_url(url, relay_url):
            continue
        replacements.append((*match.span(), ""))
        if url not in urls:
            urls.append(url)
    return urls, replacements


def _replace_media_refs(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(replacements, reverse=True):
        text = f"{text[:start]}{replacement}{text[end:]}"
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()


def _load_nostr_auth():
    """Import sibling nostr_auth loader-agnostically (the test loader imports this file as a bare module)."""
    try:
        from . import nostr_auth  # type: ignore[no-redef]
        return nostr_auth
    except ImportError:
        import importlib.util
        path = Path(__file__).with_name("nostr_auth.py")
        spec = importlib.util.spec_from_file_location("plugin_adapter_buzz_nostr_auth", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_nostr_auth = _load_nostr_auth()

# bech32 (BIP-173) npub <-> hex so mention detection and allow-lists accept either form.
from gateway.authz_mixin import (  # noqa: E402
    _BECH32_CHARSET, _bech32_hrp_expand, _bech32_polymod, _convertbits, _npub_to_hex as npub_to_hex,
)


def hex_to_npub(pubkey_hex: str) -> Optional[str]:
    """Encode a 64-char hex pubkey as an ``npub1…`` bech32 string."""
    try:
        raw = bytes.fromhex(pubkey_hex)
    except ValueError:
        return None
    if len(raw) != 32 or (data := _convertbits(raw, 8, 5)) is None:
        return None
    polymod = _bech32_polymod(_bech32_hrp_expand("npub") + data + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return "npub1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def _split_csv(raw):
    return raw.split(",") if isinstance(raw, str) else raw


def _pubkey_set(raw) -> set:
    """Normalize a csv string / list of hex pubkeys or npubs to a set of hex pubkeys."""
    return {n for entry in _split_csv(raw) if isinstance(entry, str) and (n := _normalize_user_ref(entry))}


def _setting_or(env_name: str, extra: dict, key: str, default):
    """Scoped setting read with an explicit ``None`` -> ``extra`` fallback."""
    raw = _scoped_platform_setting(env_name, extra, key)
    return extra.get(key, default) if raw is None else raw


def _ttl_get(cache: dict, key, ttl: float):
    """Value of a ``(monotonic_ts, value)`` cache entry if younger than *ttl*, else None."""
    cached = cache.get(key)
    return cached[1] if cached is not None and (time.monotonic() - cached[0]) < ttl else None


def _add_pubkey(bucket: List[str], raw) -> None:
    """Append the lowercased pubkey once (empty values are skipped)."""
    pk = str(raw or "").lower()
    if pk and pk not in bucket:
        bucket.append(pk)


def _normalize_user_ref(ref: str) -> Optional[str]:
    """Normalize a user reference (hex pubkey or npub) to lowercase hex."""
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    if ref.startswith("npub1"):
        return npub_to_hex(ref)
    return ref if _HEX64_RE.fullmatch(ref) else None


# ── buzz-cli invocation helpers ──────────────────────────────────────────────

def _reply_to_mode(config, extra: dict) -> str:
    """Reply mode ("first"/"all" thread, "off" posts flat); env overrides config, ``reply_in_thread: false`` = "off"."""
    mode = str(os.getenv("BUZZ_REPLY_TO_MODE") or getattr(config, "reply_to_mode", "first") or "first").strip().lower()
    rit = os.getenv("BUZZ_REPLY_IN_THREAD")
    if rit is None:
        rit = extra.get("reply_in_thread")
    return "off" if rit is not None and str(rit).strip().lower() in ("false", "0", "no", "off") else mode


def _configured_relay(extra: dict) -> str:
    return (_scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url") or extra.get("relay_url", "")).strip()


def _configured_home_channel(extra: dict) -> str:
    raw = _scoped_platform_setting("BUZZ_HOME_CHANNEL", extra, "home_channel")
    return (raw or str(extra.get("home_channel", "") or "")).strip()


def _configured_cli_path(extra: dict) -> str:
    raw = _scoped_platform_setting("BUZZ_CLI_PATH", extra, "cli_path")
    return _resolve_cli_path(str(raw or "").strip() or str(extra.get("cli_path", "") or ""))


def _configured_credentials_file(extra: Optional[dict]) -> str:
    # Scoped: a miss falls to the profile's own extra, never the default profile's env; unscoped keeps env precedence.
    configured = str(_get_scoped_secret("BUZZ_CREDENTIALS_FILE", "") or "").strip()
    return configured or str((extra or {}).get("credentials_file", "") or "").strip()


def _resolve_cli_path(configured: str = "") -> str:
    """Resolve the buzz binary: explicit config → ``buzz`` on PATH → ``~/bin/buzz``; "" if none."""
    if configured:
        p = Path(configured).expanduser()
        return str(p) if p.is_file() else ""
    if found := shutil.which("buzz"):
        return found
    fallback = Path.home() / "bin" / "buzz"
    return str(fallback) if fallback.is_file() else ""


def _credentials_candidates(extra: Optional[dict] = None) -> List[Path]:
    configured = _configured_credentials_file(extra)
    if configured:
        return [Path(configured).expanduser()]
    if _is_multiplex_active():
        return []
    try:
        return sorted(_DEFAULT_CREDENTIALS_DIR.glob("*credentials*.json"))
    except OSError:
        return []


_KEY_FIELDS = ("nsec", "private_key_hex", "private_key")


def _credentials_key(data: dict) -> str:
    """First non-empty private-key field in a credentials record, stripped ("" if none)."""
    for field in _KEY_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_credentials_data(extra: Optional[dict] = None) -> dict:
    """Load the first credential record containing a private key."""
    for path in _credentials_candidates(extra):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and _credentials_key(data):
            return data
    return {}


def _resolve_private_key(extra: Optional[dict] = None) -> str:
    """Resolve the Nostr private key: scoped secret first, then credentials JSON. NEVER log it."""
    key = str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip()
    return key or _credentials_key(_resolve_credentials_data(extra))


def _resolve_auth_tag(extra: Optional[dict] = None) -> str:
    """Resolve and validate the optional NIP-OA owner-attestation tag."""
    raw: Any = str(_get_scoped_secret("BUZZ_AUTH_TAG", "") or "").strip()
    if not raw:
        if str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip() and not _configured_credentials_file(extra):
            return ""
        if "auth_tag" not in (data := _resolve_credentials_data(extra)):
            return ""
        raw = data["auth_tag"]
    return json.dumps(_nostr_auth.parse_auth_tag(raw, "Buzz auth tag"), separators=(",", ":"))


async def _exec_buzz(
    cli_path: str, args: List[str], *, relay_url: str, private_key: str, auth_tag: str = "",
    input_text: Optional[str] = None, timeout: float = _CLI_TIMEOUT,
) -> Tuple[int, str, str]:
    """Run the buzz CLI (argv, never a shell) -> ``(rc, stdout, stderr)``. Key travels via env only."""
    env = os.environ.copy()
    env["BUZZ_RELAY_URL"] = relay_url
    env["BUZZ_PRIVATE_KEY"] = private_key
    env.pop("BUZZ_AUTH_TAG", None)
    if auth_tag:
        env["BUZZ_AUTH_TAG"] = auth_tag
    proc = await asyncio.create_subprocess_exec(
        cli_path, *args, stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    try:
        stdin_bytes = input_text.encode("utf-8") if input_text is not None else None
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        detail = {"error": "timeout", "message": f"buzz {args[0] if args else ''} timed out after {timeout}s"}
        return 124, "", json.dumps(detail)
    rc = proc.returncode if proc.returncode is not None else 4
    return rc, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


_MAX_CLI_MESSAGE_CHARS = 900


def _bounded_cli_message(message: str, redact_path: Optional[Path] = None) -> str:
    """Keep untrusted CLI detail useful without exposing unbounded output."""
    if redact_path is not None:
        message = message.replace(str(redact_path), redact_path.name)
    return message if len(message) <= _MAX_CLI_MESSAGE_CHARS else f"{message[: _MAX_CLI_MESSAGE_CHARS - 3]}..."


def _cli_error_message(stderr: str, returncode: int, *, redact_path: Optional[Path] = None) -> str:
    """Extract a bounded human-readable message from the CLI error contract."""
    text = (stderr or "").strip()
    data = _json_or(text, None)
    if isinstance(data, dict):
        detail, category = data.get("message"), data.get("error")
        if isinstance(detail, str) and detail.strip():
            label = category.strip() if isinstance(category, str) and category.strip() else "error"
            return _bounded_cli_message(f"{label}: {detail.strip()} (exit {returncode})", redact_path)
    return _bounded_cli_message(text or f"buzz CLI failed with exit code {returncode}", redact_path)


def _parse_send_receipt(stdout: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate the buzz-cli success receipt and return ``(event_id, error)``."""
    data = _json_or(stdout, None)
    if not isinstance(data, dict):
        return None, "invalid CLI response"
    if data.get("accepted") is False:
        detail = data.get("message")
        if not isinstance(detail, str) or not detail.strip():
            detail = "message was not accepted"
        return None, _bounded_cli_message(detail.strip())
    event_id = data.get("event_id")
    if data.get("accepted") is not True or not isinstance(event_id, str) or not event_id.strip():
        return None, "invalid CLI response"
    return event_id.strip(), None


def _json_or(text: str, default):
    """``json.loads`` of CLI stdout, or *default* when empty/malformed."""
    try:
        return json.loads(text or json.dumps(default))
    except ValueError:
        return default


def _parse_json_list(stdout: str) -> List[dict]:
    """Parse CLI stdout expected to be a JSON array of objects."""
    data = _json_or(stdout, [])
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _e_tags(event: dict):
    """Yield ``(target, marker)`` for each well-formed NIP-10 ``e`` tag (raw values, unstripped)."""
    tags = event.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, (list, tuple)) and len(tag) >= 2 and str(tag[0]) == "e":
                yield tag[1], (str(tag[3]) if len(tag) > 3 else "")


def _event_reply_parent_id(event: dict) -> Optional[str]:
    """Direct parent id from NIP-10 ``e`` tags: ``reply`` marker, then ``root``, else last positional."""
    reply_id = root_id = last_e = None
    for raw_target, marker in _e_tags(event):
        target = str(raw_target or "").strip()
        if not target:
            continue
        last_e = target
        if marker == "reply":
            reply_id = target
        elif marker == "root":
            root_id = target
    return reply_id or root_id or last_e


def _p_tagged(event: dict, pubkey: str) -> bool:
    """True when the event carries a ``p`` tag equal to *pubkey* (case-insensitive)."""
    tags = event.get("tags")
    return isinstance(tags, list) and any(
        isinstance(tag, (list, tuple)) and len(tag) > 1 and tag[0] == "p" and str(tag[1]).lower() == pubkey for tag in tags)


# Cap stored parent content snippets (gateway reply injection also clips).
_EVENT_META_CONTENT_CAP = 500
_MEDIA_KIND_PRIORITY = (("image", MessageType.PHOTO), ("audio", MessageType.AUDIO), ("video", MessageType.VIDEO))
_ATTACHMENT_KIND_TYPES = {"image": MessageType.PHOTO, "video": MessageType.VIDEO, "audio": MessageType.AUDIO, "document": MessageType.DOCUMENT}


class BuzzAdapter(BasePlatformAdapter):
    """Buzz adapter (WebSocket push with poll fallback) for the BasePlatformAdapter interface."""

    def __init__(self, config, **kwargs):
        super().__init__(config=config, platform=Platform("buzz"))
        extra = getattr(config, "extra", {}) or {}
        self._extra = extra
        # Env overrides config.yaml, except under a secondary profile scope where extra wins (_scoped_platform_setting).
        self.relay_url = _configured_relay(extra)
        hosts = _split_csv(extra.get("attachment_hosts", []))
        origins = (_attachment_origin(h) for h in hosts if isinstance(h, str))
        self._attachment_origins = {o for o in origins if o is not None}
        relay_origin = _attachment_origin(self.relay_url)
        if relay_origin is not None:
            self._attachment_origins.add(relay_origin)
        self.cli_path = _configured_cli_path(extra)
        # Channels to watch: env csv > extra list/csv; empty = all joined channels
        raw_channels = _split_csv(_setting_or("BUZZ_CHANNELS", extra, "channels", []))
        self.channels: List[str] = [c.strip() for c in raw_channels if isinstance(c, str) and c.strip()]
        self.home_channel = _configured_home_channel(extra)
        _pi_raw = _scoped_platform_setting("BUZZ_POLL_INTERVAL", extra, "poll_interval")
        try:
            self.poll_interval = max(_MIN_POLL_INTERVAL, float(_pi_raw or extra.get("poll_interval", _DEFAULT_POLL_INTERVAL)))
        except (TypeError, ValueError):
            self.poll_interval = max(_MIN_POLL_INTERVAL, _DEFAULT_POLL_INTERVAL)
        # Channel messages must @mention the agent unless disabled; DMs always dispatch.
        _rm_cfg = _setting_or("BUZZ_REQUIRE_MENTION", extra, "require_mention", True)
        self.require_mention = str(_rm_cfg).strip().lower() not in ("false", "0", "no", "off")
        self._reply_to_mode: str = _reply_to_mode(config, extra)
        # Inbound transport: "auto" (WebSocket with poll fallback), "websocket" (required), "poll".
        _transport_raw = _scoped_platform_setting("BUZZ_TRANSPORT", extra, "transport")
        _transport = (_transport_raw or str(extra.get("transport", "auto") or "auto")).strip().lower()
        self.transport = _transport if _transport in ("auto", "websocket", "poll") else "auto"
        # Entries may be hex or npub (normalized to hex). Reaction-only identities get a 👀 on explicit tags but
        # never dispatch; allowed_users wins on overlap.
        self._allowed_pubkeys: set = _pubkey_set(_setting_or("BUZZ_ALLOWED_USERS", extra, "allowed_users", []))
        self._reaction_only_pubkeys: set = _pubkey_set(os.getenv("BUZZ_REACTION_ONLY_USERS") or extra.get("reaction_only_users", []))
        # Secret — resolved lazily (never at import time, never logged); connect() re-resolves.
        self._private_key = self._auth_tag = ""
        # Identity — filled in by connect() from ``buzz users get``
        self._self_pubkey = self._self_npub = self._display_name = ""
        self._poll_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_ready: Optional[asyncio.Event] = None
        self._membership_since = self._poll_count = 0
        self._lock_key: Optional[str] = None
        # Channels the relay permanently rejected ("restricted"); persists across reconnects so we never re-subscribe.
        # channel_id -> { "chat_type", "last_ts", "seen": OrderedDict[event_id, None], "event_meta":
        # OrderedDict[event_id, (author_pubkey, content_snippet)], } event_meta backs NIP-10 reply-parent
        # resolution for require_mention (thread replies to our own messages count as addressed — #75826).
        # "restricted: not a channel member").
        self._restricted_channels: set = set()
        # channel_id -> {"chat_type", "last_ts", "seen": OrderedDict[event_id, None], "event_meta":
        #   OrderedDict[event_id, (author_pubkey, snippet)]}; event_meta backs NIP-10 reply-parent resolution.
        self._channel_state: Dict[str, dict] = {}
        # Cursors read from disk at connect(), consumed by each channel's first seed.
        self._restored_cursors: Dict[str, dict] = {}
        self._channel_names: Dict[str, str] = {}
        # channel_id -> raw ``channels list`` entry; drives DM-vs-channel classification.
        self._channel_meta: Dict[str, dict] = {}
        self._user_names: Dict[str, str] = {}
        self._member_cache: Dict[str, Tuple[float, List[str]]] = {}  # (monotonic, pubkeys)
        self._profile_name_cache: Dict[str, Tuple[float, str]] = {}
        # inbound event_id -> thread root (None when top-level), so send() joins the user's thread instead of nesting.
        self._thread_roots: "OrderedDict[str, Optional[str]]" = OrderedDict()

    @property
    def name(self) -> str:
        return "Buzz"

    @staticmethod
    def normalize_user_id(user_id: str) -> Optional[str]:
        """Normalize a user reference (hex or npub) to hex — authz_mixin allowlist hook.

        Optional hook consumed by ``gateway/authz_mixin`` when matching the profile allowlist carried in
        ``config.extra.allowed_users`` (#98738): entries may be npubs while inbound ``user_id`` is always
        the hex pubkey, so a plain string compare would deny listed users.
        """
        return _normalize_user_ref(user_id)

    # ── buzz-cli plumbing ─────────────────────────────────────────────────

    async def _run_cli(self, args: List[str], *, input_text: Optional[str] = None) -> Tuple[int, str, str]:
        if not self._private_key:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        return await _exec_buzz(self.cli_path, args, relay_url=self.relay_url, private_key=self._private_key,
                                auth_tag=self._auth_tag, input_text=input_text)

    async def _cli_json(self, args: List[str], default):
        """``_run_cli`` -> parsed stdout on rc 0, else *default*."""
        code, out, _err = await self._run_cli(args)
        return _json_or(out, default) if code == 0 else default

    # ── Connection lifecycle ──────────────────────────────────────────────

    def _connect_failed(self, code: str, detail: str, log: str, *log_args, retryable: bool = False) -> bool:
        """Log and record a fatal connect() error; always returns False."""
        logger.error(log, *log_args)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Verify relay credentials, seed high-water marks, start polling."""
        if not self.relay_url:
            return self._connect_failed("config_missing", "BUZZ_RELAY_URL must be set", "Buzz: relay URL must be configured")
        if not self.cli_path:
            return self._connect_failed(
                "cli_missing", "buzz CLI binary not found", "Buzz: buzz CLI binary not found (set BUZZ_CLI_PATH or put 'buzz' on PATH)"
            )
        try:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        except ValueError as exc:
            return self._connect_failed("config_invalid", str(exc), "Buzz: invalid owner-auth configuration — %s", exc)
        if not self._private_key:
            return self._connect_failed(
                "config_missing", "BUZZ_PRIVATE_KEY must be set", "Buzz: no private key (set BUZZ_PRIVATE_KEY or a credentials file)"
            )
        # Own identity: pubkey drives self-echo suppression, display name drives mention gating.
        code, out, err = await self._run_cli(["users", "get"])
        if code != 0:
            message = _cli_error_message(err, code)
            return self._connect_failed(
                "connect_failed", message, "Buzz: failed to fetch own profile from %s — %s",
                self.relay_url, message, retryable=code == 2,
            )
        profiles = _parse_json_list(out)
        if not profiles or not profiles[0].get("pubkey"):
            return self._connect_failed(
                "connect_failed", "buzz users get returned no profile",
                "Buzz: 'users get' returned no profile — is the key a member of this community?", retryable=True,
            )
        self._self_pubkey = str(profiles[0]["pubkey"]).lower()
        self._display_name = str(profiles[0].get("display_name") or "").strip()
        self._self_npub = hex_to_npub(self._self_pubkey) or ""
        # Two profiles must not drive the same identity on one relay (duplicate replies, split de-dupe state).
        try:
            from gateway.status import acquire_scoped_lock
            lock_key = f"{self.relay_url}:{self._self_pubkey}"
            if not acquire_scoped_lock("buzz", lock_key):
                return self._connect_failed(
                    "lock_conflict", "Buzz identity in use by another profile",
                    "Buzz: identity %s… on %s already in use by another profile", self._self_pubkey[:8], self.relay_url,
                )
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module not available (e.g. tests)
        # Map channel ids to names and pick the watch set.
        code, out, err = await self._run_cli(["channels", "list"])
        if code != 0:
            message = _cli_error_message(err, code)
            return self._connect_failed(
                "connect_failed", message, "Buzz: failed to list channels — %s", message, retryable=code == 2
            )
        self._channel_names = {}
        for ch in _parse_json_list(out):
            if ch_id := ch.get("channel_id"):
                self._channel_names[str(ch_id)] = str(ch.get("name") or ch_id)
                self._channel_meta[str(ch_id)] = ch
        watch = self.channels or list(self._channel_names)
        if not watch:
            return self._connect_failed(
                "config_missing", "no Buzz channels to watch", "Buzz: no channels to watch (configure BUZZ_CHANNELS or join a channel)"
            )
        # Seed high-water marks so a (re)start never replays history — except where a restored cursor lets
        # events that landed while down still dispatch.
        # Skip any channel the relay has permanently rejected in a previous session (e.g. "restricted: not a
        # channel member") so we don't reconnect-loop on them. See #90464.
        self._load_cursors()
        for channel_id in watch:
            if channel_id in self._restricted_channels:
                logger.debug("Buzz: skipping restricted channel %s (relay rejected subscription)", channel_id)
                continue
            await self._seed_channel(channel_id, chat_type="group")
        await self._discover_dms(seed=True)
        self._save_cursors()
        # Prefer the NIP-42 WebSocket push; poll when it can't be established (auto) or the user pinned "poll".
        transport_used = "poll"
        if self.transport in ("auto", "websocket"):
            if await self._start_websocket():
                transport_used = "websocket"
            elif self.transport == "websocket":
                self._set_fatal_error(
                    "ws_auth_failed", "Buzz WebSocket transport did not authenticate (transport=websocket)", retryable=True
                )
                await self.disconnect()
                return False
        if transport_used == "poll":
            self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info(
            "Buzz: connected to %s as %s, watching %d channel(s) via %s%s",
            self.relay_url, self._display_name or self._self_npub[:16], len(self._channel_state),
            transport_used, "" if transport_used == "websocket" else f", poll interval {self.poll_interval:.1f}s",
        )
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        """Stop the inbound transport and drop runtime state."""
        self._mark_disconnected()
        lock_key = getattr(self, "_lock_key", None)
        if lock_key:
            try:
                from gateway.status import release_scoped_lock
                release_scoped_lock("buzz", lock_key)
            except Exception:
                pass
            self._lock_key = None
        await self._cancel_task(self._ws_task)
        self._ws_task = None
        await self._cancel_task(self._poll_task)
        self._poll_task = None
        self._channel_state = {}
        self._poll_count = 0

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ── Sending ───────────────────────────────────────────────────────────

    async def _channel_member_pubkeys(self, chat_id: str) -> List[str]:
        """Mention candidates: ``channels members`` (a non-member ``--mention`` is rejected by the CLI), else
        recent traffic, which over-approximates — ``send()`` recovers by retrying without mentions."""
        cache = self._member_cache
        if (cached := _ttl_get(cache, str(chat_id), _MEMBER_CACHE_TTL)) is not None:
            return list(cached)
        pks: List[str] = []
        for row in await self._cli_json(["channels", "members", "--channel", str(chat_id)], []):
            _add_pubkey(pks, row.get("pubkey") if isinstance(row, dict) else row)
        if pks:
            cache[str(chat_id)] = (time.monotonic(), list(pks))
            return pks
        candidates: List[str] = []
        for msg in await self._cli_json(["messages", "get", "--channel", str(chat_id), "--limit", "50"], []):
            _add_pubkey(candidates, msg.get("pubkey"))
            for t in msg.get("tags") or []:
                if isinstance(t, list) and len(t) > 1 and t[0] == "p":
                    _add_pubkey(candidates, str(t[1]))
        if candidates:
            cache[str(chat_id)] = (time.monotonic(), list(candidates))
        return candidates

    async def _profile_display_name(self, pubkey: str) -> str:
        """Display name via ``users get --pubkey`` (bare ``users get`` may return only our own profile), TTL-cached."""
        cache = self._profile_name_cache
        if (cached := _ttl_get(cache, pubkey, _PROFILE_NAME_TTL)) is not None:
            return cached
        name = ""
        profiles = await self._cli_json(["users", "get", "--pubkey", pubkey], [])
        if profiles and isinstance(p0 := profiles[0], dict):
            name = str(p0.get("display_name") or p0.get("name") or "").strip()
            if not name and p0.get("content"):
                prof = _json_or(p0["content"], {})
                name = str(prof.get("display_name") or prof.get("name") or "").strip()
        cache[pubkey] = (time.monotonic(), name)
        return name

    async def _mention_pubkeys_for(self, chat_id: str, content: str) -> List[str]:
        """Resolve ``@Name`` tokens to member pubkeys so genuine mentions notify while @-prose stays text.
        Word-bounded ("email@Fizz", "@@Fizz", "@FizzBuzz" don't wake Fizz; "@Riley!!" does); longer names
        match first and consume their span; ambiguous names tag nobody."""
        if "@" not in content:
            return []
        by_name: Dict[str, List[str]] = {}
        display: Dict[str, str] = {}
        self_pk = getattr(self, "_self_pubkey", None)
        for pk in await self._channel_member_pubkeys(chat_id):
            if pk == self_pk:
                continue
            name = await self._profile_display_name(pk)
            if not name:
                continue
            key = name.lower()
            pks = by_name.setdefault(key, [])
            if pk not in pks:
                pks.append(pk)
            display.setdefault(key, name)
        found: List[str] = []
        text = content
        for key in sorted(by_name, key=len, reverse=True):
            pattern = re.compile(r"(?<![\w@])@" + re.escape(display[key]) + r"(?!\w)", re.IGNORECASE)
            if pattern.search(text):
                pks = by_name[key]
                if len(pks) == 1 and pks[0] not in found:
                    found.append(pks[0])
                # Consume the span either way so a shorter prefix name can't double-match
                # and an ambiguous name stays presentation-only.
                text = pattern.sub("\x00", text)
        return found

    async def _run_message_send(self, args: List[str], content: str, mention_pubkeys: Optional[List[str]] = None):
        """Send with bounded recovery (each rung once): explicit ``--mention``s; on "not channel members" retry
        without; escape an unresolvable ``@token`` and retry; finally ``--mention <self>`` (downgrades @names to text).

        1. publish with explicit ``--mention`` pubkeys resolved from the content (#83414) so genuine member
        mentions carry p-tags and mention-subscribed agents actually wake; 2. if the CLI rejects because a
        resolved pubkey is no longer a member (membership drift), retry without the explicit mentions —
        deliver the message rather than lose it; 3. if the CLI's preflight rejects an unresolvable
        presentation ``@token`` in prose, escape exactly that token with an invisible separator and retry
        (#82646 / #78797); 4. if the error persists and we know our own pubkey, retry once with ``--mention
        <self>`` — supplying any explicit identity downgrades unresolvable @names to presentation-only text
        (#83414); the echo de-dupe already suppresses self-notification.
        """
        mention_args: List[str] = []
        for pk in mention_pubkeys or []:
            mention_args += ["--mention", pk]
        code, out, err = await self._run_cli(args + mention_args, input_text=content)
        if code == 0:
            return code, out, err
        if mention_args and "not channel members" in (err or ""):
            code, out, err = await self._run_cli(args, input_text=content)
            if code == 0:
                return code, out, err
        escaped = _escape_unresolved_presentation_mention(content, err)
        if escaped is not None:
            logger.info("Buzz: retrying message after unresolved presentation-mention preflight")
            code, out, err = await self._run_cli(args, input_text=escaped)
            if code == 0:
                return code, out, err
        if "does not match a current channel member" in (err or "") and getattr(self, "_self_pubkey", None):
            code, out, err = await self._run_cli(args + ["--mention", self._self_pubkey], input_text=content)
        return code, out, err

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if not content:
            return SendResult(success=False, error="Empty message")
        # Anchor: metadata.thread_id, then metadata.reply_to_message_id (stream/progress sends), then reply_to.
        meta = metadata or {}
        args = ["messages", "send", "--channel", str(chat_id), "--content", "-"]
        args += self._reply_args(meta.get("thread_id") or meta.get("reply_to_message_id") or reply_to)
        mention_pubkeys = await self._mention_pubkeys_for(chat_id, content)
        code, out, err = await self._run_message_send(args, content, mention_pubkeys)
        result = self._send_result(chat_id, code, out, err)
        if result.success:
            # Record event_meta so a thread reply to this send matches even if the echo never arrives.
            self._remember_event_meta(str(chat_id), result.message_id, self._self_pubkey, content)
        return result

    def _reply_args(self, anchor: Optional[str]) -> List[str]:
        """``--reply-to`` CLI args for *anchor*, honoring ``reply_to_mode``."""
        reply_target = self._resolve_reply_anchor(anchor)
        return ["--reply-to", str(reply_target)] if reply_target and self._reply_to_mode != "off" else []

    def _send_result(self, chat_id: str, code: int, out: str, err: str, *, redact_path: Optional[Path] = None) -> SendResult:
        """``messages send`` result -> SendResult; marks the verified id seen (echo suppression belt-and-braces)."""
        if code != 0:
            return SendResult(success=False, error=_cli_error_message(err, code, redact_path=redact_path), retryable=code == 2)
        event_id, receipt_error = _parse_send_receipt(out)
        if receipt_error:
            return SendResult(success=False, error=receipt_error)
        assert event_id is not None
        # Belt-and-braces echo suppression: the poll loop already skips our own pubkey, but marking the
        # verified id seen makes de-dupe explicit. Also record event_meta so a thread reply to this send
        # matches even if the WS/poll echo never arrives (#75826).
        self._mark_seen(str(chat_id), event_id)
        return SendResult(success=True, message_id=event_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Buzz has no typing indicator API — no-op."""

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Best-effort reaction via buzz-cli; failures are logged, never raised."""
        if not self.cli_path or not emoji or not message_id:
            return False
        # The event id IS the dispatched message_id; channel is not a parameter here.
        code, _out, err = await self._run_cli(["reactions", "add", "--event", str(message_id), "--emoji", emoji])
        if code != 0:
            logger.debug("Buzz: reaction add failed for message %s in %s — %s", message_id[:12], chat_id, _cli_error_message(err, code))
        return code == 0

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        """Edit a sent message (streamed replies). The CLI reports a NEW event id but the stream consumer
        keeps addressing the original, so return the given id, never the CLI's."""
        if not message_id:
            return SendResult(success=False, error="Buzz edit needs a message id")
        if not content:
            return SendResult(success=False, error="Empty message")
        args = ["messages", "edit", "--event", str(message_id), "--content", "-"]
        code, out, err = await self._run_cli(args, input_text=content)
        if code != 0:
            return SendResult(success=False, error=_cli_error_message(err, code), retryable=code == 2)
        data = _json_or(out, {})
        if data.get("event_id"):
            # The edit is itself a relay event that echoes back on our subscription.
            self._mark_seen(str(chat_id), str(data["event_id"]))
        return SendResult(success=bool(data.get("accepted", True)), message_id=str(message_id), raw_response=data)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a sent message (stream consumer's fresh-final cleanup path)."""
        if not message_id:
            return False
        code, out, _err = await self._run_cli(["messages", "delete", "--event", str(message_id)])
        if code != 0:
            return False
        try:
            data = json.loads(out or "{}")
        except ValueError:
            return True
        if data.get("event_id"):
            self._mark_seen(str(chat_id), str(data["event_id"]))
        return bool(data.get("accepted", True))

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image: local files upload via --file, URLs go as a link."""
        local = Path(image_url).expanduser() if not image_url.startswith(("http://", "https://")) else None
        if local is not None and local.is_file():
            return await self._send_file_attachment(chat_id, local, caption=caption, reply_to=reply_to, metadata=metadata, probe=False)
        # Markdown renders in Buzz, so a URL arrives as a clickable image link.
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def _send_file_attachment(
        self, chat_id: str, file_path: Path, *, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, probe: bool = True,
    ) -> SendResult:
        """Upload a local file as a native attachment; ``probe=False`` when the caller already verified it (a re-probe could race).

        See #74999.
        """
        local = Path(file_path).expanduser()
        if probe and not local.is_file():
            # Never leak host filesystem paths into chat-visible errors.
            return SendResult(success=False, error="Media file not found")
        args = ["messages", "send", "--channel", str(chat_id), "--file", str(local), "--content", "-"]
        args += self._reply_args((metadata or {}).get("thread_id") or reply_to)
        code, out, err = await self._run_message_send(args, caption or "")
        return self._send_result(chat_id, code, out, err, redact_path=local)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Upload a local image via ``--file``; missing paths keep the Base fallback so host paths never reach chat.

        See #74999.
        """
        local = Path(image_path).expanduser()
        if local.is_file():
            return await self._send_file_attachment(chat_id, local, caption=caption, reply_to=reply_to, metadata=metadata, probe=False)
        return await super().send_image_file(chat_id=chat_id, image_path=image_path, caption=caption, reply_to=reply_to, metadata=metadata, **kwargs)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Upload a local document through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(chat_id, Path(file_path), caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Upload a local video through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(chat_id, Path(video_path), caption=caption, reply_to=reply_to, metadata=metadata)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, **kwargs) -> SendResult:
        """Upload a local audio file through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(chat_id, Path(audio_path), caption=caption, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        state = self._channel_state.get(chat_id)
        if (name := self._channel_names.get(chat_id)) is None and self.cli_path:
            data = await self._cli_json(["channels", "get", "--channel", chat_id], {})
            if isinstance(data, dict) and data.get("name"):
                name = self._channel_names[chat_id] = str(data["name"])
        return {"name": name or chat_id, "type": state["chat_type"] if state else "group", "chat_id": chat_id}

    # ── Inbound: WebSocket transport (NIP-42) — same _handle_event() as the poll loop ──────

    # ── Inbound: WebSocket transport (NIP-42 authenticated) ────────────── Push transport contributed in PR
    # #73636 by @ScaleLeanChris, adapted to dispatch through the same _handle_event() machinery as the poll
    # loop so de-dupe, mention gating, DM latching, and the allow-list behave identically on both
    # transports.
    def _websocket_url(self) -> str:
        parsed = urlsplit(self.relay_url.strip())
        scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
        if scheme not in ("ws", "wss") or not parsed.netloc:
            raise ValueError("Buzz relay URL must use http(s) or ws(s)")
        return urlunsplit((scheme, parsed.netloc, parsed.path or "", parsed.query, ""))

    async def _start_websocket(self) -> bool:
        """Start the WS loop; True when it authenticates within the timeout."""
        try:
            import websockets  # noqa: F401  (availability probe)
            self._websocket_url()
        except Exception as e:
            logger.info("Buzz: WebSocket transport unavailable (%s); falling back to polling", e)
            return False
        self._ws_ready = asyncio.Event()
        self._membership_since = int(time.time())
        self._ws_task = asyncio.create_task(self._websocket_loop())
        try:
            await asyncio.wait_for(self._ws_ready.wait(), timeout=_WS_AUTH_TIMEOUT + 5)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Buzz: WebSocket did not authenticate in time")
            await self._cancel_task(self._ws_task)
            self._ws_task = None
            return False

    async def _authenticate_websocket(self, websocket) -> None:
        """NIP-42: await the AUTH challenge, answer with a signed kind-22242 event (+ optional NIP-OA tag), await OK."""
        message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT))
        if not isinstance(message, list) or len(message) < 2 or message[0] != "AUTH":
            raise ConnectionError("Buzz relay did not send a NIP-42 AUTH challenge")
        # BUZZ_AUTH_TAG is per-identity: a scoped profile without one fails closed to "" rather than borrowing
        # the default profile's env tag. Resolved lazily so a re-auth on a bare adapter stays scope-correct.
        # BUZZ_AUTH_TAG is per-identity NIP-OA owner attestation, so it must resolve through the profile
        # secret scope (#98738): inside a scoped multiplex profile a missing tag fails closed to "" instead
        # of attaching the default profile's tag from os.environ, while single-profile and unscoped
        # default-profile reads keep the legacy env behavior. connect() populates ``self._auth_tag`` via
        # ``_resolve_auth_tag`` (scope-aware read + credentials-file fallback, #79514); resolve lazily here
        # as well so a re-auth on a bare adapter stays scope-correct.
        auth_tag = getattr(self, "_auth_tag", "") or ""
        if not auth_tag:
            try:
                auth_tag = _resolve_auth_tag(getattr(self, "_extra", None))
            except ValueError:
                auth_tag = ""
        event = _nostr_auth.build_auth_event(private_key=self._private_key, challenge=str(message[1]), relay_url=self._websocket_url(), auth_tag_json=auth_tag)
        await websocket.send(json.dumps(["AUTH", event], separators=(",", ":")))
        while True:
            response = json.loads(await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT))
            if not isinstance(response, list) or not response:
                continue
            if response[0] == "OK" and len(response) >= 4 and response[1] == event["id"]:
                if response[2] is True:
                    return
                raise ConnectionError(f"Buzz WebSocket AUTH rejected: {response[3]}")
            if response[0] in ("NOTICE", "CLOSED"):
                raise ConnectionError(f"Buzz WebSocket AUTH failed: {response[-1] if len(response) > 1 else 'authentication failed'}")

    @staticmethod
    async def _send_req(websocket, subscription_id: str, request_filter: dict) -> None:
        await websocket.send(json.dumps(["REQ", subscription_id, request_filter], separators=(",", ":")))

    async def _send_channel_subscription(self, websocket, subscription_id: str, channel_id: str) -> None:
        state = self._channel_state.get(channel_id) or {}
        last_ts = int(state.get("last_ts") or 0)
        # A conversation adopted mid-run with no high-water mark is fresh: its history IS the conversation,
        # so subscribe from the beginning instead of `since ≈ now` — otherwise the message that *created*
        # the conversation (created_at fractionally before this subscription) is silently dropped (#78429).
        # `limit` bounds the replay to the same window the poll transport fetches; the seed path gives real
        # channels a non-zero last_ts, so they never take this branch.
        request_filter = {"kinds": sorted(_DISPATCH_KINDS), "#h": [channel_id]}
        if last_ts:
            # Resume from the high-water mark (same-second overlap de-duped by id).
            request_filter["since"] = max(last_ts - 1, 0)
        else:
            # A conversation adopted mid-run with no high-water mark is fresh: its history IS the conversation,
            # so subscribe from the start or the message that *created* it is dropped. Seeded channels have last_ts != 0.
            request_filter["limit"] = _FETCH_LIMIT
        await self._send_req(websocket, subscription_id, request_filter)

    async def _subscribe_websocket(self, websocket) -> Dict[str, Optional[str]]:
        """Subscribe to every watched conversation plus membership events (kind 44100 p-tagged to us) for DM discovery."""
        subscriptions: Dict[str, Optional[str]] = {}
        for index, channel_id in enumerate(list(self._channel_state)):
            if channel_id in self._restricted_channels:
                continue
            subscriptions[f"hermes-buzz-{index}"] = channel_id
            await self._send_channel_subscription(websocket, f"hermes-buzz-{index}", channel_id)
        if self._self_pubkey:
            membership = {"kinds": [_WS_MEMBERSHIP_KIND], "#p": [self._self_pubkey], "since": max(self._membership_since - 1, 0)}
            await self._send_req(websocket, _WS_MEMBERSHIP_SUB_ID, membership)
            subscriptions[_WS_MEMBERSHIP_SUB_ID] = None
        return subscriptions

    async def _rediscover_and_subscribe(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Rediscover conversations and subscribe to any adopted since (fresh DMs dispatch from their start)."""
        before = set(self._channel_state)
        await self._discover_dms(seed=False)
        for channel_id in list(self._channel_state):
            if channel_id in before:
                continue
            subscription_id = f"hermes-buzz-dm-{len(subscriptions)}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
            logger.info("Buzz: subscribed to new conversation %s", channel_id)

    async def _ws_discovery_loop(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Periodic discovery on the poll cadence: relays don't guarantee a kind-44100 event for every new
        conversation. Failures retry next tick; the read loop alone owns connection health.

        The kind-44100 membership subscription is the fast path, but relays do not guarantee a membership
        event for every conversation that materializes mid-session (#93557) — some emit none at all for new
        DM-shaped conversations. The poll transport papers over this by re-running discovery every
        ``_DM_DISCOVERY_EVERY`` sweeps; this loop gives the WS transport the same guarantee on the same
        cadence.
        """
        interval = max(self.poll_interval * _DM_DISCOVERY_EVERY, _MIN_POLL_INTERVAL)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._rediscover_and_subscribe(websocket, subscriptions)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Buzz: WebSocket discovery sweep failed", exc_info=True)

    async def _websocket_loop(self) -> None:
        """Persistent authenticated subscription with bounded reconnect backoff; `since` filters resume on reconnect."""
        import websockets
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self._websocket_url(), open_timeout=_WS_AUTH_TIMEOUT, close_timeout=5,
                    ping_interval=20, ping_timeout=20, max_size=_WS_MAX_MESSAGE_BYTES,
                ) as websocket:
                    await self._authenticate_websocket(websocket)
                    subscriptions = await self._subscribe_websocket(websocket)
                    if self._ws_ready is not None:
                        self._ws_ready.set()
                    backoff = 1.0
                    discovery_task = asyncio.create_task(self._ws_discovery_loop(websocket, subscriptions))
                    try:
                        await self._ws_read_loop(websocket, subscriptions)
                    finally:
                        discovery_task.cancel()
                        try:
                            await discovery_task
                        except (asyncio.CancelledError, Exception):
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Buzz: WebSocket disconnected; retrying in %.1fs: %s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _ws_read_loop(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Read frames until the relay closes; an idle read raises ConnectionError to reconnect."""
        frame_iter = websocket.__aiter__()
        while True:
            try:
                raw = await asyncio.wait_for(frame_iter.__anext__(), timeout=_WS_READ_IDLE_TIMEOUT)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise ConnectionError(f"no WebSocket frame for {_WS_READ_IDLE_TIMEOUT:.0f}s; assuming the connection went silent") from None
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                logger.warning("Buzz: ignoring malformed WebSocket frame")
                continue
            if isinstance(message, list) and message:
                await self._handle_ws_message(websocket, subscriptions, message)

    async def _handle_ws_message(self, websocket, subscriptions: Dict[str, Optional[str]], message: list) -> None:
        """Route one parsed relay frame (EVENT / CLOSED / NOTICE)."""
        if message[0] == "EVENT" and len(message) >= 3:
            subscription_id, event = str(message[1]), message[2]
            if not isinstance(event, dict):
                return
            if subscription_id == _WS_MEMBERSHIP_SUB_ID:
                # A membership event p-tagged to us: rediscover and subscribe to new conversations.
                self._membership_since = max(self._membership_since, int(event.get("created_at") or 0))
                await self._rediscover_and_subscribe(websocket, subscriptions)
                return
            channel_id = subscriptions.get(subscription_id)
            state = self._channel_state.get(channel_id or "")
            if channel_id and state is not None:
                await self._handle_events(channel_id, state, [event])
        elif message[0] == "CLOSED":
            detail = message[-1] if len(message) > 2 else "subscription closed"
            sub_id = str(message[1]) if len(message) > 1 else ""
            closed_channel = subscriptions.get(sub_id)
            # A membership rejection is permanent — drop the channel instead of reconnect-looping.
            rejected = any(m in str(detail).lower() for m in ("restricted", "not a channel member", "auth-required"))
            if not (rejected and closed_channel):
                raise ConnectionError(str(detail))
            logger.warning("Buzz: relay permanently rejected channel %s (%s) — removing from watch list", closed_channel, detail)
            self._restricted_channels.add(closed_channel)
            del subscriptions[sub_id]
            self._channel_state.pop(closed_channel, None)
        elif message[0] == "NOTICE":
            logger.warning("Buzz: relay notice: %s", message[-1])

    # ── Inbound polling ───────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll every watched channel for new events until cancelled."""
        while True:
            await asyncio.sleep(self.poll_interval)
            self._poll_count += 1
            try:
                if self._poll_count % _DM_DISCOVERY_EVERY == 0:
                    await self._discover_dms(seed=False)
                for channel_id in list(self._channel_state):
                    await self._poll_channel(channel_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Buzz: poll sweep failed", exc_info=True)

    def _new_channel_state(self, chat_type: str) -> dict:
        return {"chat_type": chat_type, "last_ts": 0, "seen": OrderedDict(), "event_meta": OrderedDict()}

    # ── Durable channel cursors ───────────────────────────────────────────

    @staticmethod
    def _cursor_path() -> Path:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / _CURSOR_STATE_SUBDIR / _CURSOR_STATE_FILENAME

    def _load_cursors(self) -> None:
        """Read persisted cursors; another identity/relay's file is ignored (ids collide), failures seed from history."""
        self._restored_cursors = {}
        try:
            if not (path := self._cursor_path()).exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Buzz: could not read channel cursors", exc_info=True)
            return
        if not isinstance(data, dict) or data.get("identity") != self._self_pubkey or data.get("relay") != self.relay_url:
            return
        if not isinstance(channels := data.get("channels"), dict):
            return
        for channel_id, entry in channels.items():
            if not isinstance(entry, dict):
                continue
            try:
                last_ts = int(entry.get("last_ts") or 0)
            except (TypeError, ValueError):
                continue
            raw_seen = entry.get("seen")
            seen = [str(event_id) for event_id in raw_seen][-_SEEN_CAP:] if isinstance(raw_seen, list) else []
            self._restored_cursors[str(channel_id)] = {"chat_type": str(entry.get("chat_type") or ""), "last_ts": last_ts, "seen": seen}

    def _save_cursors(self) -> None:
        """Persist every watched channel's cursor.  Never raises."""
        channels = {
            channel_id: {
                "chat_type": state.get("chat_type") or "group", "last_ts": int(state.get("last_ts") or 0),
                "seen": list(state.get("seen") or ()),
            }
            for channel_id, state in self._channel_state.items()
        }
        payload = {"identity": self._self_pubkey, "relay": self.relay_url, "channels": channels}
        try:
            from utils import atomic_json_write
            atomic_json_write(self._cursor_path(), payload, indent=None)
        except Exception:
            logger.debug("Buzz: could not persist channel cursors", exc_info=True)

    @staticmethod
    def _cursor_mark(state: dict) -> tuple:
        """Cheap change detector for one channel's cursor."""
        seen = state.get("seen") or ()
        return int(state.get("last_ts") or 0), len(seen), (next(reversed(seen), None) if seen else None)

    def _restore_channel_state(self, channel_id: str, chat_type: str) -> bool:
        """Install a persisted cursor (True when one existed): seeding would mark downtime arrivals as seen.

        Restoring is what closes the restart gap: seeding from current history instead would mark everything
        that arrived while the gateway was down as already seen, so the relay's durable copy is never
        dispatched (#90464).
        """
        restored = self._restored_cursors.pop(channel_id, None)
        if restored is None:
            return False
        state = self._new_channel_state(restored["chat_type"] or chat_type)
        state["last_ts"] = restored["last_ts"]
        state["seen"] = OrderedDict((event_id, None) for event_id in restored["seen"])
        self._channel_state[channel_id] = state
        return True

    async def _seed_channel(self, channel_id: str, chat_type: str) -> None:
        """Initialize a channel's high-water mark from its newest events."""
        if self._restore_channel_state(channel_id, chat_type):
            return
        state = self._channel_state[channel_id] = self._new_channel_state(chat_type)
        code, out, err = await self._run_cli(["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)])
        if code != 0:
            logger.warning("Buzz: could not seed channel %s — %s", channel_id, _cli_error_message(err, code))
            # "now" so a transiently unreadable channel never replays its history later.
            state["last_ts"] = int(time.time())
            return
        for event in _parse_json_list(out):
            if event_id := event.get("id"):
                state["seen"][str(event_id)] = None
            state["last_ts"] = max(state["last_ts"], int(event.get("created_at") or 0))
            # History is never dispatched but feeds event_meta (post-restart replies to us must match) and latches DMs.
            # See #75826.
            self._remember_event(state, event)
            self._maybe_latch_dm(channel_id, state, event)
        self._trim_seen(state)

    async def _discover_dms(self, *, seed: bool) -> None:
        """Watch DMs: startup ones are seeded, mid-run ones dispatch from their start. ``dms list`` is best-effort
        (some relays return ``[]``); the fallback shape is a ``channels list`` entry named "DM" with empty
        description. Named rooms and missing metadata fail closed as groups.

        ``dms list`` is only a best-effort source: on some hosted relays it returns ``[]`` even when DM
        conversations exist (#68871).
        """
        code, out, _err = await self._run_cli(["dms", "list"])
        for dm in _parse_json_list(out) if code == 0 else []:
            dm_id = str(dm.get("dm_id") or "")
            if dm_id and dm_id not in self._channel_state and dm_id not in self._restricted_channels:
                await self._adopt_conversation(dm_id, seed)
                self._channel_names.setdefault(dm_id, "DM")
        code, out, _err = await self._run_cli(["channels", "list"])
        if code != 0:
            return
        for ch in _parse_json_list(out):
            if not (ch_id := str(ch.get("channel_id") or "")):
                continue
            self._channel_meta[ch_id] = ch
            self._channel_names.setdefault(ch_id, str(ch.get("name") or ch_id))
            if ch_id in self._restricted_channels:
                continue
            if self._may_reclassify_as_dm(ch_id):
                # DM-shaped entries promote to DM — including ones already watched.
                # See #77987, #87899, #99431.
                if ch_id in self._channel_state:
                    self._channel_state[ch_id]["chat_type"] = "dm"
                else:
                    await self._adopt_conversation(ch_id, seed)
            elif ch_id not in self._channel_state and not seed and not self.channels:
                # Watch-all mode adopts channels joined mid-run (seeded: history predates us); explicit lists stay authoritative.
                # Live adoption of real community channels joined mid-run (#75107): in watch-all mode (no
                # explicit channels list) a channel the agent is added to after connect() must start
                # dispatching without a gateway restart. Unlike a fresh DM its history predates us, so it is
                # always seeded from its newest events — only messages sent after adoption dispatch.
                await self._seed_channel(ch_id, chat_type="group")
                logger.info("Buzz: adopted newly joined channel %s (%s)", ch_id, self._channel_names.get(ch_id, ch_id))

    async def _adopt_conversation(self, channel_id: str, seed: bool) -> None:
        """Start watching a DM conversation: seed at startup, else restore or start fresh."""
        if seed:
            await self._seed_channel(channel_id, chat_type="dm")
        elif not self._restore_channel_state(channel_id, "dm"):
            self._channel_state[channel_id] = self._new_channel_state("dm")

    async def _poll_channel(self, channel_id: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is None:
            return
        args = ["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)]
        if state["last_ts"]:
            # Nostr `since` is inclusive: same-second events re-fetch and de-dupe by id.
            args += ["--since", str(state["last_ts"])]
        code, out, err = await self._run_cli(args)
        if code != 0:
            logger.debug("Buzz: poll of channel %s failed — %s", channel_id, _cli_error_message(err, code))
            return
        await self._handle_events(channel_id, state, _parse_json_list(out))

    async def _handle_events(self, channel_id: str, state: dict, events: List[dict]) -> None:
        """Handle a batch, trim, and persist only when the cursor moved (idle channels don't rewrite the file)."""
        before = self._cursor_mark(state)
        for event in events:
            await self._handle_event(channel_id, state, event)
        self._trim_seen(state)
        if self._cursor_mark(state) != before:
            self._save_cursors()

    @staticmethod
    def _parse_imeta_attachments(event: dict) -> Tuple[List[dict], int]:
        """Return accepted NIP-94 metadata and the rejected ``imeta`` count."""
        tags = event.get("tags")
        if not isinstance(tags, list):
            return [], 0
        attachments: List[dict] = []
        rejected = total_declared_bytes = 0
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or not tag or tag[0] != "imeta":
                continue
            if len(attachments) >= _MAX_INBOUND_ATTACHMENTS:
                rejected += 1
                continue
            fields: Dict[str, str] = {}
            for key, separator, value in (f.partition(" ") for f in tag[1:] if isinstance(f, str)):
                if separator and key not in fields:
                    fields[key] = value.strip()
            url, digest = fields.get("url", ""), fields.get("x", "").lower()
            try:
                size = int(fields.get("size", ""))
                parsed = urlsplit(url)
                parsed_hostname = parsed.hostname
                parsed.port  # access validates malformed/non-numeric ports
            except (TypeError, ValueError):
                rejected += 1
                continue
            if (
                parsed.scheme != "https" or not parsed_hostname or parsed.username or parsed.password or parsed.fragment
                or not _HEX64_RE.fullmatch(digest) or not 0 < size <= _MAX_INBOUND_ATTACHMENT_BYTES
                or total_declared_bytes + size > _MAX_INBOUND_ATTACHMENT_BYTES
            ):
                rejected += 1
                continue
            total_declared_bytes += size
            attachments.append({"url": url, "sha256": digest, "size": size,
                                "filename": _safe_attachment_filename(fields.get("filename", "")), "mime_type": fields.get("m", "")[:255]})
        return attachments, rejected

    @staticmethod
    def _imeta_attachments(event: dict) -> List[dict]:
        """Return bounded, structurally valid NIP-94 attachment metadata."""
        return BuzzAdapter._parse_imeta_attachments(event)[0]

    @staticmethod
    def _attachment_rejection_note(rejected: int) -> str:
        """Return a fixed-width diagnostic for malformed or excess metadata."""
        return f"[{rejected if rejected <= 999 else '999+'} Buzz attachment(s) rejected as malformed or over limits.]"

    async def _download_attachment(self, metadata: dict) -> Optional[CachedMedia]:
        """Download, integrity-check, and cache one authorized Buzz attachment."""
        url = metadata["url"]
        try:
            parsed_url = urlsplit(url)
            origin = ((parsed_url.hostname or "").lower().rstrip("."), parsed_url.port or 443)
        except ValueError:
            parsed_url, origin = None, ("", 0)
        if parsed_url is None or parsed_url.scheme != "https" or origin not in self._attachment_origins:
            logger.warning("Buzz: refusing attachment from untrusted origin %s:%s", origin[0] or "<missing>", origin[1])
            return None
        import httpx
        try:
            timeout = httpx.Timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT)
            async with (
                asyncio.timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT),
                httpx.AsyncClient(follow_redirects=False, timeout=timeout, headers={"Accept-Encoding": "identity"}) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code != 200:
                    logger.warning("Buzz: attachment download returned HTTP %s", response.status_code)
                    return None
                if content_length := response.headers.get("content-length"):
                    try:
                        declared_response_size = int(content_length)
                    except ValueError:
                        return None
                    if declared_response_size != metadata["size"]:
                        logger.warning("Buzz: attachment Content-Length does not match imeta size")
                        return None
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > metadata["size"]:
                        logger.warning("Buzz: attachment exceeded its declared size")
                        return None
        except (TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning("Buzz: attachment download failed: %s", exc)
            return None
        for bad, what in ((len(data) != metadata["size"], "size"), (hashlib.sha256(data).hexdigest() != metadata["sha256"], "SHA-256")):
            if bad:
                logger.warning("Buzz: attachment %s does not match imeta", what)
                return None
        try:
            return await cache_media_bytes_async(bytes(data), filename=metadata["filename"], mime_type=metadata["mime_type"])
        except (OSError, ValueError) as exc:
            logger.warning("Buzz: attachment cache write failed: %s", exc)
            return None

    async def _cache_inbound_attachments(self, metadata_items: List[dict]) -> List[CachedMedia]:
        return [a for m in metadata_items if (a := await self._download_attachment(m)) is not None]

    async def _handle_event(self, channel_id: str, state: dict, event: dict) -> None:
        """De-dupe, filter, and dispatch a single ``messages get`` event."""
        event_id = str(event.get("id") or "")
        created_at = int(event.get("created_at") or 0)
        if not event_id or event_id in state["seen"]:
            return
        state["seen"][event_id] = None
        state["last_ts"] = max(state["last_ts"], created_at)
        if int(event.get("kind") or 0) not in _DISPATCH_KINDS:
            return
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        attachment_metadata, rejected_attachments = self._parse_imeta_attachments(event)
        if not pubkey or not isinstance(content, str) or not (content.strip() or attachment_metadata or rejected_attachments):
            return
        # Cache before any early return so self-echo and concurrent-author traffic can still be reply parents.
        self._remember_event(state, event)
        # See #75826.
        if pubkey == self._self_pubkey:
            return
        # Reclassify a leaked DM before gating so its first un-mentioned message both latches and dispatches.
        self._maybe_latch_dm(channel_id, state, event)
        is_dm = state["chat_type"] == "dm"
        reply_parent_id = _event_reply_parent_id(event)
        reply_meta = self._lookup_event_meta(state, reply_parent_id) if reply_parent_id else None
        reply_to_is_own = bool(reply_meta is not None and reply_meta[0] == self._self_pubkey)
        # Channels dispatch only when addressed (@mention or p-tag) or replying to us (Signal/WhatsApp parity),
        # unless require_mention is off. DMs always dispatch.
        if not is_dm and self.require_mention and not self._is_addressed(event) and not reply_to_is_own:
            return
        # Adapter-level allow-list (gateway also applies it centrally); empty = no filter.
        if self._allowed_pubkeys and pubkey not in self._allowed_pubkeys:
            if pubkey in self._reaction_only_pubkeys and _p_tagged(event, self._self_pubkey) and self._is_mentioned(content):
                await self.send_reaction(channel_id, event_id, "👀")
            logger.debug("Buzz: ignoring message from unauthorized pubkey %s…", pubkey[:8])
            return
        # Strip a leading @mention (DMs often open with one too) so "@Chip /whoami" is recognized as a command.
        dispatch_text = self._strip_mention(content)
        # NIP-10 root scopes the session; remember it so our reply joins the SAME thread instead of nesting.
        thread_id = self._extract_thread_root(event)
        self._record_thread_root(event_id, event)
        # Attachment fetch spends credentials: only the gateway's explicit ``True`` permits it (else fail closed).
        # The message still dispatches so GatewayRunner can apply denial/pairing.
        chat_type = "dm" if is_dm else "group"
        fetch_allowed = bool(attachment_metadata) and self._is_sender_authorized(pubkey, chat_type, channel_id) is True
        attachments = await self._cache_inbound_attachments(attachment_metadata) if fetch_allowed else []
        if rejected_attachments:
            dispatch_text = f"{dispatch_text}\n{self._attachment_rejection_note(rejected_attachments)}".strip()
        if fetch_allowed and (failed := len(attachment_metadata) - len(attachments)):
            dispatch_text = f"{dispatch_text}\n[{failed} Buzz attachment(s) could not be downloaded or failed integrity checks.]".strip()
        message_type = MessageType.TEXT
        if attachments:
            # Mixed kinds use document semantics so an audio member is not mistaken for a voice note (STT).
            kinds = {attachment.kind for attachment in attachments}
            message_type = _ATTACHMENT_KIND_TYPES.get(next(iter(kinds)), MessageType.DOCUMENT) if len(kinds) == 1 else MessageType.DOCUMENT
        await self._dispatch_message(
            text=dispatch_text, chat_id=channel_id, chat_type=chat_type, user_id=pubkey,
            user_name=await self._resolve_user_name(pubkey), message_id=event_id,
            created_at=created_at, thread_id=thread_id, reply_to_message_id=reply_parent_id,
            reply_to_text=reply_meta[1] if reply_meta else None, reply_to_author_id=reply_meta[0] if reply_meta else None,
            reply_to_is_own_message=reply_to_is_own, media_urls=[attachment.path for attachment in attachments],
            media_types=[attachment.media_type for attachment in attachments], message_type=message_type, raw_message=event,
        )

    # ── DM classification: DMs leak in via ``channels list`` as "group"; a real channel's p-tag is only addressing ──

    # ── DM classification (issue #68871) ────────────────────────────────── ``buzz dms list`` returns [] on
    # some hosted relays even when DM conversations exist, so DMs can leak in through ``channels list`` as
    # chat_type="group". Relay-materialized DMs are named "DM" with an empty description, which periodic
    # discovery promotes to DM even when messages omit recipient p-tags. Named channels and missing metadata
    # fail closed. In normal channels a p-tag is only an addressing signal and must wake the agent without
    # changing the conversation type.
    def _may_reclassify_as_dm(self, channel_id: str) -> bool:
        """True when metadata does not rule out a DM (name "DM", empty description); missing metadata fails closed."""
        meta = self._channel_meta.get(channel_id)
        if meta is None:
            return False
        return str(meta.get("name") or "").strip() == "DM" and not str(meta.get("description") or "").strip()

    def _p_tagged_to_self(self, event: dict) -> bool:
        """True when the signed event addresses this identity by pubkey."""
        return bool(self._self_pubkey) and _p_tagged(event, self._self_pubkey)

    def _is_direct_message_event(self, channel_id: str, event: dict) -> bool:
        """Kind-9 from another user, p-tagged to us, text NOT mentioning us — structural DM addressing, not a typed @."""
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        return bool(
            self._self_pubkey and self._may_reclassify_as_dm(channel_id) and int(event.get("kind") or 0) == _CHAT_KIND
            and pubkey and pubkey != self._self_pubkey and self._p_tagged_to_self(event)
            and isinstance(content, str) and not self._is_mentioned(content)
        )

    def _maybe_latch_dm(self, channel_id: str, state: dict, event: dict) -> None:
        """Latch a group conversation to "dm" once a direct message is seen; it sticks."""
        if state["chat_type"] == "dm" or not self._is_direct_message_event(channel_id, event):
            return
        state["chat_type"] = "dm"
        self._channel_names.setdefault(channel_id, "DM")
        logger.info("Buzz: conversation %s reclassified as DM (message p-tagged to self)", channel_id)

    def _is_mentioned(self, content: str) -> bool:
        """True when text explicitly addresses this agent (npub, hex, or @name)."""
        lowered = content.lower()
        patterns = []
        if self._self_pubkey and _HEX64_RE.fullmatch(self._self_pubkey):
            patterns.append(rf"(?<![0-9a-f]){re.escape(self._self_pubkey)}(?![0-9a-f])")
        if self._self_npub:
            patterns.append(rf"(?<![a-z0-9]){re.escape(self._self_npub.lower())}(?![a-z0-9])")
        if self._display_name:
            patterns.append(rf"(?<![\w@])@{re.escape(self._display_name.lower())}" + r"(?=$|[\s,;.!?:)\]}])")
        return any(re.search(p, lowered) for p in patterns)

    def _is_addressed(self, event: dict) -> bool:
        """True when a group event carries an explicit text or p-tag address."""
        content = event.get("content")
        return isinstance(content, str) and (self._is_mentioned(content) or self._p_tagged_to_self(event))

    def _strip_mention(self, content: str) -> str:
        """Strip a LEADING @mention of this agent so ``is_command()`` sees "/whoami"; mid-sentence mentions stay."""
        text = content.strip()
        # Display names require '@'; npub/hex identities are unambiguous without it.
        candidates = []
        if self._display_name:
            candidates.append(rf"@{re.escape(self._display_name)}" + r"(?=$|[\s,;.!?:)\]}])")
        if self._self_npub:
            candidates.append(rf"@?{re.escape(self._self_npub)}(?![a-z0-9])")
        if self._self_pubkey:
            candidates.append(rf"@?{re.escape(self._self_pubkey)}(?![0-9a-f])")
        if not candidates:
            return text
        return re.sub(rf"^(?:{'|'.join(candidates)})[\s:,]*", "", text, count=1, flags=re.IGNORECASE).strip()

    async def _resolve_user_name(self, pubkey: str) -> str:
        """Pubkey -> display name, cached (negatively too, so profile-less pubkeys don't re-query); npub prefix fallback."""
        if (cached := self._user_names.get(pubkey)) is not None:
            return cached
        code, out, _err = await self._run_cli(["users", "get", "--pubkey", pubkey])
        profiles = _parse_json_list(out) if code == 0 else []
        name = str(profiles[0].get("display_name") or "").strip() if profiles else ""
        name = self._user_names[pubkey] = name or (hex_to_npub(pubkey) or pubkey)[:16]
        return name

    @staticmethod
    def _trim_seen(state: dict) -> None:
        seen = state["seen"]
        while len(seen) > _SEEN_CAP:
            seen.popitem(last=False)
        meta = state.get("event_meta")
        if isinstance(meta, OrderedDict):
            while len(meta) > _SEEN_CAP:
                meta.popitem(last=False)

    def _mark_seen(self, channel_id: str, event_id: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is not None:
            state["seen"][event_id] = None
            self._trim_seen(state)

    # ── Thread anchoring: NIP-10 replies carry ["e", root, "", "root"] + ["e", parent, "", "reply"]; a thread
    # STARTER carries a lone "reply". The gateway anchors on the trigger id, which inside a thread would nest
    # a sub-thread under every answer — so remember each inbound message's ROOT and reply against that.

    _THREAD_ROOT_CACHE = 512

    @staticmethod
    def _extract_thread_root(event: dict) -> Optional[str]:
        """Return the NIP-10 thread root of ``event``, or None if top-level."""
        root = reply = None
        for target, marker in _e_tags(event):
            marker = marker.lower()
            if marker == "root":
                root = str(target)
            elif marker == "reply":
                reply = str(target)
            elif not marker and reply is None:
                reply = str(target)  # unmarked (deprecated positional) e-tag = parent
        # A lone "reply" e-tag started a thread off <reply>; that parent IS the root.
        return root or reply

    def _record_thread_root(self, event_id: str, event: dict) -> None:
        """Cache the thread root for an inbound message id."""
        if not event_id:
            return
        roots = self._thread_roots
        roots[event_id] = self._extract_thread_root(event)
        roots.move_to_end(event_id)
        while len(roots) > self._THREAD_ROOT_CACHE:
            roots.popitem(last=False)

    def _resolve_reply_anchor(self, anchor: Optional[str]) -> Optional[str]:
        """Thread root when the trigger was inside a thread (reply joins it), else the anchor unchanged."""
        return (self._thread_roots.get(str(anchor)) or anchor) if anchor else anchor

    def _remember_event(self, state: dict, event: dict) -> None:
        """Record author + content snippet for later NIP-10 parent lookup."""
        event_id = str(event.get("id") or "")
        content = event.get("content")
        if event_id:
            snippet = content[:_EVENT_META_CONTENT_CAP] if isinstance(content, str) else ""
            self._store_event_meta(state, event_id, str(event.get("pubkey") or "").lower(), snippet)

    def _remember_event_meta(self, channel_id: str, event_id: str, pubkey: str, content: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is not None:
            self._remember_event(state, {"id": event_id, "pubkey": pubkey, "content": content or ""})

    @staticmethod
    def _store_event_meta(state: dict, event_id: str, pubkey: str, snippet: str) -> None:
        cache = state.setdefault("event_meta", OrderedDict())
        if not isinstance(cache, OrderedDict):
            cache = state["event_meta"] = OrderedDict(cache)
        cache[event_id] = (pubkey, snippet)
        cache.move_to_end(event_id)
        while len(cache) > _SEEN_CAP:
            cache.popitem(last=False)

    @staticmethod
    def _lookup_event_meta(state: dict, event_id: Optional[str]) -> Optional[Tuple[str, str]]:
        entry = (state.get("event_meta") or {}).get(event_id) if event_id else None
        if not entry or not isinstance(entry, tuple) or len(entry) < 2:
            return None
        return str(entry[0] or ""), str(entry[1] or "")

    async def _localize_inbound_media(
        self, text: str, message_id: str, *, user_id: str = "", chat_type: Optional[str] = None, chat_id: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], MessageType]:
        """Authenticate and cache same-relay media refs in *text* (failures skipped per object). Spends our
        credentials on a sender-chosen URL, so it runs only on the gateway's explicit ``True``."""
        urls, replacements = _find_relay_media_refs(text, self.relay_url)
        if not urls:
            return text, [], [], MessageType.TEXT
        if self._is_sender_authorized(user_id, chat_type, chat_id) is not True:
            logger.warning("Buzz: not localizing %d media reference(s) in message %s — sender %s… is not explicitly authorized",
                           len(urls), message_id[:12], (user_id or "?")[:8])
            return text, [], [], MessageType.TEXT
        cleaned_text = _replace_media_refs(text, replacements)
        media_urls: List[str] = []
        media_types: List[str] = []
        media_kinds: List[str] = []
        from gateway.platforms.base import cache_media_bytes_async, validate_inbound_media_size
        for url in urls:
            path_match = _MEDIA_PATH_RE.fullmatch(urlsplit(url).path)
            if path_match is None:
                continue
            label = f"{path_match.group('sha')[:12]}{(path_match.group('ext') or '.bin').lower()}"
            try:
                with tempfile.TemporaryDirectory(prefix="hermes-buzz-media-") as temp_dir:
                    download_path = Path(temp_dir) / f"buzz_{label}"
                    code, _out, _err = await self._run_cli(["media", "get", "-o", str(download_path), url])
                    if code != 0 or not download_path.is_file():
                        logger.warning("Buzz: failed to localize inbound media %s (exit %d)", label, code)
                        continue
                    validate_inbound_media_size(download_path.stat().st_size, media_type="Buzz media")
                    mime_type = mimetypes.guess_type(download_path.name)[0] or "application/octet-stream"
                    # Up to the inbound media cap (128 MiB) — read off the loop too.
                    data = await asyncio.to_thread(download_path.read_bytes)
                    cached = await cache_media_bytes_async(data, filename=download_path.name, mime_type=mime_type)
            except Exception as exc:
                logger.warning("Buzz: failed to localize inbound media %s (%s)", label, type(exc).__name__)
                continue
            if cached is None:
                logger.warning("Buzz: rejected invalid inbound media %s", label)
                continue
            media_urls.append(cached.path)
            media_types.append(cached.media_type)
            media_kinds.append(cached.kind)
        if media_urls:
            logger.info("Buzz: localized %d inbound media attachment(s) for message %s", len(media_urls), message_id[:12])
        # Priority order: image > audio > video > any other kind.
        message_type = MessageType.TEXT if not media_kinds else next((mt for k, mt in _MEDIA_KIND_PRIORITY if k in media_kinds), MessageType.DOCUMENT)
        if not cleaned_text:
            cleaned_text = "(attachment)" if media_urls else "(Buzz media attachment unavailable)"
        return cleaned_text, media_urls, media_types, message_type

    async def _dispatch_message(
        self, text: str, chat_id: str, chat_type: str, user_id: str, user_name: str,
        message_id: str, created_at: int, thread_id: Optional[str] = None,
        reply_to_message_id: Optional[str] = None, reply_to_text: Optional[str] = None,
        reply_to_author_id: Optional[str] = None, reply_to_is_own_message: bool = False,
        media_urls: Optional[List[str]] = None, media_types: Optional[List[str]] = None,
        message_type: MessageType = MessageType.TEXT, raw_message: Any = None,
    ) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return
        media_urls = list(media_urls or [])
        media_types = list(media_types or [])
        # Same-relay URL refs are localized in addition to the caller's imeta attachments (both explicit-True gated).
        localized = await self._localize_inbound_media(text, message_id, user_id=user_id, chat_type=chat_type, chat_id=chat_id)
        text, localized_urls, localized_types, localized_type = localized
        for path, mime in zip(localized_urls, localized_types):
            if path not in media_urls:
                media_urls.append(path)
                media_types.append(mime)
        if message_type == MessageType.TEXT:
            message_type = localized_type
        elif localized_urls and localized_type not in (message_type, MessageType.TEXT):
            # Mixed sources use document semantics so audio isn't routed to STT.
            message_type = MessageType.DOCUMENT
        source = self.build_source(
            chat_id=chat_id, chat_name=self._channel_names.get(chat_id, chat_id), chat_type=chat_type,
            user_id=user_id, user_name=user_name, thread_id=thread_id,
        )
        event = MessageEvent(
            text=text, message_type=message_type, source=source, raw_message=raw_message, message_id=message_id,
            media_urls=list(media_urls), media_types=list(media_types), media_text_inlined=[False] * len(media_urls),
            timestamp=datetime.fromtimestamp(created_at) if created_at else datetime.now(),
            reply_to_message_id=reply_to_message_id, reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author_id, reply_to_is_own_message=reply_to_is_own_message,
        )
        await self.handle_message(event)
        # "Seen" reaction: signals the message was received and is being processed.
        try:
            await self.send_reaction(chat_id, message_id, "👀")
        except Exception:
            logger.debug("Buzz: reaction failed for message %s", message_id[:12], exc_info=True)


# ── Plugin registration ──────────────────────────────────────────────────────

def _profile_buzz_extra() -> dict:
    """``buzz.extra`` from the scoped profile's config.yaml for ``check_requirements``; failures yield {} (fail closed)."""
    if not _profile_scoped():
        return {}
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import read_user_config_raw
        cfg = read_user_config_raw(Path(get_hermes_home()) / "config.yaml")
    except Exception:
        return {}
    buzz = ((cfg.get("gateway") or {}).get("platforms") or {}).get("buzz") if isinstance(cfg, dict) else None
    extra = buzz.get("extra", buzz) if isinstance(buzz, dict) else None
    return extra if isinstance(extra, dict) else {}


def check_requirements() -> bool:
    """Check if Buzz is configured: a relay URL plus a resolvable key."""
    if _profile_scoped():
        # Secondary profile: os.environ's BUZZ_* are the default profile's and must not satisfy the gate.
        # Consult the profile's own config.yaml (via the scoped home override) and its secret scope instead;
        # an unconfigured profile fails closed. See #98738.
        extra = _profile_buzz_extra()
        return bool(str(extra.get("relay_url") or "").strip() and _resolve_private_key(extra))
    # The gate runs before per-profile scopes install; the relay can be externally managed too.
    return bool((_get_scoped_secret("BUZZ_RELAY_URL", "") or "").strip()) and bool(_resolve_private_key())


def validate_config(config) -> bool:
    """Validate that the platform config has enough information to connect."""
    extra = getattr(config, "extra", {}) or {}
    # Scoped: extra is authoritative; unscoped: env read gains the external-secret rung.
    if _profile_scoped():
        # See #98738.
        relay = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
        relay = relay if relay is not None else extra.get("relay_url", "")
    else:
        relay = _get_scoped_secret("BUZZ_RELAY_URL", "") or extra.get("relay_url", "")
    return bool(relay and _resolve_private_key(extra))


def is_connected(config) -> bool:
    """Check whether Buzz is configured (env or config.yaml)."""
    return validate_config(config)


# (extra key, env var, kind): "str" bridges truthy values as-is, "csv" joins lists,
# "flag" lowercases when present, "thread" lowercases and ignores profile scope.
_YAML_BRIDGE = (
    ("relay_url", "BUZZ_RELAY_URL", "str"), ("cli_path", "BUZZ_CLI_PATH", "str"),
    ("home_channel", "BUZZ_HOME_CHANNEL", "str"), ("transport", "BUZZ_TRANSPORT", "str"),
    ("channels", "BUZZ_CHANNELS", "csv"), ("allowed_users", "BUZZ_ALLOWED_USERS", "csv"),
    ("reaction_only_users", "BUZZ_REACTION_ONLY_USERS", "csv"), ("allow_all_users", "BUZZ_ALLOW_ALL_USERS", "flag"),
    ("require_mention", "BUZZ_REQUIRE_MENTION", "flag"), ("reply_in_thread", "BUZZ_REPLY_IN_THREAD", "thread"),
    ("reply_to_mode", "BUZZ_REPLY_TO_MODE", "thread"),
)


def _apply_yaml_config(yaml_cfg: dict, buzz_cfg: dict) -> Optional[dict]:
    """Bridge ``buzz.extra`` into ``BUZZ_*`` env so a config.yaml-only setup passes the env-reading gate.
    Env wins over YAML; ``BUZZ_PRIVATE_KEY`` is never sourced from config.yaml."""
    extra = buzz_cfg.get("extra", buzz_cfg) or {}
    if not isinstance(extra, dict):
        return None
    # A secondary profile must NOT write the process-global env (first-writer-wins would pin it for every profile).
    # Under multiplex, a secondary profile's config loads inside its runtime scope; its values must NOT be
    # written to the process-global env, where first-writer-wins would pin them for every other profile
    # (issue #72348 Telegram/Discord mirror, Buzz side of #98738). Its adapter reads the profile's
    # PlatformConfig.extra directly instead.
    skip_env_bridge = _profile_scoped()
    interval = extra.get("poll_interval")
    if interval is not None and not skip_env_bridge and not os.getenv("BUZZ_POLL_INTERVAL"):
        os.environ["BUZZ_POLL_INTERVAL"] = str(interval)
    for src, env, kind in _YAML_BRIDGE:
        val = extra.get(src)
        missing = {"str": not val, "csv": val is None}.get(kind, src not in extra)
        if missing or (kind != "thread" and skip_env_bridge) or os.getenv(env):
            continue
        if kind == "csv" and isinstance(val, (list, tuple)):
            val = ",".join(str(v) for v in val)
        os.environ[env] = str(val).lower() if kind in ("flag", "thread") else str(val)
    return None


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env so env-only setups show in gateway status; None if unconfigured."""
    if _profile_scoped():
        # Process env holds the default profile's BUZZ_*; never fabricate Buzz for a secondary profile.
        return None
    # Secondary profile scope (#98738): the process env's BUZZ_* values are the default profile's
    # configuration, not this profile's — env enablement must not fabricate a Buzz platform for a profile
    # that did not configure one.
    relay = os.getenv("BUZZ_RELAY_URL", "").strip()
    if not relay or not _resolve_private_key():
        return None
    seed: dict = {"relay_url": relay}
    if channels := os.getenv("BUZZ_CHANNELS", "").strip():
        seed["channels"] = [c.strip() for c in channels.split(",") if c.strip()]
    if interval := os.getenv("BUZZ_POLL_INTERVAL", "").strip():
        try:
            seed["poll_interval"] = float(interval)
        except ValueError:
            pass
    if cli_path := os.getenv("BUZZ_CLI_PATH", "").strip():
        seed["cli_path"] = cli_path
    # Cron delivery target; defaults to the first watched channel.
    home = os.getenv("BUZZ_HOME_CHANNEL", "").strip() or (seed.get("channels") or [""])[0]
    if home:
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("BUZZ_HOME_CHANNEL_NAME", home)}
    return seed


async def _standalone_send(
    pconfig, chat_id: str, message: str, *, thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """One-shot send without a live adapter (out-of-process ``deliver=buzz`` cron)."""
    extra = getattr(pconfig, "extra", {}) or {}
    relay = _configured_relay(extra)
    private_key = _resolve_private_key(extra)
    try:
        auth_tag = _resolve_auth_tag(extra)
    except ValueError as exc:
        return {"error": f"Buzz standalone send: {exc}"}
    cli_path = _configured_cli_path(extra)
    if not relay or not private_key:
        return {"error": "Buzz standalone send: BUZZ_RELAY_URL and BUZZ_PRIVATE_KEY must be configured"}
    if not cli_path:
        return {"error": "Buzz standalone send: buzz CLI binary not found"}
    if not (target := (chat_id or "").strip() or _configured_home_channel(extra)):
        return {"error": "Buzz standalone send: no target channel (set BUZZ_HOME_CHANNEL)"}
    args = ["messages", "send", "--channel", target, "--content", "-"]
    # Same reply_to_mode / reply_in_thread gate as the live adapter.
    if thread_id and _reply_to_mode(pconfig, extra) != "off":
        args += ["--reply-to", str(thread_id)]
    for media in media_files or []:
        args += ["--file", str(media[0] if isinstance(media, (list, tuple)) and media else media)]
    try:
        code, out, err = await _exec_buzz(cli_path, args, relay_url=relay, private_key=private_key, auth_tag=auth_tag, input_text=message)
        escaped = _escape_unresolved_presentation_mention(message, err) if code != 0 else None
        if escaped is not None:
            logger.info("Buzz: retrying standalone message after unresolved presentation-mention preflight")
            # Retry intentionally omits auth_tag (legacy behavior).
            code, out, err = await _exec_buzz(cli_path, args, relay_url=relay, private_key=private_key, input_text=escaped)
    except asyncio.CancelledError:
        raise
    except OSError as e:
        return {"error": f"Buzz standalone send failed to launch CLI: {_bounded_cli_message(str(e))}"}
    if code != 0:
        return {"error": f"Buzz standalone send failed: {_cli_error_message(err, code)}"}
    event_id, receipt_error = _parse_send_receipt(out)
    if receipt_error:
        return {"error": f"Buzz standalone send failed: {receipt_error}"}
    result = {"success": True, "message_id": event_id}
    if media_files:
        result["media_delivered"] = True
    return result


def interactive_setup() -> None:
    """Interactive ``hermes gateway setup`` flow (lazy CLI imports keep the plugin importable elsewhere)."""
    from hermes_cli.setup import (
        prompt, prompt_yes_no, save_env_value, get_env_value, print_header, print_info, print_warning, print_success,
    )
    def ask(label: str, env: str) -> str:
        return prompt(label, default=get_env_value(env) or "")

    print_header("Buzz")
    existing_relay = get_env_value("BUZZ_RELAY_URL")
    if existing_relay:
        print_info(f"Buzz: already configured (relay: {existing_relay})")
        if not prompt_yes_no("Reconfigure Buzz?", False):
            return
    print_info("Connect Hermes to a Buzz community (Block's Nostr-based human+agent platform).")
    print_info("   Requires the buzz CLI binary and a Nostr key that is a community member.")
    print()
    relay = prompt("Relay URL (e.g. https://mycommunity.communities.buzz.xyz)", default=existing_relay or "")
    if not relay:
        print_warning("Relay URL is required — skipping Buzz setup")
        return
    save_env_value("BUZZ_RELAY_URL", relay.strip())
    key = prompt("Nostr private key (nsec or hex; leave blank to keep current)", password=True)
    if key:
        save_env_value("BUZZ_PRIVATE_KEY", key.strip())
    elif not _resolve_private_key():
        print_warning("No private key configured — set BUZZ_PRIVATE_KEY before starting the gateway")
    channels = ask("Channel UUIDs to watch (comma-separated, empty = all joined channels)", "BUZZ_CHANNELS")
    if channels:
        save_env_value("BUZZ_CHANNELS", channels.replace(" ", ""))
    home = ask("Home channel UUID for cron/notification delivery (optional)", "BUZZ_HOME_CHANNEL")
    if home:
        save_env_value("BUZZ_HOME_CHANNEL", home.strip())
    print()
    print_info("🔒 Access control: restrict who can talk to the agent")
    if prompt_yes_no("Allow all community members to talk to the agent?", False):
        save_env_value("BUZZ_ALLOW_ALL_USERS", "true")
        save_env_value("BUZZ_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — anyone in the community can command the agent.")
    else:
        save_env_value("BUZZ_ALLOW_ALL_USERS", "false")
        allowed = ask("Allowed users (comma-separated npubs or hex pubkeys, empty to deny everyone)", "BUZZ_ALLOWED_USERS")
        save_env_value("BUZZ_ALLOWED_USERS", allowed.replace(" ", "") if allowed else "")
    print()
    print_success("Buzz configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="buzz", label="Buzz", adapter_factory=lambda cfg: BuzzAdapter(cfg), check_fn=check_requirements,
        validate_config=validate_config, is_connected=is_connected, required_env=["BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"],
        install_hint="Requires the buzz CLI binary (https://github.com/block/buzz) on PATH or at BUZZ_CLI_PATH",
        setup_fn=interactive_setup, env_enablement_fn=_env_enablement, apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="BUZZ_HOME_CHANNEL", standalone_sender_fn=_standalone_send,
        allowed_users_env="BUZZ_ALLOWED_USERS", allow_all_env="BUZZ_ALLOW_ALL_USERS", emoji="🐝",
        pii_safe=False,  # identities are pubkeys, not phone numbers
        allow_update_command=True,
        platform_hint=(
            "You are collaborating in a Buzz workspace (Block's Nostr-based "
            "human+agent platform). Markdown IS supported. Users address you "
            "by @-mentioning your name or npub in channels; direct messages "
            "reach you without a mention. Keep responses conversational."
        ),
    )
