"""
Buzz Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that connects to a Buzz community relay
(Block's open-source human+agent collaboration platform, built on the
Nostr protocol) and relays messages to/from the Hermes agent.

The adapter does not speak Nostr itself — it shells out to the ``buzz``
CLI binary ("JSON in, JSON out") via ``asyncio.create_subprocess_exec``.
Inbound delivery uses a poll loop (the CLI is request/response); see the
"Known limitations" note in the platform docs.

Configuration in config.yaml::

    gateway:
      platforms:
        buzz:
          enabled: true
          extra:
            relay_url: https://mycommunity.communities.buzz.xyz
            channels:                  # channel UUIDs to watch (empty = all joined)
              - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
            home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
            poll_interval: 4           # seconds between poll sweeps
            cli_path: ""               # path to the buzz binary (default: PATH, then ~/bin/buzz)
            credentials_file: ""       # JSON file holding the nsec (fallback for BUZZ_PRIVATE_KEY)
            allowed_users: []          # empty = allow all; entries are hex pubkeys or npubs
            reply_in_thread: true      # false = post replies flat to the channel timeline
            reaction_only_users: []    # acknowledge explicit tags without dispatching; allowed_users wins on overlap

Or via environment variables (overrides config.yaml):
    BUZZ_RELAY_URL, BUZZ_CHANNELS, BUZZ_HOME_CHANNEL, BUZZ_POLL_INTERVAL,
    BUZZ_CLI_PATH, BUZZ_CREDENTIALS_FILE, BUZZ_ALLOWED_USERS,
    BUZZ_REACTION_ONLY_USERS, BUZZ_ALLOW_ALL_USERS, BUZZ_REPLY_IN_THREAD,
    BUZZ_REPLY_TO_MODE

The only secret is BUZZ_PRIVATE_KEY (nsec or hex) — it belongs in
``~/.hermes/.env``.  It is passed to the CLI via the subprocess
environment and is never logged.
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

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import current_secret_scope as _current_secret_scope
from agent.secret_scope import get_secret as _scoped_get_secret
from agent.secret_scope import is_multiplex_active as _is_multiplex_active


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read with the default-profile startup fallback.

    Secondary profiles construct their adapters under a profile secret
    scope -- the scope is authoritative and a scoped miss returns ``default``
    (no cross-profile borrow from ``os.environ``, which may hold another
    profile's value). The DEFAULT profile's adapter constructs and sends
    *unscoped* under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash this path; there ``os.environ`` is that
    profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and
    ``gateway/platforms/whatsapp_common.py::_get_wsecret``.

    The no-scope path has one more rung for the platform requirement gate:
    ``check_requirements()`` runs at gateway startup BEFORE any per-profile
    secret scope is installed, and ``get_secret`` without a scope simply
    reads ``os.environ`` — so a Bitwarden-managed ``BUZZ_PRIVATE_KEY``
    (only ``BWS_ACCESS_TOKEN`` in ``.env``) was invisible to the check and
    Buzz was silently skipped (#95216). When no scope is active and the
    process env has no value, consult a one-shot build of the profile's
    secret mapping (``build_profile_secret_scope`` resolves external secret
    sources) so externally managed credentials pass the gate. An ACTIVE
    scope still shadows this rung entirely — it never runs under
    multiplexing, so cross-profile isolation is unchanged.
    """
    try:
        val = _scoped_get_secret(name, None)
    except _UnscopedSecretError:
        val = os.getenv(name)
    if val is None and _current_secret_scope() is None:
        val = _unscoped_profile_secrets().get(name)
    return val if val is not None else default


_UNSCOPED_PROFILE_SECRETS: Optional[Dict[str, str]] = None


def _unscoped_profile_secrets() -> Dict[str, str]:
    """One-shot build of the active profile's secret mapping.

    Cached for the process: the build shells out to external secret
    resolvers (Bitwarden via ``BWS_ACCESS_TOKEN``), and the requirement
    gate / validate / is_connected probes all want the same snapshot. Any
    failure degrades to an empty mapping — callers then simply report the
    platform as not configured, which is the pre-fix behavior. The cache
    is startup-gate-only: it pins whatever ``get_hermes_home()`` resolved
    on first build, so it must not be reused off the startup path (where
    a profile scope is always active and shadows it anyway).
    """
    global _UNSCOPED_PROFILE_SECRETS
    if _UNSCOPED_PROFILE_SECRETS is None:
        try:
            from agent.secret_scope import build_profile_secret_scope
            from hermes_constants import get_hermes_home

            _UNSCOPED_PROFILE_SECRETS = dict(
                build_profile_secret_scope(get_hermes_home())
            )
        except Exception:
            logger.warning(
                "Buzz requirement probe could not build the profile secret "
                "scope; Bitwarden-managed credentials will not be visible "
                "to the startup gate (#95216)",
                exc_info=True,
            )
            _UNSCOPED_PROFILE_SECRETS = {}
    return _UNSCOPED_PROFILE_SECRETS


def _profile_scoped() -> bool:
    """True when running inside a multiplexed secondary profile's scope.

    Secondary-profile adapters are constructed, connected, and reloaded
    inside ``_profile_runtime_scope`` (secret scope installed + multiplex
    active) — the same discriminator as the Discord adapter's
    ``_profile_scoped_config_load`` (#72348). The DEFAULT profile under
    multiplexing runs unscoped: ``os.environ`` holds its own bridge output
    there and keeps its legacy precedence.
    """
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        return bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        return False


def _scoped_platform_setting(env_name, extra, key):
    """Raw read of a non-secret Buzz setting, multiplex-profile-correct.

    Inside a secondary profile scope ``os.environ`` holds the DEFAULT
    profile's YAML-to-env bridge output (#98738), so the profile's
    ``PlatformConfig.extra`` is authoritative and env is not consulted: a
    missing key yields ``None`` and callers fail closed to their default
    instead of silently borrowing the default profile's relay, channels, or
    allowlist. Everywhere else — single-profile gateways, the default
    profile under multiplexing — the legacy ``os.getenv`` read is returned
    unchanged, so env-over-config precedence is preserved.
    """
    if _profile_scoped():
        return (extra or {}).get(key)
    return os.getenv(env_name)


logger = logging.getLogger(__name__)

from gateway.platforms.base import (
    BasePlatformAdapter,
    CachedMedia,
    SendResult,
    MessageEvent,
    MessageType,
    cache_media_bytes_async,
)
from gateway.config import Platform


# Buzz chat messages are Nostr kind 9 events.  ``buzz messages get`` also
# returns housekeeping kinds (joins, canvas updates, …) — only kind 9 is
# dispatched to the agent.
_CHAT_KIND = 9
# Kinds that carry agent-relevant conversation content and are dispatched
# (#90309): chat messages (9) plus the Buzz forum kinds — 45001 is a forum
# post (thread root) and 45003 a comment reply on it.  Block's own ACP
# harness documents this set (``buzz-acp --kinds 9,46010,40007,45001,
# 45002,45003``); the stream kinds (46010/40007/45002) are left out until
# their dispatch semantics are confirmed.  ``_is_direct_message_event``
# deliberately keeps the kind-9-only check: widening it there would let a
# p-tagged forum post be reclassified as a DM and bypass mention gating.
_DISPATCH_KINDS = frozenset({_CHAT_KIND, 45001, 45003})
_UNRESOLVED_MENTION_ERROR_RE = re.compile(
    r"mention '@(?P<name>[^']+)' does not match a current channel member"
)
_BUZZ_PRESENTATION_MENTION_SEPARATOR = "\u200b"


def _escape_unresolved_presentation_mention(content: str, error: str) -> Optional[str]:
    """Make one CLI-rejected ``@name`` token presentation-only.

    Buzz resolves whitespace-prefixed ``@name`` tokens into notification
    p-tags before signing or publishing. Ordinary prose such as a Hermes
    ``@session:...`` link can therefore fail mention preflight. Insert an
    invisible separator only after the rejected ``@`` so the rendered text
    remains readable while valid member mentions remain unchanged.

    Return ``None`` for unrelated errors or absent tokens. Callers retry at
    most once.
    """
    match = _UNRESOLVED_MENTION_ERROR_RE.search(error or "")
    if match is None:
        return None
    name = match.group("name")
    if not name:
        return None
    token = re.compile(
        rf"(?<!\S)@{re.escape(name)}(?=$|[^A-Za-z0-9._-])",
        re.IGNORECASE,
    )
    escaped, count = token.subn(
        lambda found: "@" + _BUZZ_PRESENTATION_MENTION_SEPARATOR + found.group(0)[1:],
        content,
    )
    return escaped if count else None

# How many events to request per poll / seed call.
_FETCH_LIMIT = 50
# Bound on the per-channel de-dupe set (events, not bytes).
_SEEN_CAP = 500
# Where the per-channel cursors survive a restart, relative to HERMES_HOME.
_CURSOR_STATE_SUBDIR = "buzz"
_CURSOR_STATE_FILENAME = "channel-cursors.json"
# Re-run DM discovery (``dms list`` plus the channels-list fallback) every
# N poll sweeps to pick up conversations opened mid-run.
_DM_DISCOVERY_EVERY = 5

_DEFAULT_POLL_INTERVAL = 4.0
_MIN_POLL_INTERVAL = 1.0
_CLI_TIMEOUT = 30.0

# Mention-resolution caches: member lists are cheap to refetch but hit on
# every publish containing "@", so a short TTL amortizes the CLI round-trip;
# display names change rarely, but must not survive a rename forever.
_MEMBER_CACHE_TTL = 60.0
_PROFILE_NAME_TTL = 300.0
# Inbound attachment limits. Attachments are downloaded only after the sender,
# mention, and allow-list gates pass; each one must declare and match an exact
# size and SHA-256 in its NIP-94 ``imeta`` tag.
_MAX_INBOUND_ATTACHMENTS = 4
_MAX_INBOUND_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ATTACHMENT_DOWNLOAD_TIMEOUT = 30.0
_MAX_ATTACHMENT_FILENAME_BYTES = 120


def _safe_attachment_filename(value: str) -> str:
    """Return a basename that is safe for cache files and agent context."""
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(
        character
        for character in name
        if ord(character) >= 32 and character != "\x7f"
    ).strip()
    if name in {"", ".", ".."}:
        return "attachment.bin"

    suffix = Path(name).suffix
    if len(suffix.encode("utf-8")) > 20:
        suffix = ""
    stem = name[:-len(suffix)] if suffix else name
    byte_budget = _MAX_ATTACHMENT_FILENAME_BYTES - len(suffix.encode("utf-8"))
    safe_stem = (
        stem.encode("utf-8")[:byte_budget]
        .decode("utf-8", errors="ignore")
        .rstrip(" .")
    )
    if not safe_stem:
        safe_stem = "attachment"
    return f"{safe_stem}{suffix}"


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
    if parsed.scheme and parsed.scheme not in {"https", "wss"}:
        return None
    if not host:
        return None
    return host, port

# WebSocket transport (NIP-42 authenticated Nostr subscription).
# kind 44100 is Buzz's channel-membership event — used for live DM discovery.
_WS_AUTH_TIMEOUT = 20.0
# Last-resort bound on how long the read loop may wait for a frame. The
# library keepalive (ping_interval/ping_timeout below) should catch a dead
# relay first, but a relay-side close the transport never surfaces (observed
# as a CLOSE_WAIT socket with the loop parked on recv, #98097) leaves the
# gateway "connected" while inbound stops; this timeout forces the normal
# reconnect path instead.
_WS_READ_IDLE_TIMEOUT = 300.0
_WS_MAX_MESSAGE_BYTES = 2_000_000
_WS_MEMBERSHIP_KIND = 44100
_WS_MEMBERSHIP_SUB_ID = "hermes-buzz-membership"

# Where to look for a credentials JSON (keys: nsec / private_key_hex) when
# BUZZ_PRIVATE_KEY is not set.  Module-level so tests can point it at a tmpdir.
_DEFAULT_CREDENTIALS_DIR = Path("~/.config/buzz").expanduser()

# Buzz-hosted Blossom media is private to the community. Inbound messages
# carry media as markdown or bare relay URLs, so the adapter must authenticate
# and localise those references before the gateway hands them to vision.
_MEDIA_URL_PATTERN = (
    r"https?://[^\s<>\[\]()]+/media/"
    r"[0-9a-f]{64}(?:\.[a-z0-9]{1,10})?(?:\?[^\s<>\[\]()]*)?"
)
_MARKDOWN_MEDIA_RE = re.compile(
    rf"!\[(?P<alt>[^\]]*)\]\(\s*(?P<url>{_MEDIA_URL_PATTERN})"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)",
    re.IGNORECASE,
)
_BARE_MEDIA_RE = re.compile(_MEDIA_URL_PATTERN, re.IGNORECASE)
_MEDIA_PATH_RE = re.compile(
    r"^/media/(?P<sha>[0-9a-f]{64})(?P<ext>\.[a-z0-9]{1,10})?/?$",
    re.IGNORECASE,
)


def _effective_port(parsed) -> Optional[int]:
    try:
        if parsed.port is not None:
            return parsed.port
    except ValueError:
        return None
    if parsed.scheme in ("https", "wss"):
        return 443
    if parsed.scheme in ("http", "ws"):
        return 80
    return None


def _is_relay_media_url(url: str, relay_url: str) -> bool:
    """Return whether *url* is a Buzz media object on the configured relay."""
    candidate = urlsplit(url)
    relay = urlsplit(relay_url)
    if candidate.scheme not in ("http", "https"):
        return False
    if not candidate.hostname or not relay.hostname:
        return False
    if candidate.hostname.lower() != relay.hostname.lower():
        return False
    if _effective_port(candidate) != _effective_port(relay):
        return False
    return bool(_MEDIA_PATH_RE.fullmatch(candidate.path))


def _find_relay_media_refs(
    text: str, relay_url: str
) -> Tuple[List[str], List[Tuple[int, int, str]]]:
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
        if any(
            start <= match.start() and match.end() <= end
            for start, end in markdown_spans
        ):
            continue
        url = match.group(0)
        if not _is_relay_media_url(url, relay_url):
            continue
        replacements.append((*match.span(), ""))
        if url not in urls:
            urls.append(url)

    return urls, replacements


def _replace_media_refs(text: str, replacements: List[Tuple[int, int, str]]) -> str:
    cleaned = text
    for start, end, replacement in sorted(replacements, reverse=True):
        cleaned = f"{cleaned[:start]}{replacement}{cleaned[end:]}"
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _load_nostr_auth():
    """Import the sibling nostr_auth module in a loader-agnostic way.

    The adapter is imported both as a package module
    (``plugins.platforms.buzz.adapter``) and as a bare single-file module by
    the test plugin loader, where relative imports have no parent package.
    """
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


# ---------------------------------------------------------------------------
# bech32 (BIP-173) helpers — used to convert between npub and hex pubkeys so
# mention detection and allow-lists accept either form.  Pure stdlib.
# ---------------------------------------------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values: List[int]) -> int:
    generator = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> List[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def _convertbits(data, frombits: int, tobits: int, pad: bool = True) -> Optional[List[int]]:
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def hex_to_npub(pubkey_hex: str) -> Optional[str]:
    """Encode a 64-char hex pubkey as an ``npub1…`` bech32 string."""
    try:
        raw = bytes.fromhex(pubkey_hex)
    except ValueError:
        return None
    if len(raw) != 32:
        return None
    data = _convertbits(raw, 8, 5)
    if data is None:
        return None
    values = _bech32_hrp_expand("npub") + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return "npub1" + "".join(_BECH32_CHARSET[d] for d in data + checksum)


def npub_to_hex(npub: str) -> Optional[str]:
    """Decode an ``npub1…`` bech32 string to a 64-char hex pubkey."""
    npub = npub.strip().lower()
    if not npub.startswith("npub1"):
        return None
    data_part = npub[len("npub1"):]
    try:
        data = [_BECH32_CHARSET.index(c) for c in data_part]
    except ValueError:
        return None
    if _bech32_polymod(_bech32_hrp_expand("npub") + data) != 1:
        return None
    decoded = _convertbits(data[:-6], 5, 8, pad=False)
    if decoded is None or len(decoded) != 32:
        return None
    return bytes(decoded).hex()


def _normalize_user_ref(ref: str) -> Optional[str]:
    """Normalize a user reference (hex pubkey or npub) to lowercase hex."""
    ref = (ref or "").strip().lower()
    if not ref:
        return None
    if ref.startswith("npub1"):
        return npub_to_hex(ref)
    if re.fullmatch(r"[0-9a-f]{64}", ref):
        return ref
    return None


# ---------------------------------------------------------------------------
# buzz-cli invocation helpers
# ---------------------------------------------------------------------------

def _resolve_cli_path(configured: str = "") -> str:
    """Resolve the buzz CLI binary path portably.

    Order: explicit config value → ``buzz`` on PATH → ``~/bin/buzz``.
    Returns "" when nothing is found so callers can raise a config error.
    """
    if configured:
        p = Path(configured).expanduser()
        return str(p) if p.is_file() else ""
    found = shutil.which("buzz")
    if found:
        return found
    fallback = Path.home() / "bin" / "buzz"
    return str(fallback) if fallback.is_file() else ""


def _credentials_candidates(extra: Optional[dict] = None) -> List[Path]:
    # Scope-aware read (#98738/#95216): inside a secondary profile scope the
    # scope is authoritative (a miss falls to the profile's own config extra,
    # never the default profile's os.environ); unscoped reads keep env
    # precedence plus the external-secret rung.
    configured = str(_get_scoped_secret("BUZZ_CREDENTIALS_FILE", "") or "").strip() or str(
        (extra or {}).get("credentials_file", "") or ""
    ).strip()
    if configured:
        return [Path(configured).expanduser()]
    if _is_multiplex_active():
        return []
    try:
        return sorted(_DEFAULT_CREDENTIALS_DIR.glob("*credentials*.json"))
    except OSError:
        return []


def _resolve_credentials_data(extra: Optional[dict] = None) -> dict:
    """Load the first credential record containing a private key."""
    for path in _credentials_candidates(extra):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if any(isinstance(data.get(field), str) and data[field].strip() for field in ("nsec", "private_key_hex", "private_key")):
            return data
    return {}


def _resolve_private_key(extra: Optional[dict] = None) -> str:
    """Resolve the Nostr private key: scoped secret first, then credentials JSON.

    NEVER log the return value.
    """
    key = str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip()
    if key:
        return key
    data = _resolve_credentials_data(extra)
    for field in ("nsec", "private_key_hex", "private_key"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_auth_tag(extra: Optional[dict] = None) -> str:
    """Resolve and validate the optional NIP-OA owner-attestation tag."""
    configured = str(_get_scoped_secret("BUZZ_AUTH_TAG", "") or "").strip()
    if configured:
        raw: Any = configured
    else:
        credentials_file = str(_get_scoped_secret("BUZZ_CREDENTIALS_FILE", "") or "").strip() or str(
            (extra or {}).get("credentials_file", "") or ""
        ).strip()
        direct_key = str(_get_scoped_secret("BUZZ_PRIVATE_KEY", "") or "").strip()
        if direct_key and not credentials_file:
            return ""
        data = _resolve_credentials_data(extra)
        if "auth_tag" not in data:
            return ""
        raw = data["auth_tag"]

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Buzz auth tag is not valid JSON") from exc
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or raw[0] != "auth"
        or not all(isinstance(part, str) for part in raw)
    ):
        raise ValueError("Buzz auth tag must be a four-string auth tag")
    return json.dumps(raw, separators=(",", ":"))


async def _exec_buzz(
    cli_path: str,
    args: List[str],
    *,
    relay_url: str,
    private_key: str,
    auth_tag: str = "",
    input_text: Optional[str] = None,
    timeout: float = _CLI_TIMEOUT,
) -> Tuple[int, str, str]:
    """Run the buzz CLI with an argument list (never a shell) and return
    ``(returncode, stdout, stderr)``.

    The private key travels via the subprocess environment only — it never
    appears in argv, so process listings and error logs stay clean.
    """
    env = os.environ.copy()
    env["BUZZ_RELAY_URL"] = relay_url
    env["BUZZ_PRIVATE_KEY"] = private_key
    env.pop("BUZZ_AUTH_TAG", None)
    if auth_tag:
        env["BUZZ_AUTH_TAG"] = auth_tag
    proc = await asyncio.create_subprocess_exec(
        cli_path,
        *args,
        stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_text.encode("utf-8") if input_text is not None else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", json.dumps({"error": "timeout", "message": f"buzz {args[0] if args else ''} timed out after {timeout}s"})
    return (
        proc.returncode if proc.returncode is not None else 4,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


_MAX_CLI_MESSAGE_CHARS = 900


def _bounded_cli_message(message: str, redact_path: Optional[Path] = None) -> str:
    """Keep untrusted CLI detail useful without exposing unbounded output."""
    if redact_path is not None:
        message = message.replace(str(redact_path), redact_path.name)
    if len(message) <= _MAX_CLI_MESSAGE_CHARS:
        return message
    return f"{message[: _MAX_CLI_MESSAGE_CHARS - 3]}..."


def _cli_error_message(
    stderr: str,
    returncode: int,
    *,
    redact_path: Optional[Path] = None,
) -> str:
    """Extract a bounded human-readable message from the CLI error contract."""
    text = (stderr or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            detail = data.get("message")
            category = data.get("error")
            if isinstance(detail, str) and detail.strip():
                label = category.strip() if isinstance(category, str) and category.strip() else "error"
                return _bounded_cli_message(
                    f"{label}: {detail.strip()} (exit {returncode})",
                    redact_path,
                )
    except ValueError:
        pass
    return _bounded_cli_message(
        text or f"buzz CLI failed with exit code {returncode}",
        redact_path,
    )


def _parse_send_receipt(stdout: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate the buzz-cli success receipt and return ``(event_id, error)``."""
    try:
        data = json.loads(stdout or "{}")
    except ValueError:
        return None, "invalid CLI response"
    if not isinstance(data, dict):
        return None, "invalid CLI response"
    if data.get("accepted") is False:
        detail = data.get("message")
        if not isinstance(detail, str) or not detail.strip():
            detail = "message was not accepted"
        return None, _bounded_cli_message(detail.strip())
    if data.get("accepted") is not True:
        return None, "invalid CLI response"
    event_id = data.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        return None, "invalid CLI response"
    return event_id.strip(), None


def _parse_json_list(stdout: str) -> List[dict]:
    """Parse CLI stdout expected to be a JSON array of objects."""
    try:
        data = json.loads(stdout or "[]")
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _event_reply_parent_id(event: dict) -> Optional[str]:
    """Resolve a chat event's direct parent event id (NIP-10 ``e`` tags).

    Prefer a ``reply``-marked tag, then a ``root``-marked tag, else the last
    positional ``e`` tag. Buzz Desktop thread replies typically carry both
    root and reply markers; the reply marker is the direct parent.
    """
    tags = event.get("tags")
    if not isinstance(tags, list):
        return None
    reply_id: Optional[str] = None
    root_id: Optional[str] = None
    last_e: Optional[str] = None
    for tag in tags:
        if not isinstance(tag, (list, tuple)) or len(tag) < 2 or tag[0] != "e":
            continue
        target = str(tag[1] or "").strip()
        if not target:
            continue
        marker = str(tag[3] or "") if len(tag) > 3 else ""
        last_e = target
        if marker == "reply":
            reply_id = target
        elif marker == "root":
            root_id = target
    return reply_id or root_id or last_e


# Cap stored parent content snippets (gateway reply injection also clips).
_EVENT_META_CONTENT_CAP = 500


# ---------------------------------------------------------------------------
# Buzz Adapter
# ---------------------------------------------------------------------------

class BuzzAdapter(BasePlatformAdapter):
    """Poll-based Buzz adapter implementing the BasePlatformAdapter interface.

    Instantiated by the adapter_factory passed to register_platform().
    """

    def __init__(self, config, **kwargs):
        platform = Platform("buzz")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        self._extra = extra

        # Connection settings (env vars override config.yaml; under a
        # secondary multiplex profile scope the profile's extra wins and
        # env — the default profile's bridge output — is not consulted)
        _relay_raw = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
        self.relay_url = (_relay_raw or extra.get("relay_url", "")).strip()
        
        configured_attachment_hosts = extra.get("attachment_hosts", [])
        if isinstance(configured_attachment_hosts, str):
            configured_attachment_hosts = configured_attachment_hosts.split(",")
        configured_origins = (
            _attachment_origin(host)
            for host in configured_attachment_hosts
            if isinstance(host, str)
        )
        self._attachment_origins = {
            origin for origin in configured_origins if origin is not None
        }
        relay_origin = _attachment_origin(self.relay_url)
        if relay_origin is not None:
            self._attachment_origins.add(relay_origin)
        _cli_raw = _scoped_platform_setting("BUZZ_CLI_PATH", extra, "cli_path")
        self.cli_path = _resolve_cli_path(
            str(_cli_raw or "").strip() or str(extra.get("cli_path", "") or "")
        )

        # Channels to watch: env csv > extra list/csv; empty = all joined channels
        raw_channels = _scoped_platform_setting("BUZZ_CHANNELS", extra, "channels")
        if raw_channels is None:
            raw_channels = extra.get("channels", [])
        if isinstance(raw_channels, str):
            raw_channels = raw_channels.split(",")
        self.channels: List[str] = [c.strip() for c in raw_channels if isinstance(c, str) and c.strip()]

        _home_raw = _scoped_platform_setting("BUZZ_HOME_CHANNEL", extra, "home_channel")
        self.home_channel = (_home_raw or str(extra.get("home_channel", "") or "")).strip()

        _pi_raw = _scoped_platform_setting("BUZZ_POLL_INTERVAL", extra, "poll_interval")
        try:
            interval = float(_pi_raw or extra.get("poll_interval", _DEFAULT_POLL_INTERVAL))
        except (TypeError, ValueError):
            interval = _DEFAULT_POLL_INTERVAL
        self.poll_interval = max(_MIN_POLL_INTERVAL, interval)

        # Whether channel messages must @mention the agent to get a response.
        # Defaults to True (respond only when addressed). Set False to make the
        # agent respond to every message in a watched channel. DMs always
        # dispatch regardless. Env (BUZZ_REQUIRE_MENTION) overrides config.yaml.
        _rm_raw = _scoped_platform_setting("BUZZ_REQUIRE_MENTION", extra, "require_mention")
        if _rm_raw is None:
            _rm_cfg = extra.get("require_mention", True)
        else:
            _rm_cfg = _rm_raw
        self.require_mention = str(_rm_cfg).strip().lower() not in ("false", "0", "no", "off")

        # Reply anchoring: "first"/"all" thread the reply onto the parent event
        # id, "off" posts every reply as a normal top-level channel message.
        # Mirrors the Discord/Telegram adapters, which already honor this
        # PlatformConfig field; without it Buzz threaded unconditionally.
        # Env (BUZZ_REPLY_TO_MODE) overrides config.yaml.
        _rtm = (os.getenv("BUZZ_REPLY_TO_MODE") or getattr(config, "reply_to_mode", "first")
                or "first")
        self._reply_to_mode: str = str(_rtm).strip().lower()
        # Slack-convention alias: platforms.buzz.extra.reply_in_thread: false
        # (the key users already know from Slack) opts out of threading the
        # same way reply_to_mode: off does. Env (BUZZ_REPLY_IN_THREAD)
        # overrides config.yaml. See #95842 / #75082.
        _rit_raw = os.getenv("BUZZ_REPLY_IN_THREAD")
        _rit = extra.get("reply_in_thread") if _rit_raw is None else _rit_raw
        if _rit is not None and str(_rit).strip().lower() in ("false", "0", "no", "off"):
            self._reply_to_mode = "off"

        # Inbound transport: "auto" (WebSocket with poll fallback, default),
        # "websocket" (require WS; fail connect when it can't authenticate),
        # or "poll" (CLI polling only). Env (BUZZ_TRANSPORT) overrides
        # config.yaml.
        _transport_raw = _scoped_platform_setting("BUZZ_TRANSPORT", extra, "transport")
        _transport = (
            _transport_raw or str(extra.get("transport", "auto") or "auto")
        ).strip().lower()
        self.transport = _transport if _transport in ("auto", "websocket", "poll") else "auto"

        # Auth: entries may be hex pubkeys or npubs; normalized to hex
        raw_allowed = _scoped_platform_setting("BUZZ_ALLOWED_USERS", extra, "allowed_users")
        if raw_allowed is None:
            raw_allowed = extra.get("allowed_users", [])
        if isinstance(raw_allowed, str):
            raw_allowed = raw_allowed.split(",")
        self._allowed_pubkeys: set = {
            normalized
            for entry in raw_allowed
            if isinstance(entry, str) and (normalized := _normalize_user_ref(entry))
        }

        # Verified local-agent identities may acknowledge explicit tags without
        # gaining prompt/dispatch authority. This keeps the human allow-list
        # intact while providing receipt visibility for agent-authored notes.
        # If a pubkey appears in both sets, allowed_users takes precedence: the
        # normal authorized dispatch path runs and this reaction-only path does not.
        raw_reaction_only = (
            os.getenv("BUZZ_REACTION_ONLY_USERS")
            or extra.get("reaction_only_users", [])
        )
        if isinstance(raw_reaction_only, str):
            raw_reaction_only = raw_reaction_only.split(",")
        self._reaction_only_pubkeys: set = {
            normalized
            for entry in raw_reaction_only
            if isinstance(entry, str) and (normalized := _normalize_user_ref(entry))
        }

        # Secret — resolved lazily (never at import/registration time and
        # never logged).  connect() re-resolves it to fail fast with a clear
        # error when it is missing.
        self._private_key: str = ""
        self._auth_tag: str = ""

        # Identity — filled in by connect() from ``buzz users get``
        self._self_pubkey: str = ""
        self._self_npub: str = ""
        self._display_name: str = ""

        # Runtime state
        self._poll_task: Optional[asyncio.Task] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_ready: Optional[asyncio.Event] = None
        self._ws_active = False  # True while the WS loop owns inbound delivery
        self._membership_since = 0
        self._lock_key: Optional[str] = None
        # channel_id -> {
        #   "chat_type", "last_ts",
        #   "seen": OrderedDict[event_id, None],
        #   "event_meta": OrderedDict[event_id, (author_pubkey, content_snippet)],
        # }
        # event_meta backs NIP-10 reply-parent resolution for require_mention
        # (thread replies to our own messages count as addressed — #75826).
        # Channels the relay has permanently rejected (e.g. "restricted: not a
        # channel member").  Persists across reconnects so we never re-subscribe
        # to a channel the relay won't serve us.
        self._restricted_channels: set = set()
        self._channel_state: Dict[str, dict] = {}
        # Cursors read back from disk at connect() and consumed by the first
        # seed of each channel; empty on a first-ever run.
        self._restored_cursors: Dict[str, dict] = {}
        self._channel_names: Dict[str, str] = {}
        # channel_id -> raw ``channels list`` entry; drives DM-vs-channel
        # classification (see _may_reclassify_as_dm).
        self._channel_meta: Dict[str, dict] = {}
        self._user_names: Dict[str, str] = {}
        self._poll_count = 0
        # inbound event_id -> thread root event id, or None when that message
        # was itself top-level.  Lets send() mirror the user's own threading
        # instead of opening a new thread under every reply (see _thread_root).
        self._thread_roots: "OrderedDict[str, Optional[str]]" = OrderedDict()

    @property
    def name(self) -> str:
        return "Buzz"

    @staticmethod
    def normalize_user_id(user_id: str) -> Optional[str]:
        """Normalize a Buzz user reference (hex pubkey or npub) to hex.

        Optional hook consumed by ``gateway/authz_mixin`` when matching the
        profile allowlist carried in ``config.extra.allowed_users`` (#98738):
        entries may be npubs while inbound ``user_id`` is always the hex
        pubkey, so a plain string compare would deny listed users.
        """
        return _normalize_user_ref(user_id)

    # ── buzz-cli plumbing ─────────────────────────────────────────────────

    async def _run_cli(self, args: List[str], *, input_text: Optional[str] = None) -> Tuple[int, str, str]:
        if not self._private_key:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        return await _exec_buzz(
            self.cli_path,
            args,
            relay_url=self.relay_url,
            private_key=self._private_key,
            auth_tag=self._auth_tag,
            input_text=input_text,
        )

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Verify relay credentials, seed high-water marks, start polling."""
        if not self.relay_url:
            logger.error("Buzz: relay URL must be configured")
            self._set_fatal_error("config_missing", "BUZZ_RELAY_URL must be set", retryable=False)
            return False
        if not self.cli_path:
            logger.error("Buzz: buzz CLI binary not found (set BUZZ_CLI_PATH or put 'buzz' on PATH)")
            self._set_fatal_error("cli_missing", "buzz CLI binary not found", retryable=False)
            return False
        try:
            self._private_key = _resolve_private_key(self._extra)
            self._auth_tag = _resolve_auth_tag(self._extra)
        except ValueError as exc:
            logger.error("Buzz: invalid owner-auth configuration — %s", exc)
            self._set_fatal_error("config_invalid", str(exc), retryable=False)
            return False
        if not self._private_key:
            logger.error("Buzz: no private key (set BUZZ_PRIVATE_KEY or a credentials file)")
            self._set_fatal_error("config_missing", "BUZZ_PRIVATE_KEY must be set", retryable=False)
            return False

        # Learn our own identity: pubkey drives self-echo suppression and
        # display name drives channel mention gating.
        code, out, err = await self._run_cli(["users", "get"])
        if code != 0:
            message = _cli_error_message(err, code)
            logger.error("Buzz: failed to fetch own profile from %s — %s", self.relay_url, message)
            self._set_fatal_error("connect_failed", message, retryable=code == 2)
            return False
        profiles = _parse_json_list(out)
        if not profiles or not profiles[0].get("pubkey"):
            logger.error("Buzz: 'users get' returned no profile — is the key a member of this community?")
            self._set_fatal_error("connect_failed", "buzz users get returned no profile", retryable=True)
            return False
        self._self_pubkey = str(profiles[0]["pubkey"]).lower()
        self._display_name = str(profiles[0].get("display_name") or "").strip()
        self._self_npub = hex_to_npub(self._self_pubkey) or ""

        # Prevent two profiles from driving the same Buzz identity on the
        # same relay (duplicate replies, split de-dupe state). Mirrors the
        # IRC adapter's scoped-lock pattern.
        try:
            from gateway.status import acquire_scoped_lock

            lock_key = f"{self.relay_url}:{self._self_pubkey}"
            if not acquire_scoped_lock("buzz", lock_key):
                logger.error(
                    "Buzz: identity %s… on %s already in use by another profile",
                    self._self_pubkey[:8],
                    self.relay_url,
                )
                self._set_fatal_error(
                    "lock_conflict", "Buzz identity in use by another profile", retryable=False
                )
                return False
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module not available (e.g. tests)

        # Map channel ids to names and pick the watch set.
        code, out, err = await self._run_cli(["channels", "list"])
        if code != 0:
            message = _cli_error_message(err, code)
            logger.error("Buzz: failed to list channels — %s", message)
            self._set_fatal_error("connect_failed", message, retryable=code == 2)
            return False
        listed = _parse_json_list(out)
        self._channel_names = {
            str(ch.get("channel_id")): str(ch.get("name") or ch.get("channel_id"))
            for ch in listed
            if ch.get("channel_id")
        }
        for ch in listed:
            if ch.get("channel_id"):
                self._channel_meta[str(ch["channel_id"])] = ch
        watch = self.channels or list(self._channel_names)
        if not watch:
            logger.error("Buzz: no channels to watch (configure BUZZ_CHANNELS or join a channel)")
            self._set_fatal_error("config_missing", "no Buzz channels to watch", retryable=False)
            return False

        # Seed high-water marks from the newest events so a (re)start never
        # replays channel history into the agent — except where a previous run
        # left a cursor, which is restored instead so the events that landed
        # while we were down still dispatch (#90464).  Skip any channel the
        # relay has permanently rejected in a previous session (e.g.
        # "restricted: not a channel member") so we don't reconnect-loop on
        # them.
        self._load_cursors()
        for channel_id in watch:
            if channel_id in self._restricted_channels:
                logger.debug("Buzz: skipping restricted channel %s (relay rejected subscription)", channel_id)
                continue
            await self._seed_channel(channel_id, chat_type="group")
        await self._discover_dms(seed=True)
        self._save_cursors()

        # Inbound transport: prefer the NIP-42-authenticated WebSocket
        # subscription (push, near-zero latency); fall back to CLI polling
        # when the WS can't be established (transport="auto") or when the
        # user pinned transport="poll".
        transport_used = "poll"
        if self.transport in ("auto", "websocket"):
            if await self._start_websocket():
                transport_used = "websocket"
            elif self.transport == "websocket":
                self._set_fatal_error(
                    "ws_auth_failed",
                    "Buzz WebSocket transport did not authenticate (transport=websocket)",
                    retryable=True,
                )
                await self.disconnect()
                return False
        if transport_used == "poll":
            self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info(
            "Buzz: connected to %s as %s, watching %d channel(s) via %s%s",
            self.relay_url,
            self._display_name or self._self_npub[:16],
            len(self._channel_state),
            transport_used,
            "" if transport_used == "websocket" else f", poll interval {self.poll_interval:.1f}s",
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
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
        self._ws_active = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        self._ws_task = None
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._poll_task = None
        self._channel_state = {}
        self._poll_count = 0

    # ── Sending ───────────────────────────────────────────────────────────

    async def _channel_member_pubkeys(self, chat_id: str) -> List[str]:
        """Candidate pubkeys for mention resolution, membership-accurate.

        Primary source is ``channels members`` — the relay's membership
        contract — because a ``--mention`` for a non-member makes the CLI
        reject the whole publish.  CLIs without that subcommand fall back
        to harvesting recent channel traffic (authors plus prior mention
        tags), which can over-approximate; ``send()`` recovers from any
        resulting non-member mention by retrying without mention flags.

        The member list is cached per channel for ``_MEMBER_CACHE_TTL``
        seconds so a chatty agent doesn't pay a CLI round-trip on every
        publish; membership drift inside the TTL window is covered by the
        same ``send()`` recovery retry.
        """
        cache: Dict[str, Tuple[float, List[str]]] = getattr(
            self, "_member_cache", {}
        )
        self._member_cache = cache
        cached = cache.get(str(chat_id))
        if cached is not None and (time.monotonic() - cached[0]) < _MEMBER_CACHE_TTL:
            return list(cached[1])
        code, out, _err = await self._run_cli(
            ["channels", "members", "--channel", str(chat_id)]
        )
        if code == 0:
            pks: List[str] = []
            try:
                rows = json.loads(out or "[]")
            except ValueError:
                rows = []
            for row in rows:
                pk = row.get("pubkey") if isinstance(row, dict) else row
                pk = str(pk or "").lower()
                if pk and pk not in pks:
                    pks.append(pk)
            if pks:
                cache[str(chat_id)] = (time.monotonic(), list(pks))
                return pks
        candidates: List[str] = []
        code, out, _err = await self._run_cli(
            ["messages", "get", "--channel", str(chat_id), "--limit", "50"]
        )
        if code == 0:
            try:
                for msg in json.loads(out or "[]"):
                    pk = str(msg.get("pubkey") or "").lower()
                    if pk and pk not in candidates:
                        candidates.append(pk)
                    for t in msg.get("tags") or []:
                        if isinstance(t, list) and len(t) > 1 and t[0] == "p":
                            tpk = str(t[1]).lower()
                            if tpk and tpk not in candidates:
                                candidates.append(tpk)
            except ValueError:
                pass
        if candidates:
            cache[str(chat_id)] = (time.monotonic(), list(candidates))
        return candidates

    async def _profile_display_name(self, pubkey: str) -> str:
        """Display name for *pubkey* via ``users get --pubkey``, cached.

        Bare ``users get`` may return only our own profile
        (relay-dependent), so lookups are per-pubkey.  Entries expire after
        ``_PROFILE_NAME_TTL`` seconds so a renamed member resolves under
        their new display name without a process restart.
        """
        cache: Dict[str, Tuple[float, str]] = getattr(
            self, "_profile_name_cache", {}
        )
        self._profile_name_cache = cache
        cached = cache.get(pubkey)
        if cached is not None and (time.monotonic() - cached[0]) < _PROFILE_NAME_TTL:
            return cached[1]
        name = ""
        code, out, _err = await self._run_cli(["users", "get", "--pubkey", pubkey])
        if code == 0:
            try:
                profiles = json.loads(out or "[]")
            except ValueError:
                profiles = []
            if profiles and isinstance(profiles[0], dict):
                p0 = profiles[0]
                name = str(p0.get("display_name") or p0.get("name") or "").strip()
                if not name and p0.get("content"):
                    try:
                        prof = json.loads(p0["content"])
                        name = str(
                            prof.get("display_name") or prof.get("name") or ""
                        ).strip()
                    except ValueError:
                        pass
        cache[pubkey] = (time.monotonic(), name)
        return name

    async def _mention_pubkeys_for(self, chat_id: str, content: str) -> List[str]:
        """Resolve ``@Name`` references in *content* to member pubkeys.

        The CLI hard-fails a publish when any @token fails to resolve to a
        current member, and LLM prose is full of @-shaped tokens — including
        real mentions with trailing punctuation ("@Riley!!") the CLI's own
        parser rejects.  Passing explicit ``--mention`` pubkeys for every
        member name we find keeps genuine mentions notifying (p-tags intact)
        while downgrading everything unresolvable to presentation-only text.

        Matching is mention-token semantics, not substring, bounded on both
        sides with Unicode-aware word classes: the ``@`` must start a token
        ("email@Fizz", "x@Fizz", "@@Fizz", and "山田@Fizz" do NOT wake
        Fizz) and the name must be followed by a non-word character or
        end-of-text ("@Riley!!" tags Riley; "@FizzBuzz" does NOT tag a
        member named Fizz).  Longer names match first and consume their
        span, so "@Hermes Matt" prefers the member "Hermes Matt" over a
        member "Hermes".

        Duplicate display names are ambiguous: the span is consumed but no
        one is tagged (presentation-only), mirroring how Buzz treats
        ambiguous names — never pick an arbitrary member.
        """
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
            by_name.setdefault(key, [])
            if pk not in by_name[key]:
                by_name[key].append(pk)
            display.setdefault(key, name)
        found: List[str] = []
        text = content
        for key in sorted(by_name, key=len, reverse=True):
            pattern = re.compile(
                r"(?<![\w@])@" + re.escape(display[key]) + r"(?!\w)",
                re.IGNORECASE,
            )
            if pattern.search(text):
                pks = by_name[key]
                if len(pks) == 1 and pks[0] not in found:
                    found.append(pks[0])
                # Consume the span either way: a shorter member name that is
                # a prefix of this one must not double-match, and an
                # ambiguous name must stay presentation-only rather than
                # falling through to a partial match.
                text = pattern.sub("\x00", text)
        return found

    async def _run_message_send(
        self,
        args: List[str],
        content: str,
        mention_pubkeys: Optional[List[str]] = None,
    ):
        """Run one send with bounded mention-failure recovery.

        Ladder (each rung fires at most once):

        1. publish with explicit ``--mention`` pubkeys resolved from the
           content (#83414) so genuine member mentions carry p-tags and
           mention-subscribed agents actually wake;
        2. if the CLI rejects because a resolved pubkey is no longer a
           member (membership drift), retry without the explicit mentions —
           deliver the message rather than lose it;
        3. if the CLI's preflight rejects an unresolvable presentation
           ``@token`` in prose, escape exactly that token with an invisible
           separator and retry (#82646 / #78797);
        4. if the error persists and we know our own pubkey, retry once with
           ``--mention <self>`` — supplying any explicit identity downgrades
           unresolvable @names to presentation-only text (#83414); the echo
           de-dupe already suppresses self-notification.
        """
        mention_args: List[str] = []
        for pk in mention_pubkeys or []:
            mention_args += ["--mention", pk]
        code, out, err = await self._run_cli(args + mention_args, input_text=content)
        if code == 0:
            return code, out, err
        if mention_args and "not channel members" in (err or ""):
            # Membership drifted between resolution and publish (or the
            # fallback candidate source over-approximated): never let a
            # stale mention kill the message.
            code, out, err = await self._run_cli(args, input_text=content)
            if code == 0:
                return code, out, err
        escaped = _escape_unresolved_presentation_mention(content, err)
        if escaped is not None:
            logger.info(
                "Buzz: retrying message after unresolved presentation-mention preflight"
            )
            code, out, err = await self._run_cli(args, input_text=escaped)
            if code == 0:
                return code, out, err
        if (
            code != 0
            and "does not match a current channel member" in (err or "")
            and getattr(self, "_self_pubkey", None)
        ):
            code, out, err = await self._run_cli(
                args + ["--mention", self._self_pubkey], input_text=content
            )
        return code, out, err

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content:
            return SendResult(success=False, error="Empty message")
        args = ["messages", "send", "--channel", str(chat_id), "--content", "-"]
        # Prefer the stable thread anchor from metadata.thread_id (Slack-style),
        # then metadata.reply_to_message_id (gateway stream consumer /
        # progress sends), then the explicit reply_to argument.  Without
        # reply_to_message_id, interim commentary posts flat in the channel.
        meta = metadata or {}
        reply_target = self._resolve_reply_anchor(
            meta.get("thread_id") or meta.get("reply_to_message_id") or reply_to
        )
        if reply_target and self._reply_to_mode != "off":
            args += ["--reply-to", str(reply_target)]
        mention_pubkeys = await self._mention_pubkeys_for(chat_id, content)
        code, out, err = await self._run_message_send(args, content, mention_pubkeys)
        if code != 0:
            return SendResult(
                success=False,
                error=_cli_error_message(err, code),
                retryable=code == 2,
            )
        event_id, receipt_error = _parse_send_receipt(out)
        if receipt_error:
            return SendResult(success=False, error=receipt_error)
        assert event_id is not None
        # Belt-and-braces echo suppression: the poll loop already skips our own
        # pubkey, but marking the verified id seen makes de-dupe explicit.
        # Also record event_meta so a thread reply to this send matches even
        # if the WS/poll echo never arrives (#75826).
        self._mark_seen(str(chat_id), event_id)
        self._remember_event_meta(
            str(chat_id), event_id, self._self_pubkey, content
        )
        return SendResult(success=True, message_id=event_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Buzz has no typing indicator API — no-op."""
        pass

    async def send_reaction(self, chat_id: str, message_id: str, emoji: str) -> bool:
        """Add a reaction to a message via buzz-cli.

        Returns True on success, False on failure. Errors are logged but not
        raised — reactions are best-effort and should never block the main
        message flow.
        """
        if not self.cli_path or not emoji or not message_id:
            return False
        # buzz-cli: `reactions add --event <64-char hex event id> --emoji <e>`.
        # The event id IS the message_id we recorded on dispatch; channel is
        # not a parameter to this subcommand.
        args = [
            "reactions", "add",
            "--event", str(message_id),
            "--emoji", emoji,
        ]
        code, _out, err = await self._run_cli(args)
        if code != 0:
            logger.debug(
                "Buzz: reaction add failed for message %s in %s — %s",
                message_id[:12], chat_id, _cli_error_message(err, code),
            )
            return False
        return True

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message.

        Implementing this is what lets the gateway stream a reply on Buzz: the
        stream consumer sends a first partial message and then re-edits that one
        message as tokens arrive.  Without it the adapter inherits the base
        stub, which returns ``success=False``, and the whole answer is delivered
        in one block when the turn finishes.

        ``buzz-cli`` reports a NEW event id for the edit itself, but the edit
        TARGET stays the original id, and the stream consumer holds a single
        ``message_id`` across the whole stream.  So this returns the id it was
        given, not the one the CLI reports; returning the CLI's id would make
        every edit after the first address a message that was never sent.

        ``finalize`` is a no-op here.  Buzz edits carry no lifecycle state, the
        same as Telegram, Slack and Discord.
        """
        if not message_id:
            return SendResult(success=False, error="Buzz edit needs a message id")
        if not content:
            return SendResult(success=False, error="Empty message")
        args = ["messages", "edit", "--event", str(message_id), "--content", "-"]
        code, out, err = await self._run_cli(args, input_text=content)
        if code != 0:
            return SendResult(
                success=False,
                error=_cli_error_message(err, code),
                retryable=code == 2,
            )
        try:
            data = json.loads(out or "{}")
        except ValueError:
            data = {}
        edit_event_id = data.get("event_id")
        if edit_event_id:
            # The edit is itself an event on the relay and comes back on our own
            # subscription; mark it seen so the de-dupe does not treat our own
            # edit as inbound traffic.
            self._mark_seen(str(chat_id), str(edit_event_id))
        return SendResult(
            success=bool(data.get("accepted", True)),
            message_id=str(message_id),
            raw_response=data,
        )

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a previously sent message.

        Used by the stream consumer's fresh-final cleanup path, which replaces a
        long-lived preview with a completed reply rather than editing in place.
        """
        if not message_id:
            return False
        code, out, _err = await self._run_cli(
            ["messages", "delete", "--event", str(message_id)]
        )
        if code != 0:
            return False
        try:
            data = json.loads(out or "{}")
        except ValueError:
            return True
        event_id = data.get("event_id")
        if event_id:
            self._mark_seen(str(chat_id), str(event_id))
        return bool(data.get("accepted", True))

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload one local file through the Buzz CLI and verify its receipt."""
        local = Path(file_path).expanduser()
        if not local.is_file():
            return SendResult(success=False, error="Media file not found")
        args = [
            "messages", "send",
            "--channel", str(chat_id),
            "--file", str(local),
            "--content", "-",
        ]
        reply_target = self._resolve_reply_anchor(
            (metadata or {}).get("thread_id") or reply_to
        )
        if reply_target and self._reply_to_mode != "off":
            args += ["--reply-to", str(reply_target)]
        code, out, err = await self._run_message_send(args, caption or "")
        if code != 0:
            return SendResult(
                success=False,
                error=_cli_error_message(err, code, redact_path=local),
                retryable=code == 2,
            )
        event_id, receipt_error = _parse_send_receipt(out)
        if receipt_error:
            return SendResult(success=False, error=receipt_error)
        assert event_id is not None
        self._mark_seen(str(chat_id), event_id)
        return SendResult(success=True, message_id=event_id)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image: local files upload via --file, URLs go as a link."""
        local = Path(image_url).expanduser() if not image_url.startswith(("http://", "https://")) else None
        if local is not None and local.is_file():
            return await self._send_file_attachment(
                chat_id,
                local,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
                probe=False,
            )
        # Markdown renders in Buzz, so a URL arrives as a clickable image link.
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def _send_file_attachment(
        self,
        chat_id: str,
        file_path: Path,
        *,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        probe: bool = True,
    ) -> SendResult:
        """Upload a local file and publish it as a native Buzz attachment.

        ``probe=False`` skips the existence re-check when the caller already
        verified the file — a second probe could race into a false
        "not found" if the file disappears between checks (#74999).
        """
        local = Path(file_path).expanduser()
        if probe and not local.is_file():
            # Never leak host filesystem paths into chat-visible errors.
            return SendResult(success=False, error="Media file not found")
        args = [
            "messages", "send",
            "--channel", str(chat_id),
            "--file", str(local),
            "--content", "-",
        ]
        reply_target = self._resolve_reply_anchor(
            (metadata or {}).get("thread_id") or reply_to
        )
        if reply_target and self._reply_to_mode != "off":
            args += ["--reply-to", str(reply_target)]
        code, out, err = await self._run_message_send(args, caption or "")
        if code != 0:
            return SendResult(
                success=False,
                error=_cli_error_message(err, code, redact_path=local),
                retryable=code == 2,
            )
        event_id, receipt_error = _parse_send_receipt(out)
        if receipt_error:
            return SendResult(success=False, error=receipt_error)
        assert event_id is not None
        self._mark_seen(str(chat_id), event_id)
        return SendResult(success=True, message_id=event_id)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local image through Buzz's native ``--file`` path.

        Missing or non-file paths retain the Base fallback so host
        filesystem paths are never echoed into chat (#74999).
        """
        local = Path(image_path).expanduser()
        if local.is_file():
            return await self._send_file_attachment(
                chat_id,
                local,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
                probe=False,
            )
        return await super().send_image_file(
            chat_id=chat_id,
            image_path=image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            **kwargs,
        )

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
        """Upload a local document through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id,
            Path(file_path),
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local video through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id,
            Path(video_path),
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Upload a local audio file through Buzz's native ``--file`` path."""
        return await self._send_file_attachment(
            chat_id,
            Path(audio_path),
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        state = self._channel_state.get(chat_id)
        chat_type = state["chat_type"] if state else "group"
        name = self._channel_names.get(chat_id)
        if name is None and self.cli_path:
            code, out, _err = await self._run_cli(["channels", "get", "--channel", chat_id])
            if code == 0:
                try:
                    data = json.loads(out or "{}")
                    if isinstance(data, dict) and data.get("name"):
                        name = str(data["name"])
                        self._channel_names[chat_id] = name
                except ValueError:
                    pass
        return {"name": name or chat_id, "type": chat_type, "chat_id": chat_id}

    # ── Inbound: WebSocket transport (NIP-42 authenticated) ──────────────
    #
    # Push transport contributed in PR #73636 by @ScaleLeanChris, adapted to
    # dispatch through the same _handle_event() machinery as the poll loop so
    # de-dupe, mention gating, DM latching, and the allow-list behave
    # identically on both transports.

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
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("Buzz: WebSocket did not authenticate in time")
            self._ws_active = False
            if self._ws_task and not self._ws_task.done():
                self._ws_task.cancel()
                try:
                    await self._ws_task
                except asyncio.CancelledError:
                    pass
            self._ws_task = None
            return False
        return True

    async def _authenticate_websocket(self, websocket) -> None:
        """NIP-42: wait for the relay's AUTH challenge, answer with a signed
        kind-22242 event (plus the optional NIP-OA owner-attestation tag from
        BUZZ_AUTH_TAG), and wait for the OK acknowledgment."""
        build_auth_event = _load_nostr_auth().build_auth_event

        raw = await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT)
        message = json.loads(raw)
        if not isinstance(message, list) or len(message) < 2 or message[0] != "AUTH":
            raise ConnectionError("Buzz relay did not send a NIP-42 AUTH challenge")
        # BUZZ_AUTH_TAG is per-identity NIP-OA owner attestation, so it must
        # resolve through the profile secret scope (#98738): inside a scoped
        # multiplex profile a missing tag fails closed to "" instead of
        # attaching the default profile's tag from os.environ, while
        # single-profile and unscoped default-profile reads keep the legacy
        # env behavior. connect() populates ``self._auth_tag`` via
        # ``_resolve_auth_tag`` (scope-aware read + credentials-file
        # fallback, #79514); resolve lazily here as well so a re-auth on a
        # bare adapter stays scope-correct.
        auth_tag = getattr(self, "_auth_tag", "") or ""
        if not auth_tag:
            try:
                auth_tag = _resolve_auth_tag(getattr(self, "_extra", None))
            except ValueError:
                auth_tag = ""
        event = build_auth_event(
            private_key=self._private_key,
            challenge=str(message[1]),
            relay_url=self._websocket_url(),
            auth_tag_json=auth_tag,
        )
        await websocket.send(json.dumps(["AUTH", event], separators=(",", ":")))
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=_WS_AUTH_TIMEOUT)
            response = json.loads(raw)
            if not isinstance(response, list) or not response:
                continue
            if response[0] == "OK" and len(response) >= 4 and response[1] == event["id"]:
                if response[2] is True:
                    return
                raise ConnectionError(f"Buzz WebSocket AUTH rejected: {response[3]}")
            if response[0] in ("NOTICE", "CLOSED"):
                detail = response[-1] if len(response) > 1 else "authentication failed"
                raise ConnectionError(f"Buzz WebSocket AUTH failed: {detail}")

    async def _send_channel_subscription(self, websocket, subscription_id: str, channel_id: str) -> None:
        state = self._channel_state.get(channel_id) or {}
        last_ts = int(state.get("last_ts") or 0)
        if last_ts:
            # Resume from the channel's high-water mark (same-second overlap
            # is de-duped by event id).
            request_filter = {"kinds": sorted(_DISPATCH_KINDS), "#h": [channel_id], "since": max(last_ts - 1, 0)}
        else:
            # A conversation adopted mid-run with no high-water mark is fresh:
            # its history IS the conversation, so subscribe from the beginning
            # instead of `since ≈ now` — otherwise the message that *created*
            # the conversation (created_at fractionally before this
            # subscription) is silently dropped (#78429). `limit` bounds the
            # replay to the same window the poll transport fetches; the seed
            # path gives real channels a non-zero last_ts, so they never take
            # this branch.
            request_filter = {"kinds": sorted(_DISPATCH_KINDS), "#h": [channel_id], "limit": _FETCH_LIMIT}
        request = [
            "REQ",
            subscription_id,
            request_filter,
        ]
        await websocket.send(json.dumps(request, separators=(",", ":")))

    async def _subscribe_websocket(self, websocket) -> Dict[str, Optional[str]]:
        """Subscribe to every watched conversation plus membership events
        (kind 44100 p-tagged to us) for live DM discovery."""
        subscriptions: Dict[str, Optional[str]] = {}
        for index, channel_id in enumerate(list(self._channel_state)):
            if channel_id in self._restricted_channels:
                continue
            subscription_id = f"hermes-buzz-{index}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
        if self._self_pubkey:
            request = [
                "REQ",
                _WS_MEMBERSHIP_SUB_ID,
                {
                    "kinds": [_WS_MEMBERSHIP_KIND],
                    "#p": [self._self_pubkey],
                    "since": max(self._membership_since - 1, 0),
                },
            ]
            await websocket.send(json.dumps(request, separators=(",", ":")))
            subscriptions[_WS_MEMBERSHIP_SUB_ID] = None
        return subscriptions

    async def _subscribe_new_conversations(
        self, websocket, subscriptions: Dict[str, Optional[str]], before: set
    ) -> None:
        """Subscribe to every conversation adopted since *before* was taken."""
        for channel_id in list(self._channel_state):
            if channel_id in before:
                continue
            subscription_id = f"hermes-buzz-dm-{len(subscriptions)}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
            logger.info("Buzz: subscribed to new conversation %s", channel_id)

    async def _handle_membership_event(self, websocket, subscriptions: Dict[str, Optional[str]], event: dict) -> None:
        """A membership event p-tagged to us: rediscover conversations and
        subscribe to any new ones (fresh DMs dispatch from their beginning)."""
        self._membership_since = max(self._membership_since, int(event.get("created_at") or 0))
        before = set(self._channel_state)
        await self._discover_dms(seed=False)
        await self._subscribe_new_conversations(websocket, subscriptions, before)

    async def _ws_discovery_loop(self, websocket, subscriptions: Dict[str, Optional[str]]) -> None:
        """Periodic conversation discovery for the WebSocket transport.

        The kind-44100 membership subscription is the fast path, but relays do
        not guarantee a membership event for every conversation that
        materializes mid-session (#93557) — some emit none at all for new
        DM-shaped conversations. The poll transport papers over this by
        re-running discovery every ``_DM_DISCOVERY_EVERY`` sweeps; this loop
        gives the WS transport the same guarantee on the same cadence.
        Failures are logged and retried next tick; the read loop is the sole
        owner of connection health.
        """
        interval = max(self.poll_interval * _DM_DISCOVERY_EVERY, _MIN_POLL_INTERVAL)
        while True:
            await asyncio.sleep(interval)
            try:
                before = set(self._channel_state)
                await self._discover_dms(seed=False)
                await self._subscribe_new_conversations(websocket, subscriptions, before)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Buzz: WebSocket discovery sweep failed", exc_info=True)

    async def _websocket_loop(self) -> None:
        """Persistent authenticated subscription with bounded reconnect
        backoff. Events route through _handle_event() — identical semantics
        to the poll loop. On reconnect, per-channel `since` filters resume
        from the last observed timestamps (same-second overlap de-duped by
        event id)."""
        import websockets

        backoff = 1.0
        try:
            while True:
                try:
                    async with websockets.connect(
                        self._websocket_url(),
                        open_timeout=_WS_AUTH_TIMEOUT,
                        close_timeout=5,
                        ping_interval=20,
                        ping_timeout=20,
                        max_size=_WS_MAX_MESSAGE_BYTES,
                    ) as websocket:
                        await self._authenticate_websocket(websocket)
                        subscriptions = await self._subscribe_websocket(websocket)
                        self._ws_active = True
                        if self._ws_ready is not None:
                            self._ws_ready.set()
                        backoff = 1.0
                        # Companion sweep: relays don't guarantee a membership
                        # event per new conversation (#93557), so discovery
                        # also runs on the poll transport's cadence.
                        discovery_task = asyncio.create_task(
                            self._ws_discovery_loop(websocket, subscriptions)
                        )
                        try:
                            frame_iter = websocket.__aiter__()
                            while True:
                                try:
                                    raw = await asyncio.wait_for(
                                        frame_iter.__anext__(),
                                        timeout=_WS_READ_IDLE_TIMEOUT,
                                    )
                                except StopAsyncIteration:
                                    break
                                except asyncio.TimeoutError:
                                    raise ConnectionError(
                                        f"no WebSocket frame for {_WS_READ_IDLE_TIMEOUT:.0f}s; "
                                        "assuming the connection went silent"
                                    ) from None
                                try:
                                    message = json.loads(raw)
                                except (ValueError, TypeError):
                                    logger.warning("Buzz: ignoring malformed WebSocket frame")
                                    continue
                                if not isinstance(message, list) or not message:
                                    continue
                                if message[0] == "EVENT" and len(message) >= 3:
                                    subscription_id = str(message[1])
                                    event = message[2]
                                    if not isinstance(event, dict):
                                        continue
                                    if subscription_id == _WS_MEMBERSHIP_SUB_ID:
                                        await self._handle_membership_event(websocket, subscriptions, event)
                                        continue
                                    channel_id = subscriptions.get(subscription_id)
                                    state = self._channel_state.get(channel_id or "")
                                    if channel_id and state is not None:
                                        before = self._cursor_mark(state)
                                        await self._handle_event(channel_id, state, event)
                                        self._trim_seen(state)
                                        if self._cursor_mark(state) != before:
                                            self._save_cursors()
                                elif message[0] == "CLOSED":
                                    detail = message[-1] if len(message) > 2 else "subscription closed"
                                    sub_id = str(message[1]) if len(message) > 1 else ""
                                    closed_channel = subscriptions.get(sub_id)
                                    detail_l = str(detail).lower()
                                    # A membership rejection ("restricted: not a
                                    # channel member", bare "not a channel member",
                                    # or "auth-required") means the relay will
                                    # never serve this subscription — drop it
                                    # permanently rather than reconnecting and
                                    # repeating the same rejection in a tight loop.
                                    is_membership_rejection = (
                                        "restricted" in detail_l
                                        or "not a channel member" in detail_l
                                        or "auth-required" in detail_l
                                    )
                                    if is_membership_rejection and closed_channel:
                                        logger.warning(
                                            "Buzz: relay permanently rejected channel %s (%s) — "
                                            "removing from watch list",
                                            closed_channel, detail,
                                        )
                                        self._restricted_channels.add(closed_channel)
                                        del subscriptions[sub_id]
                                        self._channel_state.pop(closed_channel, None)
                                    else:
                                        raise ConnectionError(str(detail))
                                elif message[0] == "NOTICE":
                                    logger.warning("Buzz: relay notice: %s", message[-1])
                        finally:
                            discovery_task.cancel()
                            try:
                                await discovery_task
                            except (asyncio.CancelledError, Exception):
                                pass
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._ws_active = False
                    logger.warning("Buzz: WebSocket disconnected; retrying in %.1fs: %s", backoff, e)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
        finally:
            self._ws_active = False

    # ── Inbound polling ───────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Poll every watched channel for new events until cancelled."""
        try:
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
        except asyncio.CancelledError:
            raise

    def _new_channel_state(self, chat_type: str) -> dict:
        return {
            "chat_type": chat_type,
            "last_ts": 0,
            "seen": OrderedDict(),
            "event_meta": OrderedDict(),
        }

    # ── Durable channel cursors ───────────────────────────────────────────

    @staticmethod
    def _cursor_path() -> Path:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / _CURSOR_STATE_SUBDIR / _CURSOR_STATE_FILENAME

    def _load_cursors(self) -> None:
        """Read back the cursors a previous run persisted.

        A file written by a different identity or against a different relay is
        ignored rather than trusted: channel ids would collide while the event
        stream behind them is a different one.  Any read/parse failure leaves
        the cursors empty, which degrades to the old seed-from-history
        behaviour instead of failing the connect.
        """
        self._restored_cursors = {}
        try:
            path = self._cursor_path()
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Buzz: could not read channel cursors", exc_info=True)
            return
        if not isinstance(data, dict):
            return
        if (
            data.get("identity") != self._self_pubkey
            or data.get("relay") != self.relay_url
        ):
            return
        channels = data.get("channels")
        if not isinstance(channels, dict):
            return
        for channel_id, entry in channels.items():
            if not isinstance(entry, dict):
                continue
            try:
                last_ts = int(entry.get("last_ts") or 0)
            except (TypeError, ValueError):
                continue
            raw_seen = entry.get("seen")
            seen = (
                [str(event_id) for event_id in raw_seen][-_SEEN_CAP:]
                if isinstance(raw_seen, list)
                else []
            )
            self._restored_cursors[str(channel_id)] = {
                "chat_type": str(entry.get("chat_type") or ""),
                "last_ts": last_ts,
                "seen": seen,
            }

    def _save_cursors(self) -> None:
        """Persist every watched channel's cursor.  Never raises."""
        payload = {
            "identity": self._self_pubkey,
            "relay": self.relay_url,
            "channels": {
                channel_id: {
                    "chat_type": state.get("chat_type") or "group",
                    "last_ts": int(state.get("last_ts") or 0),
                    "seen": list(state.get("seen") or ()),
                }
                for channel_id, state in self._channel_state.items()
            },
        }
        try:
            from utils import atomic_json_write

            atomic_json_write(self._cursor_path(), payload, indent=None)
        except Exception:
            logger.debug("Buzz: could not persist channel cursors", exc_info=True)

    @staticmethod
    def _cursor_mark(state: dict) -> tuple:
        """Cheap change detector for one channel's cursor."""
        seen = state.get("seen") or ()
        return (
            int(state.get("last_ts") or 0),
            len(seen),
            next(reversed(seen), None) if seen else None,
        )

    def _restore_channel_state(self, channel_id: str, chat_type: str) -> bool:
        """Install a persisted cursor for *channel_id*; True when one existed.

        Restoring is what closes the restart gap: seeding from current history
        instead would mark everything that arrived while the gateway was down
        as already seen, so the relay's durable copy is never dispatched
        (#90464).
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
        state = self._new_channel_state(chat_type)
        self._channel_state[channel_id] = state
        code, out, err = await self._run_cli(
            ["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)]
        )
        if code != 0:
            logger.warning(
                "Buzz: could not seed channel %s — %s", channel_id, _cli_error_message(err, code)
            )
            # Fall back to "now" so a transiently unreadable channel does not
            # replay its whole history once it becomes readable.
            state["last_ts"] = int(time.time())
            return
        for event in _parse_json_list(out):
            event_id = event.get("id")
            created_at = int(event.get("created_at") or 0)
            if event_id:
                state["seen"][str(event_id)] = None
            state["last_ts"] = max(state["last_ts"], created_at)
            # History is never dispatched, but it still classifies and feeds
            # the event_meta cache so post-restart thread replies to messages
            # we sent before the gateway came up still match (#75826).
            self._remember_event(state, event)
            # History is never dispatched, but it still classifies: a DM that
            # leaked in via ``channels list`` latches to chat_type="dm" here,
            # so it bypasses the mention gate from the very first poll.
            self._maybe_latch_dm(channel_id, state, event)
        self._trim_seen(state)

    async def _discover_dms(self, *, seed: bool) -> None:
        """Watch DM conversations.  New ones found mid-run dispatch from their
        beginning (a fresh conversation has no history worth suppressing);
        ones present at startup are seeded like channels.

        ``dms list`` is only a best-effort source: on some hosted relays it
        returns ``[]`` even when DM conversations exist (#68871).  Those DMs
        DO surface in ``channels list`` as entries named "DM" with an empty
        description, so that exact metadata shape is the fallback.  Named
        rooms and missing metadata still fail closed as groups.
        """
        code, out, _err = await self._run_cli(["dms", "list"])
        if code == 0:
            for dm in _parse_json_list(out):
                dm_id = str(dm.get("dm_id") or "")
                if not dm_id or dm_id in self._channel_state or dm_id in self._restricted_channels:
                    continue
                if seed:
                    await self._seed_channel(dm_id, chat_type="dm")
                elif not self._restore_channel_state(dm_id, "dm"):
                    self._channel_state[dm_id] = self._new_channel_state("dm")
                self._channel_names.setdefault(dm_id, "DM")

        code, out, _err = await self._run_cli(["channels", "list"])
        if code != 0:
            return
        for ch in _parse_json_list(out):
            ch_id = str(ch.get("channel_id") or "")
            if not ch_id:
                continue
            self._channel_meta[ch_id] = ch
            self._channel_names.setdefault(ch_id, str(ch.get("name") or ch_id))
            if ch_id in self._restricted_channels:
                continue
            if self._may_reclassify_as_dm(ch_id):
                # DM-shaped channels-list entries promote to DM (#87899/#77987,
                # landed via #99431) — including ones already being watched.
                if ch_id in self._channel_state:
                    self._channel_state[ch_id]["chat_type"] = "dm"
                elif seed:
                    await self._seed_channel(ch_id, chat_type="dm")
                elif not self._restore_channel_state(ch_id, "dm"):
                    self._channel_state[ch_id] = self._new_channel_state("dm")
                continue
            if ch_id in self._channel_state:
                continue
            # Live adoption of real community channels joined mid-run
            # (#75107): in watch-all mode (no explicit channels list) a
            # channel the agent is added to after connect() must start
            # dispatching without a gateway restart. Unlike a fresh DM its
            # history predates us, so it is always seeded from its newest
            # events — only messages sent after adoption dispatch. Explicit
            # watch lists stay authoritative: the user chose that set.
            if not seed and not self.channels:
                await self._seed_channel(ch_id, chat_type="group")
                logger.info(
                    "Buzz: adopted newly joined channel %s (%s)",
                    ch_id,
                    self._channel_names.get(ch_id, ch_id),
                )

    async def _poll_channel(self, channel_id: str) -> None:
        state = self._channel_state.get(channel_id)
        if state is None:
            return
        args = ["messages", "get", "--channel", channel_id, "--limit", str(_FETCH_LIMIT)]
        if state["last_ts"]:
            # Nostr `since` is inclusive: same-second events are re-fetched
            # and de-duped by id below.
            args += ["--since", str(state["last_ts"])]
        code, out, err = await self._run_cli(args)
        if code != 0:
            logger.debug(
                "Buzz: poll of channel %s failed — %s", channel_id, _cli_error_message(err, code)
            )
            return
        before = self._cursor_mark(state)
        for event in _parse_json_list(out):
            await self._handle_event(channel_id, state, event)
        self._trim_seen(state)
        # Persist only when the sweep actually moved the cursor, so an idle
        # channel does not rewrite the file every poll interval.
        if self._cursor_mark(state) != before:
            self._save_cursors()

    @staticmethod
    def _parse_imeta_attachments(event: dict) -> Tuple[List[dict], int]:
        """Return accepted NIP-94 metadata and the rejected ``imeta`` count."""
        tags = event.get("tags")
        if not isinstance(tags, list):
            return [], 0
        attachments: List[dict] = []
        rejected = 0
        total_declared_bytes = 0
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or not tag or tag[0] != "imeta":
                continue
            if len(attachments) >= _MAX_INBOUND_ATTACHMENTS:
                rejected += 1
                continue
            fields: Dict[str, str] = {}
            for raw_field in tag[1:]:
                if not isinstance(raw_field, str):
                    continue
                key, separator, value = raw_field.partition(" ")
                if separator and key not in fields:
                    fields[key] = value.strip()
            url = fields.get("url", "")
            digest = fields.get("x", "").lower()
            filename = fields.get("filename", "")
            mime_type = fields.get("m", "")
            try:
                size = int(fields.get("size", ""))
                parsed = urlsplit(url)
                parsed_hostname = parsed.hostname
                # Access validates malformed/non-numeric ports even though exact
                # origin authorization occurs immediately before downloading.
                parsed.port
            except (TypeError, ValueError):
                rejected += 1
                continue
            if (
                parsed.scheme != "https"
                or not parsed_hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not 0 < size <= _MAX_INBOUND_ATTACHMENT_BYTES
                or total_declared_bytes + size > _MAX_INBOUND_ATTACHMENT_BYTES
            ):
                rejected += 1
                continue
            total_declared_bytes += size
            attachments.append(
                {
                    "url": url,
                    "sha256": digest,
                    "size": size,
                    "filename": _safe_attachment_filename(filename),
                    "mime_type": mime_type[:255],
                }
            )
        return attachments, rejected

    @staticmethod
    def _imeta_attachments(event: dict) -> List[dict]:
        """Return bounded, structurally valid NIP-94 attachment metadata."""
        attachments, _rejected = BuzzAdapter._parse_imeta_attachments(event)
        return attachments

    @staticmethod
    def _attachment_rejection_note(rejected: int) -> str:
        """Return a fixed-width diagnostic for malformed or excess metadata."""
        shown = str(rejected) if rejected <= 999 else "999+"
        return f"[{shown} Buzz attachment(s) rejected as malformed or over limits.]"

    async def _download_attachment(self, metadata: dict) -> Optional[CachedMedia]:
        """Download, integrity-check, and cache one authorized Buzz attachment."""
        url = metadata["url"]
        try:
            parsed_url = urlsplit(url)
            host = (parsed_url.hostname or "").lower().rstrip(".")
            origin = (host, parsed_url.port or 443)
        except ValueError:
            parsed_url = None
            origin = ("", 0)
        if (
            parsed_url is None
            or parsed_url.scheme != "https"
            or origin not in self._attachment_origins
        ):
            logger.warning(
                "Buzz: refusing attachment from untrusted origin %s:%s",
                origin[0] or "<missing>",
                origin[1],
            )
            return None

        import httpx

        try:
            timeout = httpx.Timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT)
            async with asyncio.timeout(_ATTACHMENT_DOWNLOAD_TIMEOUT):
                async with httpx.AsyncClient(
                    follow_redirects=False,
                    timeout=timeout,
                    headers={"Accept-Encoding": "identity"},
                ) as client:
                    async with client.stream("GET", url) as response:
                        if response.status_code != 200:
                            logger.warning(
                                "Buzz: attachment download returned HTTP %s",
                                response.status_code,
                            )
                            return None
                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                declared_response_size = int(content_length)
                            except ValueError:
                                return None
                            if declared_response_size != metadata["size"]:
                                logger.warning(
                                    "Buzz: attachment Content-Length does not match imeta size"
                                )
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

        if len(data) != metadata["size"]:
            logger.warning("Buzz: attachment size does not match imeta")
            return None
        if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            logger.warning("Buzz: attachment SHA-256 does not match imeta")
            return None
        try:
            return await cache_media_bytes_async(
                bytes(data),
                filename=metadata["filename"],
                mime_type=metadata["mime_type"],
            )
        except (OSError, ValueError) as exc:
            logger.warning("Buzz: attachment cache write failed: %s", exc)
            return None

    async def _cache_inbound_attachments(
        self,
        metadata_items: List[dict],
    ) -> List[CachedMedia]:
        cached: List[CachedMedia] = []
        for metadata in metadata_items:
            attachment = await self._download_attachment(metadata)
            if attachment is not None:
                cached.append(attachment)
        return cached

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
        has_imeta = bool(attachment_metadata or rejected_attachments)
        if (
            not pubkey
            or not isinstance(content, str)
            or (not content.strip() and not has_imeta)
        ):
            return

        # Feed the per-channel event cache before any early return so self-echo
        # and concurrent-author traffic can still be reply parents (#75826).
        self._remember_event(state, event)

        # Suppress self-echo: never dispatch our own messages back to the agent.
        if pubkey == self._self_pubkey:
            return

        # Reclassify a leaked DM before gating so its first un-mentioned
        # message both latches the conversation and dispatches.
        self._maybe_latch_dm(channel_id, state, event)

        is_dm = state["chat_type"] == "dm"
        reply_parent_id = _event_reply_parent_id(event)
        reply_meta = self._lookup_event_meta(state, reply_parent_id) if reply_parent_id else None
        reply_to_is_own = bool(
            reply_meta is not None and reply_meta[0] == self._self_pubkey
        )
        # In shared channels, respond only when addressed — unless
        # require_mention is disabled, in which case respond to every message.
        # A NIP-10 thread reply whose direct parent is one of our messages is
        # treated as addressed (parity with Signal/WhatsApp; fixes #75826 —
        # e.g. Desktop "/approve session" replies that never type @name).
        # Explicit addressing is a text @mention OR a signed recipient p-tag
        # (#92781). DMs always dispatch.
        if (
            not is_dm
            and self.require_mention
            and not self._is_addressed(event)
            and not reply_to_is_own
        ):
            return

        # Adapter-level allow-list (the gateway applies BUZZ_ALLOWED_USERS /
        # BUZZ_ALLOW_ALL_USERS centrally as well; empty list = no filter here).
        if self._allowed_pubkeys and pubkey not in self._allowed_pubkeys:
            explicitly_tagged = any(
                isinstance(tag, (list, tuple))
                and len(tag) > 1
                and tag[0] == "p"
                and str(tag[1]).lower() == self._self_pubkey
                for tag in event.get("tags") or []
            )
            if (
                pubkey in self._reaction_only_pubkeys
                and explicitly_tagged
                and self._is_mentioned(content)
            ):
                await self.send_reaction(channel_id, event_id, "👀")
            logger.debug("Buzz: ignoring message from unauthorized pubkey %s…", pubkey[:8])
            return

        # Strip a leading @mention so slash commands (@Chip /whoami ->
        # /whoami) and clean prompts are recognized. DM messages often still
        # open with "@Chip" even though no mention is required there, so the
        # strip applies to both chat types.
        dispatch_text = self._strip_mention(content)
        # NIP-10 thread root for session scoping: replies inside a thread all
        # share the root as their thread_id, so the gateway groups them into
        # one thread session (marked "root" tag preferred, legacy fallback).
        thread_id = self._extract_thread_root(event)

        # Remember where this message sits in the thread graph so our reply
        # can join the SAME thread rather than nesting a new one under it.
        self._record_thread_root(event_id, event)
        # Attachment fetch/cache is a security-sensitive side effect. Only the
        # gateway's authoritative callback can permit it, and only an explicit
        # True is permission: false, absent, or failed checks all fail closed.
        # The message still dispatches so GatewayRunner can apply denial/pairing.
        chat_type = "dm" if is_dm else "group"
        attachment_fetch_allowed = bool(attachment_metadata) and (
            self._is_sender_authorized(pubkey, chat_type, channel_id) is True
        )
        attachments = (
            await self._cache_inbound_attachments(attachment_metadata)
            if attachment_fetch_allowed
            else []
        )
        if rejected_attachments:
            dispatch_text = (
                f"{dispatch_text}\n"
                f"{self._attachment_rejection_note(rejected_attachments)}"
            ).strip()
        if (
            attachment_fetch_allowed
            and len(attachments) < len(attachment_metadata)
        ):
            failed = len(attachment_metadata) - len(attachments)
            dispatch_text = (
                f"{dispatch_text}\n"
                f"[{failed} Buzz attachment(s) could not be downloaded or failed integrity checks.]"
            ).strip()

        message_type = MessageType.TEXT
        if attachments:
            attachment_kinds = {attachment.kind for attachment in attachments}
            if len(attachment_kinds) == 1:
                message_type = {
                    "image": MessageType.PHOTO,
                    "video": MessageType.VIDEO,
                    "audio": MessageType.AUDIO,
                    "document": MessageType.DOCUMENT,
                }.get(next(iter(attachment_kinds)), MessageType.DOCUMENT)
            else:
                # Mixed media must use document semantics so an audio member is
                # not mistaken for a voice note and sent through STT.
                message_type = MessageType.DOCUMENT

        await self._dispatch_message(
            text=dispatch_text,
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=pubkey,
            user_name=await self._resolve_user_name(pubkey),
            message_id=event_id,
            created_at=created_at,
            thread_id=thread_id,
            reply_to_message_id=reply_parent_id,
            reply_to_text=reply_meta[1] if reply_meta else None,
            reply_to_author_id=reply_meta[0] if reply_meta else None,
            reply_to_is_own_message=reply_to_is_own,
            media_urls=[attachment.path for attachment in attachments],
            media_types=[attachment.media_type for attachment in attachments],
            message_type=message_type,
            raw_message=event,
        )

    # ── DM classification (issue #68871) ──────────────────────────────────
    #
    # ``buzz dms list`` returns [] on some hosted relays even when DM
    # conversations exist, so DMs can leak in through ``channels list`` as
    # chat_type="group".  Relay-materialized DMs are named "DM" with an empty
    # description, which periodic discovery promotes to DM even when messages
    # omit recipient p-tags.  Named channels and missing metadata fail closed.
    # In normal channels a p-tag is only an addressing signal and must wake the
    # agent without changing the conversation type.

    def _may_reclassify_as_dm(self, channel_id: str) -> bool:
        """True when the conversation's metadata does not rule out a DM.

        Known real community channels (real name or non-empty description in
        ``channels list``) must never turn into DMs just because a message
        p-tags us.  Missing metadata fails closed rather than allowing a named
        channel to latch as a DM before its metadata arrives.
        """
        meta = self._channel_meta.get(channel_id)
        if meta is None:
            return False
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        return name == "DM" and not description

    def _p_tagged_to_self(self, event: dict) -> bool:
        """True when the signed event addresses this identity by pubkey."""
        if not self._self_pubkey:
            return False
        tags = event.get("tags")
        if not isinstance(tags, list):
            return False
        return any(
            isinstance(tag, (list, tuple))
            and len(tag) > 1
            and tag[0] == "p"
            and str(tag[1]).lower() == self._self_pubkey
            for tag in tags
        )

    def _is_direct_message_event(self, channel_id: str, event: dict) -> bool:
        """True when ``event`` is shaped like a direct message to us: a chat
        message from another user, p-tagged to our pubkey, whose content does
        NOT visibly mention us — i.e. the p-tag is structural DM addressing,
        not the artifact of a typed @mention (see block comment above)."""
        if not self._self_pubkey or not self._may_reclassify_as_dm(channel_id):
            return False
        if int(event.get("kind") or 0) != _CHAT_KIND:
            return False
        pubkey = str(event.get("pubkey") or "").lower()
        if not pubkey or pubkey == self._self_pubkey:
            return False
        if not self._p_tagged_to_self(event):
            return False
        content = event.get("content")
        return isinstance(content, str) and not self._is_mentioned(content)

    def _maybe_latch_dm(self, channel_id: str, state: dict, event: dict) -> None:
        """Latch a group conversation to chat_type="dm" once any direct
        message is seen; the classification then sticks so subsequent
        un-mentioned messages in the conversation dispatch too."""
        if state["chat_type"] == "dm" or not self._is_direct_message_event(channel_id, event):
            return
        state["chat_type"] = "dm"
        self._channel_names.setdefault(channel_id, "DM")
        logger.info("Buzz: conversation %s reclassified as DM (message p-tagged to self)", channel_id)

    def _is_mentioned(self, content: str) -> bool:
        """True when text explicitly addresses this agent (npub, hex, or @name)."""
        lowered = content.lower()
        if self._self_pubkey and re.fullmatch(r"[0-9a-f]{64}", self._self_pubkey):
            pattern = rf"(?<![0-9a-f]){re.escape(self._self_pubkey)}(?![0-9a-f])"
            if re.search(pattern, lowered):
                return True
        if self._self_npub:
            pattern = rf"(?<![a-z0-9]){re.escape(self._self_npub.lower())}(?![a-z0-9])"
            if re.search(pattern, lowered):
                return True
        if self._display_name:
            pattern = (
                rf"(?<![\w@])@{re.escape(self._display_name.lower())}"
                r"(?=$|[\s,;.!?:)\]}])"
            )
            if re.search(pattern, lowered):
                return True
        return False

    def _is_addressed(self, event: dict) -> bool:
        """True when a group event carries an explicit text or p-tag address."""
        content = event.get("content")
        return (
            isinstance(content, str)
            and (self._is_mentioned(content) or self._p_tagged_to_self(event))
        )

    def _strip_mention(self, content: str) -> str:
        """Remove a leading @mention of this agent so the remaining text can be
        recognized as a slash command or clean prompt.

        Mirrors the Discord adapter, which strips its own ``<@id>`` mention
        before dispatch. Without this a channel message like ``@Chip /whoami``
        arrives with a leading ``@Chip``; the gateway's ``is_command()`` checks
        ``text.lstrip().startswith("/")`` and never fires the command. Only a
        LEADING mention is stripped (case-insensitive); mentions mid-sentence
        are left intact so normal prose is unaffected.
        """
        text = content.strip()
        candidates = []
        if self._display_name:
            candidates.append(
                rf"@{re.escape(self._display_name)}" + r"(?=$|[\s,;.!?:)\]}])"
            )
        if self._self_npub:
            candidates.append(rf"@?{re.escape(self._self_npub)}(?![a-z0-9])")
        if self._self_pubkey:
            candidates.append(rf"@?{re.escape(self._self_pubkey)}(?![0-9a-f])")
        if not candidates:
            return text
        # Display names require '@'; npub and hex identities are already
        # unambiguous and may optionally include it.
        pattern = rf"^(?:{'|'.join(candidates)})[\s:,]*"
        stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        return stripped.strip()

    async def _resolve_user_name(self, pubkey: str) -> str:
        """Resolve a pubkey to a display name (cached; falls back to npub prefix).

        Failures are cached too (negative caching): without it, every message
        from a profile-less pubkey re-runs ``users get`` each poll sweep,
        which amplifies badly when several adapter instances poll in one
        process.
        """
        cached = self._user_names.get(pubkey)
        if cached is not None:
            return cached
        name = ""
        code, out, _err = await self._run_cli(["users", "get", "--pubkey", pubkey])
        if code == 0:
            profiles = _parse_json_list(out)
            if profiles:
                name = str(profiles[0].get("display_name") or "").strip()
        if not name:
            name = (hex_to_npub(pubkey) or pubkey)[:16]
        self._user_names[pubkey] = name
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

    # ── Thread anchoring ──────────────────────────────────────────────────
    #
    # NIP-10 marked ``e`` tags: a reply carries ["e", <root>, "", "root"] plus
    # ["e", <parent>, "", "reply"]; a message that STARTS a thread carries a
    # single ["e", <parent>, "", "reply"] and no root marker.
    #
    # The gateway hands adapters the triggering message's own id as the reply
    # anchor.  Anchoring to that id is correct for a top-level message (it
    # opens the thread the user expects), but inside an existing thread it
    # nests a fresh sub-thread under every single answer.  Buzz renders that
    # as an endless ladder of one-message threads.
    #
    # Fix: remember each inbound message's thread ROOT.  When the trigger was
    # already inside a thread, reply against that root so our answer lands in
    # the same thread the user is typing in.  When it was top-level, keep the
    # existing behaviour and anchor to the message itself.

    _THREAD_ROOT_CACHE = 512

    @staticmethod
    def _extract_thread_root(event: dict) -> Optional[str]:
        """Return the NIP-10 thread root of ``event``, or None if top-level."""
        tags = event.get("tags")
        if not isinstance(tags, list):
            return None
        root = None
        reply = None
        for tag in tags:
            if not isinstance(tag, (list, tuple)) or len(tag) < 2:
                continue
            if str(tag[0]) != "e":
                continue
            marker = str(tag[3]).lower() if len(tag) > 3 else ""
            if marker == "root":
                root = str(tag[1])
            elif marker == "reply":
                reply = str(tag[1])
            elif not marker and reply is None:
                # Unmarked (deprecated positional) e-tag: treat as the parent.
                reply = str(tag[1])
        if root:
            return root
        # A lone "reply" e-tag means this message started a thread hanging off
        # <reply>; that parent IS the thread root for anything that follows.
        return reply

    def _record_thread_root(self, event_id: str, event: dict) -> None:
        """Cache the thread root for an inbound message id."""
        if not event_id:
            return
        roots = getattr(self, "_thread_roots", None)
        if roots is None:
            roots = self._thread_roots = OrderedDict()
        roots[event_id] = self._extract_thread_root(event)
        roots.move_to_end(event_id)
        while len(roots) > self._THREAD_ROOT_CACHE:
            roots.popitem(last=False)

    def _resolve_reply_anchor(self, anchor: Optional[str]) -> Optional[str]:
        """Map a gateway reply anchor onto the right Buzz thread anchor.

        Returns the thread root when the triggering message was already inside
        a thread (so the reply joins it), otherwise the anchor unchanged (so a
        reply to a top-level message opens one thread, as before).
        """
        if not anchor:
            return anchor
        roots = getattr(self, "_thread_roots", None) or {}
        return roots.get(str(anchor)) or anchor
    def _remember_event(self, state: dict, event: dict) -> None:
        """Record author + content snippet for later NIP-10 parent lookup."""
        event_id = str(event.get("id") or "")
        if not event_id:
            return
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        snippet = content[:_EVENT_META_CONTENT_CAP] if isinstance(content, str) else ""
        self._store_event_meta(state, event_id, pubkey, snippet)

    def _remember_event_meta(
        self,
        channel_id: str,
        event_id: str,
        pubkey: str,
        content: str,
    ) -> None:
        state = self._channel_state.get(channel_id)
        if state is None or not event_id:
            return
        snippet = (content or "")[:_EVENT_META_CONTENT_CAP]
        self._store_event_meta(state, event_id, (pubkey or "").lower(), snippet)

    @staticmethod
    def _store_event_meta(
        state: dict,
        event_id: str,
        pubkey: str,
        snippet: str,
    ) -> None:
        cache = state.setdefault("event_meta", OrderedDict())
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict(cache)
            state["event_meta"] = cache
        cache[event_id] = (pubkey, snippet)
        cache.move_to_end(event_id)
        while len(cache) > _SEEN_CAP:
            cache.popitem(last=False)

    @staticmethod
    def _lookup_event_meta(state: dict, event_id: Optional[str]) -> Optional[Tuple[str, str]]:
        if not event_id:
            return None
        cache = state.get("event_meta") or {}
        entry = cache.get(event_id)
        if not entry or not isinstance(entry, tuple) or len(entry) < 2:
            return None
        return str(entry[0] or ""), str(entry[1] or "")
    async def _localize_inbound_media(
        self,
        text: str,
        message_id: str,
        *,
        user_id: str = "",
        chat_type: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str], MessageType]:
        """Authenticate and cache same-relay media references in *text*.

        Each object is independent: one failed download is logged and skipped
        without discarding the caption or any other successfully cached files.

        Downloading spends this agent's Buzz credentials on a URL chosen by
        the sender, so it runs only when the gateway's authorization callback
        returns an explicit ``True``. The adapter's own ``allowed_users``
        list is a pre-filter, not a substitute: a missing, failed, or
        negative gateway decision leaves the text untouched and no
        credentialed request is made.
        """
        urls, replacements = _find_relay_media_refs(text, self.relay_url)
        if not urls:
            return text, [], [], MessageType.TEXT

        if self._is_sender_authorized(user_id, chat_type, chat_id) is not True:
            logger.warning(
                "Buzz: not localizing %d media reference(s) in message %s — "
                "sender %s… is not explicitly authorized",
                len(urls), message_id[:12], (user_id or "?")[:8],
            )
            return text, [], [], MessageType.TEXT

        cleaned_text = _replace_media_refs(text, replacements)
        media_urls: List[str] = []
        media_types: List[str] = []
        media_kinds: List[str] = []

        from gateway.platforms.base import (
            cache_media_bytes_async,
            validate_inbound_media_size,
        )

        for url in urls:
            path_match = _MEDIA_PATH_RE.fullmatch(urlsplit(url).path)
            if path_match is None:
                continue
            ext = (path_match.group("ext") or ".bin").lower()
            label = f"{path_match.group('sha')[:12]}{ext}"
            try:
                with tempfile.TemporaryDirectory(prefix="hermes-buzz-media-") as temp_dir:
                    download_path = Path(temp_dir) / f"buzz_{label}"
                    code, _out, _err = await self._run_cli(
                        ["media", "get", "-o", str(download_path), url]
                    )
                    if code != 0 or not download_path.is_file():
                        logger.warning(
                            "Buzz: failed to localize inbound media %s (exit %d)",
                            label,
                            code,
                        )
                        continue
                    validate_inbound_media_size(
                        download_path.stat().st_size,
                        media_type="Buzz media",
                    )
                    mime_type = (
                        mimetypes.guess_type(download_path.name)[0]
                        or "application/octet-stream"
                    )
                    # Up to the inbound media cap (128 MiB) — read off the loop too.
                    data = await asyncio.to_thread(download_path.read_bytes)
                    cached = await cache_media_bytes_async(
                        data,
                        filename=download_path.name,
                        mime_type=mime_type,
                    )
            except Exception as exc:
                logger.warning(
                    "Buzz: failed to localize inbound media %s (%s)",
                    label,
                    type(exc).__name__,
                )
                continue

            if cached is None:
                logger.warning("Buzz: rejected invalid inbound media %s", label)
                continue
            media_urls.append(cached.path)
            media_types.append(cached.media_type)
            media_kinds.append(cached.kind)

        if media_urls:
            logger.info(
                "Buzz: localized %d inbound media attachment(s) for message %s",
                len(media_urls),
                message_id[:12],
            )

        if "image" in media_kinds:
            message_type = MessageType.PHOTO
        elif "audio" in media_kinds:
            message_type = MessageType.AUDIO
        elif "video" in media_kinds:
            message_type = MessageType.VIDEO
        elif media_kinds:
            message_type = MessageType.DOCUMENT
        else:
            message_type = MessageType.TEXT

        if not cleaned_text:
            cleaned_text = (
                "(attachment)"
                if media_urls
                else "(Buzz media attachment unavailable)"
            )
        return cleaned_text, media_urls, media_types, message_type

    async def _dispatch_message(
        self,
        text: str,
        chat_id: str,
        chat_type: str,
        user_id: str,
        user_name: str,
        message_id: str,
        created_at: int,
        thread_id: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        reply_to_author_id: Optional[str] = None,
        reply_to_is_own_message: bool = False,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
        message_type: MessageType = MessageType.TEXT,
        raw_message: Any = None,
    ) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return

        media_urls = list(media_urls or [])
        media_types = list(media_types or [])

        # Localize authenticated same-relay URL references embedded in the
        # text, in addition to any native imeta attachments already verified
        # and cached by the caller. Both paths are gated on the gateway's
        # explicit-True authorization decision.
        text, localized_urls, localized_types, localized_type = await self._localize_inbound_media(
            text,
            message_id,
            user_id=user_id,
            chat_type=chat_type,
            chat_id=chat_id,
        )
        for path, mime in zip(localized_urls, localized_types):
            if path not in media_urls:
                media_urls.append(path)
                media_types.append(mime)
        if message_type == MessageType.TEXT:
            message_type = localized_type
        elif localized_urls and localized_type not in (message_type, MessageType.TEXT):
            # Mixed media sources must use document semantics so an audio
            # member is not mistaken for a voice note and sent through STT.
            message_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=chat_id,
            chat_name=self._channel_names.get(chat_id, chat_id),
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_id,
        )

        event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            raw_message=raw_message,
            message_id=message_id,
            media_urls=list(media_urls or []),
            media_types=list(media_types or []),
            media_text_inlined=[False] * len(media_urls or []),
            timestamp=datetime.fromtimestamp(created_at) if created_at else datetime.now(),
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            reply_to_author_id=reply_to_author_id,
            reply_to_is_own_message=reply_to_is_own_message,
        )

        await self.handle_message(event)

        # Add a "seen" reaction after dispatching — signals to the user that
        # their message was received and is being processed.
        try:
            await self.send_reaction(chat_id, message_id, "👀")
        except Exception:
            logger.debug("Buzz: reaction failed for message %s", message_id[:12], exc_info=True)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def _profile_buzz_extra() -> dict:
    """Read ``buzz.extra`` from the active profile's config.yaml (scoped path).

    Only meaningful inside a secondary profile scope, where the hermes-home
    override points at that profile's home. Used by ``check_requirements``
    (which has no PlatformConfig argument) so the multiplex gate consults the
    profile's own configuration instead of the process env. Best-effort: any
    failure yields an empty mapping and the caller fails closed.
    """
    if not _profile_scoped():
        return {}
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import read_user_config_raw

        cfg = read_user_config_raw(Path(get_hermes_home()) / "config.yaml")
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    buzz = ((cfg.get("gateway") or {}).get("platforms") or {}).get("buzz")
    if not isinstance(buzz, dict):
        return {}
    extra = buzz.get("extra", buzz)
    return extra if isinstance(extra, dict) else {}


def check_requirements() -> bool:
    """Check if Buzz is configured: a relay URL plus a resolvable key."""
    if _profile_scoped():
        # Multiplexed secondary profile (#98738): os.environ's BUZZ_* values
        # are the default profile's bridge output and must not satisfy this
        # gate for another profile. Consult the profile's own config.yaml
        # (via the scoped home override) and its secret scope instead; an
        # unconfigured profile fails closed.
        extra = _profile_buzz_extra()
        relay = str(extra.get("relay_url") or "").strip()
        return bool(relay and _resolve_private_key(extra))
    # Scope-aware read: the gate runs before per-profile scopes install, and
    # BUZZ_RELAY_URL can be externally managed just like the key (#95216).
    if not (_get_scoped_secret("BUZZ_RELAY_URL", "") or "").strip():
        return False
    return bool(_resolve_private_key())


def validate_config(config) -> bool:
    """Validate that the platform config has enough information to connect."""
    extra = getattr(config, "extra", {}) or {}
    # Inside a secondary profile scope, extra is authoritative (#98738);
    # unscoped, the env read gains the external-secret rung so a managed
    # relay passes too (#95216).
    if _profile_scoped():
        relay = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
        relay = relay if relay is not None else extra.get("relay_url", "")
    else:
        relay = _get_scoped_secret("BUZZ_RELAY_URL", "") or extra.get("relay_url", "")
    return bool(relay and _resolve_private_key(extra))


def is_connected(config) -> bool:
    """Check whether Buzz is configured (env or config.yaml)."""
    return validate_config(config)


def _apply_yaml_config(yaml_cfg: dict, buzz_cfg: dict) -> Optional[dict]:
    """Translate ``config.yaml`` ``buzz.extra`` keys into ``BUZZ_*`` env vars.

    Implements the ``apply_yaml_config_fn`` contract.  ``check_requirements``
    and the adapter's connect path read configuration from the environment, so
    a config.yaml-only setup (no ``BUZZ_*`` env vars beyond the secret) would
    otherwise fail the ``check_fn`` gate and be silently skipped at gateway
    startup.  This hook bridges the ``extra`` block into env, mirroring the
    Slack/Telegram pattern.  Env vars win over YAML — every assignment is
    guarded by ``not os.getenv(...)`` so explicit env overrides survive a
    config.yaml update.  ``BUZZ_PRIVATE_KEY`` is a secret and stays in ``.env``;
    it is never sourced from config.yaml here.
    """
    extra = buzz_cfg.get("extra", buzz_cfg) or {}
    if not isinstance(extra, dict):
        return None
    # Under multiplex, a secondary profile's config loads inside its runtime
    # scope; its values must NOT be written to the process-global env, where
    # first-writer-wins would pin them for every other profile (issue #72348
    # Telegram/Discord mirror, Buzz side of #98738). Its adapter reads the
    # profile's PlatformConfig.extra directly instead.
    _skip_env_bridge = _profile_scoped()
    _str_keys = {
        "relay_url": "BUZZ_RELAY_URL",
        "cli_path": "BUZZ_CLI_PATH",
        "home_channel": "BUZZ_HOME_CHANNEL",
        "transport": "BUZZ_TRANSPORT",
    }
    for src, env in _str_keys.items():
        val = extra.get(src)
        if val and not _skip_env_bridge and not os.getenv(env):
            os.environ[env] = str(val)
    interval = extra.get("poll_interval")
    if interval is not None and not _skip_env_bridge and not os.getenv("BUZZ_POLL_INTERVAL"):
        os.environ["BUZZ_POLL_INTERVAL"] = str(interval)
    channels = extra.get("channels")
    if channels is not None and not _skip_env_bridge and not os.getenv("BUZZ_CHANNELS"):
        if isinstance(channels, (list, tuple)):
            channels = ",".join(str(c) for c in channels)
        os.environ["BUZZ_CHANNELS"] = str(channels)
    allowed = extra.get("allowed_users")
    if allowed is not None and not _skip_env_bridge and not os.getenv("BUZZ_ALLOWED_USERS"):
        if isinstance(allowed, (list, tuple)):
            allowed = ",".join(str(a) for a in allowed)
        os.environ["BUZZ_ALLOWED_USERS"] = str(allowed)
    reaction_only = extra.get("reaction_only_users")
    if reaction_only is not None and not _skip_env_bridge and not os.getenv("BUZZ_REACTION_ONLY_USERS"):
        if isinstance(reaction_only, (list, tuple)):
            reaction_only = ",".join(str(v) for v in reaction_only)
        os.environ["BUZZ_REACTION_ONLY_USERS"] = str(reaction_only)
    if "allow_all_users" in extra and not _skip_env_bridge and not os.getenv("BUZZ_ALLOW_ALL_USERS"):
        os.environ["BUZZ_ALLOW_ALL_USERS"] = str(extra["allow_all_users"]).lower()
    if "require_mention" in extra and not _skip_env_bridge and not os.getenv("BUZZ_REQUIRE_MENTION"):
        os.environ["BUZZ_REQUIRE_MENTION"] = str(extra["require_mention"]).lower()
    if "reply_in_thread" in extra and not os.getenv("BUZZ_REPLY_IN_THREAD"):
        os.environ["BUZZ_REPLY_IN_THREAD"] = str(extra["reply_in_thread"]).lower()
    if "reply_to_mode" in extra and not os.getenv("BUZZ_REPLY_TO_MODE"):
        os.environ["BUZZ_REPLY_TO_MODE"] = str(extra["reply_to_mode"]).lower()
    return None


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called BEFORE adapter construction so env-only setups show up in
    ``hermes gateway status`` and ``get_connected_platforms()``.  Returns
    ``None`` when Buzz isn't minimally configured.

    The special ``home_channel`` key is handled by the core hook — it becomes
    a proper ``HomeChannel`` on the ``PlatformConfig``.
    """
    if _profile_scoped():
        # Secondary profile scope (#98738): the process env's BUZZ_* values
        # are the default profile's configuration, not this profile's — env
        # enablement must not fabricate a Buzz platform for a profile that
        # did not configure one.
        return None
    relay = os.getenv("BUZZ_RELAY_URL", "").strip()
    if not relay or not _resolve_private_key():
        return None
    seed: dict = {"relay_url": relay}
    channels = os.getenv("BUZZ_CHANNELS", "").strip()
    if channels:
        seed["channels"] = [c.strip() for c in channels.split(",") if c.strip()]
    interval = os.getenv("BUZZ_POLL_INTERVAL", "").strip()
    if interval:
        try:
            seed["poll_interval"] = float(interval)
        except ValueError:
            pass
    cli_path = os.getenv("BUZZ_CLI_PATH", "").strip()
    if cli_path:
        seed["cli_path"] = cli_path
    # Home channel for deliver=buzz cron jobs; defaults to the first watched
    # channel so env-only setups get a sensible target without extra config.
    home = os.getenv("BUZZ_HOME_CHANNEL", "").strip() or (seed.get("channels") or [""])[0]
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("BUZZ_HOME_CHANNEL_NAME", home),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """One-shot send without a live adapter (out-of-process cron delivery).

    Used by ``tools/send_message_tool`` when ``hermes cron`` runs separately
    from the gateway process.  Without this hook, ``deliver=buzz`` cron jobs
    fail with ``No live adapter for platform 'buzz'``.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    _relay_raw = _scoped_platform_setting("BUZZ_RELAY_URL", extra, "relay_url")
    relay = (_relay_raw or extra.get("relay_url", "")).strip()
    private_key = _resolve_private_key(extra)
    _cli_raw = _scoped_platform_setting("BUZZ_CLI_PATH", extra, "cli_path")
    try:
        auth_tag = _resolve_auth_tag(extra)
    except ValueError as exc:
        return {"error": f"Buzz standalone send: {exc}"}
    cli_path = _resolve_cli_path(
        str(_cli_raw or "").strip() or str(extra.get("cli_path", "") or "")
    )
    if not relay or not private_key:
        return {"error": "Buzz standalone send: BUZZ_RELAY_URL and BUZZ_PRIVATE_KEY must be configured"}
    if not cli_path:
        return {"error": "Buzz standalone send: buzz CLI binary not found"}
    _home_raw = _scoped_platform_setting("BUZZ_HOME_CHANNEL", extra, "home_channel")
    target = (chat_id or "").strip() or (_home_raw or str(extra.get("home_channel", "") or "")).strip()
    if not target:
        return {"error": "Buzz standalone send: no target channel (set BUZZ_HOME_CHANNEL)"}

    args = ["messages", "send", "--channel", target, "--content", "-"]
    # Same reply_to_mode / reply_in_thread gate as the live adapter, so
    # out-of-process cron delivery (deliver=buzz) doesn't thread when the
    # operator asked for flat channel replies.
    _rtm = (os.getenv("BUZZ_REPLY_TO_MODE")
            or getattr(pconfig, "reply_to_mode", "first") or "first")
    _rtm = str(_rtm).strip().lower()
    _rit = os.getenv("BUZZ_REPLY_IN_THREAD")
    if _rit is None:
        _rit = extra.get("reply_in_thread")
    if _rit is not None and str(_rit).strip().lower() in ("false", "0", "no", "off"):
        _rtm = "off"
    if thread_id and _rtm != "off":
        args += ["--reply-to", str(thread_id)]
    for media in media_files or []:
        path = media[0] if isinstance(media, (list, tuple)) and media else media
        args += ["--file", str(path)]
    try:
        code, out, err = await _exec_buzz(
            cli_path,
            args,
            relay_url=relay,
            private_key=private_key,
            auth_tag=auth_tag,
            input_text=message,
        )
        if code != 0:
            escaped = _escape_unresolved_presentation_mention(message, err)
            if escaped is not None:
                logger.info(
                    "Buzz: retrying standalone message after unresolved "
                    "presentation-mention preflight"
                )
                code, out, err = await _exec_buzz(
                    cli_path,
                    args,
                    relay_url=relay,
                    private_key=private_key,
                    input_text=escaped,
                )
    except asyncio.CancelledError:
        raise
    except OSError as e:
        detail = _bounded_cli_message(str(e))
        return {"error": f"Buzz standalone send failed to launch CLI: {detail}"}
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
    """Interactive ``hermes gateway setup`` flow for the Buzz platform.

    Lazy-imports ``hermes_cli.setup`` helpers so the plugin stays importable
    in non-CLI contexts (gateway runtime, tests).
    """
    from hermes_cli.setup import (
        prompt,
        prompt_yes_no,
        save_env_value,
        get_env_value,
        print_header,
        print_info,
        print_warning,
        print_success,
    )

    print_header("Buzz")
    existing_relay = get_env_value("BUZZ_RELAY_URL")
    if existing_relay:
        print_info(f"Buzz: already configured (relay: {existing_relay})")
        if not prompt_yes_no("Reconfigure Buzz?", False):
            return

    print_info("Connect Hermes to a Buzz community (Block's Nostr-based human+agent platform).")
    print_info("   Requires the buzz CLI binary and a Nostr key that is a community member.")
    print()

    relay = prompt(
        "Relay URL (e.g. https://mycommunity.communities.buzz.xyz)",
        default=existing_relay or "",
    )
    if not relay:
        print_warning("Relay URL is required — skipping Buzz setup")
        return
    save_env_value("BUZZ_RELAY_URL", relay.strip())

    key = prompt("Nostr private key (nsec or hex; leave blank to keep current)", password=True)
    if key:
        save_env_value("BUZZ_PRIVATE_KEY", key.strip())
    elif not _resolve_private_key():
        print_warning("No private key configured — set BUZZ_PRIVATE_KEY before starting the gateway")

    channels = prompt(
        "Channel UUIDs to watch (comma-separated, empty = all joined channels)",
        default=get_env_value("BUZZ_CHANNELS") or "",
    )
    if channels:
        save_env_value("BUZZ_CHANNELS", channels.replace(" ", ""))

    home = prompt(
        "Home channel UUID for cron/notification delivery (optional)",
        default=get_env_value("BUZZ_HOME_CHANNEL") or "",
    )
    if home:
        save_env_value("BUZZ_HOME_CHANNEL", home.strip())

    print()
    print_info("🔒 Access control: restrict who can talk to the agent")
    allow_all = prompt_yes_no("Allow all community members to talk to the agent?", False)
    if allow_all:
        save_env_value("BUZZ_ALLOW_ALL_USERS", "true")
        save_env_value("BUZZ_ALLOWED_USERS", "")
        print_warning("⚠️  Open access — anyone in the community can command the agent.")
    else:
        save_env_value("BUZZ_ALLOW_ALL_USERS", "false")
        allowed = prompt(
            "Allowed users (comma-separated npubs or hex pubkeys, empty to deny everyone)",
            default=get_env_value("BUZZ_ALLOWED_USERS") or "",
        )
        save_env_value("BUZZ_ALLOWED_USERS", allowed.replace(" ", "") if allowed else "")

    print()
    print_success("Buzz configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="buzz",
        label="Buzz",
        adapter_factory=lambda cfg: BuzzAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"],
        install_hint="Requires the buzz CLI binary (https://github.com/block/buzz) on PATH or at BUZZ_CLI_PATH",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra with
        # relay/channels/poll interval + home_channel so env-only setups show
        # up in gateway status without instantiating the adapter.
        env_enablement_fn=_env_enablement,
        # Bridge config.yaml buzz.extra -> BUZZ_* env vars so check_fn and the
        # env-driven connect path work for config.yaml-only setups (secret stays
        # in .env). Without this the check_fn gate skips Buzz at startup.
        apply_yaml_config_fn=_apply_yaml_config,
        # Cron home-channel delivery support (deliver=buzz).
        cron_deliver_env_var="BUZZ_HOME_CHANNEL",
        # Out-of-process cron delivery.  Without this hook, deliver=buzz
        # cron jobs fail with "No live adapter" when cron runs separately
        # from the gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
        # Display
        emoji="🐝",
        # Buzz identities are pubkeys, not phone numbers
        pii_safe=False,
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are collaborating in a Buzz workspace (Block's Nostr-based "
            "human+agent platform). Markdown IS supported. Users address you "
            "by @-mentioning your name or npub in channels; direct messages "
            "reach you without a mention. Keep responses conversational."
        ),
    )
