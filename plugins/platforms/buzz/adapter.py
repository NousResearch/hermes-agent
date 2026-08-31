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

Or via environment variables (overrides config.yaml):
    BUZZ_RELAY_URL, BUZZ_CHANNELS, BUZZ_HOME_CHANNEL, BUZZ_POLL_INTERVAL,
    BUZZ_CLI_PATH, BUZZ_CREDENTIALS_FILE, BUZZ_ALLOWED_USERS,
    BUZZ_ALLOW_ALL_USERS, BUZZ_REPLY_IN_THREAD, BUZZ_REPLY_TO_MODE

The only secret is BUZZ_PRIVATE_KEY (nsec or hex) — it belongs in
``~/.hermes/.env``.  It is passed to the CLI via the subprocess
environment and is never logged.
"""

import asyncio
import json
import logging
import os
import re
import shutil
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
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.config import Platform


# Buzz chat messages are Nostr kind 9 events.  ``buzz messages get`` also
# returns housekeeping kinds (joins, canvas updates, …) — only kind 9 is
# dispatched to the agent.
_CHAT_KIND = 9
# How many events to request per poll / seed call.
_FETCH_LIMIT = 50
# Bound on the per-channel de-dupe set (events, not bytes).
_SEEN_CAP = 500
# Re-run DM discovery (``dms list`` plus the channels-list fallback) every
# N poll sweeps to pick up conversations opened mid-run.
_DM_DISCOVERY_EVERY = 5

_DEFAULT_POLL_INTERVAL = 4.0
_MIN_POLL_INTERVAL = 1.0
_CLI_TIMEOUT = 30.0

# WebSocket transport (NIP-42 authenticated Nostr subscription).
# kind 44100 is Buzz's channel-membership event — used for live DM discovery.
_WS_AUTH_TIMEOUT = 20.0
_WS_MAX_MESSAGE_BYTES = 2_000_000
_WS_MEMBERSHIP_KIND = 44100
_WS_MEMBERSHIP_SUB_ID = "hermes-buzz-membership"

# Where to look for a credentials JSON (keys: nsec / private_key_hex) when
# BUZZ_PRIVATE_KEY is not set.  Module-level so tests can point it at a tmpdir.
_DEFAULT_CREDENTIALS_DIR = Path("~/.config/buzz").expanduser()


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


def _cli_error_message(stderr: str, returncode: int) -> str:
    """Extract the human-readable message from the CLI's JSON error contract.

    stderr is ``{"error": "<category>", "message": "<detail>"}`` on failure;
    fall back to the raw (stripped) stderr when it isn't JSON.
    """
    text = (stderr or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("message"):
            return f"{data.get('error', 'error')}: {data['message']} (exit {returncode})"
    except ValueError:
        pass
    return text or f"buzz CLI failed with exit code {returncode}"


def _parse_json_list(stdout: str) -> List[dict]:
    """Parse CLI stdout expected to be a JSON array of objects."""
    try:
        data = json.loads(stdout or "[]")
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


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
        # channel_id -> {"chat_type", "last_ts", "seen": OrderedDict[event_id, None]}
        self._channel_state: Dict[str, dict] = {}
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
        # replays channel history into the agent.
        for channel_id in watch:
            await self._seed_channel(channel_id, chat_type="group")
        await self._discover_dms(seed=True)

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
        event_id = data.get("event_id")
        if event_id:
            # Belt-and-braces echo suppression: the poll loop already skips
            # our own pubkey, but marking the id seen makes de-dupe explicit.
            self._mark_seen(str(chat_id), str(event_id))
        return SendResult(
            success=bool(data.get("accepted", True)),
            message_id=str(event_id) if event_id else None,
            raw_response=data,
        )

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
            code, out, err = await self._run_cli(args, input_text=caption or "")
            if code != 0:
                return SendResult(success=False, error=_cli_error_message(err, code), retryable=code == 2)
            try:
                data = json.loads(out or "{}")
            except ValueError:
                data = {}
            event_id = data.get("event_id")
            if event_id:
                self._mark_seen(str(chat_id), str(event_id))
            return SendResult(
                success=bool(data.get("accepted", True)),
                message_id=str(event_id) if event_id else None,
                raw_response=data,
            )
        # Markdown renders in Buzz, so a URL arrives as a clickable image link.
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

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
        since = max(int(state.get("last_ts") or time.time()) - 1, 0)
        request = [
            "REQ",
            subscription_id,
            {"kinds": [_CHAT_KIND], "#h": [channel_id], "since": since},
        ]
        await websocket.send(json.dumps(request, separators=(",", ":")))

    async def _subscribe_websocket(self, websocket) -> Dict[str, Optional[str]]:
        """Subscribe to every watched conversation plus membership events
        (kind 44100 p-tagged to us) for live DM discovery."""
        subscriptions: Dict[str, Optional[str]] = {}
        for index, channel_id in enumerate(list(self._channel_state)):
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

    async def _handle_membership_event(self, websocket, subscriptions: Dict[str, Optional[str]], event: dict) -> None:
        """A membership event p-tagged to us: rediscover conversations and
        subscribe to any new ones (fresh DMs dispatch from their beginning)."""
        self._membership_since = max(self._membership_since, int(event.get("created_at") or 0))
        before = set(self._channel_state)
        await self._discover_dms(seed=False)
        for channel_id in self._channel_state:
            if channel_id in before:
                continue
            subscription_id = f"hermes-buzz-dm-{len(subscriptions)}"
            subscriptions[subscription_id] = channel_id
            await self._send_channel_subscription(websocket, subscription_id, channel_id)
            logger.info("Buzz: subscribed to new conversation %s", channel_id)

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
                        async for raw in websocket:
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
                                    await self._handle_event(channel_id, state, event)
                                    self._trim_seen(state)
                            elif message[0] == "CLOSED":
                                detail = message[-1] if len(message) > 2 else "subscription closed"
                                raise ConnectionError(str(detail))
                            elif message[0] == "NOTICE":
                                logger.warning("Buzz: relay notice: %s", message[-1])
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

    async def _seed_channel(self, channel_id: str, chat_type: str) -> None:
        """Initialize a channel's high-water mark from its newest events."""
        state = {"chat_type": chat_type, "last_ts": 0, "seen": OrderedDict()}
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
        description, so that listing is scanned as a fallback.  Fallback
        finds are watched as ``group`` and latch to ``dm`` via p-tag
        detection (_is_direct_message_event) rather than trusting the name
        alone to unlock the mention-free DM path.
        """
        code, out, _err = await self._run_cli(["dms", "list"])
        if code == 0:
            for dm in _parse_json_list(out):
                dm_id = str(dm.get("dm_id") or "")
                if not dm_id or dm_id in self._channel_state:
                    continue
                if seed:
                    await self._seed_channel(dm_id, chat_type="dm")
                else:
                    self._channel_state[dm_id] = {"chat_type": "dm", "last_ts": 0, "seen": OrderedDict()}
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
            if ch_id in self._channel_state or not self._may_reclassify_as_dm(ch_id):
                continue
            if seed:
                await self._seed_channel(ch_id, chat_type="group")
            else:
                self._channel_state[ch_id] = {"chat_type": "group", "last_ts": 0, "seen": OrderedDict()}

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
        for event in _parse_json_list(out):
            await self._handle_event(channel_id, state, event)
        self._trim_seen(state)

    async def _handle_event(self, channel_id: str, state: dict, event: dict) -> None:
        """De-dupe, filter, and dispatch a single ``messages get`` event."""
        event_id = str(event.get("id") or "")
        created_at = int(event.get("created_at") or 0)
        if not event_id or event_id in state["seen"]:
            return
        state["seen"][event_id] = None
        state["last_ts"] = max(state["last_ts"], created_at)

        if int(event.get("kind") or 0) != _CHAT_KIND:
            return
        pubkey = str(event.get("pubkey") or "").lower()
        content = event.get("content")
        if not pubkey or not isinstance(content, str) or not content.strip():
            return

        # Suppress self-echo: never dispatch our own messages back to the agent.
        if pubkey == self._self_pubkey:
            return

        # Reclassify a leaked DM before gating so its first un-mentioned
        # message both latches the conversation and dispatches.
        self._maybe_latch_dm(channel_id, state, event)

        is_dm = state["chat_type"] == "dm"
        # In shared channels, respond only when addressed — unless
        # require_mention is disabled, in which case respond to every message.
        # DMs always dispatch.
        if not is_dm and self.require_mention and not self._is_mentioned(content):
            return

        # Adapter-level allow-list (the gateway applies BUZZ_ALLOWED_USERS /
        # BUZZ_ALLOW_ALL_USERS centrally as well; empty list = no filter here).
        if self._allowed_pubkeys and pubkey not in self._allowed_pubkeys:
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

        await self._dispatch_message(
            text=dispatch_text,
            chat_id=channel_id,
            chat_type="dm" if is_dm else "group",
            user_id=pubkey,
            user_name=await self._resolve_user_name(pubkey),
            message_id=event_id,
            created_at=created_at,
            thread_id=thread_id,
        )

    # ── DM classification (issue #68871) ──────────────────────────────────
    #
    # ``buzz dms list`` returns [] on some hosted relays even when DM
    # conversations exist, so DMs leak in via ``channels list`` and get
    # watched as chat_type="group" — which wrongly puts them behind the
    # channel mention gate.  Classification therefore keys off the Nostr
    # tags of the messages themselves.  Observed on a live hosted relay:
    #
    #   * every message another user sends IN A DM carries a structural
    #     ["p", <our pubkey>] tag, even when the text never mentions us
    #     (recipient addressing);
    #   * in a real channel, a ["p", <our pubkey>] tag appears only when the
    #     text visibly @mentions us (typed mention, with or without a reply
    #     ["e", ...] tag) — never on plain broadcasts.
    #
    # So "p-tagged to self WITHOUT a visible mention in the content" is the
    # DM discriminator: in a channel that combination does not occur, and a
    # channel reply/mention that p-tags us is excluded because the mention
    # is right there in the text.  As a second, independent guard, a
    # conversation whose ``channels list`` metadata looks like a real
    # community channel (real name / non-empty description) is never
    # reclassified at all, whereas relay-materialized DMs are always named
    # "DM" with an empty description.  Nothing is lost while unlatched: a
    # DM message that DOES mention us dispatches through the mention gate
    # anyway, so the latch flips exactly on the first message that needs it.

    def _may_reclassify_as_dm(self, channel_id: str) -> bool:
        """True when the conversation's metadata does not rule out a DM.

        Known real community channels (real name or non-empty description in
        ``channels list``) must never turn into DMs just because a message
        p-tags us.  A conversation with no metadata at all is trusted only
        when the user did not explicitly configure it as a watched channel.
        """
        meta = self._channel_meta.get(channel_id)
        if meta is None:
            return channel_id not in self.channels
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        return name == "DM" and not description

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
        tags = event.get("tags")
        if not isinstance(tags, list):
            return False
        p_tagged_to_self = any(
            isinstance(tag, (list, tuple))
            and len(tag) > 1
            and tag[0] == "p"
            and str(tag[1]).lower() == self._self_pubkey
            for tag in tags
        )
        if not p_tagged_to_self:
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
        """True when the message addresses this agent (npub, hex, or name)."""
        lowered = content.lower()
        if self._self_pubkey and self._self_pubkey in lowered:
            return True
        if self._self_npub and self._self_npub in lowered:
            return True
        if self._display_name:
            pattern = rf"(?<!\w)@?{re.escape(self._display_name.lower())}(?!\w)"
            if re.search(pattern, lowered):
                return True
        return False

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
            candidates.append(re.escape(self._display_name))
        if self._self_npub:
            candidates.append(re.escape(self._self_npub))
        if self._self_pubkey:
            candidates.append(re.escape(self._self_pubkey))
        if not candidates:
            return text
        # Optional leading '@', one of the identity forms, optional trailing
        # ':' or ',' and surrounding whitespace.
        pattern = rf"^@?(?:{'|'.join(candidates)})[\s:,]*"
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
    ) -> None:
        """Build a MessageEvent and hand it to the base class handler."""
        if not self._message_handler:
            return

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
            message_type=MessageType.TEXT,
            source=source,
            message_id=message_id,
            timestamp=datetime.fromtimestamp(created_at) if created_at else datetime.now(),
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
    media_files: Optional[List[str]] = None,
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
    for path in media_files or []:
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
    except asyncio.CancelledError:
        raise
    except OSError as e:
        return {"error": f"Buzz standalone send failed to launch CLI: {e}"}
    if code != 0:
        return {"error": f"Buzz standalone send failed: {_cli_error_message(err, code)}"}
    try:
        data = json.loads(out or "{}")
    except ValueError:
        data = {}
    return {"success": True, "message_id": str(data.get("event_id") or "")}


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
