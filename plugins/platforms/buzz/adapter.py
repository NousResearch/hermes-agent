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

Or via environment variables (overrides config.yaml):
    BUZZ_RELAY_URL, BUZZ_CHANNELS, BUZZ_HOME_CHANNEL, BUZZ_POLL_INTERVAL,
    BUZZ_CLI_PATH, BUZZ_CREDENTIALS_FILE, BUZZ_ALLOWED_USERS,
    BUZZ_ALLOW_ALL_USERS

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

# Where to look for a credentials JSON (keys: nsec / private_key_hex) when
# BUZZ_PRIVATE_KEY is not set.  Module-level so tests can point it at a tmpdir.
_DEFAULT_CREDENTIALS_DIR = Path("~/.config/buzz").expanduser()


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


def _resolve_private_key(extra: Optional[dict] = None) -> str:
    """Resolve the Nostr private key: env first, then a credentials JSON.

    NEVER log the return value.
    """
    key = os.getenv("BUZZ_PRIVATE_KEY", "").strip()
    if key:
        return key
    configured = os.getenv("BUZZ_CREDENTIALS_FILE", "").strip() or (extra or {}).get("credentials_file", "")
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        try:
            candidates = sorted(_DEFAULT_CREDENTIALS_DIR.glob("*credentials*.json"))
        except OSError:
            candidates = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for field in ("nsec", "private_key_hex", "private_key"):
            value = data.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


async def _exec_buzz(
    cli_path: str,
    args: List[str],
    *,
    relay_url: str,
    private_key: str,
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

        # Connection settings (env vars override config.yaml)
        self.relay_url = (os.getenv("BUZZ_RELAY_URL") or extra.get("relay_url", "")).strip()
        self.cli_path = _resolve_cli_path(
            os.getenv("BUZZ_CLI_PATH", "").strip() or str(extra.get("cli_path", "") or "")
        )

        # Channels to watch: env csv > extra list/csv; empty = all joined channels
        raw_channels = os.getenv("BUZZ_CHANNELS") or extra.get("channels", [])
        if isinstance(raw_channels, str):
            raw_channels = raw_channels.split(",")
        self.channels: List[str] = [c.strip() for c in raw_channels if isinstance(c, str) and c.strip()]

        self.home_channel = (os.getenv("BUZZ_HOME_CHANNEL") or str(extra.get("home_channel", "") or "")).strip()

        try:
            interval = float(os.getenv("BUZZ_POLL_INTERVAL") or extra.get("poll_interval", _DEFAULT_POLL_INTERVAL))
        except (TypeError, ValueError):
            interval = _DEFAULT_POLL_INTERVAL
        self.poll_interval = max(_MIN_POLL_INTERVAL, interval)

        # Whether channel messages must @mention the agent to get a response.
        # Defaults to True (respond only when addressed). Set False to make the
        # agent respond to every message in a watched channel. DMs always
        # dispatch regardless. Env (BUZZ_REQUIRE_MENTION) overrides config.yaml.
        _rm_raw = os.getenv("BUZZ_REQUIRE_MENTION")
        if _rm_raw is None:
            _rm_cfg = extra.get("require_mention", True)
        else:
            _rm_cfg = _rm_raw
        self.require_mention = str(_rm_cfg).strip().lower() not in ("false", "0", "no", "off")

        # Auth: entries may be hex pubkeys or npubs; normalized to hex
        raw_allowed = os.getenv("BUZZ_ALLOWED_USERS") or extra.get("allowed_users", [])
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

        # Identity — filled in by connect() from ``buzz users get``
        self._self_pubkey: str = ""
        self._self_npub: str = ""
        self._display_name: str = ""

        # Runtime state
        self._poll_task: Optional[asyncio.Task] = None
        # channel_id -> {"chat_type", "last_ts", "seen": OrderedDict[event_id, None]}
        self._channel_state: Dict[str, dict] = {}
        self._channel_names: Dict[str, str] = {}
        # channel_id -> raw ``channels list`` entry; drives DM-vs-channel
        # classification (see _may_reclassify_as_dm).
        self._channel_meta: Dict[str, dict] = {}
        self._user_names: Dict[str, str] = {}
        self._poll_count = 0

    @property
    def name(self) -> str:
        return "Buzz"

    # ── buzz-cli plumbing ─────────────────────────────────────────────────

    async def _run_cli(self, args: List[str], *, input_text: Optional[str] = None) -> Tuple[int, str, str]:
        if not self._private_key:
            self._private_key = _resolve_private_key(self._extra)
        return await _exec_buzz(
            self.cli_path,
            args,
            relay_url=self.relay_url,
            private_key=self._private_key,
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
        self._private_key = _resolve_private_key(self._extra)
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

        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info(
            "Buzz: connected to %s as %s, watching %d channel(s), poll interval %.1fs",
            self.relay_url,
            self._display_name or self._self_npub[:16],
            len(self._channel_state),
            self.poll_interval,
        )
        return True

    async def disconnect(self) -> None:
        """Stop the poll loop and drop runtime state."""
        self._mark_disconnected()
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
        reply_target = reply_to or (metadata or {}).get("thread_id")
        if reply_target:
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
            if reply_to:
                args += ["--reply-to", str(reply_to)]
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

        await self._dispatch_message(
            text=dispatch_text,
            chat_id=channel_id,
            chat_type="dm" if is_dm else "group",
            user_id=pubkey,
            user_name=await self._resolve_user_name(pubkey),
            message_id=event_id,
            created_at=created_at,
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
        """Resolve a pubkey to a display name (cached; falls back to npub prefix)."""
        cached = self._user_names.get(pubkey)
        if cached:
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

    async def _dispatch_message(
        self,
        text: str,
        chat_id: str,
        chat_type: str,
        user_id: str,
        user_name: str,
        message_id: str,
        created_at: int,
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

def check_requirements() -> bool:
    """Check if Buzz is configured: a relay URL plus a resolvable key."""
    if not os.getenv("BUZZ_RELAY_URL", "").strip():
        return False
    return bool(_resolve_private_key())


def validate_config(config) -> bool:
    """Validate that the platform config has enough info to connect."""
    extra = getattr(config, "extra", {}) or {}
    relay = os.getenv("BUZZ_RELAY_URL") or extra.get("relay_url", "")
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
    _str_keys = {
        "relay_url": "BUZZ_RELAY_URL",
        "cli_path": "BUZZ_CLI_PATH",
        "home_channel": "BUZZ_HOME_CHANNEL",
    }
    for src, env in _str_keys.items():
        val = extra.get(src)
        if val and not os.getenv(env):
            os.environ[env] = str(val)
    interval = extra.get("poll_interval")
    if interval is not None and not os.getenv("BUZZ_POLL_INTERVAL"):
        os.environ["BUZZ_POLL_INTERVAL"] = str(interval)
    channels = extra.get("channels")
    if channels is not None and not os.getenv("BUZZ_CHANNELS"):
        if isinstance(channels, (list, tuple)):
            channels = ",".join(str(c) for c in channels)
        os.environ["BUZZ_CHANNELS"] = str(channels)
    allowed = extra.get("allowed_users")
    if allowed is not None and not os.getenv("BUZZ_ALLOWED_USERS"):
        if isinstance(allowed, (list, tuple)):
            allowed = ",".join(str(a) for a in allowed)
        os.environ["BUZZ_ALLOWED_USERS"] = str(allowed)
    if "allow_all_users" in extra and not os.getenv("BUZZ_ALLOW_ALL_USERS"):
        os.environ["BUZZ_ALLOW_ALL_USERS"] = str(extra["allow_all_users"]).lower()
    if "require_mention" in extra and not os.getenv("BUZZ_REQUIRE_MENTION"):
        os.environ["BUZZ_REQUIRE_MENTION"] = str(extra["require_mention"]).lower()
    return None


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called BEFORE adapter construction so env-only setups show up in
    ``hermes gateway status`` and ``get_connected_platforms()``.  Returns
    ``None`` when Buzz isn't minimally configured.

    The special ``home_channel`` key is handled by the core hook — it becomes
    a proper ``HomeChannel`` on the ``PlatformConfig``.
    """
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
    relay = (os.getenv("BUZZ_RELAY_URL") or extra.get("relay_url", "")).strip()
    private_key = _resolve_private_key(extra)
    cli_path = _resolve_cli_path(
        os.getenv("BUZZ_CLI_PATH", "").strip() or str(extra.get("cli_path", "") or "")
    )
    if not relay or not private_key:
        return {"error": "Buzz standalone send: BUZZ_RELAY_URL and BUZZ_PRIVATE_KEY must be configured"}
    if not cli_path:
        return {"error": "Buzz standalone send: buzz CLI binary not found"}
    target = (chat_id or "").strip() or (os.getenv("BUZZ_HOME_CHANNEL") or str(extra.get("home_channel", "") or "")).strip()
    if not target:
        return {"error": "Buzz standalone send: no target channel (set BUZZ_HOME_CHANNEL)"}

    args = ["messages", "send", "--channel", target, "--content", "-"]
    if thread_id:
        args += ["--reply-to", str(thread_id)]
    for path in media_files or []:
        args += ["--file", str(path)]
    try:
        code, out, err = await _exec_buzz(
            cli_path, args, relay_url=relay, private_key=private_key, input_text=message
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
