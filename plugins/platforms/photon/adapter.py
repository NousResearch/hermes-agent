"""
Photon Spectrum (iMessage) platform adapter for Hermes Agent.

Both directions of traffic flow through a small supervised Node sidecar
(see ``sidecar/index.mjs``) that runs the ``spectrum-ts`` SDK — the SDK is
TypeScript-only and there is no public HTTP message API, so a sidecar is
unavoidable.

Inbound:
    The SDK's ``app.messages`` is a long-lived **gRPC** stream. The sidecar
    serializes each message to a normalized JSON event and streams it to this
    adapter over a loopback ``GET /inbound`` (NDJSON). A background task here
    consumes that stream, dedupes on ``messageId``, and dispatches a
    ``MessageEvent`` to the gateway via ``BasePlatformAdapter.handle_message``.
    No webhook, no public URL, no signing secret.

Outbound:
    ``send`` / ``send_typing`` are loopback POSTs to the sidecar's control
    endpoints, authenticated with a shared bearer token.  Outbound media
    (images, voice notes, video, documents) goes through spectrum-ts'
    ``attachment()`` / ``voice()`` content builders via the sidecar's
    ``/send-attachment`` endpoint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # Type checkers see ``httpx`` as the always-imported module, so every use
    # site type-checks cleanly. The runtime fallback below keeps the optional
    # dependency truly optional (each use site is guarded by HTTPX_AVAILABLE).
    import httpx
    HTTPX_AVAILABLE = True
else:
    try:
        import httpx
        HTTPX_AVAILABLE = True
    except ImportError:  # pragma: no cover - httpx is already a Hermes dep
        HTTPX_AVAILABLE = False
        httpx = None

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.platforms.helpers import strip_markdown

from .auth import load_project_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants

_DEFAULT_SIDECAR_PORT = 8789
_DEFAULT_SIDECAR_BIND = "127.0.0.1"

# Photon iMessage messages from the SDK side have no documented hard
# limit, but the underlying iMessage protocol limits practical message
# size to ~16 KB.  Keep a conservative cap that matches BlueBubbles.
_MAX_MESSAGE_LENGTH = 8000

# Dedup parameters — the gRPC stream is at-least-once, and a sidecar
# reconnect can replay, so keep at least 1k ids for ~48h.
_DEDUP_MAX_SIZE = 4000
_DEDUP_WINDOW_SECONDS = 48 * 3600

_SIDECAR_DIR = Path(__file__).parent / "sidecar"

# Photon / Envoy / spectrum-ts error substrings that indicate a transient
# upstream overload rather than a permanent failure.  These are not in the
# core _RETRYABLE_ERROR_PATTERNS because they are specific to this adapter.
_PHOTON_RETRYABLE_PATTERNS = (
    "internal sidecar error",
    "upstream connect error",
    "reset reason: overflow",
)

# Minimum seconds between typing-indicator calls for the same chat.
# iMessage is a personal channel — suppressing rapid repeats reduces
# upstream gRPC pressure during Photon overflow events.
_TYPING_COOLDOWN_SECONDS = 5.0

# Group-chat mention wake words. When ``require_mention`` is enabled, group
# messages are ignored unless they match one of these patterns — same
# behavior and defaults as the BlueBubbles iMessage channel so the two
# iMessage adapters gate group chats identically.
_DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]


# ---------------------------------------------------------------------------
# Module-level helpers — also used by check_fn / standalone send

def _coerce_port(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_requirements() -> bool:
    """Return True when both Python deps and the Node sidecar are available."""
    if not HTTPX_AVAILABLE:
        return False
    if not shutil.which(os.getenv("PHOTON_NODE_BIN") or "node"):
        return False
    if not (_SIDECAR_DIR / "node_modules").exists():
        # spectrum-ts not installed yet — `hermes photon setup` will
        # install it.  check_fn still returns False so the gateway
        # surfaces the missing-deps state in `hermes setup` / status.
        return False
    return True


def validate_config(cfg: PlatformConfig) -> bool:
    extra = cfg.extra or {}
    project_id = extra.get("project_id") or os.getenv("PHOTON_PROJECT_ID")
    project_secret = extra.get("project_secret") or os.getenv("PHOTON_PROJECT_SECRET")
    if not project_id or not project_secret:
        # Fall back to auth.json
        stored_id, stored_sec = load_project_credentials()
        return bool(stored_id and stored_sec)
    return True


def is_connected(cfg: PlatformConfig) -> bool:
    return validate_config(cfg)


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so env-only setups appear in status.

    The special ``home_channel`` key is handled by the core plugin hook and
    becomes a proper ``HomeChannel`` on ``PlatformConfig``.
    """
    project_id, project_secret = load_project_credentials()
    if not (project_id and project_secret):
        return None
    seed: dict = {"project_id": project_id, "project_secret": project_secret}
    home = os.getenv("PHOTON_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("PHOTON_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


def _markdown_enabled() -> bool:
    """Send agent replies as markdown (spectrum-ts ``markdown()`` builder).

    iMessage renders it natively; other Spectrum platforms degrade to
    readable plain text. On-device rendering can't be unit-tested, so
    ``PHOTON_MARKDOWN=false`` is the kill-switch back to stripped plain
    text without a release.
    """
    return os.getenv("PHOTON_MARKDOWN", "true").strip().lower() not in {
        "false", "0", "no",
    }


# ---------------------------------------------------------------------------
# Adapter

class PhotonAdapter(BasePlatformAdapter):
    """Bidirectional bridge to Photon Spectrum via the Node spectrum-ts sidecar.

    Inbound: consume the sidecar's ``/inbound`` gRPC stream.
    Outbound: loopback POSTs to the sidecar's control channel.
    """

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("photon"))
        extra = config.extra or {}

        # Project credentials (env wins, then config.extra, then auth.json).
        # ``project_id`` here is the project's spectrumProjectId — the value
        # the spectrum-ts SDK authenticates with.
        stored_id, stored_sec = load_project_credentials()
        self._project_id: str = (
            os.getenv("PHOTON_PROJECT_ID")
            or extra.get("project_id")
            or stored_id
            or ""
        )
        self._project_secret: str = (
            os.getenv("PHOTON_PROJECT_SECRET")
            or extra.get("project_secret")
            or stored_sec
            or ""
        )

        # Sidecar
        self._sidecar_port = _coerce_port(
            extra.get("sidecar_port") or os.getenv("PHOTON_SIDECAR_PORT"),
            _DEFAULT_SIDECAR_PORT,
        )
        self._sidecar_bind = _DEFAULT_SIDECAR_BIND
        self._sidecar_token = (
            os.getenv("PHOTON_SIDECAR_TOKEN") or secrets.token_hex(16)
        )
        self._autostart_sidecar = str(
            os.getenv("PHOTON_SIDECAR_AUTOSTART", "true")
        ).lower() not in ("0", "false", "no")
        self._node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node") or "node"

        # With markdown on, format_message preserves fences and the sidecar's
        # markdown() builder renders them (or degrades them readably).
        self.supports_code_blocks = _markdown_enabled()

        # Runtime state
        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._sidecar_supervisor_task: Optional[asyncio.Task] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._inbound_running = False
        self._http_client: Optional["httpx.AsyncClient"] = None
        # Lightweight in-memory dedup. The gRPC stream is at-least-once, so we
        # may see the same messageId more than once (e.g. after a reconnect).
        self._seen_messages: Dict[str, float] = {}
        # Ids of messages WE sent (bounded, insertion-order eviction). Inbound
        # reaction events are only routed to the agent when they target one of
        # these — a tapback on a human↔human message is not addressed to us.
        self._sent_message_ids: Dict[str, float] = {}
        # Latest inbound message id per chat (bounded). Lets the agent-facing
        # react action default to "the message that triggered me" without
        # requiring the model to thread message ids through tool calls.
        self._last_inbound_by_chat: Dict[str, str] = {}
        # Last time we sent a typing indicator per chat, for cooldown gating.
        self._typing_last_sent: Dict[str, float] = {}

        # Group-chat mention gating (parity with BlueBubbles). When enabled,
        # group messages are ignored unless they match a wake word; DMs are
        # always processed. Config key wins, then env var.
        _require_mention = extra.get("require_mention")
        if _require_mention is None:
            _require_mention = os.getenv("PHOTON_REQUIRE_MENTION")
        self.require_mention = str(_require_mention).strip().lower() in {
            "true", "1", "yes", "on",
        }
        self._mention_patterns = self._compile_mention_patterns(
            extra["mention_patterns"]
            if "mention_patterns" in extra
            else os.getenv("PHOTON_MENTION_PATTERNS")
        )

    # -- Group-mention gating (parity with BlueBubbles) -------------------

    @staticmethod
    def _compile_mention_patterns(raw: Any) -> "list[re.Pattern]":
        """Compile group-mention wake words from config/env.

        ``raw`` is a list (config or env JSON), a string (env var: JSON
        list, or comma/newline-separated), or None (use Hermes defaults).
        Mirrors the BlueBubbles implementation so both iMessage channels
        accept the same configuration shapes.
        """
        if raw is None:
            patterns = list(_DEFAULT_MENTION_PATTERNS)
        elif isinstance(raw, str):
            text = raw.strip()
            try:
                loaded = json.loads(text) if text else []
            except Exception:
                loaded = None
            patterns = loaded if isinstance(loaded, list) else [
                part.strip()
                for line in text.splitlines()
                for part in line.split(",")
            ]
        elif isinstance(raw, list):
            patterns = raw
        else:
            patterns = [raw]

        compiled: "list[re.Pattern]" = []
        for pattern in patterns:
            text = str(pattern).strip()
            if not text:
                continue
            try:
                compiled.append(re.compile(text, re.IGNORECASE))
            except re.error as exc:
                logger.warning("[photon] Invalid mention pattern %r: %s", text, exc)
        return compiled

    def _message_matches_mention_patterns(self, text: str) -> bool:
        if not text or not self._mention_patterns:
            return False
        return any(pattern.search(text) for pattern in self._mention_patterns)

    def _clean_mention_text(self, text: str) -> str:
        """Strip a leading wake word before dispatch.

        Custom mention patterns are regexes, so we only strip a leading
        match to avoid deleting ordinary words later in the prompt.
        """
        if not text:
            return text
        for pattern in self._mention_patterns:
            match = pattern.match(text.lstrip())
            if match:
                cleaned = text.lstrip()[match.end():].lstrip(" ,:-")
                return cleaned or text
        return text

    # -- Connection lifecycle ---------------------------------------------

    async def connect(self) -> bool:
        if not HTTPX_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP", "httpx not installed", retryable=False
            )
            return False
        if not self._project_id or not self._project_secret:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "PHOTON_PROJECT_ID and PHOTON_PROJECT_SECRET are required. "
                "Run: hermes photon setup",
                retryable=False,
            )
            return False

        client = httpx.AsyncClient(timeout=30.0)
        self._http_client = client

        # The sidecar holds the gRPC stream for BOTH directions, so it is
        # required now (not just for outbound).
        if self._autostart_sidecar:
            try:
                await self._start_sidecar()
            except Exception as e:
                self._set_fatal_error(
                    "SIDECAR_FAILED",
                    f"failed to start Photon sidecar: {e}",
                    retryable=True,
                )
                await client.aclose()
                self._http_client = None
                return False
        else:
            logger.warning(
                "[photon] sidecar autostart disabled — inbound + outbound will fail"
            )

        # Start consuming the inbound gRPC stream from the sidecar.
        self._inbound_running = True
        self._inbound_task = asyncio.get_event_loop().create_task(
            self._inbound_loop()
        )

        self._mark_connected()
        logger.info(
            "[photon] connected — sidecar on %s:%d, streaming inbound over gRPC",
            self._sidecar_bind, self._sidecar_port,
        )
        return True

    async def disconnect(self) -> None:
        self._inbound_running = False
        if self._inbound_task is not None:
            self._inbound_task.cancel()
            try:
                await self._inbound_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._inbound_task = None
        await self._stop_sidecar()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._mark_disconnected()

    # -- Inbound stream consumer ------------------------------------------

    async def _inbound_loop(self) -> None:
        """Consume the sidecar's ``/inbound`` NDJSON stream, with reconnect.

        The sidecar owns the gRPC reconnect/heartbeat to Photon; this loop
        only has to re-open the loopback HTTP stream if it drops (e.g. the
        sidecar restarts).
        """
        client = self._http_client
        if client is None:
            return
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}/inbound"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        backoff = 1.0
        while self._inbound_running:
            try:
                async with client.stream(
                    "GET", url, headers=headers, timeout=None,
                ) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"/inbound returned {resp.status_code}")
                    backoff = 1.0  # reset on a successful connect
                    async for line in resp.aiter_lines():
                        if not self._inbound_running:
                            break
                        line = line.strip()
                        if not line:
                            continue  # heartbeat
                        await self._on_inbound_line(line)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._inbound_running:
                    break
                logger.warning(
                    "[photon] inbound stream dropped (%s); reconnecting in %.1fs",
                    e, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _on_inbound_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("[photon] skipping non-JSON inbound line")
            return
        msg_id = event.get("messageId")
        if msg_id and self._is_duplicate(msg_id):
            return
        try:
            await self._dispatch_inbound(event)
        except Exception:
            logger.exception("[photon] inbound dispatch failed")

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        seen = self._seen_messages
        t = seen.get(msg_id)
        if t is not None and now - t < _DEDUP_WINDOW_SECONDS:
            return True  # seen, unexpired
        # New or expired: record and enforce a HARD size bound (evict oldest,
        # insertion-order) so a burst of unique ids within the window can't grow
        # the dict without limit — not just the expired-only prune.
        if msg_id in seen:
            del seen[msg_id]  # refresh insertion order
        seen[msg_id] = now
        if len(seen) > _DEDUP_MAX_SIZE:
            for old in list(seen.keys())[: len(seen) - _DEDUP_MAX_SIZE]:
                del seen[old]
        return False

    async def _dispatch_inbound(self, event: Dict[str, Any]) -> None:
        """Normalize a sidecar inbound event and dispatch it to the gateway.

        Event shape (from ``sidecar/index.mjs``)::

            {
              "messageId": "...",
              "platform": "iMessage",
              "space": {"id": "...", "type": "dm"|"group", "phone": "+E164"},
              "sender": {"id": "+E164"},
              "content": {"type": "text", "text": "..."}
                       | {"type": "attachment"|"voice", "id", "name",
                          "mimeType", "size", "duration"?, "data"?,
                          "encoding"?}
                       | {"type": "reaction", "emoji": "❤️",
                          "targetMessageId": "..." | null,
                          "targetDirection": "inbound"|"outbound" | null},
              "timestamp": "2026-05-14T19:06:32.000Z"

        Attachment and voice content carry the bytes inline as base64 ``data``
        (with ``encoding == "base64"``) when the sidecar could read them
        within its size cap; otherwise only metadata is present and we surface
        a marker.
            }
        """
        space = event.get("space") or {}
        sender = event.get("sender") or {}
        content = event.get("content") or {}

        space_id = space.get("id") or ""
        if not space_id:
            logger.warning("[photon] inbound missing space.id")
            return

        # iMessage spaces carry their type directly — no id string-sniffing.
        chat_type = "group" if space.get("type") == "group" else "dm"
        sender_id = sender.get("id") or space.get("phone") or space_id

        ts_str = event.get("timestamp") or ""
        try:
            timestamp = (
                datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_str
                else datetime.now(tz=timezone.utc)
            )
        except ValueError:
            timestamp = datetime.now(tz=timezone.utc)

        # Media attachments (local cached paths) handed to the agent via the
        # gateway's image-routing path, exactly like the BlueBubbles channel.
        media_urls: List[str] = []
        media_types: List[str] = []

        def _normalize_binary_payload(
            payload: Dict[str, Any]
        ) -> tuple[str, MessageType, List[str], List[str]]:
            is_voice = payload.get("type") == "voice"
            name = payload.get("name") or ("voice" if is_voice else "(unnamed)")
            mime = payload.get("mimeType") or ""
            mtype = MessageType.VOICE if is_voice else _attachment_message_type(mime)
            cached = _cache_inbound_attachment(
                payload, name, mime, force_audio=is_voice
            )
            if cached:
                return (
                    "(voice)" if is_voice else "(attachment)",
                    mtype,
                    [cached],
                    [mime or ("audio/mp4" if is_voice else "application/octet-stream")],
                )
            label = "voice" if is_voice else "attachment"
            duration = payload.get("duration")
            duration_text = (
                f", duration: {duration}s"
                if isinstance(duration, (int, float))
                else ""
            )
            return (
                f"[Photon {label} received: {name} "
                f"({mime or 'unknown MIME'}{duration_text})]",
                mtype,
                [],
                [],
            )

        ctype = content.get("type")
        if ctype == "reaction":
            # Route only tapbacks on messages WE sent — those are implicitly
            # addressed to the bot (feishu precedent: synthetic text event).
            # Reactions on human↔human messages are not for us. Checked before
            # the mention gate: a tapback never carries a wake word.
            target_id = content.get("targetMessageId")
            is_ours = content.get("targetDirection") == "outbound" or (
                target_id and target_id in self._sent_message_ids
            )
            if not is_ours:
                logger.debug(
                    "[photon] ignoring reaction on a message we didn't send"
                )
                return
            emoji = content.get("emoji") or ""
            source = self.build_source(
                chat_id=space_id,
                chat_name=space_id,
                chat_type=chat_type,
                user_id=sender_id,
                user_name=sender_id or None,
            )
            await self.handle_message(
                MessageEvent(
                    text=f"reaction:added:{emoji}",
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=event.get("messageId"),
                    raw_message=event,
                    timestamp=timestamp,
                )
            )
            return
        # Anything past here is a real (reactable) message — remember it as
        # the chat's latest inbound so `add_reaction` can target it when the
        # caller doesn't pass an explicit message id. Recorded before the
        # mention gate: a reaction to a non-wake-word group message is valid.
        self._record_last_inbound(space_id, event.get("messageId"))
        if ctype == "text":
            text = content.get("text") or ""
            mtype = MessageType.TEXT
        elif ctype in {"attachment", "voice"}:
            text, mtype, media_urls, media_types = _normalize_binary_payload(content)
        elif ctype == "group":
            text_parts: List[str] = []
            mtype = MessageType.TEXT
            for item in content.get("items") or []:
                if not isinstance(item, dict):
                    continue
                item_content = item.get("content") or {}
                if not isinstance(item_content, dict):
                    continue
                item_type = item_content.get("type")
                if item_type == "text":
                    item_text = item_content.get("text") or ""
                    if item_text:
                        text_parts.append(item_text)
                    continue
                if item_type in {"attachment", "voice"}:
                    marker, item_mtype, item_urls, item_types = _normalize_binary_payload(
                        item_content
                    )
                    if mtype == MessageType.TEXT:
                        mtype = item_mtype
                    media_urls.extend(item_urls)
                    media_types.extend(item_types)
                    if not item_urls:
                        text_parts.append(marker)
                    continue
                if item_type:
                    text_parts.append(f"[Photon content type not handled: {item_type}]")
            if media_urls and mtype == MessageType.TEXT:
                mtype = MessageType.DOCUMENT
            text = "\n".join(part for part in text_parts if part).strip()
            if not text:
                text = "(attachment)" if media_urls else "[Photon empty group received]"
        else:
            text = f"[Photon content type not handled: {ctype}]"
            mtype = MessageType.TEXT

        # Group-mention gating (parity with BlueBubbles). In group chats with
        # require_mention enabled, drop messages that don't hit a wake word;
        # strip the leading wake word from the ones that do. DMs are never
        # gated.
        if chat_type == "group" and self.require_mention:
            if not self._message_matches_mention_patterns(text):
                logger.debug(
                    "[photon] ignoring group message "
                    "(require_mention=true, no mention pattern matched)"
                )
                return
            text = self._clean_mention_text(text)

        source = self.build_source(
            chat_id=space_id,
            chat_name=space_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_id or None,
        )
        message_event = MessageEvent(
            text=text,
            message_type=mtype,
            source=source,
            message_id=event.get("messageId"),
            raw_message=event,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
        )
        await self.handle_message(message_event)

    # -- Sidecar lifecycle -------------------------------------------------

    @staticmethod
    def _find_listener_pids(port: int) -> List[int]:
        """PIDs listening on a local TCP port (empty if none/undeterminable)."""
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [int(tok) for tok in out.stdout.split() if tok.strip().isdigit()]

    @staticmethod
    def _pid_is_sidecar(pid: int) -> bool:
        """True if ``pid``'s command line is a Photon sidecar process."""
        try:
            out = subprocess.run(  # noqa: S603, S607
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        # Checkout-agnostic: any Hermes checkout's sidecar entry point.
        return "photon/sidecar/index.mjs" in out.stdout

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)  # windows-footgun: ok — only called from _reap_stale_sidecar which win32-guards early
            return True
        except OSError:
            return False

    async def _reap_stale_sidecar(self) -> None:
        """Kill an orphaned sidecar squatting our port before spawning ours.

        A hard gateway exit (crash, SIGKILL, supervisor restart) used to leave
        the detached sidecar running with a token the new gateway doesn't
        know, so it can't be told to ``/shutdown`` — and every replacement
        spawn died on EADDRINUSE, failing each reconnect attempt. The
        stdin-EOF watch prevents new orphans; this reclaims the port from
        orphans that predate it (or survived it). Listeners are verified by
        command line before being signalled.
        """
        if sys.platform == "win32":  # lsof/ps; orphaning is a POSIX-only path
            return
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    f"http://{self._sidecar_bind}:{self._sidecar_port}/healthz",
                    headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                )
        except httpx.RequestError:
            return  # nothing listening — the normal case
        pids = self._find_listener_pids(self._sidecar_port)
        stale = [pid for pid in pids if self._pid_is_sidecar(pid)]
        foreign = [pid for pid in pids if pid not in stale]
        if not stale:
            raise RuntimeError(
                f"port {self._sidecar_port} is in use by another process "
                f"(pids: {foreign or 'unknown'}, not a Photon sidecar) — "
                f"free it or set PHOTON_SIDECAR_PORT to a different port"
            )
        for pid in stale:
            logger.warning(
                "[photon] reaping orphaned sidecar (pid %d) on port %d",
                pid, self._sidecar_port,
            )
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.time() + 3.0
        while time.time() < deadline and any(self._pid_alive(p) for p in stale):
            await asyncio.sleep(0.1)
        for pid in stale:
            if self._pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)  # windows-footgun: ok — unreachable on win32 (early return above)
                except OSError:
                    pass
        # Give the OS a beat to release the listening socket.
        await asyncio.sleep(0.2)
        if foreign:
            raise RuntimeError(
                f"port {self._sidecar_port} is also held by non-sidecar "
                f"processes (pids: {foreign}) — free it or set "
                f"PHOTON_SIDECAR_PORT to a different port"
            )

    async def _start_sidecar(self) -> None:
        if not (_SIDECAR_DIR / "node_modules").exists():
            raise RuntimeError(
                f"Photon sidecar deps not installed. Run: "
                f"cd {_SIDECAR_DIR} && npm install   (or `hermes photon setup`)"
            )
        await self._reap_stale_sidecar()

        env = os.environ.copy()
        env["PHOTON_PROJECT_ID"] = self._project_id
        env["PHOTON_PROJECT_SECRET"] = self._project_secret
        env["PHOTON_SIDECAR_PORT"] = str(self._sidecar_port)
        env["PHOTON_SIDECAR_BIND"] = self._sidecar_bind
        env["PHOTON_SIDECAR_TOKEN"] = self._sidecar_token
        # The sidecar exits when its stdin (the pipe below) hits EOF, so a
        # gateway death of ANY kind — including SIGKILL, where disconnect()
        # never runs — can't leave it orphaned on the port.
        env["PHOTON_SIDECAR_WATCH_STDIN"] = "1"

        try:
            patch = subprocess.run(  # noqa: S603
                [
                    self._node_bin,
                    str(_SIDECAR_DIR / "patch-spectrum-mixed-attachments.mjs"),
                    str(_SIDECAR_DIR),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if patch.returncode != 0:
                raise RuntimeError((patch.stderr or patch.stdout or "").strip())
            if patch.stderr.strip():
                logger.debug("[photon] %s", patch.stderr.strip())
        except Exception as exc:
            logger.warning(
                "[photon] failed to apply Spectrum mixed attachment patch: %s",
                exc,
            )

        self._sidecar_proc = subprocess.Popen(  # noqa: S603
            [self._node_bin, str(_SIDECAR_DIR / "index.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(sys.platform != "win32"),
        )

        # Pump sidecar stderr/stdout into our logger so users see crashes.
        loop = asyncio.get_event_loop()
        self._sidecar_supervisor_task = loop.create_task(
            self._supervise_sidecar(self._sidecar_proc)
        )

        # Wait for /healthz to come up. The 4.x WS transport's cold
        # handshake to fusor-ws can take ~17s, so allow a 45s margin.
        deadline = time.time() + 45.0
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if self._sidecar_proc.poll() is not None:
                    raise RuntimeError(
                        f"Photon sidecar exited with code "
                        f"{self._sidecar_proc.returncode} before becoming ready"
                    )
                try:
                    resp = await client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/healthz",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                    )
                    if resp.status_code == 200:
                        return
                except httpx.RequestError as e:
                    last_err = e
                await asyncio.sleep(0.2)
        raise RuntimeError(
            f"Photon sidecar did not become ready within 45s: {last_err}"
        )

    async def _supervise_sidecar(self, proc: subprocess.Popen) -> None:
        """Pump the sidecar's stdout/stderr into our logger."""
        if proc.stdout is None:  # subprocess was launched without stdout=PIPE
            return
        stdout = proc.stdout
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, stdout.readline)
                if not line:
                    break
                logger.info("[photon-sidecar] %s", line.decode("utf-8", "replace").rstrip())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[photon-sidecar] supervisor exited: %s", e)
        if self._inbound_running:
            exit_code = proc.poll()
            logger.error(
                "[photon] sidecar exited unexpectedly (code %s) — triggering reconnect",
                exit_code,
            )
            self._set_fatal_error(
                "SIDECAR_CRASHED",
                f"Photon sidecar exited unexpectedly (code {exit_code})",
                retryable=True,
            )
            try:
                await self._notify_fatal_error()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[photon] fatal-error notification failed: %s", exc)

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        if proc is None:
            return
        try:
            # Closing our end of the stdin pipe is itself a shutdown signal
            # (the sidecar watches for EOF), and covers the case where the
            # HTTP call below can't get through.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            # Polite shutdown first.
            if self._http_client is not None:
                try:
                    await self._http_client.post(
                        f"http://{self._sidecar_bind}:{self._sidecar_port}/shutdown",
                        headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
                        timeout=2.0,
                    )
                except Exception:
                    pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                if sys.platform != "win32":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # windows-footgun: ok
                    except (ProcessLookupError, PermissionError):
                        proc.terminate()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            self._sidecar_proc = None
            if self._sidecar_supervisor_task is not None:
                self._sidecar_supervisor_task.cancel()
                self._sidecar_supervisor_task = None

    # -- Outbound ----------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        return await self._sidecar_send(chat_id, self.format_message(content))

    # -- Outbound media (parity with the BlueBubbles iMessage channel) -----
    #
    # Photon ships outbound attachments via spectrum-ts' `attachment()` /
    # `voice()` content builders. The sidecar's `/send-attachment` endpoint
    # wraps `space.send(attachment(path, {...}))`. These overrides mirror
    # BlueBubbles: URL-based helpers cache to a local path first, file-based
    # helpers pass the path straight through.

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            from gateway.platforms.base import cache_image_from_url

            local_path = await cache_image_from_url(image_url)
        except Exception:
            # Couldn't fetch the URL — fall back to sending it as text.
            return await super().send_image(chat_id, image_url, caption, reply_to)
        return await self._sidecar_send_attachment(
            chat_id, local_path, caption=caption,
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._sidecar_send_attachment(
            chat_id, image_path, caption=caption,
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
        return await self._sidecar_send_attachment(
            chat_id, audio_path, caption=caption, kind="voice",
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
        return await self._sidecar_send_attachment(
            chat_id, video_path, caption=caption,
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
        return await self._sidecar_send_attachment(
            chat_id, file_path, name=file_name, caption=caption,
        )

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # iMessage renders GIFs inline as ordinary image attachments.
        return await self.send_image(
            chat_id, animation_url, caption, reply_to, metadata,
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        now = time.time()
        if now - self._typing_last_sent.get(chat_id, 0.0) < _TYPING_COOLDOWN_SECONDS:
            return
        self._typing_last_sent[chat_id] = now
        try:
            await self._sidecar_call(
                "/typing", {"spaceId": chat_id, "state": "start"}
            )
        except Exception as e:
            logger.debug("[photon] send_typing failed: %s", e)

    async def stop_typing(self, chat_id: str) -> None:
        self._typing_last_sent.pop(chat_id, None)
        try:
            await self._sidecar_call(
                "/typing", {"spaceId": chat_id, "state": "stop"}
            )
        except Exception as e:
            logger.debug("[photon] stop_typing failed: %s", e)

    # -- Reactions (tapbacks) -----------------------------------------------
    #
    # Same lifecycle-hook pattern as Telegram/Discord: 👀 while processing,
    # swapped for 👍/👎 on completion. Opt-in via PHOTON_REACTIONS — iMessage
    # is a personal-texting channel, and a tapback on every text is noisy.

    _SENT_IDS_MAX = 1000
    _LAST_INBOUND_CHATS_MAX = 200

    def _record_sent_message(self, message_id: Optional[str]) -> None:
        if not message_id:
            return
        sent = self._sent_message_ids
        if message_id in sent:
            del sent[message_id]  # refresh insertion order
        sent[message_id] = time.time()
        if len(sent) > self._SENT_IDS_MAX:
            for old in list(sent.keys())[: len(sent) - self._SENT_IDS_MAX]:
                del sent[old]

    # A DM space is addressable two ways — the chat GUID (`any;-;+1555...`)
    # that inbound events carry, and the bare E.164 phone that home-channel
    # config typically uses. The sidecar's resolveSpace treats them as the
    # same space; normalize to the bare phone so the last-inbound tracker
    # does too (mirrors phoneTargetFromSpaceId in sidecar/index.mjs).
    _DM_CHAT_GUID_RE = re.compile(r"^any;-;(\+\d{6,})$")

    @classmethod
    def _normalize_chat_key(cls, chat_id: str) -> str:
        match = cls._DM_CHAT_GUID_RE.match(chat_id)
        return match.group(1) if match else chat_id

    def _record_last_inbound(
        self, chat_id: Optional[str], message_id: Optional[str]
    ) -> None:
        if not chat_id or not message_id:
            return
        key = self._normalize_chat_key(chat_id)
        last = self._last_inbound_by_chat
        if key in last:
            del last[key]  # refresh insertion order
        last[key] = message_id
        if len(last) > self._LAST_INBOUND_CHATS_MAX:
            for old in list(last.keys())[
                : len(last) - self._LAST_INBOUND_CHATS_MAX
            ]:
                del last[old]

    def _reactions_enabled(self) -> bool:
        return os.getenv("PHOTON_REACTIONS", "false").strip().lower() in {
            "true", "1", "yes", "on",
        }

    async def _add_reaction(
        self, chat_id: str, message_id: str, emoji: str
    ) -> bool:
        """Tapback ``emoji`` onto a message. Soft-fails (False), never raises."""
        try:
            await self._sidecar_call(
                "/react",
                {"spaceId": chat_id, "messageId": message_id, "emoji": emoji},
            )
            return True
        except Exception as e:
            logger.debug("[photon] add_reaction failed: %s", e)
            return False

    async def _remove_reaction(self, chat_id: str, message_id: str) -> bool:
        """Retract our tapback from a message. Soft-fails (False), never raises.

        The sidecar tracks one reaction handle per target message; after a
        sidecar restart the handle is gone and removal is best-effort (the
        stale tapback self-heals when the next reaction replaces it).
        """
        try:
            await self._sidecar_call(
                "/unreact", {"spaceId": chat_id, "messageId": message_id},
            )
            return True
        except Exception as e:
            logger.debug("[photon] remove_reaction failed: %s", e)
            return False

    # -- Agent-facing reactions (send_message action="react") ---------------
    #
    # Unlike the lifecycle hooks below, these are deliberate agent intents,
    # so they are NOT gated by PHOTON_REACTIONS (that env var exists to mute
    # the automatic per-message tapback noise, not explicit requests).

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tapback ``emoji`` onto a message in ``chat_id``.

        Without ``message_id``, targets the chat's most recent inbound
        message (typically the one the agent is responding to). iMessage
        maps ❤️👍👎😂‼️❓ to native tapbacks; anything else uses Apple's
        custom-emoji reaction.
        """
        target = message_id or self._last_inbound_by_chat.get(
            self._normalize_chat_key(chat_id)
        )
        if not target:
            return {
                "success": False,
                "error": "no message to react to — pass message_id (no "
                "inbound message seen in this chat since the gateway started)",
            }
        ok = await self._add_reaction(chat_id, target, emoji)
        if not ok:
            return {
                "success": False,
                "error": "reaction failed (see gateway debug log)",
            }
        return {"success": True, "message_id": target}

    async def remove_reaction(
        self, chat_id: str, message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retract our tapback from a message (best-effort)."""
        target = message_id or self._last_inbound_by_chat.get(
            self._normalize_chat_key(chat_id)
        )
        if not target:
            return {
                "success": False,
                "error": "no message to unreact — pass message_id",
            }
        ok = await self._remove_reaction(chat_id, target)
        if not ok:
            return {
                "success": False,
                "error": "unreact failed (see gateway debug log)",
            }
        return {"success": True, "message_id": target}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Tapback 👀 on the triggering message while the agent works."""
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id:
            await self._add_reaction(chat_id, message_id, "\U0001f440")

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """Swap the 👀 progress tapback for a 👍/👎 result.

        Remove-then-add rather than a bare replace: deterministic whether the
        platform replaces a sender's previous tapback or stacks them, and it
        keeps the sidecar's reaction-handle slot coherent.
        """
        if not self._reactions_enabled():
            return
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if not chat_id or not message_id:
            return
        await self._remove_reaction(chat_id, message_id)
        if outcome == ProcessingOutcome.SUCCESS:
            await self._add_reaction(chat_id, message_id, "\U0001f44d")
        elif outcome == ProcessingOutcome.FAILURE:
            await self._add_reaction(chat_id, message_id, "\U0001f44e")
        # CANCELLED: leave the message unreacted.

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return whatever we know about a Spectrum space id.

        Photon's ``space.id`` is opaque; the inbound event also carries the
        DM/group type, but here we only have the id, so infer conservatively.
        """
        return {"name": chat_id, "type": "dm", "id": chat_id}

    def format_message(self, content: str) -> str:
        # Markdown is passed through verbatim — the sidecar sends it with the
        # markdown() builder and iMessage renders it. The strip path remains
        # as the PHOTON_MARKDOWN=false kill-switch.
        if _markdown_enabled():
            return content
        return strip_markdown(content)

    @staticmethod
    def _is_retryable_error(error: Optional[str]) -> bool:
        if BasePlatformAdapter._is_retryable_error(error):
            return True
        if not error:
            return False
        lowered = error.lower()
        return any(pat in lowered for pat in _PHOTON_RETRYABLE_PATTERNS)

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
        max_retries: int = 1,
        base_delay: float = 2.0,
    ) -> SendResult:
        """Retry sends without the generic Markdown banner.

        Photon replies are markdown (rendered by iMessage) or stripped plain
        text under ``PHOTON_MARKDOWN=false`` — either way the gateway's
        generic banner never applies.
        """
        text = self.format_message(content)
        result = await self.send(
            chat_id=chat_id,
            content=text,
            reply_to=reply_to,
            metadata=metadata,
        )
        if result.success:
            return result

        error_str = result.error or ""
        is_network = result.retryable or self._is_retryable_error(error_str)
        if not is_network and self._is_timeout_error(error_str):
            return result

        if is_network:
            for attempt in range(1, max_retries + 1):
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[photon] Send failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt, max_retries, delay, error_str,
                )
                await asyncio.sleep(delay)
                result = await self.send(
                    chat_id=chat_id,
                    content=text,
                    reply_to=reply_to,
                    metadata=metadata,
                )
                if result.success:
                    return result
                error_str = result.error or ""
                if not (result.retryable or self._is_retryable_error(error_str)):
                    break
            else:
                logger.error(
                    "[photon] Failed to deliver response after %d retries: %s",
                    max_retries, error_str,
                )
                return result

        logger.warning(
            "[photon] Send failed: %s - retrying plain-text message",
            error_str,
        )
        fallback_result = await self.send(
            chat_id=chat_id,
            content=text[: self.MAX_MESSAGE_LENGTH],
            reply_to=reply_to,
            metadata=metadata,
        )
        if not fallback_result.success:
            logger.error("[photon] Plain-text retry also failed: %s", fallback_result.error)
        return fallback_result

    async def _sidecar_send(self, space_id: str, text: str) -> SendResult:
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[photon] truncating outbound from %d to %d chars",
                len(text), self.MAX_MESSAGE_LENGTH,
            )
            text = text[: self.MAX_MESSAGE_LENGTH]
        body: Dict[str, Any] = {"spaceId": space_id, "text": text}
        # Omit the key when disabled so an older sidecar (pre-`format`)
        # keeps accepting the body during a half-upgraded restart.
        if _markdown_enabled():
            body["format"] = "markdown"
        try:
            data = await self._sidecar_call("/send", body)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send_attachment(
        self,
        space_id: str,
        path: str,
        *,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        caption: Optional[str] = None,
        kind: str = "attachment",
    ) -> SendResult:
        """POST a local file to the sidecar's ``/send-attachment`` endpoint.

        ``kind`` is ``"voice"`` for audio sent as a voice note (downgrades
        to a plain audio attachment on platforms without voice notes),
        otherwise ``"attachment"``. spectrum-ts infers ``name`` and
        ``mimeType`` from the file extension; we only pass overrides when
        Hermes supplied them.
        """
        # Defense-in-depth: re-validate the path before handing it to the
        # Node sidecar. The gateway already filters MEDIA paths, but
        # send_*_file / cron callers may pass arbitrary strings.
        safe_path = self.validate_media_delivery_path(str(path))
        if not safe_path:
            return SendResult(
                success=False, error=f"unsafe or missing attachment path: {path}"
            )
        if not mime_type:
            import mimetypes

            guessed, _ = mimetypes.guess_type(safe_path)
            mime_type = guessed or None
        body: Dict[str, Any] = {
            "spaceId": space_id,
            "path": safe_path,
            "kind": "voice" if kind == "voice" else "attachment",
        }
        if name:
            body["name"] = name
        if mime_type:
            body["mimeType"] = mime_type
        if caption:
            body["caption"] = caption
        try:
            data = await self._sidecar_call("/send-attachment", body)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        self._record_sent_message(data.get("messageId"))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        # Guard: adapter not yet connected (no sidecar address known).
        if self._http_client is None:
            raise RuntimeError("Photon adapter not connected")
        # Use a fresh client per call so this method is safe when invoked from
        # a worker thread that owns a different event loop than the one the
        # persistent _http_client was created on (e.g. via _run_async in
        # send_message_tool).  The inbound streaming loop continues to use
        # _http_client directly — it always runs on the gateway's loop.
        url = f"http://{self._sidecar_bind}:{self._sidecar_port}{path}"
        headers = {"X-Hermes-Sidecar-Token": self._sidecar_token}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Photon sidecar {path} returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json() or {}
        if not data.get("ok"):
            raise RuntimeError(
                f"Photon sidecar {path} reported error: {data.get('error')}"
            )
        return data


# ---------------------------------------------------------------------------
# Helpers

def _attachment_message_type(mime: str) -> MessageType:
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    if mime.startswith("application/"):
        return MessageType.DOCUMENT
    return MessageType.DOCUMENT


# MIME → file-extension maps for caching inbound attachment bytes. These mirror
# the BlueBubbles iMessage channel so both adapters name cached media the same.
_IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".jpg",
    "image/heif": ".jpg",
    "image/tiff": ".jpg",
}
_AUDIO_EXT_BY_MIME = {
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-caf": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".m4a",
}


def _cache_inbound_attachment(
    content: Dict[str, Any],
    name: str,
    mime: str,
    *,
    force_audio: bool = False,
) -> Optional[str]:
    """Decode a base64-inlined inbound attachment and cache it locally.

    The sidecar inlines the attachment bytes as ``content["data"]`` (base64).
    We decode them and route to the shared media cache by MIME type, returning
    the cached absolute path so the caller can populate ``media_urls`` (which
    the gateway then hands to the model). Returns ``None`` when there are no
    bytes (over the sidecar's inline cap or a failed read) or when caching
    fails, so the caller can fall back to a text marker.
    """
    data_b64 = content.get("data")
    if not data_b64:
        return None
    try:
        raw = base64.b64decode(data_b64)
    except (ValueError, TypeError) as exc:
        logger.warning("[photon] failed to decode inbound attachment bytes: %s", exc)
        return None

    from gateway.platforms.base import (
        cache_audio_from_bytes,
        cache_document_from_bytes,
        cache_image_from_bytes,
    )

    mime = (mime or "").lower()
    # Prefer the real extension from the filename; fall back to the MIME map.
    suffix = Path(name).suffix if name else ""
    try:
        if mime.startswith("image/"):
            ext = suffix or _IMAGE_EXT_BY_MIME.get(mime, ".jpg")
            try:
                return cache_image_from_bytes(raw, ext)
            except ValueError:
                # Bytes don't look like a supported image (e.g. HEIC magic) —
                # still deliver them as a document rather than dropping them.
                return cache_document_from_bytes(raw, name)
        if force_audio or mime.startswith("audio/"):
            ext = suffix or _AUDIO_EXT_BY_MIME.get(
                mime, ".m4a" if force_audio else ".mp3"
            )
            return cache_audio_from_bytes(raw, ext)
        # Video, application/*, and everything else → document cache.
        return cache_document_from_bytes(raw, name)
    except Exception as exc:
        logger.warning("[photon] failed to cache inbound attachment %s: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Standalone (out-of-process) send for cron deliveries when the gateway
# is not co-resident.  Reuses a live sidecar already listening on the
# configured port (cron processes cannot spawn the sidecar themselves).

async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,  # noqa: ARG001 — Spectrum has no threads yet
    media_files: Optional[list] = None,
    force_document: bool = False,  # noqa: ARG001 — iMessage auto-detects file kind
) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not installed"}
    port = _coerce_port(
        (pconfig.extra or {}).get("sidecar_port") or os.getenv("PHOTON_SIDECAR_PORT"),
        _DEFAULT_SIDECAR_PORT,
    )
    token = os.getenv("PHOTON_SIDECAR_TOKEN")
    if not token:
        return {
            "error": (
                "Photon standalone send requires a running sidecar with "
                "PHOTON_SIDECAR_TOKEN set in the environment. Cron processes "
                "cannot spawn the sidecar themselves."
            )
        }
    base = f"http://{_DEFAULT_SIDECAR_BIND}:{port}"
    headers = {"X-Hermes-Sidecar-Token": token}
    last_message_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Text body first (if any), so it leads the conversation.
            if message:
                send_body: Dict[str, Any] = {
                    "spaceId": chat_id,
                    "text": message[:_MAX_MESSAGE_LENGTH],
                }
                if _markdown_enabled():
                    send_body["format"] = "markdown"
                resp = await client.post(
                    f"{base}/send", json=send_body, headers=headers,
                )
                if resp.status_code != 200:
                    return {"error": f"sidecar returned {resp.status_code}: {resp.text[:200]}"}
                data = resp.json() or {}
                if not data.get("ok"):
                    return {"error": data.get("error") or "sidecar reported failure"}
                last_message_id = data.get("messageId")

            # 2. Each attachment as a separate /send-attachment call.
            #    media_files is List[Tuple[path, is_voice]] (see
            #    BasePlatformAdapter.filter_media_delivery_paths).
            import mimetypes

            for media_path, is_voice in media_files or []:
                safe_path = BasePlatformAdapter.validate_media_delivery_path(str(media_path))
                if not safe_path:
                    logger.warning("[photon] standalone send skipping unsafe path")
                    continue
                guessed, _ = mimetypes.guess_type(safe_path)
                att_body: Dict[str, Any] = {
                    "spaceId": chat_id,
                    "path": safe_path,
                    "kind": "voice" if is_voice else "attachment",
                }
                if guessed:
                    att_body["mimeType"] = guessed
                resp = await client.post(
                    f"{base}/send-attachment", json=att_body, headers=headers,
                )
                if resp.status_code != 200:
                    return {"error": f"sidecar returned {resp.status_code}: {resp.text[:200]}"}
                data = resp.json() or {}
                if not data.get("ok"):
                    return {"error": data.get("error") or "sidecar reported failure"}
                last_message_id = data.get("messageId") or last_message_id

        return {"success": True, "message_id": last_message_id}
    except Exception as e:
        return {"error": f"Photon standalone send failed: {e}"}


# ---------------------------------------------------------------------------
# Plugin entry point

def register(ctx) -> None:
    """Called by the Hermes plugin loader at startup."""
    # Local import to avoid argparse work at module load; reused for both the
    # gateway-setup hook and the `hermes photon` CLI command below.
    from . import cli as _cli

    ctx.register_platform(
        name="photon",
        label="iMessage via Photon",
        adapter_factory=lambda cfg: PhotonAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["PHOTON_PROJECT_ID", "PHOTON_PROJECT_SECRET"],
        install_hint=(
            "Run: hermes photon setup  (logs in via device flow, creates a "
            "Spectrum project, links your phone number, installs the "
            "spectrum-ts sidecar)."
        ),
        # Surfaces Photon in `hermes gateway setup` alongside every other
        # channel — same unified onboarding wizard, no Photon-only detour.
        setup_fn=_cli.gateway_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="PHOTON_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="PHOTON_ALLOWED_USERS",
        allow_all_env="PHOTON_ALLOW_ALL_USERS",
        max_message_length=_MAX_MESSAGE_LENGTH,
        emoji="📱",
        # iMessage carries E.164 phone numbers — treat session descriptions
        # as PII-sensitive so they get redacted before reaching the LLM
        # (matches the BlueBubbles iMessage channel in _PII_SAFE_PLATFORMS).
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via Photon Spectrum (iMessage). "
            "Treat replies like regular text messages — short and friendly. "
            "Markdown is rendered (bold, italics, lists, code), but keep "
            "formatting light and conversational. Recipient identifiers are "
            "E.164 phone numbers; never expose them in responses unless the "
            "user asked. Attachments arrive as metadata only."
        ),
    )

    # Register CLI subcommands — `hermes photon ...`
    ctx.register_cli_command(
        name="photon",
        help="Set up and manage the Photon iMessage integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )
