"""
Photon Spectrum (iMessage) platform adapter for Hermes Agent.

Inbound:
    Photon delivers signed JSON ``POST``s to a URL we register.  The
    adapter spins up an aiohttp server on ``PHOTON_WEBHOOK_PORT``,
    verifies ``X-Spectrum-Signature`` (HMAC-SHA256 of
    ``v0:{timestamp}:{body}`` keyed by the per-URL signing secret),
    rejects deliveries with a timestamp drift > 5 minutes, dedupes on
    ``message.id``, and dispatches a normalized ``MessageEvent`` to the
    gateway runner via ``BasePlatformAdapter.handle_message``.

Outbound:
    Photon does not currently expose a public HTTP send-message
    endpoint, so the adapter spawns a small Node sidecar (see
    ``sidecar/index.mjs``) that runs the ``spectrum-ts`` SDK.  Each
    ``send`` / ``send_typing`` / attachment call from Hermes is a
    loopback POST to the sidecar with a shared bearer token.  Outbound
    media (images, voice notes, video, documents) goes through
    spectrum-ts' ``attachment()`` / ``voice()`` content builders.

When Photon ships an HTTP send endpoint we can collapse the sidecar
into ``_send_via_http`` and drop the Node dependency entirely.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
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
from typing import Any, Dict, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx is already a Hermes dep
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from .auth import (
    DEFAULT_SPECTRUM_HOST,
    load_project_credentials,
    _spectrum_host,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants

_DEFAULT_WEBHOOK_PORT = 8788
_DEFAULT_WEBHOOK_PATH = "/photon/webhook"
_DEFAULT_WEBHOOK_BIND = "0.0.0.0"

_DEFAULT_SIDECAR_PORT = 8789
_DEFAULT_SIDECAR_BIND = "127.0.0.1"

# Photon iMessage messages from the SDK side have no documented hard
# limit, but the underlying iMessage protocol limits practical message
# size to ~16 KB.  Keep a conservative cap that matches BlueBubbles.
_MAX_MESSAGE_LENGTH = 8000

# Spec says reject deliveries older than ~5 minutes for replay protection.
_TIMESTAMP_DRIFT_SECONDS = 300

# Dedup parameters — keep at least 1k IDs for ~48h per Photon's
# at-least-once guidance.
_DEDUP_MAX_SIZE = 4000
_DEDUP_WINDOW_SECONDS = 48 * 3600

_SIDECAR_DIR = Path(__file__).parent / "sidecar"

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
    if not HTTPX_AVAILABLE or not AIOHTTP_AVAILABLE:
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
    """Seed PlatformConfig.extra from env so env-only setups appear in status."""
    project_id, project_secret = load_project_credentials()
    if not (project_id and project_secret):
        return None
    return {
        "project_id": project_id,
        "project_secret": project_secret,
        "webhook_port": _coerce_port(os.getenv("PHOTON_WEBHOOK_PORT"), _DEFAULT_WEBHOOK_PORT),
        "webhook_path": os.getenv("PHOTON_WEBHOOK_PATH") or _DEFAULT_WEBHOOK_PATH,
    }


# ---------------------------------------------------------------------------
# Signature verification

def verify_signature(
    *,
    body: bytes,
    timestamp_header: str,
    signature_header: str,
    signing_secret: str,
    now: Optional[float] = None,
    drift: int = _TIMESTAMP_DRIFT_SECONDS,
) -> bool:
    """Constant-time verify a Photon webhook signature.

    Returns True iff the timestamp is within ``drift`` of *now* AND
    ``signature_header == "v0=" + hmac_sha256(secret, "v0:{ts}:{body}")``.

    Exposed at module scope so tests can exercise it without an adapter
    instance.
    """
    if not timestamp_header or not signature_header or not signing_secret:
        return False
    try:
        ts = int(timestamp_header)
    except ValueError:
        return False
    if abs((now or time.time()) - ts) > drift:
        return False
    if not signature_header.startswith("v0="):
        return False
    expected = hmac.new(
        signing_secret.encode("utf-8"),
        f"v0:{ts}:".encode("utf-8") + body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header[3:])


# ---------------------------------------------------------------------------
# Adapter

class PhotonAdapter(BasePlatformAdapter):
    """Inbound: signed webhook on aiohttp. Outbound: Node sidecar via loopback HTTP."""

    MAX_MESSAGE_LENGTH = _MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("photon"))
        extra = config.extra or {}

        # Project credentials (env wins, then config.extra, then auth.json).
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

        # Webhook receiver
        self._webhook_port = _coerce_port(
            extra.get("webhook_port") or os.getenv("PHOTON_WEBHOOK_PORT"),
            _DEFAULT_WEBHOOK_PORT,
        )
        self._webhook_path = (
            extra.get("webhook_path")
            or os.getenv("PHOTON_WEBHOOK_PATH")
            or _DEFAULT_WEBHOOK_PATH
        )
        self._webhook_bind = (
            extra.get("webhook_bind")
            or os.getenv("PHOTON_WEBHOOK_BIND")
            or _DEFAULT_WEBHOOK_BIND
        )
        self._webhook_secret: str = (
            os.getenv("PHOTON_WEBHOOK_SECRET")
            or extra.get("webhook_secret")
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

        # Runtime state
        self._runner: Optional["web.AppRunner"] = None
        self._sidecar_proc: Optional[subprocess.Popen] = None
        self._sidecar_supervisor_task: Optional[asyncio.Task] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        # Lightweight in-memory dedup. Photon's at-least-once guarantee
        # means we WILL see the same message.id more than once.
        self._seen_messages: Dict[str, float] = {}

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
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error(
                "MISSING_DEP",
                "aiohttp not installed. Run: pip install aiohttp",
                retryable=False,
            )
            return False
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

        # Start the aiohttp receiver first; without it the sidecar would
        # be able to forward inbound traffic to a closed port.
        try:
            await self._start_webhook_server()
        except OSError as e:
            self._set_fatal_error(
                "PORT_IN_USE",
                f"webhook port {self._webhook_port} unavailable: {e}",
                retryable=True,
            )
            return False

        # Spin up the Node sidecar (required for outbound).
        if self._autostart_sidecar:
            try:
                await self._start_sidecar()
            except Exception as e:
                self._set_fatal_error(
                    "SIDECAR_FAILED",
                    f"failed to start Photon sidecar: {e}",
                    retryable=True,
                )
                await self._stop_webhook_server()
                return False
        else:
            logger.info("[photon] sidecar autostart disabled — outbound will fail")

        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._mark_connected()
        logger.info(
            "[photon] connected — webhook at %s:%d%s, sidecar on %s:%d",
            self._webhook_bind, self._webhook_port, self._webhook_path,
            self._sidecar_bind, self._sidecar_port,
        )
        return True

    async def disconnect(self) -> None:
        await self._stop_sidecar()
        await self._stop_webhook_server()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._mark_disconnected()

    # -- Webhook server ----------------------------------------------------

    async def _start_webhook_server(self) -> None:
        app = web.Application()
        app.router.add_post(self._webhook_path, self._handle_webhook)
        app.router.add_get("/healthz", lambda _: web.Response(text="ok"))
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_bind, self._webhook_port)
        await site.start()

    async def _stop_webhook_server(self) -> None:
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        body = await request.read()
        if self._webhook_secret:
            ts = request.headers.get("X-Spectrum-Timestamp", "")
            sig = request.headers.get("X-Spectrum-Signature", "")
            if not verify_signature(
                body=body,
                timestamp_header=ts,
                signature_header=sig,
                signing_secret=self._webhook_secret,
            ):
                logger.warning("[photon] rejected webhook with bad signature")
                return web.Response(status=401, text="invalid signature")
        else:
            logger.warning(
                "[photon] PHOTON_WEBHOOK_SECRET unset — accepting unsigned "
                "deliveries. Set the per-URL signing secret returned by "
                "register-webhook to enable verification."
            )

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")
        if payload.get("event") != "messages":
            # Photon currently emits only `messages`; any future event
            # types are ack'd 200 so they don't retry.
            return web.Response(text="ok")

        msg = payload.get("message") or {}
        msg_id = msg.get("id")
        if not msg_id:
            return web.Response(status=400, text="missing message.id")
        if self._is_duplicate(msg_id):
            return web.Response(text="ok (dup)")

        try:
            await self._dispatch_inbound(payload)
        except Exception:
            logger.exception("[photon] inbound dispatch failed")
            # 200 anyway — we own the dedup; failing here would cause
            # Photon to retry the same id.
        return web.Response(text="ok")

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > _DEDUP_MAX_SIZE:
            cutoff = now - _DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    async def _dispatch_inbound(self, payload: Dict[str, Any]) -> None:
        msg = payload.get("message") or {}
        space = msg.get("space") or payload.get("space") or {}
        sender = msg.get("sender") or {}
        content = msg.get("content") or {}

        space_id = space.get("id") or ""
        sender_id = sender.get("id") or ""
        if not space_id:
            logger.warning("[photon] inbound missing space.id")
            return

        # Space type — Photon documents iMessage DM ids as `any;-;+E164`
        # and group ids as `any;+;<chat-guid>`.  Use that as the
        # heuristic; everything else is treated as DM.
        chat_type = "group" if ";+;" in space_id else "dm"

        # Timestamp — ISO 8601 from the platform.
        ts_str = msg.get("timestamp") or ""
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime.now(tz=timezone.utc)

        # Content normalization.  Spectrum is a discriminated union;
        # text vs attachment metadata.  Attachments are metadata-only
        # today (no download URL) — log + carry the name so the agent
        # at least knows something was sent.
        if content.get("type") == "text":
            text = content.get("text") or ""
            mtype = MessageType.TEXT
        elif content.get("type") == "attachment":
            name = content.get("name") or "(unnamed)"
            mime = content.get("mimeType") or ""
            text = f"[Photon attachment received: {name} ({mime}) — no download URL yet]"
            mtype = _attachment_message_type(mime)
        else:
            text = f"[Photon content type not handled: {content.get('type')}]"
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
            user_id=sender_id or space_id,
            user_name=sender_id or None,
        )
        event = MessageEvent(
            text=text,
            message_type=mtype,
            source=source,
            message_id=msg.get("id"),
            raw_message=payload,
            timestamp=timestamp,
        )
        await self.handle_message(event)

    # -- Sidecar lifecycle -------------------------------------------------

    async def _start_sidecar(self) -> None:
        if not (_SIDECAR_DIR / "node_modules").exists():
            raise RuntimeError(
                f"Photon sidecar deps not installed. Run: "
                f"cd {_SIDECAR_DIR} && npm install   (or `hermes photon setup`)"
            )
        env = os.environ.copy()
        env["PHOTON_PROJECT_ID"] = self._project_id
        env["PHOTON_PROJECT_SECRET"] = self._project_secret
        env["PHOTON_SIDECAR_PORT"] = str(self._sidecar_port)
        env["PHOTON_SIDECAR_BIND"] = self._sidecar_bind
        env["PHOTON_SIDECAR_TOKEN"] = self._sidecar_token

        self._sidecar_proc = subprocess.Popen(  # noqa: S603
            [self._node_bin, str(_SIDECAR_DIR / "index.mjs")],
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

        # Wait for /healthz to come up — give it up to 15s on cold start.
        deadline = time.time() + 15.0
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
            f"Photon sidecar did not become ready within 15s: {last_err}"
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

    async def _stop_sidecar(self) -> None:
        proc = self._sidecar_proc
        if proc is None:
            return
        try:
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
        return await self._sidecar_send(chat_id, content, reply_to=reply_to)

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
            chat_id, local_path, caption=caption, reply_to=reply_to,
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
            chat_id, image_path, caption=caption, reply_to=reply_to,
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
            chat_id, audio_path, caption=caption, reply_to=reply_to, kind="voice",
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
            chat_id, video_path, caption=caption, reply_to=reply_to,
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
            chat_id, file_path, name=file_name, caption=caption, reply_to=reply_to,
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
        try:
            await self._sidecar_call("/typing", {"spaceId": chat_id})
        except Exception as e:
            logger.debug("[photon] send_typing failed: %s", e)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return whatever we know about a Spectrum space id.

        Photon's `space.id` is opaque (`any;-;+E164` for DMs,
        `any;+;<guid>` for groups). We surface that shape directly so
        the gateway has something to show in session pickers / logs.
        """
        chat_type = "group" if ";+;" in chat_id else "dm"
        return {"name": chat_id, "type": chat_type, "id": chat_id}

    async def _sidecar_send(
        self, space_id: str, text: str, *, reply_to: Optional[str] = None,
    ) -> SendResult:
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                "[photon] truncating outbound from %d to %d chars",
                len(text), self.MAX_MESSAGE_LENGTH,
            )
            text = text[: self.MAX_MESSAGE_LENGTH]
        body: Dict[str, Any] = {"spaceId": space_id, "text": text}
        if reply_to:
            body["replyTo"] = reply_to
        try:
            data = await self._sidecar_call("/send", body)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_send_attachment(
        self,
        space_id: str,
        path: str,
        *,
        name: Optional[str] = None,
        mime_type: Optional[str] = None,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
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
        if reply_to:
            body["replyTo"] = reply_to
        try:
            data = await self._sidecar_call("/send-attachment", body)
        except Exception as e:
            return SendResult(success=False, error=str(e))
        return SendResult(success=True, message_id=data.get("messageId"))

    async def _sidecar_call(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if self._http_client is None:
            raise RuntimeError("Photon adapter not connected")
        resp = await self._http_client.post(
            f"http://{self._sidecar_bind}:{self._sidecar_port}{path}",
            json=body,
            headers={"X-Hermes-Sidecar-Token": self._sidecar_token},
            timeout=30.0,
        )
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


# ---------------------------------------------------------------------------
# Standalone (out-of-process) send for cron deliveries when the gateway
# is not co-resident.  Spins up an ephemeral sidecar call by spawning
# the existing sidecar binary one-shot; if a live sidecar is already
# listening on the configured port we reuse it.

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
                resp = await client.post(
                    f"{base}/send",
                    json={"spaceId": chat_id, "text": message[:_MAX_MESSAGE_LENGTH]},
                    headers=headers,
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
        label="Photon iMessage",
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
            "Treat replies like regular text messages — short, friendly, no "
            "markdown rendering. Recipient identifiers are E.164 phone "
            "numbers; never expose them in responses unless the user asked. "
            "Attachments arrive as metadata only (no download URL yet)."
        ),
    )

    # Register CLI subcommands — `hermes photon ...`
    ctx.register_cli_command(
        name="photon",
        help="Set up and manage the Photon iMessage integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )
