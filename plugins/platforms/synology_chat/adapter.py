"""Synology Chat gateway adapter.

Connects a Synology NAS (DSM 7.x) Chat application to Hermes:

- Inbound: DSM outgoing webhooks (bot DMs and per-channel trigger-word
  webhooks) POST form-urlencoded payloads to a small aiohttp server this
  adapter runs on a dedicated local port.
- Outbound: DSM ``SYNO.Chat.External`` API — ``method=chatbot`` for DMs
  (requires ``user_ids``), or each channel's own incoming-webhook URL for
  channel messages.

DSM specifics this adapter encodes:

- Every DSM integration carries its OWN verification token (the bot plus
  one outgoing token per channel).  Inbound validation binds the presented
  token to the claimed source: a channel's token only authorizes messages
  from that channel, and the bot token only authorizes DMs.  The
  configured-channels table doubles as the channel whitelist.
- DSM only fires a channel outgoing webhook for messages containing the
  trigger word configured on the DSM side — mention-gating happens at the
  source, so none is implemented here.
- DSM expects ``payload=<url-encoded JSON>`` in a form-urlencoded body; a
  raw JSON body is silently ignored.
- DSM rate-limits outbound posts (~2 messages/second; API error code 411).

chat_id namespacing: DSM user_ids and channel_ids live in the same small
integer space, while ``send()`` only receives a chat_id (never the type).
To route outbound messages unambiguously — and never leak a private DM into
a public channel — inbound chat_ids are tagged ``dm:<user_id>`` /
``ch:<channel>`` (the same disambiguation Google Chat does with
``users/`` vs ``spaces/``).  Bare ids (cron / home-channel config) resolve
by membership in the configured-channels table.

Security model (documented for operators): the outgoing webhook is plain
HTTP and its only inbound authentication is the per-integration token, so a
token holder on the network segment can spoof any ``user_id``.  The
verification token is therefore the real authorization boundary; the
gateway user allowlist is a convenience filter on top of it.  Restrict the
port to the trusted LAN (firewall rule) and prefer a CA-pinned TLS endpoint.

Environment variables:
    SYNOLOGY_CHAT_TOKEN               Bot verification token
    SYNOLOGY_CHAT_INCOMING_URL        Bot incoming webhook URL (method=chatbot)
    SYNOLOGY_CHAT_WEBHOOK_PORT        Inbound listen port (default 8645)
    SYNOLOGY_CHAT_WEBHOOK_HOST        Inbound bind interface (default 0.0.0.0)
    SYNOLOGY_CHAT_HOME_CHANNEL       Channel ID for cron delivery
    SYNOLOGY_CHAT_ALLOWED_USERS      Comma-separated user IDs (gateway-enforced)
    SYNOLOGY_CHAT_ALLOW_ALL_USERS    Allow all users (gateway-enforced)
    SYNOLOGY_CHAT_CA_BUNDLE          CA cert path for the NAS TLS endpoint
    SYNOLOGY_CHAT_ALLOW_INSECURE_SSL Skip outbound TLS verification
    SYNOLOGY_CHANNEL_TOKEN_<id>      Outgoing-webhook token of channel <id>
    SYNOLOGY_CHANNEL_WEBHOOK_<id>    Incoming-webhook URL of channel <id>
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import time
import urllib.parse
from collections import OrderedDict, deque
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Synology Chat renders messages as plain text with Slack-like link syntax.
MAX_MESSAGE_LENGTH = 2000

# Inbound webhook body cap (DSM payloads are tiny; LINE uses a similar guard).
_MAX_BODY_BYTES = 65536

# Inbound per-IP rate limit (Feishu-webhook precedent: window over ALL
# requests, no invalid-token lockout that could reject a valid sender).
_IP_WINDOW_SECONDS = 60.0
_IP_WINDOW_MAX = 120
_IP_TRACKED_MAX = 4096

# Outbound pacing: DSM silently drops bursts; ~2 msg/s is the documented
# safe rate (API error 411 when exceeded).
_MIN_SEND_INTERVAL = 0.5
_SEND_RETRIES = 3
_SEND_BACKOFF_BASE = 0.3

# chat_id type prefixes (see module docstring).
_DM_PREFIX = "dm:"
_CHANNEL_PREFIX = "ch:"

_ENV_CHANNEL_TOKEN_PREFIX = "SYNOLOGY_CHANNEL_TOKEN_"
_ENV_CHANNEL_WEBHOOK_PREFIX = "SYNOLOGY_CHANNEL_WEBHOOK_"


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def check_synology_chat_requirements() -> bool:
    """Return True if the Synology Chat adapter can be used."""
    if not os.getenv("SYNOLOGY_CHAT_TOKEN", "").strip():
        logger.debug("Synology Chat: SYNOLOGY_CHAT_TOKEN not set")
        return False
    if not os.getenv("SYNOLOGY_CHAT_INCOMING_URL", "").strip():
        logger.debug("Synology Chat: SYNOLOGY_CHAT_INCOMING_URL not set")
        return False
    return True


def _channels_from_env() -> Dict[str, Dict[str, str]]:
    """Collect SYNOLOGY_CHANNEL_TOKEN_<id> / SYNOLOGY_CHANNEL_WEBHOOK_<id>."""
    channels: Dict[str, Dict[str, str]] = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_CHANNEL_TOKEN_PREFIX) and value.strip():
            channels.setdefault(key[len(_ENV_CHANNEL_TOKEN_PREFIX):], {})["token"] = value.strip()
        elif key.startswith(_ENV_CHANNEL_WEBHOOK_PREFIX) and value.strip():
            channels.setdefault(key[len(_ENV_CHANNEL_WEBHOOK_PREFIX):], {})["incoming_url"] = value.strip()
    return channels


def _merge_channels(extra: dict) -> Dict[str, Dict[str, str]]:
    """Build the {channel_id: {token, incoming_url}} table from YAML + env."""
    channels: Dict[str, Dict[str, str]] = {}
    yaml_channels = extra.get("channels")
    if isinstance(yaml_channels, dict):
        for cid, cval in yaml_channels.items():
            if isinstance(cval, dict):
                channels[str(cid)] = {
                    k: str(v) for k, v in cval.items() if k in ("token", "incoming_url")
                }
    for cid, cval in _channels_from_env().items():
        channels.setdefault(cid, {}).update(cval)
    return channels


class SynologyChatAdapter(BasePlatformAdapter):
    """Gateway adapter for Synology Chat (DSM outgoing/incoming webhooks)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("synology_chat"))
        extra = config.extra or {}

        self._token: str = (config.token or os.getenv("SYNOLOGY_CHAT_TOKEN", "")).strip()
        self._incoming_url: str = (
            extra.get("incoming_url") or os.getenv("SYNOLOGY_CHAT_INCOMING_URL", "")
        ).strip()
        self._webhook_port: int = int(
            extra.get("webhook_port") or os.getenv("SYNOLOGY_CHAT_WEBHOOK_PORT", "8645")
        )
        self._webhook_host: str = (
            extra.get("webhook_host") or os.getenv("SYNOLOGY_CHAT_WEBHOOK_HOST", "0.0.0.0")
        ).strip()
        self._ca_bundle: str = (
            extra.get("ca_bundle") or os.getenv("SYNOLOGY_CHAT_CA_BUNDLE", "")
        ).strip()
        self._allow_insecure_ssl: bool = _parse_bool(
            extra.get("allow_insecure_ssl")
            or os.getenv("SYNOLOGY_CHAT_ALLOW_INSECURE_SSL", "false")
        )

        # {channel_id: {"token": ..., "incoming_url": ...}} — this table IS the
        # channel whitelist: inbound messages from unconfigured channels fail
        # token binding, and outbound routing checks membership here.
        self._channels: Dict[str, Dict[str, str]] = _merge_channels(extra)

        # Runtime state
        self._runner = None  # aiohttp AppRunner
        self._http_session = None  # aiohttp.ClientSession (outbound)
        self._send_lock = asyncio.Lock()
        self._last_send_time = 0.0
        # LRU-ish per-IP windows (bounded; evicts oldest, never fails open).
        self._ip_windows: "OrderedDict[str, deque]" = OrderedDict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self._token or not self._incoming_url:
            logger.error(
                "Synology Chat: SYNOLOGY_CHAT_TOKEN and SYNOLOGY_CHAT_INCOMING_URL are required"
            )
            return False
        if not self._acquire_platform_lock(
            "synology_chat_port", str(self._webhook_port),
            f"Synology Chat webhook port {self._webhook_port}",
        ):
            return False

        import aiohttp
        from aiohttp import web

        try:
            ssl_ctx = None
            if self._ca_bundle:
                import ssl as _ssl
                ssl_ctx = _ssl.create_default_context(cafile=self._ca_bundle)
            elif self._allow_insecure_ssl:
                ssl_ctx = False
                logger.warning(
                    "Synology Chat: outbound TLS verification DISABLED "
                    "(allow_insecure_ssl) — prefer SYNOLOGY_CHAT_CA_BUNDLE"
                )

            self._http_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ssl_ctx),
                timeout=aiohttp.ClientTimeout(total=30),
            )

            # client_max_size caps the body DURING accumulation (aiohttp raises
            # 413 before buffering the whole payload) — the manual len() check
            # below is only a secondary guard.
            app = web.Application(client_max_size=_MAX_BODY_BYTES)
            app.router.add_post("/", self._handle_inbound)
            app.router.add_get("/health", self._handle_health)

            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
            await site.start()
        except Exception as exc:  # noqa: BLE001 — surface a clean failure to the gateway
            logger.error("Synology Chat: failed to start webhook server: %s", exc, exc_info=True)
            if self._http_session is not None and not self._http_session.closed:
                await self._http_session.close()
            self._http_session = None
            if self._runner is not None:
                await self._runner.cleanup()
                self._runner = None
            self._release_platform_lock()
            return False

        self._mark_connected()
        logger.info(
            "Synology Chat: webhook server listening on %s:%d (%d channel(s) configured)",
            self._webhook_host, self._webhook_port, len(self._channels),
        )
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None
        self._release_platform_lock()
        self._mark_disconnected()

    # ------------------------------------------------------------------
    # Inbound (DSM outgoing webhook -> Hermes)
    # ------------------------------------------------------------------

    async def _handle_health(self, request):
        from aiohttp import web
        return web.json_response({"status": "ok"})

    def _ip_rate_limited(self, remote: str) -> bool:
        """Sliding window over all requests per source IP (no token lockout).

        Bounded by an LRU: when the table is full the oldest IP is evicted
        rather than failing open, and empty windows are dropped eagerly so
        the table tracks only currently-active sources.
        """
        now = time.monotonic()
        window = self._ip_windows.get(remote)
        if window is None:
            window = deque()
            self._ip_windows[remote] = window
            while len(self._ip_windows) > _IP_TRACKED_MAX:
                self._ip_windows.popitem(last=False)
        else:
            self._ip_windows.move_to_end(remote)
        while window and now - window[0] > _IP_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= _IP_WINDOW_MAX:
            return True
        window.append(now)
        return False

    @staticmethod
    def _strip_trigger_word(text: str, trigger_word: str) -> str:
        """Remove the leading DSM trigger word (mention-equivalent) from text."""
        stripped = (text or "").strip()
        if not trigger_word:
            return stripped
        tw = trigger_word.strip()
        if stripped.lower().startswith(tw.lower()):
            stripped = stripped[len(tw):].lstrip(" \t:,;-")
        return stripped.strip()

    def _validate_inbound_token(self, token: str, channel_key: str) -> bool:
        """Bind the presented token to the claimed source.

        DM (no channel) -> must match the bot token.
        Channel <id>    -> must match that channel's configured outgoing token.
        Constant-time comparison; fail-closed when nothing is configured.
        """
        if not token:
            return False
        if channel_key:
            expected = (self._channels.get(channel_key) or {}).get("token", "")
        else:
            expected = self._token
        if not expected:
            return False
        return hmac.compare_digest(token.encode(), expected.encode())

    async def _handle_inbound(self, request):
        from aiohttp import web

        remote = request.remote or "?"
        if self._ip_rate_limited(remote):
            logger.warning("Synology Chat: inbound rate limit exceeded for %s", remote)
            return web.Response(status=429)

        if request.content_length and request.content_length > _MAX_BODY_BYTES:
            return web.Response(status=413)

        try:
            raw = await asyncio.wait_for(request.read(), timeout=5.0)
        except asyncio.TimeoutError:
            return web.Response(status=408)
        except Exception as exc:  # noqa: BLE001 — body read errors -> 400, never crash the server
            logger.debug("Synology Chat: inbound read failed from %s: %s", remote, exc)
            return web.Response(status=400)
        if len(raw) > _MAX_BODY_BYTES:
            return web.Response(status=413)

        payload = self._parse_payload(raw, request.content_type or "")
        if payload is None:
            return web.Response(status=400)

        token = payload.get("token", "")
        user_id = payload.get("user_id", "")
        if not token or not user_id or "text" not in payload:
            return web.Response(status=400)

        # DM payloads carry neither channel_id nor trigger_word; some DSM
        # versions send channel_name without channel_id — treat any truthy
        # channel identifier as a channel message.
        channel_key = (payload.get("channel_id") or "").strip() or (
            payload.get("channel_name") or ""
        ).strip()

        if not self._validate_inbound_token(token, channel_key):
            logger.warning(
                "Synology Chat: rejected inbound from %s (invalid token, channel=%r)",
                remote, channel_key[:32] or "<dm>",
            )
            return web.Response(status=401)

        clean_text = self._strip_trigger_word(payload.get("text", ""), payload.get("trigger_word", ""))

        if channel_key:
            source = self.build_source(
                chat_id=f"{_CHANNEL_PREFIX}{channel_key}",
                chat_name=(payload.get("channel_name") or "").strip() or None,
                chat_type="group",
                user_id=user_id,
                user_name=payload.get("username") or None,
            )
        else:
            source = self.build_source(
                chat_id=f"{_DM_PREFIX}{user_id}",
                chat_type="dm",
                user_id=user_id,
                user_name=payload.get("username") or None,
            )

        event = MessageEvent(
            text=clean_text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=payload.get("post_id") or None,
        )

        # The LLM run is already backgrounded by base.handle_message, so this
        # returns in milliseconds — well within the short DSM webhook timeout.
        try:
            await self.handle_message(event)
        except Exception as exc:  # noqa: BLE001 — never bounce a webhook on agent errors
            logger.error("Synology Chat: handle_message failed: %s", exc, exc_info=True)

        return web.Response(status=204)

    @staticmethod
    def _parse_payload(raw: bytes, content_type: str) -> Optional[Dict[str, str]]:
        """Parse a DSM outgoing-webhook body (form-urlencoded, JSON fallback)."""
        try:
            body = raw.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
        if "json" in content_type.lower():
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(data, dict):
                return None
            return {str(k): str(v) if v is not None else "" for k, v in data.items()}
        parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
        if not parsed:
            return None
        return {k: v[0] if v else "" for k, v in parsed.items()}

    # ------------------------------------------------------------------
    # Outbound (Hermes -> DSM incoming webhook / chatbot API)
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content:
            return SendResult(success=True)
        formatted = self.format_message(content)
        if not formatted:
            return SendResult(success=True)
        url, payload_extra, err = self._resolve_destination(chat_id)
        if err is not None:
            logger.error("Synology Chat: cannot route chat_id=%s: %s", str(chat_id)[:32], err)
            return SendResult(success=False, error=err)
        result = SendResult(success=True)
        for chunk in self.truncate_message(formatted, MAX_MESSAGE_LENGTH):
            result = await self._send_chunk(url, payload_extra, chunk, chat_id)
            if not result.success:
                return result
        return result

    def _resolve_destination(
        self, chat_id: str
    ) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        """Resolve (url, payload_extra, error) for a chat_id.

        ``dm:<id>``  -> chatbot API with user_ids (DSM requires it).
        ``ch:<id>``  -> that channel's incoming webhook.
        bare id      -> channel webhook if configured, else chatbot (cron /
                        home-channel targets that bypass the inbound prefixing).
        """
        chat_id = str(chat_id)
        if chat_id.startswith(_DM_PREFIX):
            return self._dm_destination(chat_id[len(_DM_PREFIX):])
        if chat_id.startswith(_CHANNEL_PREFIX):
            return self._channel_destination(chat_id[len(_CHANNEL_PREFIX):])
        if chat_id in self._channels:
            return self._channel_destination(chat_id)
        return self._dm_destination(chat_id)

    def _dm_destination(self, user_id: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        try:
            return self._incoming_url, {"user_ids": [int(user_id)]}, None
        except ValueError:
            # DSM method=chatbot requires numeric user_ids (error 800 otherwise).
            return None, {}, f"non-numeric DM target '{user_id}' — chatbot API needs an integer user_id"

    def _channel_destination(self, channel_id: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
        channel = self._channels.get(channel_id)
        if not channel or not channel.get("incoming_url"):
            return None, {}, (
                f"channel '{channel_id}' has no incoming_url — set "
                f"SYNOLOGY_CHANNEL_WEBHOOK_{channel_id} (or synology_chat.channels.{channel_id}.incoming_url)"
            )
        return channel["incoming_url"], {}, None

    async def _send_chunk(
        self, url: str, payload_extra: Dict[str, Any], text: str, chat_id: str
    ) -> SendResult:
        import aiohttp

        payload = {"text": text, **payload_extra}
        body = "payload=" + urllib.parse.quote(json.dumps(payload, ensure_ascii=False))

        # Hold the lock only to compute/await the send pacing, then release it
        # before the (possibly slow) POST so unrelated chats aren't blocked.
        async with self._send_lock:
            elapsed = time.monotonic() - self._last_send_time
            if elapsed < _MIN_SEND_INTERVAL:
                await asyncio.sleep(_MIN_SEND_INTERVAL - elapsed)
            self._last_send_time = time.monotonic()

        last_error = "unknown"
        for attempt in range(_SEND_RETRIES):
            try:
                async with self._http_session.post(
                    url,
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        last_error = f"HTTP {resp.status}"
                    elif resp.status >= 400:
                        body_text = await resp.text()
                        return SendResult(success=False, error=f"DSM HTTP {resp.status}: {body_text[:200]}")
                    else:
                        api_error = await self._api_error(resp)
                        if api_error is None:
                            return SendResult(success=True)
                        code = api_error.get("code")
                        if code == 411:  # DSM rate limit — retryable
                            last_error = "DSM API error 411 (rate limit)"
                        else:
                            return SendResult(
                                success=False, error=f"DSM API error {code}", raw_response=api_error,
                            )
            except asyncio.TimeoutError:
                # No retry on timeout: the message may already be delivered.
                # retryable=False so base._send_with_retry does NOT re-send
                # (avoids double delivery); "timed out" in the error string so
                # the base timeout heuristic skips the plain-text fallback too.
                logger.warning("Synology Chat: send timed out for chat_id=%s", str(chat_id)[:24])
                return SendResult(success=False, error="DSM request timed out", retryable=False)
            except aiohttp.ClientError as exc:
                last_error = f"transport: {exc}"

            if attempt < _SEND_RETRIES - 1:
                await asyncio.sleep(_SEND_BACKOFF_BASE * (2 ** attempt))

        logger.error(
            "Synology Chat: send failed after %d attempts for chat_id=%s (%s)",
            _SEND_RETRIES, str(chat_id)[:24], last_error,
        )
        return SendResult(success=False, error=last_error, retryable=True)

    @staticmethod
    async def _api_error(resp) -> Optional[Dict[str, Any]]:
        """DSM replies HTTP 200 with ``{"success": false, "error": {...}}`` on
        API-level failures.  Returns the error dict, or None on success.  A 200
        with a non-JSON body is treated as success (incoming webhooks return
        plain bodies)."""
        try:
            data = json.loads(await resp.text())
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(data, dict) and data.get("success") is False:
            return data.get("error") or {"code": "unknown"}
        return None

    # ------------------------------------------------------------------
    # Formatting / misc
    # ------------------------------------------------------------------

    def format_message(self, content: str) -> str:
        """Synology Chat renders plain text only: flatten Markdown, convert
        links to the Slack-like ``<url|text>`` syntax DSM understands."""
        text = content or ""
        text = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<\2|\1>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"`([^`\n]+)`", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_id = str(chat_id)
        if chat_id.startswith(_CHANNEL_PREFIX) or chat_id in self._channels:
            return {"name": chat_id, "type": "group"}
        return {"name": chat_id, "type": "dm"}


# ---------------------------------------------------------------------------
# Standalone sender (out-of-process cron delivery)
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
) -> dict:
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    incoming_url = (extra.get("incoming_url") or os.getenv("SYNOLOGY_CHAT_INCOMING_URL", "")).strip()
    if not incoming_url:
        return {"error": "SYNOLOGY_CHAT_INCOMING_URL not configured"}

    chat_id = str(chat_id)
    channels = _merge_channels(extra)

    # Mirror the in-process routing (prefixes + membership fallback).
    if chat_id.startswith(_CHANNEL_PREFIX):
        channel_id = chat_id[len(_CHANNEL_PREFIX):]
        channel = channels.get(channel_id) or {}
        if not channel.get("incoming_url"):
            return {"error": f"channel '{channel_id}' has no incoming_url"}
        url, payload = channel["incoming_url"], {"text": message}
    elif chat_id.startswith(_DM_PREFIX):
        url, payload = _dm_payload(incoming_url, chat_id[len(_DM_PREFIX):])
    elif chat_id in channels and channels[chat_id].get("incoming_url"):
        url, payload = channels[chat_id]["incoming_url"], {"text": message}
    else:
        url, payload = _dm_payload(incoming_url, chat_id)
    payload["text"] = message

    ca_bundle = (extra.get("ca_bundle") or os.getenv("SYNOLOGY_CHAT_CA_BUNDLE", "")).strip()
    ssl_ctx = None
    if ca_bundle:
        import ssl as _ssl
        ssl_ctx = _ssl.create_default_context(cafile=ca_bundle)
    elif _parse_bool(extra.get("allow_insecure_ssl")
                     or os.getenv("SYNOLOGY_CHAT_ALLOW_INSECURE_SSL", "false")):
        ssl_ctx = False

    body = "payload=" + urllib.parse.quote(json.dumps(payload, ensure_ascii=False))
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            async with session.post(
                url, data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"DSM HTTP {resp.status}: {text[:200]}"}
                # DSM signals API-level failure as 200 + success:false.
                try:
                    data = json.loads(await resp.text())
                except (json.JSONDecodeError, ValueError):
                    data = None
                if isinstance(data, dict) and data.get("success") is False:
                    err = data.get("error") or {"code": "unknown"}
                    return {"error": f"DSM API error {err.get('code')}"}
                return {"success": True, "platform": "synology_chat", "chat_id": chat_id}
    except Exception as exc:  # noqa: BLE001 — standalone path reports, never raises
        return {"error": f"Synology Chat standalone send failed: {exc}"}


def _dm_payload(incoming_url: str, user_id: str) -> Tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {}
    try:
        payload["user_ids"] = [int(user_id)]
    except ValueError:
        pass
    return incoming_url, payload


# ---------------------------------------------------------------------------
# Plugin hooks
# ---------------------------------------------------------------------------


def _env_enablement() -> Optional[dict]:
    token = os.getenv("SYNOLOGY_CHAT_TOKEN", "").strip()
    incoming_url = os.getenv("SYNOLOGY_CHAT_INCOMING_URL", "").strip()
    if not (token and incoming_url):
        return None
    seed: Dict[str, Any] = {"token": token, "incoming_url": incoming_url}
    channels = _channels_from_env()
    if channels:
        seed["channels"] = channels
    home = os.getenv("SYNOLOGY_CHAT_HOME_CHANNEL", "").strip()
    if home:
        # Bare id: outbound routing resolves it by membership in the
        # configured-channels table, so both a channel id and a user id work.
        seed["home_channel"] = {"chat_id": home, "name": "Home"}
    return seed


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    simple_map = {
        "webhook_port": "SYNOLOGY_CHAT_WEBHOOK_PORT",
        "webhook_host": "SYNOLOGY_CHAT_WEBHOOK_HOST",
        "allow_insecure_ssl": "SYNOLOGY_CHAT_ALLOW_INSECURE_SSL",
        "ca_bundle": "SYNOLOGY_CHAT_CA_BUNDLE",
    }
    for yaml_key, env_key in simple_map.items():
        if yaml_key in platform_cfg and not os.getenv(env_key):
            os.environ[env_key] = str(platform_cfg[yaml_key])

    if "allowed_user_ids" in platform_cfg and not os.getenv("SYNOLOGY_CHAT_ALLOWED_USERS"):
        val = platform_cfg["allowed_user_ids"]
        if isinstance(val, list):
            val = ",".join(str(v) for v in val)
        os.environ["SYNOLOGY_CHAT_ALLOWED_USERS"] = str(val)

    extra: Dict[str, Any] = {}
    channels_cfg = platform_cfg.get("channels")
    if isinstance(channels_cfg, dict):
        channels: Dict[str, Dict[str, str]] = {}
        for ch_id, ch_val in channels_cfg.items():
            if not isinstance(ch_val, dict):
                continue
            entry = {k: str(ch_val[k]) for k in ("token", "incoming_url") if k in ch_val}
            if entry:
                channels[str(ch_id)] = entry
        if channels:
            extra["channels"] = channels
    return extra or None


def _is_connected(config) -> bool:
    import hermes_cli.gateway as gateway_mod
    return bool(
        (gateway_mod.get_env_value("SYNOLOGY_CHAT_TOKEN") or "").strip()
        and (gateway_mod.get_env_value("SYNOLOGY_CHAT_INCOMING_URL") or "").strip()
    )


def interactive_setup() -> bool:
    """Interactive setup: prompt the two required values and persist them."""
    from hermes_cli.config import save_env_value

    print("Synology Chat setup — values from DSM > Chat > Integration")
    if os.getenv("SYNOLOGY_CHAT_TOKEN") and os.getenv("SYNOLOGY_CHAT_INCOMING_URL"):
        if input("Already configured. Reconfigure? (y/N): ").strip().lower() != "y":
            return True

    token = input("Bot token: ").strip()
    incoming_url = input("Bot incoming webhook URL (method=chatbot): ").strip()
    if not token or not incoming_url:
        print("Both values are required.")
        return False

    save_env_value("SYNOLOGY_CHAT_TOKEN", token)
    save_env_value("SYNOLOGY_CHAT_INCOMING_URL", incoming_url)
    print(
        "Configured. Point the DSM bot outgoing webhook at "
        "http://<hermes-host>:8645/ (SYNOLOGY_CHAT_WEBHOOK_PORT to change). "
        "Restrict that port to your trusted LAN."
    )
    return True


def _build_adapter(config):
    """Factory wrapper that constructs SynologyChatAdapter from a PlatformConfig."""
    return SynologyChatAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="synology_chat",
        label="Synology Chat",
        adapter_factory=_build_adapter,
        check_fn=check_synology_chat_requirements,
        is_connected=_is_connected,
        required_env=["SYNOLOGY_CHAT_TOKEN", "SYNOLOGY_CHAT_INCOMING_URL"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="SYNOLOGY_CHAT_ALLOWED_USERS",
        allow_all_env="SYNOLOGY_CHAT_ALLOW_ALL_USERS",
        cron_deliver_env_var="SYNOLOGY_CHAT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        platform_hint=(
            "You are chatting via Synology Chat on a Synology NAS. "
            "No markdown rendering: avoid bold, italic, and code blocks. "
            "Links use <URL|text> syntax. "
            "Messages are auto-chunked at 2000 characters. "
            "No thread support, no message editing."
        ),
        emoji="💬",
        allow_update_command=True,
    )
