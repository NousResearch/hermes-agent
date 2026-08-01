"""
Zalo Bot Platform adapter — long polling (getUpdates) or webhook.

Uses the Zalo Bot HTTP API — not Zalo Official Account (OA).
See https://bot.zapps.me/docs/

Requires:
    pip install httpx
    ZALO_BOT_TOKEN in ~/.hermes/.env (from Zalo Bot Creator)

Modes:
    - polling (default): getUpdates long polling. Does not work if a webhook is
      already set — call deleteWebhook first (Zalo docs).
    - webhook: local aiohttp server + setWebhook with a public HTTPS URL.
      Requires aiohttp, ZALO_WEBHOOK_PUBLIC_URL (https), ZALO_WEBHOOK_SECRET (8–256 chars).

API base: https://bot-api.zaloplatforms.com/bot<BOT_TOKEN>/<method>
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

import httpx

from agent.secret_scope import get_secret

# httpx is a Hermes core dependency.  Keep the flag for compatibility with
# the original adapter's requirement probe and for focused tests.
HTTPX_AVAILABLE = True

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
    cache_image_from_url,
)

logger = logging.getLogger(__name__)

# https://bot.zapps.me/docs/apis/sendMessage/
MAX_MESSAGE_LENGTH = 2000

ZALO_API_BASE = "https://bot-api.zaloplatforms.com/bot"
DEFAULT_POLL_TIMEOUT = 30
WEBHOOK_MAX_BYTES = 256 * 1024

DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 1000

# Backoff after transport/API errors (seconds). Reset on successful poll cycle.
_POLL_BACKOFF_SEC = (1, 2, 4, 8, 16, 30, 45, 60, 90, 120)
_MAX_BACKOFF_JITTER_RATIO = 0.25


def _env_value(name: str, default: str = "") -> str:
    """Resolve Zalo settings from the active profile's isolated secret scope."""
    value = get_secret(name, default)
    return default if value is None else str(value)


def _api_url(token: str, method: str, api_base: str = ZALO_API_BASE) -> str:
    return f"{api_base.rstrip('/')}{token}/{method}"


def _validated_https_url(value: str, *, default: str = "") -> str:
    """Accept credential-bearing endpoints only on well-formed HTTPS origins."""
    candidate = str(value or "").strip()
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return default
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return default
    return candidate.rstrip("/")


def _safe_api_error(
    body: Any,
    status_code: Any,
    *secrets: str,
    limit: int = 300,
) -> str:
    """Return a bounded API error without reflecting request URLs or secrets."""
    if isinstance(body, dict):
        raw = body.get("description") or body.get("error_code")
    else:
        raw = None
    text = str(raw or f"HTTP {status_code}")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:limit]


def check_requirements() -> bool:
    """Return whether the core HTTP dependency is available.

    Credential checks live in :func:`validate_config`; keeping the plugin's
    dependency and configuration gates separate lets config.yaml-only setups
    work with the platform registry.
    """
    return HTTPX_AVAILABLE


def check_zalo_requirements() -> bool:
    """Backward-compatible combined dependency + env credential probe."""
    return check_requirements() and bool(_env_value("ZALO_BOT_TOKEN", "").strip())


def _iter_updates(result: Any) -> Iterator[Dict[str, Any]]:
    """Normalize getUpdates `result` into per-update dicts."""
    if result is None:
        return
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                yield item
    elif isinstance(result, dict):
        if "event_name" in result or "message" in result:
            yield result


def _parse_webhook_path(public_url: str, path_override: Optional[str]) -> str:
    if path_override and path_override.strip():
        p = path_override.strip()
        return p if p.startswith("/") else f"/{p}"
    parsed = urlparse(public_url)
    path = parsed.path or "/"
    if path == "/":
        return "/zalo/webhook"
    return path


def _webhook_url_for_path(public_url: str, path: str) -> str:
    """Return the registered public URL for the actual local listener path."""
    parsed = urlparse(public_url)
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _ensure_webhook_dependency() -> bool:
    """Load aiohttp for webhook mode, lazy-installing its pinned extra once."""
    global AIOHTTP_AVAILABLE, web
    if AIOHTTP_AVAILABLE and web is not None:
        return True
    try:
        from tools.lazy_deps import ensure

        ensure("platform.zalo", prompt=False)
        from aiohttp import web as aiohttp_web
    except Exception:
        return False
    web = aiohttp_web
    AIOHTTP_AVAILABLE = True
    return True


class ZaloBotAdapter(BasePlatformAdapter):
    """Zalo Bot Platform — polling or webhook; text, images, stickers."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("zalo"))
        extra = config.extra or {}
        self._token: str = str(
            _env_value("ZALO_BOT_TOKEN") or config.token or extra.get("bot_token") or ""
        ).strip()
        requested_api_base = str(
            _env_value("ZALO_API_BASE") or extra.get("api_base") or ZALO_API_BASE
        )
        self._api_base = _validated_https_url(requested_api_base, default=ZALO_API_BASE)
        try:
            poll_timeout = int(
                _env_value("ZALO_POLL_TIMEOUT")
                or extra.get("poll_timeout")
                or DEFAULT_POLL_TIMEOUT
            )
        except (TypeError, ValueError):
            poll_timeout = DEFAULT_POLL_TIMEOUT
        self._poll_timeout = max(1, min(poll_timeout, 50))

        self._http_client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._seen_messages: Dict[str, float] = {}

        self._connection_mode: str = (
            str(
                _env_value("ZALO_CONNECTION_MODE")
                or extra.get("connection_mode")
                or "polling"
            )
        ).lower()
        if self._connection_mode not in ("polling", "webhook"):
            self._connection_mode = "polling"

        self._webhook_public_url: str = str(
            _env_value("ZALO_WEBHOOK_PUBLIC_URL")
            or extra.get("webhook_public_url")
            or ""
        ).strip()
        self._webhook_secret: str = str(
            _env_value("ZALO_WEBHOOK_SECRET") or extra.get("webhook_secret") or ""
        ).strip()
        self._webhook_host: str = str(
            _env_value("ZALO_WEBHOOK_HOST") or extra.get("webhook_host") or "0.0.0.0"
        )
        try:
            webhook_port = int(
                _env_value("ZALO_WEBHOOK_PORT") or extra.get("webhook_port") or 8790
            )
        except (TypeError, ValueError):
            webhook_port = 8790
        self._webhook_port = webhook_port if 1 <= webhook_port <= 65535 else 8790
        path_ov = _env_value("ZALO_WEBHOOK_PATH") or extra.get("webhook_path")
        self._webhook_path: str = _parse_webhook_path(
            self._webhook_public_url, path_ov if path_ov else None
        )

        self._webhook_runner: Any = None
        self._webhook_site: Any = None
        self._webhook_registered: bool = False
        self._delete_webhook_on_disconnect: bool = False

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        _ = is_reconnect
        if not HTTPX_AVAILABLE:
            logger.warning(
                "[%s] httpx not installed. Run: pip install httpx", self.name
            )
            return False
        if not self._token:
            logger.warning("[%s] ZALO_BOT_TOKEN is not set", self.name)
            return False

        if not self._acquire_platform_lock("zalo-token", self._token, "Zalo bot token"):
            return False

        try:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._poll_timeout + 35.0, connect=15.0),
                headers={"User-Agent": "HermesAgent/1.0 (Zalo Bot)"},
            )
        except Exception as e:
            logger.error(
                "[%s] Failed to create HTTP client: %s", self.name, type(e).__name__
            )
            self._release_platform_lock()
            return False

        if not await self._probe_bot():
            try:
                await self._close_http_client()
            finally:
                self._release_platform_lock()
            return False

        try:
            if self._connection_mode == "webhook":
                connected = await self._connect_webhook()
            else:
                connected = await self._connect_polling()
        except Exception as exc:
            logger.error(
                "[%s] Transport startup failed: %s", self.name, type(exc).__name__
            )
            await self._cleanup_failed_connect()
            return False
        if not connected:
            await self._cleanup_failed_connect()
        return connected

    async def _cleanup_failed_connect(self) -> None:
        try:
            await self._disconnect_webhook_server()
        except Exception as exc:
            logger.warning(
                "[%s] Failed-connect webhook cleanup failed: %s",
                self.name,
                type(exc).__name__,
            )
        finally:
            try:
                await self._close_http_client()
            finally:
                self._release_platform_lock()

    async def _close_http_client(self) -> None:
        client = self._http_client
        self._http_client = None
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as exc:
            logger.warning(
                "[%s] HTTP client close failed: %s", self.name, type(exc).__name__
            )

    async def _probe_bot(self) -> bool:
        """Validate the token before declaring the adapter connected."""
        assert self._http_client is not None
        try:
            response = await self._http_client.post(
                _api_url(self._token, "getMe", self._api_base)
            )
            body = response.json()
        except Exception as exc:
            logger.error("[%s] getMe failed: %s", self.name, type(exc).__name__)
            return False
        if not isinstance(body, dict) or not body.get("ok"):
            error = _safe_api_error(
                body, response.status_code, self._token, self._webhook_secret
            )
            logger.error("[%s] getMe rejected the bot token: %s", self.name, error)
            self._set_fatal_error(
                "zalo_auth_failed",
                f"Zalo getMe rejected the bot token: {error}",
                retryable=False,
            )
            return False
        return True

    async def _connect_polling(self) -> bool:
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._mark_connected()
        logger.info("[%s] Long polling started (getUpdates)", self.name)
        return True

    async def _connect_webhook(self) -> bool:
        if not _ensure_webhook_dependency() or web is None:
            logger.warning(
                "[%s] Webhook mode requires aiohttp. "
                "Run: pip install 'hermes-agent[zalo]'",
                self.name,
            )
            await self._http_client.aclose()
            self._http_client = None
            return False
        validated_public_url = _validated_https_url(self._webhook_public_url)
        if not validated_public_url:
            logger.error(
                "[%s] ZALO_WEBHOOK_PUBLIC_URL must be an https:// URL (Zalo requirement)",
                self.name,
            )
            await self._http_client.aclose()
            self._http_client = None
            return False
        self._webhook_public_url = _webhook_url_for_path(
            validated_public_url, self._webhook_path
        )
        slen = len(self._webhook_secret)
        if slen < 8 or slen > 256:
            logger.error(
                "[%s] ZALO_WEBHOOK_SECRET must be 8–256 characters (see setWebhook docs)",
                self.name,
            )
            await self._http_client.aclose()
            self._http_client = None
            return False

        app = web.Application(client_max_size=WEBHOOK_MAX_BYTES)
        app.router.add_post(self._webhook_path, self._handle_webhook_post)

        self._webhook_runner = web.AppRunner(app)
        await self._webhook_runner.setup()
        self._webhook_site = web.TCPSite(
            self._webhook_runner, self._webhook_host, self._webhook_port
        )
        try:
            await self._webhook_site.start()
        except OSError as e:
            logger.error(
                "[%s] Webhook bind failed %s:%s: %s",
                self.name,
                self._webhook_host,
                self._webhook_port,
                e,
            )
            await self._webhook_runner.cleanup()
            self._webhook_runner = None
            self._webhook_site = None
            await self._http_client.aclose()
            self._http_client = None
            return False

        sw_url = _api_url(self._token, "setWebhook", self._api_base)
        try:
            resp = await self._http_client.post(
                sw_url,
                json={
                    "url": self._webhook_public_url,
                    "secret_token": self._webhook_secret,
                },
            )
            body = resp.json() if resp.content else {}
            if not body.get("ok"):
                error = _safe_api_error(
                    body, resp.status_code, self._token, self._webhook_secret
                )
                logger.error("[%s] setWebhook failed: %s", self.name, error)
                await self._disconnect_webhook_server()
                await self._http_client.aclose()
                self._http_client = None
                return False
        except Exception as e:
            logger.error(
                "[%s] setWebhook request failed: %s", self.name, type(e).__name__
            )
            await self._disconnect_webhook_server()
            await self._http_client.aclose()
            self._http_client = None
            return False

        self._webhook_registered = True
        self._delete_webhook_on_disconnect = True
        self._mark_connected()
        logger.info(
            "[%s] Webhook listening on http://%s:%s%s (public URL registered with Zalo)",
            self.name,
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
        )
        return True

    async def _handle_webhook_post(self, request: Any) -> Any:
        token_hdr = request.headers.get("X-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(
            token_hdr.encode("utf-8"), self._webhook_secret.encode("utf-8")
        ):
            logger.warning("[%s] Webhook rejected: bad secret token", self.name)
            return web.Response(status=403, text="Forbidden")

        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="Bad JSON")

        if not isinstance(data, dict):
            return web.Response(text="ok")

        if not data.get("ok", True):
            return web.Response(text="ok")

        result = data.get("result")
        if isinstance(result, dict) and (
            result.get("event_name") or result.get("message")
        ):
            await self._dispatch_update(result)
        return web.Response(text="ok")

    async def _disconnect_webhook_server(self) -> None:
        site = self._webhook_site
        self._webhook_site = None
        if site:
            try:
                await site.stop()
            except Exception as exc:
                logger.warning(
                    "[%s] Webhook site cleanup failed: %s",
                    self.name,
                    type(exc).__name__,
                )
        runner = self._webhook_runner
        self._webhook_runner = None
        if runner:
            try:
                await runner.cleanup()
            except Exception as exc:
                logger.warning(
                    "[%s] Webhook runner cleanup failed: %s",
                    self.name,
                    type(exc).__name__,
                )

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()
        try:
            if (
                self._connection_mode == "webhook"
                and self._delete_webhook_on_disconnect
                and self._http_client
            ):
                try:
                    dw = _api_url(self._token, "deleteWebhook", self._api_base)
                    await self._http_client.post(dw)
                    logger.info("[%s] deleteWebhook called", self.name)
                except Exception as e:
                    logger.warning(
                        "[%s] deleteWebhook failed (non-fatal): %s",
                        self.name,
                        type(e).__name__,
                    )
            self._webhook_registered = False
            self._delete_webhook_on_disconnect = False
            await self._disconnect_webhook_server()

            task = self._poll_task
            self._poll_task = None
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "[%s] Poll task cleanup failed: %s",
                        self.name,
                        type(exc).__name__,
                    )
        finally:
            try:
                await self._close_http_client()
            finally:
                self._seen_messages.clear()
                self._release_platform_lock()
        logger.info("[%s] Disconnected", self.name)

    def _poll_backoff_sleep(self, backoff_idx: int) -> float:
        cap = _POLL_BACKOFF_SEC[min(backoff_idx, len(_POLL_BACKOFF_SEC) - 1)]
        jitter = cap * _MAX_BACKOFF_JITTER_RATIO * random.random()
        return cap + jitter

    async def _poll_loop(self) -> None:
        assert self._http_client is not None
        url = _api_url(self._token, "getUpdates", self._api_base)
        backoff_idx = 0
        while self._running:
            try:
                resp = await self._http_client.post(
                    url,
                    json={"timeout": str(self._poll_timeout)},
                )
                if resp.status_code != 200:
                    logger.warning(
                        "[%s] getUpdates HTTP %s", self.name, resp.status_code
                    )
                    await asyncio.sleep(self._poll_backoff_sleep(backoff_idx))
                    backoff_idx = min(backoff_idx + 1, len(_POLL_BACKOFF_SEC) - 1)
                    continue

                data = resp.json()
                if not data.get("ok"):
                    desc = _safe_api_error(
                        data, resp.status_code, self._token, self._webhook_secret
                    )
                    logger.warning("[%s] getUpdates error: %s", self.name, desc)
                    if "webhook" in desc.lower():
                        logger.warning(
                            "[%s] Webhook may be active — deleteWebhook before getUpdates, "
                            "or switch to connection_mode webhook in config.",
                            self.name,
                        )
                    await asyncio.sleep(self._poll_backoff_sleep(backoff_idx))
                    backoff_idx = min(backoff_idx + 1, len(_POLL_BACKOFF_SEC) - 1)
                    continue

                backoff_idx = 0
                for item in _iter_updates(data.get("result")):
                    await self._dispatch_update(item)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning("[%s] Poll error: %s", self.name, type(e).__name__)
                await asyncio.sleep(self._poll_backoff_sleep(backoff_idx))
                backoff_idx = min(backoff_idx + 1, len(_POLL_BACKOFF_SEC) - 1)

    def _base_source_and_meta(self, msg: Dict[str, Any]) -> Optional[tuple]:
        from_o = msg.get("from") or {}
        if from_o.get("is_bot"):
            return None

        msg_id = str(msg.get("message_id") or "")
        if msg_id and self._is_duplicate(msg_id):
            return None

        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id") or "")
        if not chat_id:
            return None

        chat_type_raw = str(chat.get("chat_type") or "PRIVATE").upper()
        chat_type = "dm" if chat_type_raw == "PRIVATE" else "group"

        user_id = str(from_o.get("id") or "")
        user_name = (from_o.get("display_name") or user_id or "user").strip()

        date_ms = msg.get("date")
        try:
            if date_ms is not None:
                ts = datetime.fromtimestamp(float(date_ms) / 1000.0, tz=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            ts = datetime.now(tz=timezone.utc)

        source = self.build_source(
            chat_id=chat_id,
            chat_name=user_name if chat_type == "dm" else None,
            chat_type=chat_type,
            user_id=user_id,
            user_name=user_name,
        )
        return source, ts, msg_id

    async def _dispatch_update(self, item: Dict[str, Any]) -> None:
        event_name = item.get("event_name")
        msg = item.get("message") or {}

        if event_name == "message.unsupported.received":
            logger.debug("[%s] Unsupported message event (policy or type)", self.name)
            return

        meta = self._base_source_and_meta(msg)
        if meta is None:
            return
        source, ts, msg_id = meta

        if event_name == "message.text.received":
            text = (msg.get("text") or "").strip()
            if not text:
                return
            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=msg_id or None,
                raw_message=item,
                timestamp=ts,
            )
            await self.handle_message(event)
            return

        if event_name == "message.image.received":
            photo_url = (msg.get("photo_url") or msg.get("photo") or "").strip()
            caption = (msg.get("caption") or "").strip()
            text = caption or "[Photo]"
            media_urls: List[str] = []
            sender_authorized = self._is_sender_authorized(
                source.user_id, source.chat_type, source.chat_id
            )
            if photo_url and sender_authorized is not False:
                try:
                    media_urls.append(await cache_image_from_url(photo_url))
                except Exception as e:
                    logger.warning(
                        "[%s] Image cache failed: %s", self.name, type(e).__name__
                    )
                    text = f"{text}\n[Image unavailable]".strip()
            elif photo_url:
                logger.info(
                    "[%s] Skipping image fetch for unauthorized sender", self.name
                )
            event = MessageEvent(
                text=text,
                message_type=MessageType.PHOTO,
                source=source,
                message_id=msg_id or None,
                raw_message=item,
                timestamp=ts,
                media_urls=media_urls,
                media_types=["image"] * len(media_urls),
            )
            await self.handle_message(event)
            return

        if event_name == "message.sticker.received":
            sticker_id = (msg.get("sticker") or "").strip()
            sticker_url = (msg.get("url") or "").strip()
            parts = [p for p in (sticker_id, sticker_url) if p]
            text = f"[Sticker] {' | '.join(parts)}" if parts else "[Sticker]"
            event = MessageEvent(
                text=text,
                message_type=MessageType.STICKER,
                source=source,
                message_id=msg_id or None,
                raw_message=item,
                timestamp=ts,
            )
            await self.handle_message(event)
            return

    def _is_duplicate(self, msg_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        _ = reply_to, metadata
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        url = _api_url(self._token, "sendMessage", self._api_base)
        last_mid: Optional[str] = None
        try:
            for chunk in chunks:
                resp = await self._http_client.post(
                    url,
                    json={"chat_id": str(chat_id), "text": chunk},
                )
                try:
                    body: Dict[str, Any] = resp.json()
                except Exception:
                    return SendResult(
                        success=False,
                        error=f"Invalid JSON from Zalo API (HTTP {resp.status_code})",
                    )
                if not body.get("ok"):
                    err = _safe_api_error(
                        body,
                        resp.status_code,
                        self._token,
                        self._webhook_secret,
                        limit=500,
                    )
                    return SendResult(success=False, error=err)
                res = body.get("result") or {}
                if isinstance(res, dict):
                    last_mid = str(res.get("message_id") or last_mid or "")
            return SendResult(success=True, message_id=last_mid)
        except httpx.TimeoutException:
            return SendResult(
                success=False, error="Timeout sending to Zalo", retryable=True
            )
        except Exception as e:
            logger.error("[%s] send error: %s", self.name, type(e).__name__)
            return SendResult(success=False, error="Zalo send failed")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        _ = metadata
        if not self._http_client:
            return
        try:
            url = _api_url(self._token, "sendChatAction", self._api_base)
            await self._http_client.post(
                url,
                json={"chat_id": str(chat_id), "action": "typing"},
            )
        except Exception as e:
            logger.debug("[%s] sendChatAction failed: %s", self.name, type(e).__name__)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        _ = reply_to, metadata
        if not image_url:
            return SendResult(success=False, error="No image URL")
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        url = _api_url(self._token, "sendPhoto", self._api_base)
        payload: Dict[str, Any] = {"chat_id": str(chat_id), "photo": image_url}
        cap = (caption or "").strip()
        if cap:
            payload["caption"] = cap[:MAX_MESSAGE_LENGTH]
        try:
            resp = await self._http_client.post(url, json=payload)
            try:
                body: Dict[str, Any] = resp.json()
            except Exception:
                return SendResult(
                    success=False, error=f"Invalid JSON (HTTP {resp.status_code})"
                )
            if not body.get("ok"):
                err = _safe_api_error(
                    body, resp.status_code, self._token, self._webhook_secret, limit=500
                )
                return SendResult(success=False, error=err)
            res = body.get("result") or {}
            mid = str(res.get("message_id") or "") if isinstance(res, dict) else None
            return SendResult(success=True, message_id=mid)
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sendPhoto", retryable=True)
        except Exception as e:
            logger.error("[%s] sendPhoto error: %s", self.name, type(e).__name__)
            return SendResult(success=False, error="Zalo sendPhoto failed")

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a sticker via sendSticker (animation_url = Zalo sticker id from stickers.zaloapp.com)."""
        _ = reply_to, metadata
        sticker = (animation_url or "").strip()
        if not sticker:
            return SendResult(success=False, error="No sticker id")
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")
        url = _api_url(self._token, "sendSticker", self._api_base)
        try:
            resp = await self._http_client.post(
                url,
                json={"chat_id": str(chat_id), "sticker": sticker},
            )
            try:
                body: Dict[str, Any] = resp.json()
            except Exception:
                return SendResult(
                    success=False, error=f"Invalid JSON (HTTP {resp.status_code})"
                )
            if not body.get("ok"):
                err = _safe_api_error(
                    body, resp.status_code, self._token, self._webhook_secret, limit=500
                )
                return SendResult(success=False, error=err)
            res = body.get("result") or {}
            mid = str(res.get("message_id") or "") if isinstance(res, dict) else None
            return SendResult(success=True, message_id=mid)
        except Exception as e:
            logger.error("[%s] sendSticker error: %s", self.name, type(e).__name__)
            return SendResult(success=False, error="Zalo sendSticker failed")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}


# Compatibility name used by early local installations of the adapter.
ZaloAdapter = ZaloBotAdapter


def validate_config(config: PlatformConfig) -> bool:
    """Validate credentials plus webhook-only requirements before startup."""
    extra = getattr(config, "extra", {}) or {}
    token_configured = bool(
        _env_value("ZALO_BOT_TOKEN", "").strip()
        or str(getattr(config, "token", "") or "").strip()
        or str(extra.get("bot_token") or "").strip()
    )
    if not token_configured:
        return False
    mode = (
        str(
            _env_value("ZALO_CONNECTION_MODE")
            or extra.get("connection_mode")
            or "polling"
        )
        .strip()
        .lower()
    )
    if mode != "webhook":
        return True
    public_url = _validated_https_url(
        str(
            _env_value("ZALO_WEBHOOK_PUBLIC_URL")
            or extra.get("webhook_public_url")
            or ""
        )
    )
    secret = str(
        _env_value("ZALO_WEBHOOK_SECRET") or extra.get("webhook_secret") or ""
    ).strip()
    return bool(public_url and 8 <= len(secret) <= 256)


def is_connected(config: PlatformConfig) -> bool:
    """Surface configured state to gateway setup/status before construction."""
    return validate_config(config)


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed a plugin ``PlatformConfig`` from ZALO_* environment variables."""
    token = _env_value("ZALO_BOT_TOKEN", "").strip()
    seed: Dict[str, Any] = {}
    if token:
        seed["bot_token"] = token
    mode = _env_value("ZALO_CONNECTION_MODE", "").strip().lower()
    if mode in {"polling", "webhook"}:
        seed["connection_mode"] = mode

    text_fields = {
        "ZALO_API_BASE": "api_base",
        "ZALO_POLL_TIMEOUT": "poll_timeout",
        "ZALO_WEBHOOK_PUBLIC_URL": "webhook_public_url",
        "ZALO_WEBHOOK_SECRET": "webhook_secret",
        "ZALO_WEBHOOK_HOST": "webhook_host",
        "ZALO_WEBHOOK_PATH": "webhook_path",
    }
    for env_name, extra_name in text_fields.items():
        value = _env_value(env_name, "").strip()
        if value:
            seed[extra_name] = value

    port = _env_value("ZALO_WEBHOOK_PORT", "").strip()
    if port:
        try:
            seed["webhook_port"] = int(port)
        except ValueError:
            pass

    home = _env_value("ZALO_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": _env_value("ZALO_HOME_CHANNEL_NAME", "").strip() or "Home",
        }
    return seed or None


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """One-shot Bot API delivery for send_message and out-of-process cron."""
    _ = thread_id, force_document
    extra = getattr(pconfig, "extra", {}) or {}
    token = str(
        _env_value("ZALO_BOT_TOKEN")
        or getattr(pconfig, "token", "")
        or extra.get("bot_token")
        or ""
    ).strip()
    if not token:
        return {"error": "ZALO_BOT_TOKEN not configured for Zalo"}
    if not chat_id:
        return {"error": "Zalo standalone send: missing chat_id"}

    content = (message or "").strip()
    if not content:
        if media_files:
            return {
                "error": (
                    "Zalo standalone send requires text; local attachments "
                    "cannot be uploaded because sendPhoto requires an HTTPS URL"
                )
            }
        return {"error": "Zalo standalone send: empty message"}

    requested_api_base = str(
        _env_value("ZALO_API_BASE") or extra.get("api_base") or ZALO_API_BASE
    )
    api_base = _validated_https_url(requested_api_base, default=ZALO_API_BASE)
    chunks = BasePlatformAdapter.truncate_message(content, MAX_MESSAGE_LENGTH)
    last_message_id: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for chunk in chunks:
                response = await client.post(
                    _api_url(token, "sendMessage", api_base),
                    json={"chat_id": str(chat_id), "text": chunk},
                )
                try:
                    body = response.json()
                except Exception:
                    return {
                        "error": (
                            f"Invalid JSON from Zalo API (HTTP {response.status_code})"
                        )
                    }
                if not isinstance(body, dict) or not body.get("ok"):
                    return {
                        "error": _safe_api_error(
                            body, response.status_code, token, limit=500
                        )
                    }
                result = body.get("result") or {}
                if isinstance(result, dict) and result.get("message_id"):
                    last_message_id = str(result["message_id"])
    except httpx.TimeoutException:
        return {"error": "Timeout sending to Zalo"}
    except Exception:
        return {"error": "Zalo send failed"}

    return {
        "success": True,
        "platform": "zalo",
        "chat_id": str(chat_id),
        "message_id": last_message_id,
    }


def interactive_setup() -> None:
    """Configure the bundled Zalo platform plugin from the gateway wizard."""
    print()
    print("Zalo Bot Platform setup")
    print("-----------------------")
    print("Create a bot with Zalo Bot Manager, then copy its Bot Token.")
    print("Docs: https://bot.zapps.me/docs/create-bot/")
    print()

    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_cli.secret_prompt import masked_secret_prompt
    except ImportError:
        print("Set ZALO_BOT_TOKEN manually in ~/.hermes/.env")
        return

    def _prompt(name: str, label: str, *, secret: bool = False) -> str:
        existing = get_env_value(name)
        suffix = " [keep current]" if existing else ""
        try:
            value = (
                masked_secret_prompt(f"{label}{suffix}: ")
                if secret
                else input(f"{label}{suffix}: ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        if value:
            save_env_value(name, value)
            return value
        return str(existing or "")

    _prompt("ZALO_BOT_TOKEN", "Bot token", secret=True)
    _prompt("ZALO_ALLOWED_USERS", "Allowed user IDs (comma-separated)")
    _prompt("ZALO_HOME_CHANNEL", "Home chat ID for cron (optional)")
    mode = (
        _prompt("ZALO_CONNECTION_MODE", "Connection mode [polling/webhook]")
        .strip()
        .lower()
    )
    if mode == "webhook":
        _prompt("ZALO_WEBHOOK_PUBLIC_URL", "Public HTTPS webhook URL")
        _prompt("ZALO_WEBHOOK_SECRET", "Webhook secret (8-256 chars)", secret=True)
        _prompt("ZALO_WEBHOOK_HOST", "Local bind host [0.0.0.0]")
        _prompt("ZALO_WEBHOOK_PORT", "Local bind port [8790]")
    print("Zalo configuration saved. Restart the gateway to apply it.")


def register(ctx) -> None:
    """Register Zalo through the current plugin platform interface."""
    ctx.register_platform(
        name="zalo",
        label="Zalo Bot",
        adapter_factory=lambda cfg: ZaloBotAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["ZALO_BOT_TOKEN"],
        install_hint=(
            "Polling uses core httpx; webhook mode uses "
            "pip install 'hermes-agent[zalo]'"
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="ZALO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="ZALO_ALLOWED_USERS",
        allow_all_env="ZALO_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Zalo Bot Platform. Keep responses concise "
            "and use plain text because Zalo Bot messages do not render "
            "Markdown. Text messages are limited to 2000 characters; Hermes "
            "will split longer replies automatically."
        ),
    )
