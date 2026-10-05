"""Chatwork gateway adapter for Hermes Agent.

Receives Chatwork Webhook events through a signature-verified aiohttp endpoint
and sends replies through the official Chatwork API v2.  The adapter is a
platform plugin, so it does not require changes to Hermes core.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

CHATWORK_API_BASE = "https://api.chatwork.com/v2"
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/chatwork/webhook"
WEBHOOK_BODY_MAX_BYTES = 1_048_576
MAX_MESSAGE_LENGTH = 10_000
_REPLY_CONTEXT_LIMIT = 1_000
_TO_TAG_RE = re.compile(r"^\[To:\d+\]")


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def verify_chatwork_signature(body: bytes, signature: str, webhook_token: str) -> bool:
    """Verify ``x-chatworkwebhooksignature`` for the raw request body.

    Chatwork exposes the webhook token as Base64.  The decoded bytes are the
    HMAC-SHA256 key; the resulting digest is Base64-encoded for comparison.
    """
    if body is None or not signature or not webhook_token:
        return False
    try:
        key = base64.b64decode(webhook_token, validate=True)
        expected = base64.b64encode(
            hmac.new(key, body, hashlib.sha256).digest()
        ).decode("ascii")
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected.encode("ascii"), signature.encode("ascii", "ignore"))


def strip_chatwork_mentions(text: str) -> str:
    """Remove Chatwork addressing tags without dropping inline message text.

    The UI commonly emits ``[To:id] display name`` on a line before the
    message, while Chatwork's webhook example uses ``[To:id]message`` inline.
    Address-only prefix lines are removed; a tag on the final/only line is
    stripped while preserving everything after it.
    """
    lines = (text or "").splitlines()
    normalized: List[str] = []
    for index, line in enumerate(lines):
        if not _TO_TAG_RE.match(line):
            normalized.append(line)
            continue
        if index < len(lines) - 1:
            continue
        normalized.append(_TO_TAG_RE.sub("", line, count=1))
    return "\n".join(normalized).strip()


class ChatworkAdapter(BasePlatformAdapter):
    """Chatwork Webhook + REST API v2 adapter."""

    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("chatwork"))
        extra = getattr(config, "extra", {}) or {}
        self.api_token = (
            os.getenv("CHATWORK_API_TOKEN") or extra.get("api_token", "")
        ).strip()
        self.webhook_token = (
            os.getenv("CHATWORK_WEBHOOK_TOKEN") or extra.get("webhook_token", "")
        ).strip()
        self.api_base = (
            os.getenv("CHATWORK_API_BASE") or extra.get("api_base", CHATWORK_API_BASE)
        ).rstrip("/")
        self.webhook_host = (
            os.getenv("CHATWORK_HOST") or extra.get("host", DEFAULT_WEBHOOK_HOST)
        ).strip()
        self.webhook_port = int(
            os.getenv("CHATWORK_PORT") or extra.get("port", DEFAULT_WEBHOOK_PORT)
        )
        path = os.getenv("CHATWORK_WEBHOOK_PATH") or extra.get(
            "webhook_path", DEFAULT_WEBHOOK_PATH
        )
        self.webhook_path = path if str(path).startswith("/") else f"/{path}"
        self.allowed_rooms = _csv_set(
            os.getenv("CHATWORK_ALLOWED_ROOMS")
            or str(extra.get("allowed_rooms", ""))
        )

        self._session: Any = None
        self._app: Any = None
        self._runner: Any = None
        self._site: Any = None
        self._account_id = ""
        self._room_cache: Dict[str, Dict[str, Any]] = {}
        self._reply_context: OrderedDict[str, str] = OrderedDict()
        self._webhook_tasks: set[asyncio.Task] = set()

    def _headers(self) -> Dict[str, str]:
        return {"x-chatworktoken": self.api_token}

    async def _api_get(self, path: str) -> Dict[str, Any]:
        import aiohttp

        if ".." in path:
            return {}
        try:
            async with self._session.get(
                f"{self.api_base}/{path.lstrip('/')}",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    logger.error(
                        "Chatwork API GET %s -> %s: %s",
                        path,
                        response.status,
                        body[:200],
                    )
                    return {}
                data = await response.json()
                return data if isinstance(data, dict) else {}
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Chatwork API GET %s failed: %s", path, exc)
            return {}

    async def _post_message(self, room_id: str, body: str) -> SendResult:
        import aiohttp

        try:
            async with self._session.post(
                f"{self.api_base}/rooms/{room_id}/messages",
                headers=self._headers(),
                data={"body": body},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_body = await response.text()
                if response.status >= 400:
                    retry_after = None
                    if response.status == 429:
                        try:
                            retry_after = float(response.headers.get("Retry-After", ""))
                        except (TypeError, ValueError):
                            pass
                    return SendResult(
                        success=False,
                        error=f"Chatwork API returned {response.status}: {response_body[:200]}",
                        retryable=response.status == 429 or response.status >= 500,
                        retry_after=retry_after,
                    )
                try:
                    data = json.loads(response_body)
                except json.JSONDecodeError:
                    data = {}
                message_id = data.get("message_id") if isinstance(data, dict) else None
                return SendResult(
                    success=True,
                    message_id=str(message_id) if message_id is not None else None,
                    raw_response=data,
                )
        except (aiohttp.ClientError, TimeoutError) as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.api_token or not self.webhook_token:
            logger.error("Chatwork: API token and webhook token are required")
            return False

        try:
            import aiohttp
            from aiohttp import web
        except ImportError:
            self._set_fatal_error(
                "missing_dep",
                "aiohttp is required for Chatwork",
                retryable=False,
            )
            return False

        self._session = aiohttp.ClientSession()
        me = await self._api_get("me")
        if not me.get("account_id"):
            logger.error("Chatwork: authentication failed; check CHATWORK_API_TOKEN")
            await self._session.close()
            self._session = None
            return False
        self._account_id = str(me["account_id"])

        self._app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        self._app.router.add_post(self.webhook_path, self._handle_webhook)
        self._app.router.add_get(f"{self.webhook_path}/health", self._handle_health)
        self._runner = web.AppRunner(self._app)
        try:
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner, self.webhook_host, self.webhook_port
            )
            await self._site.start()
        except OSError as exc:
            self._set_fatal_error(
                "bind_failed",
                f"Could not bind Chatwork webhook on {self.webhook_host}:{self.webhook_port}: {exc}",
                retryable=True,
            )
            await self.disconnect()
            return False

        self._mark_connected()
        logger.info(
            "Chatwork: authenticated as account %s; webhook listening on %s:%s%s",
            self._account_id,
            self.webhook_host,
            self.webhook_port,
            self.webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for task in list(self._webhook_tasks):
            task.cancel()
        if self._webhook_tasks:
            await asyncio.gather(*self._webhook_tasks, return_exceptions=True)
        self._webhook_tasks.clear()
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None
        self._app = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _handle_health(self, request) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok", "platform": "chatwork"})

    async def _handle_webhook(self, request) -> Any:
        from aiohttp import web

        try:
            body = await request.read()
        except Exception:
            return web.Response(status=400, text="bad request")
        if len(body) > WEBHOOK_BODY_MAX_BYTES:
            return web.Response(status=413, text="payload too large")

        signature = request.headers.get("x-chatworkwebhooksignature", "")
        if not signature:
            signature = request.query.get("chatwork_webhook_signature", "")
        if not verify_chatwork_signature(body, signature, self.webhook_token):
            return web.Response(status=401, text="invalid signature")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.Response(status=400, text="bad json")

        # Chatwork requires a response within 10 seconds and does not retry
        # failed deliveries. A room metadata lookup can take longer, so ACK the
        # authenticated payload immediately and finish dispatch in background.
        task = asyncio.create_task(self._dispatch_webhook_safe(payload))
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)
        return web.Response(status=200, text="ok")

    async def _dispatch_webhook_safe(self, payload: Dict[str, Any]) -> None:
        try:
            await self._dispatch_webhook(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Chatwork: webhook dispatch failed")

    async def _dispatch_webhook(self, payload: Dict[str, Any]) -> None:
        event_type = str(payload.get("webhook_event_type") or "")
        if event_type not in {"message_created", "mention_to_me"}:
            logger.debug("Chatwork: ignoring event type %r", event_type)
            return

        event = payload.get("webhook_event") or {}
        room_id = str(event.get("room_id") or "")
        # ``mention_to_me`` uses from_account_id, while room-wide
        # ``message_created`` events use account_id.
        sender_id = str(
            event.get("from_account_id") or event.get("account_id") or ""
        )
        message_id = str(event.get("message_id") or "")
        if not room_id or not sender_id or not message_id:
            return
        if sender_id == self._account_id:
            return
        if self.allowed_rooms and room_id not in self.allowed_rooms:
            logger.info("Chatwork: ignoring message from non-allowed room %s", room_id)
            return

        text = strip_chatwork_mentions(str(event.get("body") or ""))
        if not text:
            return

        room_info = await self.get_chat_info(room_id)
        self._reply_context[message_id] = sender_id
        self._reply_context.move_to_end(message_id)
        while len(self._reply_context) > _REPLY_CONTEXT_LIMIT:
            self._reply_context.popitem(last=False)

        source = self.build_source(
            chat_id=room_id,
            chat_name=room_info.get("name") or room_id,
            chat_type=room_info.get("type") or "group",
            user_id=sender_id,
            user_name=sender_id,
        )
        await self.handle_message(
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=payload,
                message_id=message_id,
            )
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content:
            return SendResult(success=True)
        if self._session is None:
            return SendResult(success=False, error="Chatwork adapter is not connected")

        chunks = self.truncate_message(content, MAX_MESSAGE_LENGTH)
        last_result = SendResult(success=True)
        reply_account_id = self._reply_context.get(str(reply_to or ""))
        for index, chunk in enumerate(chunks):
            body = chunk
            if index == 0 and reply_to and reply_account_id:
                body = f"[rp aid={reply_account_id} to={chat_id}-{reply_to}]\n{chunk}"
            last_result = await self._post_message(str(chat_id), body)
            if not last_result.success:
                return last_result
        return last_result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        key = str(chat_id)
        cached = self._room_cache.get(key)
        if cached is not None:
            return cached
        if self._session is None:
            return {"name": key, "type": "group"}
        data = await self._api_get(f"rooms/{key}")
        room_type = "dm" if data.get("type") in {"my", "direct"} else "group"
        info = {"name": data.get("name") or key, "type": room_type}
        self._room_cache[key] = info
        return info


def check_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        (os.getenv("CHATWORK_API_TOKEN") or extra.get("api_token"))
        and (os.getenv("CHATWORK_WEBHOOK_TOKEN") or extra.get("webhook_token"))
    )


def _env_enablement() -> Optional[Dict[str, Any]]:
    if not (os.getenv("CHATWORK_API_TOKEN") and os.getenv("CHATWORK_WEBHOOK_TOKEN")):
        return None
    seeded: Dict[str, Any] = {}
    if os.getenv("CHATWORK_HOST"):
        seeded["host"] = os.environ["CHATWORK_HOST"]
    if os.getenv("CHATWORK_PORT"):
        try:
            seeded["port"] = int(os.environ["CHATWORK_PORT"])
        except ValueError:
            pass
    if os.getenv("CHATWORK_WEBHOOK_PATH"):
        seeded["webhook_path"] = os.environ["CHATWORK_WEBHOOK_PATH"]
    home = os.getenv("CHATWORK_HOME_CHANNEL")
    if home:
        seeded["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("CHATWORK_HOME_CHANNEL_NAME", "Chatwork Home"),
        }
    return seeded


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    import aiohttp

    extra = getattr(pconfig, "extra", {}) or {}
    token = (os.getenv("CHATWORK_API_TOKEN") or extra.get("api_token", "")).strip()
    api_base = (os.getenv("CHATWORK_API_BASE") or extra.get("api_base", CHATWORK_API_BASE)).rstrip("/")
    if not token or not chat_id:
        return {"error": "Chatwork standalone send: missing API token or room ID"}

    content = message or ""
    if media_files:
        content = f"{content}\n\n" + "\n".join(f"Attachment: {path}" for path in media_files)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{api_base}/rooms/{chat_id}/messages",
                headers={"x-chatworktoken": token},
                data={"body": content},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response_body = await response.text()
                if response.status >= 400:
                    return {"error": f"Chatwork API returned {response.status}: {response_body[:200]}"}
                try:
                    payload = json.loads(response_body)
                except json.JSONDecodeError:
                    payload = {}
                return {"success": True, "message_id": payload.get("message_id")}
        except (aiohttp.ClientError, TimeoutError) as exc:
            return {"error": str(exc)}


def interactive_setup() -> None:
    print("\nChatwork setup\n--------------")
    print("Create an API token and a webhook in Chatwork, then enter both tokens.")
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_cli.secret_prompt import masked_secret_prompt
    except ImportError:
        print("Set CHATWORK_API_TOKEN and CHATWORK_WEBHOOK_TOKEN in ~/.hermes/.env")
        return

    def _secret(var: str, label: str) -> None:
        existing = get_env_value(var)
        suffix = " [keep current]" if existing else ""
        try:
            value = masked_secret_prompt(f"{label}{suffix}: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            save_env_value(var, value)

    _secret("CHATWORK_API_TOKEN", "API token")
    _secret("CHATWORK_WEBHOOK_TOKEN", "Webhook token")
    try:
        home = input("Home room ID for cron delivery (optional): ").strip()
        allowed = input("Allowed user account IDs (comma-separated): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if home:
        save_env_value("CHATWORK_HOME_CHANNEL", home)
    if allowed:
        save_env_value("CHATWORK_ALLOWED_USERS", allowed)
    print(
        "Done. Expose port 8647 over HTTPS and set the Chatwork webhook URL to "
        "https://<your-host>/chatwork/webhook."
    )


def register(ctx) -> None:
    ctx.register_platform(
        name="chatwork",
        label="Chatwork",
        adapter_factory=lambda cfg: ChatworkAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=validate_config,
        required_env=["CHATWORK_API_TOKEN", "CHATWORK_WEBHOOK_TOKEN"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="CHATWORK_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="CHATWORK_ALLOWED_USERS",
        allow_all_env="CHATWORK_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Chatwork. Chatwork does not render Markdown; "
            "keep messages concise and prefer plain text. Replies are represented "
            "with Chatwork reply tags when the triggering message is available."
        ),
    )
