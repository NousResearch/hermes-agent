"""Sendblue iMessage platform adapter for Hermes Agent.

This bundled platform plugin follows the Hermes platform-plugin path:
it registers a ``BasePlatformAdapter`` subclass, runs a Sendblue receive
webhook server for inbound messages, and sends replies through Sendblue's
REST API without touching Hermes core files.

Sendblue requirements reflected here:

* ``from_number`` is required for every send.
* Recipients should keep seeing the same Sendblue number, so the adapter
  keeps a sticky recipient -> from_number map.
* Inbound webhooks must be acknowledged quickly with a 2xx response.
* 429 responses are rate limits and should be surfaced as retryable.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by requirement tests
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.helpers import redact_phone

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://api.sendblue.com"
DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8650
DEFAULT_WEBHOOK_PATH = "/sendblue/webhook"
MAX_MESSAGE_LENGTH = 18_996
WEBHOOK_BODY_MAX_BYTES = 1_048_576
DEDUP_WINDOW_SECONDS = 300
DEDUP_MAX_SIZE = 2_000

_SECRET_HEADER_CANDIDATES = (
    "sb-signing-secret",
    "x-sendblue-webhook-secret",
    "sendblue-webhook-secret",
    "x-webhook-secret",
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str) -> List[str]:
    return [
        part.strip() for part in value.replace("\n", ",").split(",") if part.strip()
    ]


def _normalize_webhook_path(path: str) -> str:
    path = (path or DEFAULT_WEBHOOK_PATH).strip()
    if not path:
        return DEFAULT_WEBHOOK_PATH
    return path if path.startswith("/") else f"/{path}"


def _default_sticky_state_path() -> Path:
    home = os.getenv("HERMES_HOME", "").strip()
    base = Path(home).expanduser() if home else Path("~/.hermes").expanduser()
    return base / "sendblue_sticky_senders.json"


def _redact_target(value: str) -> str:
    if value.startswith("group:"):
        return f"group:{value[6:12]}..."
    return redact_phone(value)


def _chat_id_for_payload(payload: Dict[str, Any]) -> str:
    group_id = str(payload.get("group_id") or "").strip()
    if group_id:
        return f"group:{group_id}"
    return str(
        payload.get("from_number")
        or payload.get("number")
        or payload.get("to_number")
        or ""
    ).strip()


def _sender_for_payload(payload: Dict[str, Any]) -> str:
    return str(payload.get("from_number") or payload.get("number") or "").strip()


def _sendblue_number_for_payload(payload: Dict[str, Any]) -> str:
    return str(
        payload.get("sendblue_number")
        or payload.get("to_number")
        or payload.get("from_number")
        or ""
    ).strip()


def _media_url_for_payload(payload: Dict[str, Any]) -> str:
    return str(payload.get("media_url") or "").strip()


def _media_type_for_payload(payload: Dict[str, Any]) -> str:
    return str(
        payload.get("message_type")
        or payload.get("media_type")
        or payload.get("content_type")
        or "attachment"
    ).strip()


def _message_type_for_payload(payload: Dict[str, Any]) -> MessageType:
    media_type = _media_type_for_payload(payload).lower()
    if not _media_url_for_payload(payload):
        return MessageType.TEXT
    if "image" in media_type or media_type in {"photo", "picture"}:
        return MessageType.PHOTO
    if "video" in media_type:
        return MessageType.VIDEO
    if "audio" in media_type:
        return MessageType.AUDIO
    if "voice" in media_type:
        return MessageType.VOICE
    return MessageType.DOCUMENT


def _response_message_id(data: Dict[str, Any]) -> Optional[str]:
    value = data.get("message_handle") or data.get("id") or data.get("message_id")
    return str(value) if value else None


def _retry_after(headers: Any) -> Optional[float]:
    raw = None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        delay = float(str(raw).strip())
    except ValueError:
        return None
    return delay if delay >= 0 else None


def _api_headers(api_key_id: str, api_secret_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "sb-api-key-id": api_key_id,
        "sb-api-secret-key": api_secret_key,
    }


async def _post_sendblue(
    session: "aiohttp.ClientSession",
    *,
    api_base_url: str,
    api_key_id: str,
    api_secret_key: str,
    endpoint: str,
    payload: Dict[str, Any],
) -> Tuple[int, Dict[str, str], Dict[str, Any] | str]:
    url = f"{api_base_url.rstrip('/')}{endpoint}"
    async with session.post(
        url,
        json=payload,
        headers=_api_headers(api_key_id, api_secret_key),
    ) as resp:
        text = await resp.text()
        try:
            body: Dict[str, Any] | str = json.loads(text) if text else {}
        except json.JSONDecodeError:
            body = text
        return resp.status, dict(resp.headers), body


def _send_result_from_response(
    status: int,
    headers: Dict[str, str],
    body: Dict[str, Any] | str,
) -> SendResult:
    if 200 <= status < 300:
        data = body if isinstance(body, dict) else {}
        return SendResult(
            success=True, message_id=_response_message_id(data), raw_response=body
        )

    detail = body
    if isinstance(body, dict):
        detail = (
            body.get("error_message")
            or body.get("message")
            or body.get("error")
            or body
        )
    retry_after = _retry_after(headers)
    return SendResult(
        success=False,
        error=f"Sendblue HTTP {status}: {str(detail)[:300]}",
        raw_response=body,
        retryable=status == 429 or 500 <= status < 600,
        retry_after=retry_after,
        error_kind="rate_limited" if status == 429 else None,
    )


def _standalone_dict(result: SendResult, chat_id: str) -> Dict[str, Any]:
    if result.success:
        return {
            "success": True,
            "platform": "sendblue",
            "chat_id": chat_id,
            "message_id": result.message_id,
        }
    out: Dict[str, Any] = {"error": result.error or "Sendblue send failed"}
    if result.retry_after is not None:
        out["retry_after"] = result.retry_after
    if result.error_kind:
        out["error_kind"] = result.error_kind
    return out


def check_requirements() -> bool:
    return bool(
        AIOHTTP_AVAILABLE
        and os.getenv("SENDBLUE_API_KEY_ID")
        and os.getenv("SENDBLUE_API_SECRET_KEY")
        and (os.getenv("SENDBLUE_FROM_NUMBER") or os.getenv("SENDBLUE_FROM_NUMBERS"))
    )


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        (extra.get("api_key_id") or os.getenv("SENDBLUE_API_KEY_ID"))
        and (extra.get("api_secret_key") or os.getenv("SENDBLUE_API_SECRET_KEY"))
        and (
            extra.get("from_number")
            or extra.get("from_numbers")
            or os.getenv("SENDBLUE_FROM_NUMBER")
            or os.getenv("SENDBLUE_FROM_NUMBERS")
        )
    )


def is_connected(config) -> bool:
    return validate_config(config)


class SendblueAdapter(BasePlatformAdapter):
    """Sendblue REST + receive-webhook adapter."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("sendblue"))

        extra = config.extra or {}
        self._api_key_id = str(
            extra.get("api_key_id") or os.getenv("SENDBLUE_API_KEY_ID", "")
        ).strip()
        self._api_secret_key = str(
            extra.get("api_secret_key") or os.getenv("SENDBLUE_API_SECRET_KEY", "")
        ).strip()
        self._api_base_url = (
            str(
                extra.get("api_base_url")
                or os.getenv("SENDBLUE_API_BASE_URL", DEFAULT_API_BASE_URL)
            )
            .strip()
            .rstrip("/")
        )
        self._webhook_secret = str(
            extra.get("webhook_secret") or os.getenv("SENDBLUE_WEBHOOK_SECRET", "")
        ).strip()
        self._webhook_host = str(
            extra.get("webhook_host")
            or os.getenv("SENDBLUE_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST)
        ).strip()
        self._webhook_port = int(
            extra.get("webhook_port")
            or os.getenv("SENDBLUE_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT))
        )
        self._webhook_path = _normalize_webhook_path(
            str(
                extra.get("webhook_path")
                or os.getenv("SENDBLUE_WEBHOOK_PATH", DEFAULT_WEBHOOK_PATH)
            )
        )
        self._status_callback = str(
            extra.get("status_callback") or os.getenv("SENDBLUE_STATUS_CALLBACK", "")
        ).strip()
        self._seat_id = str(
            extra.get("seat_id") or os.getenv("SENDBLUE_SEAT_ID", "")
        ).strip()
        from_number = str(
            extra.get("from_number") or os.getenv("SENDBLUE_FROM_NUMBER", "")
        ).strip()
        from_numbers = extra.get("from_numbers")
        if isinstance(from_numbers, str):
            self._from_numbers = _split_csv(from_numbers)
        elif isinstance(from_numbers, list):
            self._from_numbers = [
                str(v).strip() for v in from_numbers if str(v).strip()
            ]
        else:
            self._from_numbers = _split_csv(os.getenv("SENDBLUE_FROM_NUMBERS", ""))
        if from_number and from_number not in self._from_numbers:
            self._from_numbers.insert(0, from_number)
        self._default_from_number = from_number or (
            self._from_numbers[0] if self._from_numbers else ""
        )

        self._state_path = (
            Path(
                str(
                    extra.get("sticky_state_path")
                    or os.getenv("SENDBLUE_STICKY_STATE_PATH", "")
                )
            ).expanduser()
            if (
                extra.get("sticky_state_path")
                or os.getenv("SENDBLUE_STICKY_STATE_PATH")
            )
            else _default_sticky_state_path()
        )
        self._sticky_senders: Dict[str, str] = self._load_sticky_senders()
        self._seen_messages: Dict[str, float] = {}
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._runner: Optional["web.AppRunner"] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[sendblue] aiohttp not installed")
            return False
        if not self._api_key_id or not self._api_secret_key:
            msg = "[sendblue] SENDBLUE_API_KEY_ID and SENDBLUE_API_SECRET_KEY are required"
            logger.error(msg)
            self._set_fatal_error("sendblue_missing_credentials", msg, retryable=False)
            return False
        if not self._from_numbers:
            msg = "[sendblue] SENDBLUE_FROM_NUMBER or SENDBLUE_FROM_NUMBERS is required"
            logger.error(msg)
            self._set_fatal_error("sendblue_missing_from_number", msg, retryable=False)
            return False

        insecure = _truthy(os.getenv("SENDBLUE_INSECURE_NO_SIGNATURE", ""))
        if not self._webhook_secret and not insecure:
            msg = (
                "[sendblue] Refusing to start without SENDBLUE_WEBHOOK_SECRET. "
                "Configure a Sendblue receive webhook secret, or set "
                "SENDBLUE_INSECURE_NO_SIGNATURE=true for local development only."
            )
            logger.error(msg)
            self._set_fatal_error(
                "sendblue_missing_webhook_secret", msg, retryable=False
            )
            return False
        if insecure:
            logger.warning("[sendblue] webhook secret validation is disabled")

        app = web.Application(client_max_size=WEBHOOK_BODY_MAX_BYTES)
        app.router.add_post(self._webhook_path, self._handle_webhook)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._webhook_host, self._webhook_port)
        await site.start()
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            trust_env=True,
        )
        self._mark_connected()
        logger.info(
            "[sendblue] webhook listening on %s:%d%s, from=%s",
            self._webhook_host,
            self._webhook_port,
            self._webhook_path,
            ",".join(redact_phone(n) for n in self._from_numbers),
        )
        return True

    async def disconnect(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._running = False
        logger.info("[sendblue] disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        if not self._http_session:
            return SendResult(success=False, error="Sendblue adapter not connected")

        last_result = SendResult(success=True)
        for chunk in self.truncate_message(content):
            payload, endpoint = self._build_send_payload(
                chat_id, chunk, metadata=metadata
            )
            if not payload.get("from_number"):
                return SendResult(
                    success=False, error="Sendblue from_number not configured"
                )
            if "number" not in payload and "group_id" not in payload:
                return SendResult(success=False, error="Sendblue target missing")

            status, headers, body = await _post_sendblue(
                self._http_session,
                api_base_url=self._api_base_url,
                api_key_id=self._api_key_id,
                api_secret_key=self._api_secret_key,
                endpoint=endpoint,
                payload=payload,
            )
            last_result = _send_result_from_response(status, headers, body)
            if not last_result.success:
                logger.warning(
                    "[sendblue] send failed to %s: %s",
                    _redact_target(chat_id),
                    last_result.error,
                )
                return last_result
            self._remember_sender(chat_id, str(payload["from_number"]))
        return last_result

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Sendblue typing indicators are not required for gateway replies."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": chat_id,
            "type": "group" if chat_id.startswith("group:") else "dm",
            "chat_id": chat_id,
        }

    def _build_send_payload(
        self,
        chat_id: str,
        content: str,
        *,
        metadata: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        from_number = str(
            metadata.get("from_number") or self._select_from_number(chat_id)
        ).strip()
        payload: Dict[str, Any] = {
            "content": content,
            "from_number": from_number,
        }
        if self._status_callback:
            payload["status_callback"] = self._status_callback
        if self._seat_id:
            payload["seat_id"] = self._seat_id
        if metadata.get("media_url"):
            payload["media_url"] = metadata["media_url"]
        if metadata.get("send_style"):
            payload["send_style"] = metadata["send_style"]

        if chat_id.startswith("group:"):
            payload["group_id"] = chat_id.split(":", 1)[1]
            return payload, "/api/send-group-message"

        payload["number"] = chat_id
        return payload, "/api/send-message"

    def _select_from_number(self, chat_id: str) -> str:
        sticky = self._sticky_senders.get(chat_id)
        if sticky:
            return sticky
        if not self._from_numbers:
            return ""
        if len(self._from_numbers) == 1:
            return self._from_numbers[0]
        digest = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        return self._from_numbers[int(digest, 16) % len(self._from_numbers)]

    def _remember_sender(self, chat_id: str, from_number: str) -> None:
        if not chat_id or not from_number:
            return
        if self._sticky_senders.get(chat_id) == from_number:
            return
        self._sticky_senders[chat_id] = from_number
        self._save_sticky_senders()

    def _load_sticky_senders(self) -> Dict[str, str]:
        try:
            if not self._state_path.is_file():
                return {}
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return {
                str(k): str(v)
                for k, v in data.items()
                if isinstance(k, str) and isinstance(v, str) and k and v
            }
        except Exception as exc:
            logger.warning("[sendblue] failed to read sticky sender state: %s", exc)
            return {}

    def _save_sticky_senders(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._sticky_senders, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.warning("[sendblue] failed to write sticky sender state: %s", exc)

    def _is_duplicate(self, message_handle: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if message_handle in self._seen_messages:
            return True
        self._seen_messages[message_handle] = now
        return False

    def _validate_webhook_secret(self, headers: Any) -> bool:
        if not self._webhook_secret:
            return _truthy(os.getenv("SENDBLUE_INSECURE_NO_SIGNATURE", ""))
        for header in _SECRET_HEADER_CANDIDATES:
            value = headers.get(header)
            if value and hmac.compare_digest(str(value), self._webhook_secret):
                return True
        auth = str(headers.get("authorization") or headers.get("Authorization") or "")
        if auth.lower().startswith("bearer "):
            return hmac.compare_digest(auth[7:].strip(), self._webhook_secret)
        return False

    async def _handle_webhook(self, request) -> "web.Response":
        try:
            raw = await request.read()
            if len(raw) > WEBHOOK_BODY_MAX_BYTES:
                return web.json_response({"error": "payload too large"}, status=413)
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                return web.json_response({"error": "invalid payload"}, status=400)
        except Exception as exc:
            logger.warning("[sendblue] webhook parse error: %s", exc)
            return web.json_response({"error": "invalid json"}, status=400)

        if not self._validate_webhook_secret(request.headers):
            logger.warning("[sendblue] rejected webhook: missing or invalid secret")
            return web.json_response({"error": "forbidden"}, status=403)

        if bool(payload.get("is_outbound")):
            return web.json_response({"status": "ok"})
        if str(payload.get("status") or "").upper() not in {"", "RECEIVED"}:
            return web.json_response({"status": "ok"})

        media_url = _media_url_for_payload(payload)
        media_type = _media_type_for_payload(payload) if media_url else ""
        text = str(payload.get("content") or "").strip()
        if not text and media_url:
            text = "(attachment)"
        sender = _sender_for_payload(payload)
        chat_id = _chat_id_for_payload(payload)
        message_handle = str(payload.get("message_handle") or "").strip()
        if not text or not sender or not chat_id:
            return web.json_response({"error": "missing message fields"}, status=400)
        if message_handle and self._is_duplicate(message_handle):
            return web.json_response({"status": "duplicate"})

        sendblue_number = _sendblue_number_for_payload(payload)
        if sendblue_number:
            self._remember_sender(chat_id, sendblue_number)

        is_group = chat_id.startswith("group:")
        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(payload.get("group_display_name") or chat_id),
            chat_type="group" if is_group else "dm",
            user_id=sender,
            user_name=sender,
            chat_id_alt=str(payload.get("sendblue_number") or ""),
            message_id=message_handle or None,
        )
        event = MessageEvent(
            text=text,
            message_type=_message_type_for_payload(payload),
            source=source,
            raw_message=payload,
            message_id=message_handle or None,
            media_urls=[media_url] if media_url else [],
            media_types=[media_type] if media_type else [],
        )

        logger.info(
            "[sendblue] inbound from %s to %s via %s",
            redact_phone(sender),
            _redact_target(chat_id),
            payload.get("service") or "unknown",
        )
        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.json_response({"status": "ok"})


def _env_enablement() -> Dict[str, Any] | None:
    api_key_id = os.getenv("SENDBLUE_API_KEY_ID", "").strip()
    api_secret_key = os.getenv("SENDBLUE_API_SECRET_KEY", "").strip()
    from_number = os.getenv("SENDBLUE_FROM_NUMBER", "").strip()
    from_numbers = _split_csv(os.getenv("SENDBLUE_FROM_NUMBERS", ""))
    if not (api_key_id and api_secret_key and (from_number or from_numbers)):
        return None
    seed: Dict[str, Any] = {
        "api_key_id": api_key_id,
        "api_secret_key": api_secret_key,
        "api_base_url": os.getenv("SENDBLUE_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip(
            "/"
        ),
    }
    if from_number:
        seed["from_number"] = from_number
    if from_numbers:
        seed["from_numbers"] = from_numbers
    for env_name, key in (
        ("SENDBLUE_WEBHOOK_SECRET", "webhook_secret"),
        ("SENDBLUE_STATUS_CALLBACK", "status_callback"),
        ("SENDBLUE_SEAT_ID", "seat_id"),
        ("SENDBLUE_STICKY_STATE_PATH", "sticky_state_path"),
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            seed[key] = value
    host = os.getenv("SENDBLUE_WEBHOOK_HOST", "").strip()
    if host:
        seed["webhook_host"] = host
    port = os.getenv("SENDBLUE_WEBHOOK_PORT", "").strip()
    if port:
        seed["webhook_port"] = int(port)
    path = os.getenv("SENDBLUE_WEBHOOK_PATH", "").strip()
    if path:
        seed["webhook_path"] = _normalize_webhook_path(path)
    home = os.getenv("SENDBLUE_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("SENDBLUE_HOME_CHANNEL_NAME", home),
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
    if not AIOHTTP_AVAILABLE:
        return {"error": "Sendblue standalone send: aiohttp not installed"}
    extra = getattr(pconfig, "extra", {}) or {}
    adapter = SendblueAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                **extra,
                "sticky_state_path": extra.get("sticky_state_path")
                or os.getenv(
                    "SENDBLUE_STICKY_STATE_PATH", str(_default_sticky_state_path())
                ),
            },
        )
    )
    if (
        not adapter._api_key_id
        or not adapter._api_secret_key
        or not adapter._from_numbers
    ):
        return {
            "error": (
                "Sendblue not configured "
                "(SENDBLUE_API_KEY_ID, SENDBLUE_API_SECRET_KEY, SENDBLUE_FROM_NUMBER required)"
            )
        }
    if media_files:
        message = f"{message}\n\n[{len(media_files)} attachment(s) generated; Sendblue requires public media_url delivery]"
    chunks = adapter.truncate_message(message)
    result = SendResult(success=True)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        for chunk in chunks:
            payload, endpoint = adapter._build_send_payload(chat_id, chunk, metadata={})
            status, headers, body = await _post_sendblue(
                session,
                api_base_url=adapter._api_base_url,
                api_key_id=adapter._api_key_id,
                api_secret_key=adapter._api_secret_key,
                endpoint=endpoint,
                payload=payload,
            )
            result = _send_result_from_response(status, headers, body)
            if not result.success:
                return _standalone_dict(result, chat_id)
            adapter._remember_sender(chat_id, str(payload["from_number"]))
    return _standalone_dict(result, chat_id)


def interactive_setup() -> None:
    print()
    print("Sendblue iMessage setup")
    print("-----------------------")
    print("Create API credentials and a receive webhook in the Sendblue dashboard.")
    print("Set the receive webhook URL to your public tunnel plus /sendblue/webhook.")
    print()
    try:
        from hermes_cli.config import get_env_var, set_env_var
    except ImportError:
        print(
            "hermes_cli.config not available; set SENDBLUE_* vars manually in ~/.hermes/.env"
        )
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_var(var) if callable(get_env_var) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt

                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            set_env_var(var, value)

    _prompt("SENDBLUE_API_KEY_ID", "API key id", secret=True)
    _prompt("SENDBLUE_API_SECRET_KEY", "API secret key", secret=True)
    _prompt("SENDBLUE_FROM_NUMBER", "Default from number")
    _prompt("SENDBLUE_WEBHOOK_SECRET", "Webhook secret", secret=True)
    _prompt(
        "SENDBLUE_ALLOWED_USERS",
        "Allowed E.164 numbers (comma-separated; blank=pairing)",
    )
    _prompt(
        "SENDBLUE_HOME_CHANNEL", "Home channel E.164 number or group:<id> (optional)"
    )
    print("Done. Start the gateway, then send an iMessage to your Sendblue number.")


def register(ctx) -> None:
    try:
        from . import cli as _cli
    except ImportError:  # test loader imports adapter.py outside package context
        import importlib

        _cli = importlib.import_module("plugins.platforms.sendblue.cli")

    ctx.register_platform(
        name="sendblue",
        label="Sendblue",
        adapter_factory=lambda cfg: SendblueAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[
            "SENDBLUE_API_KEY_ID",
            "SENDBLUE_API_SECRET_KEY",
            "SENDBLUE_FROM_NUMBER",
            "SENDBLUE_WEBHOOK_SECRET",
        ],
        install_hint="pip install aiohttp   # already included with Hermes messaging",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="SENDBLUE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="SENDBLUE_ALLOWED_USERS",
        allow_all_env="SENDBLUE_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating over Sendblue iMessage/SMS/RCS. "
            "Keep replies concise and natural for a phone messaging thread. "
            "The transport may fall back from iMessage to SMS/RCS depending on the recipient. "
            "Recipient identifiers are E.164 phone numbers; never expose them in "
            "responses unless the user asked."
        ),
    )
    ctx.register_cli_command(
        name="sendblue",
        help="Set up and inspect the Sendblue iMessage/SMS/RCS integration",
        setup_fn=_cli.register_cli,
        handler_fn=_cli.dispatch,
    )
