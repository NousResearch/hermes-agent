"""Optional YouPet Core bridge for the WeCom callback adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx

from hermes_constants import get_hermes_home
from gateway.platforms.base import MessageEvent, SendResult
from utils import atomic_replace

logger = logging.getLogger(__name__)

HERMES_CONSUMER = "hermes"
YOUPET_SOURCE = "hermes_wecom"
MAX_PROCESSED_EVENT_IDS = 1000
SUPPORTED_OUTBOX_EVENTS = {
    "health_plan.activated",
    "task.created",
    "task.reminder_due",
    "task.escalated",
    "alert.created",
}


class YouPetBridgeError(RuntimeError):
    """Raised when the YouPet bridge cannot complete a required side effect."""


SendCallable = Callable[[str, str], Awaitable[SendResult]]


@dataclass
class YouPetBridgeSettings:
    enabled: bool = False
    core_base_url: str = ""
    service_token: str = ""
    actor_id: str = "hermes-wecom-bridge"
    source: str = YOUPET_SOURCE
    outbox_consumer: str = HERMES_CONSUMER
    outbox_poll_enabled: bool = True
    outbox_poll_interval_seconds: float = 5.0
    outbox_limit: int = 20
    skip_agent_dispatch: bool = True
    ack_unhandled_events: bool = True
    default_chat_id: Optional[str] = None
    user_chat_map: dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.core_base_url and self.service_token)


def build_youpet_bridge_from_env(send: SendCallable) -> Optional["YouPetBridge"]:
    settings = youpet_settings_from_env()
    if not settings.enabled:
        return None
    return YouPetBridge(settings, send)


def youpet_settings_from_env() -> YouPetBridgeSettings:
    user_chat_map: dict[str, str] = {}
    raw_map = os.getenv("YOUPET_WECOM_USER_CHAT_MAP_JSON", "").strip()
    if raw_map:
        try:
            decoded = json.loads(raw_map)
            if isinstance(decoded, dict):
                user_chat_map = {
                    str(key): str(value)
                    for key, value in decoded.items()
                    if key and value
                }
        except json.JSONDecodeError:
            logger.warning("[YouPetBridge] Invalid YOUPET_WECOM_USER_CHAT_MAP_JSON")

    return YouPetBridgeSettings(
        enabled=_env_bool("YOUPET_WECOM_BRIDGE_ENABLED", default=False),
        core_base_url=os.getenv("YOUPET_CORE_BASE_URL", "").rstrip("/"),
        service_token=os.getenv("YOUPET_SERVICE_TOKEN", ""),
        actor_id=os.getenv("YOUPET_ACTOR_ID", "hermes-wecom-bridge"),
        source=os.getenv("YOUPET_WECOM_SOURCE", YOUPET_SOURCE),
        outbox_consumer=os.getenv("YOUPET_OUTBOX_CONSUMER", HERMES_CONSUMER),
        outbox_poll_enabled=_env_bool("YOUPET_OUTBOX_POLL_ENABLED", default=True),
        outbox_poll_interval_seconds=_env_float(
            "YOUPET_OUTBOX_POLL_INTERVAL_SECONDS", default=5.0,
        ),
        outbox_limit=_env_int("YOUPET_OUTBOX_LIMIT", default=20),
        skip_agent_dispatch=_env_bool("YOUPET_WECOM_SKIP_AGENT_DISPATCH", default=True),
        ack_unhandled_events=_env_bool("YOUPET_OUTBOX_ACK_UNHANDLED_EVENTS", default=True),
        default_chat_id=os.getenv("YOUPET_WECOM_DEFAULT_CHAT_ID") or None,
        user_chat_map=user_chat_map,
    )


class YouPetBridge:
    """Bridge WeCom callback events and YouPet Core outbox events."""

    def __init__(self, settings: YouPetBridgeSettings, send: SendCallable):
        self.settings = settings
        self._send = send
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._runtime_user_chat_map: dict[str, str] = {}
        self._processed_event_id_order = self._load_processed_event_ids()
        self._processed_event_ids = set(self._processed_event_id_order)

    async def start(self) -> None:
        if not self.settings.configured:
            logger.warning(
                "[YouPetBridge] Enabled but not configured; set "
                "YOUPET_CORE_BASE_URL and YOUPET_SERVICE_TOKEN",
            )
            return
        self._ensure_client()
        if self.settings.outbox_poll_enabled and self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_loop())
            logger.info("[YouPetBridge] Started Core outbox poller")

    async def stop(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def handle_wecom_event(self, event: MessageEvent, app: dict[str, Any]) -> bool:
        if not self.settings.configured:
            return False
        payload = self._build_inbound_payload(event, app)
        response = await self._post(
            "/api/v1/wecom/inbound",
            json=payload,
            headers=self._headers(
                correlation_id=f"corr_wecom_{payload['corp_id']}_{payload['message_id']}",
                idempotency_key=f"wecom:{payload['corp_id']}:{payload['message_id']}",
            ),
        )
        data = _response_json(response)
        matched_user_id = data.get("matched_user_id")
        if matched_user_id and event.source and event.source.chat_id:
            self._runtime_user_chat_map[str(matched_user_id)] = str(event.source.chat_id)
        return self.settings.skip_agent_dispatch

    async def poll_once(self) -> int:
        if not self.settings.configured:
            return 0
        response = await self._get(
            "/internal/events/outbox",
            params={
                "consumer": self.settings.outbox_consumer,
                "limit": self.settings.outbox_limit,
            },
            headers=self._headers(),
        )
        items = _response_json(response).get("items", [])
        if not isinstance(items, list):
            raise YouPetBridgeError("Core outbox response did not contain items")

        for item in items:
            event_id = str(item.get("event_id") or "")
            try:
                if event_id in self._processed_event_ids:
                    await self._ack(event_id)
                    continue
                await self._process_outbox_item(item)
                self._remember_processed_event_id(event_id)
                await self._ack(event_id)
            except Exception as exc:
                logger.warning(
                    "[YouPetBridge] Failed to process outbox event %s: %s",
                    event_id,
                    exc,
                )
                await self._nack(event_id, str(exc))
        return len(items)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception:
                logger.exception("[YouPetBridge] Outbox poll failed")
            await asyncio.sleep(self.settings.outbox_poll_interval_seconds)

    def _build_inbound_payload(self, event: MessageEvent, app: dict[str, Any]) -> dict[str, Any]:
        source = event.source
        corp_id = str(app.get("corp_id") or "")
        if not corp_id and source and source.chat_id and ":" in source.chat_id:
            corp_id = source.chat_id.split(":", 1)[0]
        user_id = str(getattr(source, "user_id", "") or "")
        return {
            "corp_id": corp_id,
            "source": self.settings.source,
            "conversation_type": "dm",
            "wecom_user_id": user_id or None,
            "wecom_group_id": None,
            "message_id": str(event.message_id or ""),
            "message_type": getattr(event.message_type, "value", str(event.message_type)),
            "text": event.text or None,
            "media": [],
            "received_at": _iso_utc(event.timestamp),
        }

    async def _process_outbox_item(self, item: dict[str, Any]) -> None:
        event_type = str(item.get("event_type") or "")
        envelope = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if not event_type:
            event_type = str(envelope.get("event_type") or "")
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}

        if event_type not in SUPPORTED_OUTBOX_EVENTS:
            if self.settings.ack_unhandled_events:
                return
            raise YouPetBridgeError(f"Unhandled YouPet event type: {event_type}")

        if event_type == "health_plan.activated":
            return

        if event_type in {"task.created", "task.reminder_due"}:
            chat_id = self._resolve_chat_id(payload, ("recipient_user_id", "owner_user_id"))
            await self._send_required(chat_id, self._render_task_message(event_type, payload))
            return

        chat_id = self._resolve_chat_id(
            payload,
            ("recipient_user_id", "owner_user_id", "assigned_to"),
        )
        await self._send_required(chat_id, self._render_alert_message(event_type, payload))

    def _load_processed_event_ids(self) -> list[str]:
        path = self._processed_event_state_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_ids = data.get("event_ids") if isinstance(data, dict) else None
        if not isinstance(raw_ids, list):
            return []
        return [str(item) for item in raw_ids[-MAX_PROCESSED_EVENT_IDS:] if item]

    def _remember_processed_event_id(self, event_id: str) -> None:
        if not event_id or event_id in self._processed_event_ids:
            return
        self._processed_event_ids.add(event_id)
        self._processed_event_id_order.append(event_id)
        if len(self._processed_event_id_order) > MAX_PROCESSED_EVENT_IDS:
            evicted = self._processed_event_id_order.pop(0)
            self._processed_event_ids.discard(evicted)

        path = self._processed_event_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps({"event_ids": self._processed_event_id_order}, indent=2) + "\n",
            encoding="utf-8",
        )
        atomic_replace(tmp_path, path)

    @staticmethod
    def _processed_event_state_path() -> Path:
        return get_hermes_home() / "integrations" / "youpet_processed_outbox_events.json"

    def _resolve_chat_id(self, payload: dict[str, Any], user_keys: tuple[str, ...]) -> str:
        for key in user_keys:
            user_id = payload.get(key)
            if not user_id:
                continue
            chat_id = (
                self._runtime_user_chat_map.get(str(user_id))
                or self.settings.user_chat_map.get(str(user_id))
            )
            if chat_id:
                return chat_id
        if self.settings.default_chat_id:
            return self.settings.default_chat_id
        raise YouPetBridgeError("No WeCom chat_id for YouPet outbox recipient")

    async def _send_required(self, chat_id: str, content: str) -> None:
        result = await self._send(chat_id, content)
        if isinstance(result, SendResult) and not result.success:
            raise YouPetBridgeError(result.error or "WeCom send failed")

    async def _ack(self, event_id: str) -> None:
        if not event_id:
            raise YouPetBridgeError("Missing outbox event_id")
        await self._post(
            f"/internal/events/outbox/{event_id}/ack",
            params={"consumer": self.settings.outbox_consumer},
            headers=self._headers(),
        )

    async def _nack(self, event_id: str, error: str) -> None:
        if not event_id:
            logger.warning("[YouPetBridge] Cannot nack outbox event without event_id")
            return
        await self._post(
            f"/internal/events/outbox/{event_id}/nack",
            params={"consumer": self.settings.outbox_consumer},
            json={"error": error[:1000]},
            headers=self._headers(),
        )

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().get(self._url(path), **kwargs)
        _raise_for_status(response, path)
        return response

    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        response = await self._ensure_client().post(self._url(path), **kwargs)
        _raise_for_status(response, path)
        return response

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    def _url(self, path: str) -> str:
        return f"{self.settings.core_base_url}{path}"

    def _headers(
        self,
        *,
        correlation_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.service_token}",
            "X-Actor-Id": self.settings.actor_id,
        }
        if correlation_id:
            headers["X-Correlation-Id"] = correlation_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    @staticmethod
    def _render_task_message(event_type: str, payload: dict[str, Any]) -> str:
        context = payload.get("message_context") if isinstance(payload.get("message_context"), dict) else {}
        pet_name = context.get("pet_name") or payload.get("pet_id") or "your pet"
        plan_title = context.get("plan_title") or payload.get("task_type") or "care task"
        due_at = payload.get("due_at")
        if event_type == "task.created":
            suffix = f" Due at {due_at}." if due_at else ""
            return f"[YouPet] New care task for {pet_name}: {plan_title}.{suffix}"
        return f"[YouPet] Reminder for {pet_name}: {plan_title}. Reply when completed."

    @staticmethod
    def _render_alert_message(event_type: str, payload: dict[str, Any]) -> str:
        severity = payload.get("severity") or "alert"
        summary = payload.get("summary") or payload.get("alert_type") or event_type
        return f"[YouPet Alert] {severity}: {summary}"


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _response_json(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception as exc:
        raise YouPetBridgeError("Core response was not JSON") from exc
    if not isinstance(data, dict):
        raise YouPetBridgeError("Core response JSON was not an object")
    return data


def _raise_for_status(response: Any, path: str) -> None:
    status_code = int(getattr(response, "status_code", 200) or 200)
    if status_code < 400:
        return
    body = getattr(response, "text", "")
    raise YouPetBridgeError(f"Core request failed {status_code} {path}: {body}")
