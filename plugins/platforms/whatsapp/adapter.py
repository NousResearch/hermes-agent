"""Plugin-owned WhatsApp override seam."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.whatsapp import (
    WhatsAppAdapter as BuiltinWhatsAppAdapter,
    check_whatsapp_requirements,
)
from gateway.whatsapp_identity import canonical_whatsapp_identifier
from gateway.whatsapp_message_store import (
    SCHEMA_VERSION,
    append_whatsapp_record,
    build_whatsapp_destination_fields,
    next_whatsapp_record_sequence,
    parse_whatsapp_event_datetime,
    utc_isoformat,
    utc_now,
)

DEFAULT_BRIDGE_HOST = "127.0.0.1"
DEFAULT_BRIDGE_PORT = 3000

logger = logging.getLogger(__name__)


def _bridge_port_from_config(config: Any) -> int:
    extra = getattr(config, "extra", {}) or {}
    raw_port = extra.get("bridge_port", DEFAULT_BRIDGE_PORT)
    try:
        return int(raw_port)
    except (TypeError, ValueError):
        return DEFAULT_BRIDGE_PORT


def bridge_base_url(config: Any = None) -> str:
    """Return the canonical local WhatsApp bridge base URL."""
    return f"http://{DEFAULT_BRIDGE_HOST}:{_bridge_port_from_config(config)}"


def bridge_health_url(config: Any = None) -> str:
    """Return the concrete local bridge health endpoint."""
    return f"{bridge_base_url(config)}/health"


def local_bridge_healthcheck_command(config: Any = None) -> str:
    """Return the founder-visible manual health probe command."""
    return f"curl {bridge_health_url(config)}"


class WhatsAppPluginAdapter(BuiltinWhatsAppAdapter):
    """Plugin override that enforces command authority and append-only logging."""

    def __init__(self, config: Any):
        super().__init__(config)
        self._plugin_bridge_health_url = bridge_health_url(config)

    @property
    def plugin_bridge_health_url(self) -> str:
        return self._plugin_bridge_health_url

    def _admin_ids_for_scope(self, *, is_group: bool) -> set[str]:
        key = "group_allow_admin_from" if is_group else "allow_admin_from"
        raw_ids = self._coerce_allow_list(self.config.extra.get(key))
        canonical_ids = set()
        for raw_id in raw_ids:
            canonical_id = canonical_whatsapp_identifier(raw_id)
            if canonical_id:
                canonical_ids.add(canonical_id)
        return canonical_ids

    def _classify_inbound_principal(self, *, chat_id: str, sender_id: str) -> tuple[str, str, str]:
        is_group = str(chat_id or "").endswith("@g.us")
        canonical_sender_id = canonical_whatsapp_identifier(sender_id)
        if (
            canonical_sender_id
            and canonical_sender_id in self._admin_ids_for_scope(is_group=is_group)
        ):
            return ("owner_operator", "command_capable", "owner_only")
        return ("external_party", "conversational_only", "none")

    def _base_record(
        self,
        *,
        record_kind: str,
        direction: str,
        chat_id: str,
        participant_role: str,
        message_classification: str,
        command_authority_scope: str,
        observed_at: datetime,
        effective_event_at: datetime | None = None,
        source_event_at: datetime | None = None,
        dispatch_group_id: str,
        dispatch_group_sequence: int,
        sender_id: str | None,
        text: str,
        message_id: str | None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], datetime]:
        effective = effective_event_at or observed_at
        destination = build_whatsapp_destination_fields(chat_id)
        record = {
            "record_kind": record_kind,
            "schema_version": SCHEMA_VERSION,
            "record_id": uuid.uuid4().hex,
            "conversation_key": destination["destination_key"],
            **destination,
            "direction": direction,
            "record_sequence": next_whatsapp_record_sequence(effective),
            "effective_event_at_utc": utc_isoformat(effective),
            "source_event_at_utc": utc_isoformat(source_event_at),
            "hermes_observed_at_utc": utc_isoformat(observed_at),
            "dispatch_group_id": dispatch_group_id,
            "dispatch_group_sequence": dispatch_group_sequence,
            "participant_role": participant_role,
            "message_classification": message_classification,
            "command_authority_scope": command_authority_scope,
            "platform": "whatsapp",
            "sender_id": sender_id,
            "text": text,
            "message_id": message_id,
        }
        if extra:
            record.update(extra)
        return record, effective

    def _append_record(self, record: dict[str, Any], effective_event_at: datetime) -> None:
        try:
            append_whatsapp_record(record, effective_event_at=effective_event_at)
        except Exception as exc:
            logger.warning("[whatsapp] Failed to append conversation record: %s", exc)

    def _append_inbound_record(self, event: MessageEvent, data: dict[str, Any]) -> None:
        observed_at = utc_now()
        source_event_at = parse_whatsapp_event_datetime(data.get("timestamp"))
        sender_id = canonical_whatsapp_identifier(
            event.source.user_id or data.get("senderId") or data.get("from") or ""
        )
        dispatch_group_id = str(
            event.message_id or data.get("messageId") or uuid.uuid4().hex
        )
        record, effective = self._base_record(
            record_kind="conversation_record",
            direction="inbound",
            chat_id=event.source.chat_id,
            participant_role=event.participant_role or "external_party",
            message_classification=(
                event.message_classification or "conversational_only"
            ),
            command_authority_scope=event.command_authority_scope or "none",
            observed_at=observed_at,
            effective_event_at=source_event_at,
            source_event_at=source_event_at,
            dispatch_group_id=dispatch_group_id,
            dispatch_group_sequence=1,
            sender_id=sender_id,
            text=event.text or "",
            message_id=event.message_id,
            extra={
                "raw_sender_id": (
                    event.source.user_id or data.get("senderId") or data.get("from")
                ),
                "sender_name": event.source.user_name,
                "reply_to_message_id": data.get("quotedMessageId"),
                "quoted_participant": data.get("quotedParticipant"),
                "media_urls": list(event.media_urls or []),
                "media_types": list(event.media_types or []),
            },
        )
        self._append_record(record, effective)

    def _append_outbound_record(
        self,
        *,
        record_kind: str,
        chat_id: str,
        text: str,
        dispatch_group_id: str,
        dispatch_group_sequence: int,
        response_data: dict[str, Any] | None,
        reply_to: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        observed_at = utc_now()
        payload_extra = {
            "reply_to_message_id": reply_to,
            "delivery": {
                "success": bool((response_data or {}).get("success", True)),
                "messageId": (response_data or {}).get("messageId"),
                "messageIds": list((response_data or {}).get("messageIds") or []),
            },
        }
        if extra:
            payload_extra.update(extra)
        record, effective = self._base_record(
            record_kind=record_kind,
            direction="outbound",
            chat_id=chat_id,
            participant_role="agent",
            message_classification="conversational_only",
            command_authority_scope="none",
            observed_at=observed_at,
            dispatch_group_id=dispatch_group_id,
            dispatch_group_sequence=dispatch_group_sequence,
            sender_id="agent",
            text=text,
            message_id=(response_data or {}).get("messageId"),
            extra=payload_extra,
        )
        self._append_record(record, effective)

    async def _build_message_event(self, data: dict[str, Any]) -> MessageEvent | None:
        event = await super()._build_message_event(data)
        if event is None:
            return None

        participant_role, message_classification, command_authority_scope = self._classify_inbound_principal(
            chat_id=event.source.chat_id,
            sender_id=(
                event.source.user_id or data.get("senderId") or data.get("from") or ""
            ),
        )
        event.participant_role = participant_role
        event.message_classification = message_classification
        event.command_authority_scope = command_authority_scope
        self._append_inbound_record(event, data)
        return event

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)

        try:
            import aiohttp

            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self._outgoing_chunk_limit())
            dispatch_group_id = uuid.uuid4().hex
            last_message_id = None

            for index, chunk in enumerate(chunks, start=1):
                payload: dict[str, Any] = {"chatId": chat_id, "message": chunk}
                if reply_to and last_message_id is None:
                    payload["replyTo"] = reply_to

                async with self._http_session.post(
                    f"http://127.0.0.1:{self._bridge_port}/send",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        return SendResult(success=False, error=await resp.text())
                    response_data = await resp.json()
                    last_message_id = response_data.get("messageId")
                    self._append_outbound_record(
                        record_kind="conversation_record",
                        chat_id=chat_id,
                        text=chunk,
                        dispatch_group_id=dispatch_group_id,
                        dispatch_group_sequence=index,
                        response_data=response_data,
                        reply_to=payload.get("replyTo"),
                    )

                if len(chunks) > 1:
                    await asyncio.sleep(0.3)

            return SendResult(success=True, message_id=last_message_id)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)

        try:
            import aiohttp

            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/edit",
                json={"chatId": chat_id, "messageId": message_id, "message": content},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return SendResult(success=False, error=await resp.text())
                response_data = await resp.json()
                self._append_outbound_record(
                    record_kind="edit_outcome",
                    chat_id=chat_id,
                    text=content,
                    dispatch_group_id=uuid.uuid4().hex,
                    dispatch_group_sequence=1,
                    response_data=response_data,
                    extra={
                        "edited_message_id": message_id,
                        "finalize": bool(finalize),
                    },
                )
                return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _send_media_to_bridge(
        self,
        chat_id: str,
        file_path: str,
        media_type: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> SendResult:
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        bridge_exit = await self._check_managed_bridge_exit()
        if bridge_exit:
            return SendResult(success=False, error=bridge_exit)

        try:
            import aiohttp

            if not os.path.exists(file_path):
                return SendResult(success=False, error=f"File not found: {file_path}")

            payload: dict[str, Any] = {
                "chatId": chat_id,
                "filePath": file_path,
                "mediaType": media_type,
            }
            if caption:
                payload["caption"] = caption
            if file_name:
                payload["fileName"] = file_name

            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send-media",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    return SendResult(success=False, error=await resp.text())
                response_data = await resp.json()
                self._append_outbound_record(
                    record_kind="conversation_record",
                    chat_id=chat_id,
                    text=caption or "",
                    dispatch_group_id=uuid.uuid4().hex,
                    dispatch_group_sequence=1,
                    response_data=response_data,
                    extra={
                        "media_type": media_type,
                        "file_path": file_path,
                        "file_name": file_name,
                    },
                )
                return SendResult(
                    success=True,
                    message_id=response_data.get("messageId"),
                    raw_response=response_data,
                )
        except Exception as exc:
            return SendResult(success=False, error=str(exc))


def register(ctx) -> None:
    """Register the plugin-owned WhatsApp override."""
    ctx.register_platform(
        name="whatsapp",
        label="WhatsApp",
        adapter_factory=lambda cfg: WhatsAppPluginAdapter(cfg),
        check_fn=check_whatsapp_requirements,
        emoji="💬",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via WhatsApp. Keep replies concise, mobile-friendly, "
            "and aware that the local bridge health endpoint is "
            f"{bridge_health_url()}."
        ),
    )
