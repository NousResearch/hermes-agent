"""Telnyx call provider (Call Control API v2) — the default provider.

Port of OpenClaw's ``src/providers/telnyx.ts``:

- outbound calls via ``POST /v2/calls`` with ``client_state`` carrying our
  call id (base64) so every webhook event binds back to the right call
- explicit ``answer`` for inbound calls (a Telnyx quirk — webhook-response
  carriers answer implicitly)
- carrier-native TTS (``actions/speak``) and transcription
  (``actions/transcription_start/stop``)
- Ed25519 webhook signature verification (``telnyx-signature-ed25519`` +
  ``telnyx-timestamp``, 5-minute skew window) via the already-pinned
  ``cryptography`` package
- bidirectional PCMU/RTP media-stream fields on dial/answer for the
  realtime phase

All HTTP is pinned to ``api.telnyx.com``.
"""

import base64
import binascii
import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from ..config import VoiceCallConfig
from ..events import CallRecord, EventType, NormalizedEvent
from ._http import guarded_json_request
from .base import (
    CallStatusResult,
    VoiceCallProvider,
    WebhookContext,
    WebhookParseResult,
    WebhookVerificationResult,
)

logger = logging.getLogger(__name__)

API_HOST = "api.telnyx.com"
BASE_URL = f"https://{API_HOST}/v2"
SIGNATURE_MAX_SKEW_S = 300

_HANGUP_CAUSE_MAP = {
    "normal_clearing": "completed",
    "normal_unspecified": "completed",
    "originator_cancel": "hangup-bot",
    "call_rejected": "busy",
    "user_busy": "busy",
    "no_answer": "no-answer",
    "no_user_response": "no-answer",
    "destination_out_of_order": "failed",
    "network_out_of_order": "failed",
    "service_unavailable": "failed",
    "recovery_on_timer_expire": "failed",
    "machine_detected": "voicemail",
    "fax_detected": "voicemail",
    "user_hangup": "hangup-user",
    "subscriber_absent": "hangup-user",
}

_EVENT_TYPE_MAP = {
    "call.initiated": EventType.CALL_INITIATED,
    "call.ringing": EventType.CALL_RINGING,
    "call.answered": EventType.CALL_ANSWERED,
    "call.bridged": EventType.CALL_ACTIVE,
    "call.speak.started": EventType.CALL_SPEAKING,
    "call.speak.ended": EventType.CALL_SPEAK_ENDED,
    "call.transcription": EventType.CALL_SPEECH,
    "call.hangup": EventType.CALL_ENDED,
    "call.dtmf.received": EventType.CALL_DTMF,
}


def _normalize_direction(direction: Optional[str]) -> Optional[str]:
    if direction in ("incoming", "inbound"):
        return "inbound"
    if direction in ("outgoing", "outbound"):
        return "outbound"
    return None


def _decode_client_state(value: str) -> Optional[str]:
    """Base64-decode ``client_state`` if it round-trips cleanly."""
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def build_streaming_fields(
    stream_url: str, auth_token: Optional[str] = None
) -> Dict[str, Any]:
    """Bidirectional PCMU/RTP media-stream fields for dial/answer requests."""
    fields: Dict[str, Any] = {
        "stream_url": stream_url,
        "stream_track": "inbound_track",
        "stream_codec": "PCMU",
        "stream_bidirectional_mode": "rtp",
        "stream_bidirectional_codec": "PCMU",
        "stream_bidirectional_sampling_rate": 8000,
        "stream_bidirectional_target_legs": "self",
    }
    if auth_token:
        fields["stream_auth_token"] = auth_token
    return fields


class TelnyxProvider(VoiceCallProvider):
    name = "telnyx"
    requires_public_webhook = True

    def __init__(self, config: VoiceCallConfig):
        super().__init__(config)
        self.api_key = config.provider_credential("api_key", "TELNYX_API_KEY")
        self.connection_id = config.provider_credential(
            "connection_id", "TELNYX_CONNECTION_ID"
        )
        self.public_key = config.provider_credential("public_key", "TELNYX_PUBLIC_KEY")
        if not self.api_key:
            raise ValueError("Telnyx API key is required (TELNYX_API_KEY)")
        if not self.connection_id:
            raise ValueError(
                "Telnyx connection ID is required (TELNYX_CONNECTION_ID)"
            )

    # -- HTTP -----------------------------------------------------------------

    async def _api(
        self,
        endpoint: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        method: str = "POST",
        allow_not_found: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return await guarded_json_request(
            method,
            f"{BASE_URL}{endpoint}",
            allowed_host=API_HOST,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body=body,
            allow_not_found=allow_not_found,
            error_prefix="Telnyx API error",
        )

    # -- webhook ----------------------------------------------------------------

    def verify_webhook(self, ctx: WebhookContext) -> WebhookVerificationResult:
        if not self.public_key:
            return WebhookVerificationResult(
                ok=False, error="missing TELNYX_PUBLIC_KEY"
            )
        signature_b64 = ctx.header("telnyx-signature-ed25519")
        timestamp = ctx.header("telnyx-timestamp")
        if not signature_b64 or not timestamp:
            return WebhookVerificationResult(
                ok=False, error="missing signature or timestamp header"
            )
        try:
            event_time = int(timestamp)
        except ValueError:
            return WebhookVerificationResult(ok=False, error="invalid timestamp header")

        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            # Accept base64url variants the way OpenClaw does.
            normalized = signature_b64.replace("-", "+").replace("_", "/")
            normalized += "=" * (-len(normalized) % 4)
            signature = base64.b64decode(normalized)
            key = Ed25519PublicKey.from_public_bytes(
                base64.b64decode(self.public_key)
            )
            signed_payload = timestamp.encode("utf-8") + b"|" + ctx.body
            try:
                key.verify(signature, signed_payload)
            except InvalidSignature:
                return WebhookVerificationResult(ok=False, error="invalid signature")
        except (ValueError, binascii.Error) as e:
            return WebhookVerificationResult(
                ok=False, error=f"verification error: {type(e).__name__}"
            )

        if abs(time.time() - event_time) > SIGNATURE_MAX_SKEW_S:
            return WebhookVerificationResult(ok=False, error="timestamp too old")

        digest = hashlib.sha256(
            timestamp.encode() + b"\n" + signature + b"\n" + ctx.body
        ).hexdigest()
        return WebhookVerificationResult(ok=True, dedupe_key=f"telnyx:{digest}")

    def parse_webhook(self, ctx: WebhookContext) -> WebhookParseResult:
        try:
            payload = json.loads(ctx.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return WebhookParseResult(response_status=400, response_body="{}")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not data.get("event_type"):
            return WebhookParseResult(response_body="{}")
        event = self._normalize_event(data)
        return WebhookParseResult(
            events=[event] if event else [], response_body="{}"
        )

    def _normalize_event(self, data: Dict[str, Any]) -> Optional[NormalizedEvent]:
        event_type = _EVENT_TYPE_MAP.get(str(data.get("event_type", "")))
        if event_type is None:
            return None  # streaming.* and other events are not call events
        body = data.get("payload") or {}

        call_id = None
        client_state = body.get("client_state")
        if client_state:
            call_id = _decode_client_state(str(client_state)) or str(client_state)

        text = None
        is_final = True
        if event_type == EventType.CALL_SPEECH:
            tdata = body.get("transcription_data") or {}
            text = tdata.get("transcript") or body.get("transcription") or ""
            is_final = bool(tdata.get("is_final", body.get("is_final", True)))
        elif event_type == EventType.CALL_SPEAKING:
            text = body.get("text") or ""

        reason = None
        if event_type == EventType.CALL_ENDED:
            cause = str(body.get("hangup_cause") or "")
            reason = _HANGUP_CAUSE_MAP.get(cause)
            if reason is None:
                if cause:
                    logger.warning("telnyx: unknown hangup cause %r", cause)
                reason = "completed"

        return NormalizedEvent(
            type=event_type,
            provider=self.name,
            provider_call_id=body.get("call_control_id"),
            call_id=call_id,
            from_number=body.get("from"),
            to_number=body.get("to"),
            direction=_normalize_direction(body.get("direction")),
            text=text,
            is_final=is_final,
            digits=body.get("digit") if event_type == EventType.CALL_DTMF else None,
            reason=reason,
            dedupe_key=str(data.get("id")) if data.get("id") else None,
            raw=data,
        )

    # -- call control ----------------------------------------------------------------

    def _streaming_kwargs(self, call: CallRecord) -> Dict[str, Any]:
        stream_url = call.metadata.get("stream_url")
        if not stream_url:
            return {}
        return build_streaming_fields(
            stream_url, call.metadata.get("stream_auth_token")
        )

    async def initiate_call(self, call: CallRecord) -> str:
        body: Dict[str, Any] = {
            "connection_id": self.connection_id,
            "to": call.to_number,
            "from": call.from_number,
            "webhook_url": self.webhook_url,
            "webhook_url_method": "POST",
            "client_state": base64.b64encode(call.call_id.encode()).decode(),
            "timeout_secs": 30,
            **self._streaming_kwargs(call),
        }
        result = await self._api("/calls", body)
        return str((result or {}).get("data", {}).get("call_control_id", ""))

    async def answer_call(self, call: CallRecord) -> None:
        body: Dict[str, Any] = {
            "command_id": f"hermes-answer-{call.call_id}",
            "client_state": base64.b64encode(call.call_id.encode()).decode(),
            **self._streaming_kwargs(call),
        }
        await self._api(f"/calls/{call.provider_call_id}/actions/answer", body)

    async def hangup_call(self, call: CallRecord) -> None:
        await self._api(
            f"/calls/{call.provider_call_id}/actions/hangup",
            {"command_id": str(uuid.uuid4())},
            allow_not_found=True,
        )

    async def speak(self, call: CallRecord, text: str) -> None:
        extra = self.config.provider_extra.get("telnyx", {})
        await self._api(
            f"/calls/{call.provider_call_id}/actions/speak",
            {
                "command_id": str(uuid.uuid4()),
                "payload": text,
                "voice": str(extra.get("voice", "female")),
                "language": str(extra.get("language", "en-US")),
            },
        )

    async def send_dtmf(self, call: CallRecord, digits: str) -> None:
        await self._api(
            f"/calls/{call.provider_call_id}/actions/send_dtmf",
            {"command_id": str(uuid.uuid4()), "digits": digits},
        )

    async def start_listening(self, call: CallRecord) -> None:
        extra = self.config.provider_extra.get("telnyx", {})
        await self._api(
            f"/calls/{call.provider_call_id}/actions/transcription_start",
            {
                "command_id": str(uuid.uuid4()),
                "language": str(extra.get("transcription_language", "en")),
            },
        )

    async def stop_listening(self, call: CallRecord) -> None:
        await self._api(
            f"/calls/{call.provider_call_id}/actions/transcription_stop",
            {"command_id": str(uuid.uuid4())},
            allow_not_found=True,
        )

    async def get_call_status(self, call: CallRecord) -> CallStatusResult:
        try:
            result = await self._api(
                f"/calls/{call.provider_call_id}", method="GET", allow_not_found=True
            )
        except Exception:  # noqa: BLE001 — transient errors must not kill restore
            return CallStatusResult(is_unknown=True, raw_status="error")
        if result is None:
            return CallStatusResult(is_terminal=True, raw_status="not-found")
        data = result.get("data") or {}
        is_alive = data.get("is_alive")
        if is_alive is None:
            return CallStatusResult(is_unknown=True, raw_status=data.get("state"))
        return CallStatusResult(
            is_active=bool(is_alive),
            is_terminal=not is_alive,
            raw_status=data.get("state"),
        )

    # -- realtime ----------------------------------------------------------------------

    def streaming_fields(self, stream_url: str, auth_token: str) -> Dict[str, Any]:
        return build_streaming_fields(stream_url, auth_token)
