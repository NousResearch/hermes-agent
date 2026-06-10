"""Plivo call provider (Voice API + Plivo XML).

Port of OpenClaw's ``src/providers/plivo.ts`` (turn-based subset):

- outbound dials via ``POST /v1/Account/{auth_id}/Call/`` with
  ``answer_url`` carrying ``?callId=<ours>``; the response's
  ``request_uuid`` is replaced by the real ``CallUUID`` when the first
  webhook arrives (the manager refreshes the mapping)
- webhook events are form-encoded: ``CallStatus`` lifecycle, ``Digits``
  DTMF, speech recognition results from ``<GetInput inputType="speech">``
- TTS uses the same transfer trick as OpenClaw: store the pending text,
  transfer the A-leg to ``?flow=xml-speak``, and answer that request with
  ``<Speak>`` + ``<GetInput>`` XML (Plivo has no reliable live-TTS +
  speech-capture combination outside XML)
- webhook signature: V3 (HMAC-SHA256 over the canonical URL+params + "." +
  nonce, ``X-Plivo-Signature-V3``/``-Nonce``) with V2 fallback
"""

import base64
import binascii
import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit
from xml.sax.saxutils import escape as xml_escape

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

API_HOST = "api.plivo.com"

XML_EMPTY = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
XML_KEEPALIVE = (
    '<?xml version="1.0" encoding="UTF-8"?><Response><Wait length="300"/></Response>'
)

_TERMINAL_STATUSES = {"completed", "busy", "no-answer", "failed", "cancelled"}

_ENDED_REASON_MAP = {
    "completed": "completed",
    "busy": "busy",
    "no-answer": "no-answer",
    "failed": "failed",
    "cancelled": "hangup-bot",
}

# Form keys Plivo uses for speech-recognition results (GetInput).
_SPEECH_KEYS = ("Speech", "SpeechResult", "Transcription", "UnstableSpeech")


def _normalize_b64(value: str) -> Optional[str]:
    """Canonicalize base64 (decode → re-encode) the way the Plivo SDK does."""
    try:
        return base64.b64encode(base64.b64decode(value)).decode()
    except (binascii.Error, ValueError):
        return None


def _sorted_query_string(pairs: List[tuple]) -> str:
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


def _sorted_params_string(pairs: List[tuple]) -> str:
    return "".join(f"{k}{v}" for k, v in sorted(pairs))


def construct_plivo_v3_base(method: str, url: str, post_params: List[tuple]) -> str:
    """Plivo V3 canonical string: url?sorted-query[.]sorted-post-params."""
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    query_string = _sorted_query_string(query_pairs)
    has_post = bool(post_params)
    if query_string or has_post:
        base = f"{base}?{query_string}"
    if query_string and has_post:
        base += "."
    if method == "GET":
        return base
    return base + _sorted_params_string(post_params)


def compute_plivo_v3_signature(
    auth_token: str, method: str, url: str, post_params: List[tuple], nonce: str
) -> str:
    payload = construct_plivo_v3_base(method, url, post_params) + "." + nonce
    digest = hmac.new(
        auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode()


def compute_plivo_v2_signature(auth_token: str, url: str, nonce: str) -> str:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    digest = hmac.new(
        auth_token.encode("utf-8"), (base + nonce).encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode()


class PlivoProvider(VoiceCallProvider):
    name = "plivo"
    requires_public_webhook = True

    def __init__(self, config: VoiceCallConfig):
        super().__init__(config)
        self.auth_id = config.provider_credential("auth_id", "PLIVO_AUTH_ID")
        self.auth_token = config.provider_credential("auth_token", "PLIVO_AUTH_TOKEN")
        if not self.auth_id:
            raise ValueError("Plivo auth ID is required (PLIVO_AUTH_ID)")
        if not self.auth_token:
            raise ValueError("Plivo auth token is required (PLIVO_AUTH_TOKEN)")
        self.base_url = f"https://{API_HOST}/v1/Account/{self.auth_id}"
        # Pending TTS texts served when the transferred leg fetches xml-speak.
        self._pending_speak: Dict[str, str] = {}

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
            f"{self.base_url}{endpoint}",
            allowed_host=API_HOST,
            auth=(self.auth_id, self.auth_token),
            json_body=body,
            allow_not_found=allow_not_found,
            error_prefix="Plivo API error",
        )

    # -- webhook ------------------------------------------------------------------

    def verify_webhook(self, ctx: WebhookContext) -> WebhookVerificationResult:
        signature_v3 = ctx.header("x-plivo-signature-v3")
        nonce_v3 = ctx.header("x-plivo-signature-v3-nonce")
        if signature_v3 and nonce_v3:
            try:
                post_params = parse_qsl(
                    ctx.body.decode("utf-8"), keep_blank_values=True
                )
            except UnicodeDecodeError:
                return WebhookVerificationResult(ok=False, error="undecodable body")
            expected = _normalize_b64(
                compute_plivo_v3_signature(
                    self.auth_token, ctx.method, ctx.url, post_params, nonce_v3
                )
            )
            # The header may carry several comma-separated signatures.
            for candidate in signature_v3.split(","):
                normalized = _normalize_b64(candidate.strip())
                if normalized and expected and hmac.compare_digest(expected, normalized):
                    digest = hashlib.sha256(
                        (ctx.url + "\n" + nonce_v3).encode()
                    ).hexdigest()
                    return WebhookVerificationResult(
                        ok=True, dedupe_key=f"plivo:v3:{digest}"
                    )
            return WebhookVerificationResult(ok=False, error="invalid Plivo V3 signature")

        signature_v2 = ctx.header("x-plivo-signature-v2")
        nonce_v2 = ctx.header("x-plivo-signature-v2-nonce")
        if signature_v2 and nonce_v2:
            expected = _normalize_b64(
                compute_plivo_v2_signature(self.auth_token, ctx.url, nonce_v2)
            )
            provided = _normalize_b64(signature_v2)
            if expected and provided and hmac.compare_digest(expected, provided):
                digest = hashlib.sha256(
                    (ctx.url + "\n" + nonce_v2).encode()
                ).hexdigest()
                return WebhookVerificationResult(
                    ok=True, dedupe_key=f"plivo:v2:{digest}"
                )
            return WebhookVerificationResult(ok=False, error="invalid Plivo V2 signature")

        return WebhookVerificationResult(
            ok=False, error="missing Plivo signature headers"
        )

    def parse_webhook(self, ctx: WebhookContext) -> WebhookParseResult:
        try:
            params = dict(parse_qsl(ctx.body.decode("utf-8"), keep_blank_values=True))
        except UnicodeDecodeError:
            return WebhookParseResult(
                response_status=400, response_body=XML_EMPTY,
                response_content_type="text/xml",
            )
        flow = ctx.query.get("flow", "")
        call_id = ctx.query.get("callId")

        # Transferred-leg XML flows return Plivo XML and produce no events.
        if flow == "xml-speak":
            text = self._pending_speak.pop(call_id or "", None)
            xml = self._speak_xml(call_id, text) if text else XML_KEEPALIVE
            return WebhookParseResult(
                response_body=xml, response_content_type="text/xml"
            )

        event = self._normalize_event(params, call_id)
        body = XML_KEEPALIVE if flow in ("answer", "getinput") else XML_EMPTY
        return WebhookParseResult(
            events=[event] if event else [],
            response_body=body,
            response_content_type="text/xml",
        )

    def _normalize_event(
        self, params: Dict[str, str], call_id_override: Optional[str]
    ) -> Optional[NormalizedEvent]:
        call_uuid = params.get("CallUUID", "")
        request_uuid = params.get("RequestUUID", "")
        direction = params.get("Direction")
        base = dict(
            provider=self.name,
            provider_call_id=call_uuid or request_uuid or None,
            call_id=call_id_override or None,
            from_number=params.get("From") or None,
            to_number=params.get("To") or None,
            direction=direction if direction in ("inbound", "outbound") else None,
            raw=dict(params),
        )

        digits = params.get("Digits")
        if digits:
            return NormalizedEvent(type=EventType.CALL_DTMF, digits=digits, **base)
        for key in _SPEECH_KEYS:
            speech = params.get(key)
            if speech:
                return NormalizedEvent(
                    type=EventType.CALL_SPEECH, text=speech, is_final=True, **base
                )

        status = params.get("CallStatus", "")
        if status == "ringing":
            return NormalizedEvent(type=EventType.CALL_RINGING, **base)
        if status == "in-progress":
            return NormalizedEvent(type=EventType.CALL_ANSWERED, **base)
        reason = _ENDED_REASON_MAP.get(status)
        if reason:
            return NormalizedEvent(type=EventType.CALL_ENDED, reason=reason, **base)
        # Plivo posts the answer_url with Event=StartApp when the call
        # connects; treat it as answered so the call can proceed.
        if params.get("Event") == "StartApp" and call_uuid:
            return NormalizedEvent(type=EventType.CALL_ANSWERED, **base)
        return None

    # -- XML builders ------------------------------------------------------------------

    def _webhook_url_with(self, call_id: Optional[str], **query) -> str:
        base = self.webhook_url or ""
        merged = {"provider": "plivo", **({"callId": call_id} if call_id else {}), **query}
        return f"{base}?{urlencode(merged)}"

    def _speak_xml(self, call_id: Optional[str], text: str) -> str:
        extra = self.config.provider_extra.get("plivo", {})
        language = xml_escape(str(extra.get("language", "en-US")))
        action = xml_escape(self._webhook_url_with(call_id, flow="getinput"))
        return (
            '<?xml version="1.0" encoding="UTF-8"?><Response>'
            f'<Speak language="{language}">{xml_escape(text)}</Speak>'
            f'<GetInput inputType="speech" method="POST" action="{action}" '
            f'language="{language}" executionTimeout="30" speechEndTimeout="1" '
            'redirect="false"></GetInput>'
            '<Wait length="300"/></Response>'
        )

    # -- call control -------------------------------------------------------------------

    async def initiate_call(self, call: CallRecord) -> str:
        result = await self._api(
            "/Call/",
            {
                "to": call.to_number,
                "from": call.from_number,
                "answer_url": self._webhook_url_with(call.call_id, flow="answer"),
                "answer_method": "POST",
                "hangup_url": self._webhook_url_with(call.call_id, flow="hangup"),
                "hangup_method": "POST",
                "ring_timeout": 30,
            },
        )
        request_uuid = (result or {}).get("request_uuid")
        if isinstance(request_uuid, list):
            request_uuid = request_uuid[0] if request_uuid else ""
        return str(request_uuid or "")

    async def hangup_call(self, call: CallRecord) -> None:
        await self._api(
            f"/Call/{call.provider_call_id}/", method="DELETE", allow_not_found=True
        )

    async def speak(self, call: CallRecord, text: str) -> None:
        # Transfer the A-leg to our xml-speak flow; the transferred leg
        # fetches <Speak> + <GetInput> XML carrying this pending text.
        self._pending_speak[call.call_id] = text
        await self._api(
            f"/Call/{call.provider_call_id}/",
            {
                "legs": "aleg",
                "aleg_url": self._webhook_url_with(call.call_id, flow="xml-speak"),
                "aleg_method": "POST",
            },
        )

    async def send_dtmf(self, call: CallRecord, digits: str) -> None:
        safe = "".join(c for c in digits if c in "0123456789*#w")
        await self._api(f"/Call/{call.provider_call_id}/DTMF/", {"digits": safe})

    async def start_listening(self, call: CallRecord) -> None:
        """No-op: conversation-mode ``speak`` XML embeds ``<GetInput>``, and
        transferring the leg here would cut the greeting off mid-word."""

    async def get_call_status(self, call: CallRecord) -> CallStatusResult:
        try:
            result = await self._api(
                f"/Call/{call.provider_call_id}/", method="GET", allow_not_found=True
            )
        except Exception:  # noqa: BLE001 — transient errors must not kill restore
            return CallStatusResult(is_unknown=True, raw_status="error")
        if result is None:
            return CallStatusResult(is_terminal=True, raw_status="not-found")
        status = str(result.get("call_status", "unknown"))
        return CallStatusResult(
            is_active=status not in _TERMINAL_STATUSES,
            is_terminal=status in _TERMINAL_STATUSES,
            raw_status=status,
        )
