"""Twilio call provider (Programmable Voice REST API + TwiML).

Port of OpenClaw's ``src/providers/twilio.ts`` (turn-based subset):

- outbound dials via ``POST /2010-04-01/Accounts/{sid}/Calls.json`` with the
  webhook URL carrying ``?callId=<ours>`` and a separate status callback
  (``&type=status``) for lifecycle events
- webhook events are form-encoded; ``CallStatus`` maps to lifecycle events,
  ``SpeechResult`` (from ``<Gather input="speech">``) to ``call.speech``,
  ``Digits`` to ``call.dtmf``
- TTS via call update with inline TwiML: ``<Say>`` followed by an embedded
  ``<Gather>`` so the next caller utterance posts back — Twilio has no
  "speak on live call" API, so each speak replaces the call's TwiML
- webhook signature: HMAC-SHA1 over ``url + sorted(key+value)`` form params
  (``X-Twilio-Signature``), retried without the port (Twilio signs both
  variants in the wild)

Media streams (``<Connect><Stream>``) land with the realtime phase.
"""

import base64
import hashlib
import hmac
import logging
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
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

API_HOST = "api.twilio.com"

EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
KEEPALIVE_TWIML = (
    '<?xml version="1.0" encoding="UTF-8"?><Response><Pause length="30"/></Response>'
)

_TERMINAL_STATUSES = {"completed", "busy", "failed", "no-answer", "canceled"}

_ENDED_REASON_MAP = {
    "completed": "completed",
    "busy": "busy",
    "failed": "failed",
    "no-answer": "no-answer",
    "canceled": "hangup-bot",
}


def compute_twilio_signature(auth_token: str, url: str, params: Dict[str, str]) -> str:
    """Twilio's documented scheme: HMAC-SHA1 over url + sorted key+value."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(
        auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1
    ).digest()
    return base64.b64encode(digest).decode()


def _strip_port(url: str) -> str:
    parts = urlsplit(url)
    if parts.port is None:
        return url
    host = parts.hostname or ""
    if ":" in host:  # IPv6
        host = f"[{host}]"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _normalize_direction(direction: Optional[str]) -> Optional[str]:
    if not direction:
        return None
    if direction.startswith("outbound"):
        return "outbound"
    if direction == "inbound":
        return "inbound"
    return None


class TwilioProvider(VoiceCallProvider):
    name = "twilio"
    requires_public_webhook = True

    def __init__(self, config: VoiceCallConfig):
        super().__init__(config)
        self.account_sid = config.provider_credential("account_sid", "TWILIO_ACCOUNT_SID")
        self.auth_token = config.provider_credential("auth_token", "TWILIO_AUTH_TOKEN")
        if not self.account_sid:
            raise ValueError("Twilio account SID is required (TWILIO_ACCOUNT_SID)")
        if not self.auth_token:
            raise ValueError("Twilio auth token is required (TWILIO_AUTH_TOKEN)")
        self.base_url = f"https://{API_HOST}/2010-04-01/Accounts/{self.account_sid}"

    # -- HTTP ---------------------------------------------------------------

    async def _api(
        self,
        endpoint: str,
        form: Optional[Dict[str, Any]] = None,
        *,
        method: str = "POST",
        allow_not_found: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return await guarded_json_request(
            method,
            f"{self.base_url}{endpoint}",
            allowed_host=API_HOST,
            auth=(self.account_sid, self.auth_token),
            form_body=form,
            allow_not_found=allow_not_found,
            error_prefix="Twilio API error",
        )

    # -- webhook ---------------------------------------------------------------

    def verify_webhook(self, ctx: WebhookContext) -> WebhookVerificationResult:
        signature = ctx.header("x-twilio-signature")
        if not signature:
            return WebhookVerificationResult(
                ok=False, error="missing X-Twilio-Signature header"
            )
        try:
            params = dict(parse_qsl(ctx.body.decode("utf-8"), keep_blank_values=True))
        except UnicodeDecodeError:
            return WebhookVerificationResult(ok=False, error="undecodable body")

        # Twilio signs the exact public URL; port inclusion varies, so try a
        # small deterministic set of variants before failing closed.
        candidates = [ctx.url, _strip_port(ctx.url)]
        for url in dict.fromkeys(candidates):
            expected = compute_twilio_signature(self.auth_token, url, params)
            if hmac.compare_digest(expected, signature):
                digest = hashlib.sha256(
                    signature.encode() + b"\n" + url.encode() + b"\n" + ctx.body
                ).hexdigest()
                return WebhookVerificationResult(
                    ok=True, dedupe_key=f"twilio:{digest}"
                )
        return WebhookVerificationResult(
            ok=False, error=f"invalid signature for URL {ctx.url}"
        )

    def parse_webhook(self, ctx: WebhookContext) -> WebhookParseResult:
        try:
            params = dict(parse_qsl(ctx.body.decode("utf-8"), keep_blank_values=True))
        except UnicodeDecodeError:
            return WebhookParseResult(response_status=400, response_body=EMPTY_TWIML,
                                      response_content_type="text/xml")
        event = self._normalize_event(params, ctx.query.get("callId"))
        # Status callbacks get an empty ack; voice-flow requests (initial
        # answer webhook, Gather actions) get a keep-alive <Pause> so the
        # call stays up while the manager speaks via call update.
        is_status = ctx.query.get("type") == "status"
        body = EMPTY_TWIML if is_status else KEEPALIVE_TWIML
        return WebhookParseResult(
            events=[event] if event else [],
            response_body=body,
            response_content_type="text/xml",
        )

    def _normalize_event(
        self, params: Dict[str, str], call_id_override: Optional[str]
    ) -> Optional[NormalizedEvent]:
        call_sid = params.get("CallSid", "")
        base = dict(
            provider=self.name,
            provider_call_id=call_sid or None,
            call_id=call_id_override or None,
            from_number=params.get("From") or None,
            to_number=params.get("To") or None,
            direction=_normalize_direction(params.get("Direction")),
            dedupe_key=None,  # the webhook layer dedupes on the signature
            raw=dict(params),
        )

        speech = params.get("SpeechResult")
        if speech:
            return NormalizedEvent(
                type=EventType.CALL_SPEECH, text=speech, is_final=True, **base
            )
        digits = params.get("Digits")
        if digits:
            return NormalizedEvent(type=EventType.CALL_DTMF, digits=digits, **base)

        status = params.get("CallStatus", "")
        if status in ("queued", "initiated"):
            return NormalizedEvent(type=EventType.CALL_INITIATED, **base)
        if status == "ringing":
            return NormalizedEvent(type=EventType.CALL_RINGING, **base)
        if status == "in-progress":
            return NormalizedEvent(type=EventType.CALL_ANSWERED, **base)
        reason = _ENDED_REASON_MAP.get(status)
        if reason:
            return NormalizedEvent(type=EventType.CALL_ENDED, reason=reason, **base)
        return None

    # -- call control --------------------------------------------------------------

    def _webhook_url_with(self, call: CallRecord, **query) -> str:
        base = self.webhook_url or ""
        return f"{base}?{urlencode({'callId': call.call_id, **query})}"

    async def initiate_call(self, call: CallRecord) -> str:
        form: Dict[str, Any] = {
            "To": call.to_number,
            "From": call.from_number,
            "Url": self._webhook_url_with(call),
            "StatusCallback": self._webhook_url_with(call, type="status"),
            "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
            "Timeout": "30",
        }
        result = await self._api("/Calls.json", form)
        return str((result or {}).get("sid", ""))

    async def hangup_call(self, call: CallRecord) -> None:
        await self._api(
            f"/Calls/{call.provider_call_id}.json",
            {"Status": "completed"},
            allow_not_found=True,
        )

    def _speak_twiml(self, call: CallRecord, text: str) -> str:
        extra = self.config.provider_extra.get("twilio", {})
        voice = xml_escape(str(extra.get("voice", "alice")))
        language = xml_escape(str(extra.get("language", "en-US")))
        gather = ""
        if call.mode == "conversation":
            action = xml_escape(self._webhook_url_with(call))
            gather = (
                f'<Gather input="speech" speechTimeout="auto" '
                f'language="{language}" action="{action}" method="POST"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?><Response>'
            f'<Say voice="{voice}" language="{language}">{xml_escape(text)}</Say>'
            f'{gather}<Pause length="60"/></Response>'
        )

    async def speak(self, call: CallRecord, text: str) -> None:
        # No live-TTS API: replace the call's TwiML with Say + embedded
        # Gather so the caller's next utterance posts back to us.
        await self._api(
            f"/Calls/{call.provider_call_id}.json",
            {"Twiml": self._speak_twiml(call, text)},
        )

    async def send_dtmf(self, call: CallRecord, digits: str) -> None:
        safe = "".join(c for c in digits if c in "0123456789*#w")
        await self._api(
            f"/Calls/{call.provider_call_id}.json",
            {
                "Twiml": (
                    '<?xml version="1.0" encoding="UTF-8"?><Response>'
                    f'<Play digits="{safe}"/><Pause length="60"/></Response>'
                )
            },
        )

    async def start_listening(self, call: CallRecord) -> None:
        """No-op: every conversation-mode ``speak`` embeds a ``<Gather>``,
        and updating the call here would cut the greeting off mid-word."""

    async def get_call_status(self, call: CallRecord) -> CallStatusResult:
        try:
            result = await self._api(
                f"/Calls/{call.provider_call_id}.json",
                method="GET",
                allow_not_found=True,
            )
        except Exception:  # noqa: BLE001 — transient errors must not kill restore
            return CallStatusResult(is_unknown=True, raw_status="error")
        if result is None:
            return CallStatusResult(is_terminal=True, raw_status="not-found")
        status = str(result.get("status", ""))
        return CallStatusResult(
            is_active=status not in _TERMINAL_STATUSES,
            is_terminal=status in _TERMINAL_STATUSES,
            raw_status=status,
        )
