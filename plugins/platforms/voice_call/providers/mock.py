"""Credential-free mock provider for development and tests.

Behaves like a well-behaved carrier without any network access:

- ``initiate_call`` returns a synthetic provider call id and (by default)
  pushes ``call.ringing`` → ``call.answered`` → ``call.active`` events
  through the runtime's event sink, like a real carrier webhook would.
- The webhook endpoint accepts pre-normalized JSON events —
  ``{"event": {...}}`` or ``{"events": [...]}`` — so inbound calls and
  caller speech can be simulated with curl.
- ``speak`` records what the bot said for assertions.

No signature verification (there is no carrier to verify against) and no
public webhook requirement.
"""

import asyncio
import json
import logging
import uuid
from typing import Awaitable, Callable, List, Optional

from ..config import VoiceCallConfig
from ..events import CallRecord, EventType, NormalizedEvent
from .base import (
    CallStatusResult,
    VoiceCallProvider,
    WebhookContext,
    WebhookParseResult,
    WebhookVerificationResult,
)

logger = logging.getLogger(__name__)

EventSink = Callable[[NormalizedEvent], Awaitable[None]]


def _parse_event_dict(data: dict) -> Optional[NormalizedEvent]:
    """Build a NormalizedEvent from a JSON dict; None when type is unknown."""
    try:
        event_type = EventType(str(data.get("type", "")))
    except ValueError:
        return None
    direction = data.get("direction")
    if direction not in ("inbound", "outbound"):
        direction = None
    return NormalizedEvent(
        type=event_type,
        provider="mock",
        provider_call_id=data.get("provider_call_id"),
        call_id=data.get("call_id"),
        from_number=data.get("from") or data.get("from_number"),
        to_number=data.get("to") or data.get("to_number"),
        direction=direction,
        text=data.get("text"),
        is_final=bool(data.get("is_final", True)),
        digits=data.get("digits"),
        reason=data.get("reason"),
        retryable=bool(data.get("retryable", False)),
        dedupe_key=data.get("dedupe_key") or data.get("event_id"),
        raw=data,
    )


class MockProvider(VoiceCallProvider):
    name = "mock"
    requires_public_webhook = False

    def __init__(self, config: VoiceCallConfig):
        super().__init__(config)
        extra = config.provider_extra.get("mock", {})
        # Auto-advance outbound calls to answered unless disabled (tests
        # that exercise ring timeouts turn this off).
        self.auto_answer: bool = bool(extra.get("auto_answer", True))
        self.fail_initiate: bool = bool(extra.get("fail_initiate", False))
        self.event_sink: Optional[EventSink] = None
        # Observability for tests.
        self.spoken: List[tuple] = []          # (provider_call_id, text)
        self.hangups: List[str] = []           # provider_call_id
        self.dtmf_sent: List[tuple] = []       # (provider_call_id, digits)
        self.listening: List[str] = []         # provider_call_id currently listening
        self.answered: List[str] = []          # provider_call_id
        self._terminal: set = set()

    # -- webhook ----------------------------------------------------------

    def verify_webhook(self, ctx: WebhookContext) -> WebhookVerificationResult:
        return WebhookVerificationResult(ok=True)

    def parse_webhook(self, ctx: WebhookContext) -> WebhookParseResult:
        try:
            data = json.loads(ctx.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return WebhookParseResult(
                response_status=400, response_body='{"error": "invalid json"}'
            )
        raw_events = data.get("events") if isinstance(data, dict) else None
        if raw_events is None:
            raw_events = [data.get("event")] if isinstance(data, dict) and data.get("event") else []
        events = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            event = _parse_event_dict(raw)
            if event is None:
                return WebhookParseResult(
                    response_status=400,
                    response_body=json.dumps(
                        {"error": f"unknown event type {raw.get('type')!r}"}
                    ),
                )
            events.append(event)
        return WebhookParseResult(events=events, response_body='{"ok": true}')

    # -- call control ------------------------------------------------------

    async def initiate_call(self, call: CallRecord) -> str:
        if self.fail_initiate:
            raise RuntimeError("mock provider: initiate_call forced failure")
        provider_call_id = f"mock-{uuid.uuid4().hex[:12]}"
        if self.auto_answer and self.event_sink is not None:
            asyncio.get_running_loop().create_task(
                self._auto_answer(provider_call_id, call)
            )
        return provider_call_id

    async def _auto_answer(self, provider_call_id: str, call: CallRecord) -> None:
        # Yield first so the manager finishes registering the call id mapping.
        await asyncio.sleep(0)
        sink = self.event_sink
        if sink is None:
            return
        for event_type in (
            EventType.CALL_RINGING,
            EventType.CALL_ANSWERED,
            EventType.CALL_ACTIVE,
        ):
            if provider_call_id in self._terminal:
                return
            await sink(
                NormalizedEvent(
                    type=event_type,
                    provider=self.name,
                    provider_call_id=provider_call_id,
                    call_id=call.call_id,
                    direction="outbound",
                    from_number=call.from_number,
                    to_number=call.to_number,
                )
            )

    async def answer_call(self, call: CallRecord) -> None:
        if call.provider_call_id:
            self.answered.append(call.provider_call_id)

    async def hangup_call(self, call: CallRecord) -> None:
        if call.provider_call_id:
            self.hangups.append(call.provider_call_id)
            self._terminal.add(call.provider_call_id)

    async def speak(self, call: CallRecord, text: str) -> None:
        self.spoken.append((call.provider_call_id, text))

    async def send_dtmf(self, call: CallRecord, digits: str) -> None:
        self.dtmf_sent.append((call.provider_call_id, digits))

    async def start_listening(self, call: CallRecord) -> None:
        if call.provider_call_id and call.provider_call_id not in self.listening:
            self.listening.append(call.provider_call_id)

    async def stop_listening(self, call: CallRecord) -> None:
        if call.provider_call_id in self.listening:
            self.listening.remove(call.provider_call_id)

    async def get_call_status(self, call: CallRecord) -> CallStatusResult:
        if call.provider_call_id in self._terminal:
            return CallStatusResult(is_terminal=True, raw_status="completed")
        return CallStatusResult(is_active=True, raw_status="active")
