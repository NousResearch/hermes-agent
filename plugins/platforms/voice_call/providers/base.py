"""Provider abstraction for the voice_call platform.

Everything carrier-specific lives behind :class:`VoiceCallProvider`:
API clients, webhook signature verification, wire-format parsing into
``NormalizedEvent``, speak/TTS, transcription control, and (for carriers
that support media streams) realtime streaming field construction.

The call manager and webhook server consume only this interface.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import VoiceCallConfig
from ..events import CallRecord, NormalizedEvent


@dataclass
class WebhookContext:
    """One inbound HTTP request to the webhook server, provider-agnostic."""

    method: str
    path: str
    body: bytes
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, str] = field(default_factory=dict)
    remote_ip: str = ""
    # Full external URL as the carrier signed it (public URL + path),
    # needed by HMAC schemes that sign the URL (Twilio).
    url: str = ""

    def header(self, name: str) -> Optional[str]:
        """Case-insensitive header lookup."""
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


@dataclass
class WebhookVerificationResult:
    ok: bool
    error: Optional[str] = None
    # Provider-derived replay/dedupe key (e.g. Plivo nonce, Telnyx
    # signature+timestamp). Falls back to a body hash in the server.
    dedupe_key: Optional[str] = None


@dataclass
class WebhookParseResult:
    """Normalized events plus the HTTP response the carrier expects."""

    events: List[NormalizedEvent] = field(default_factory=list)
    response_status: int = 200
    response_body: str = ""
    response_content_type: str = "application/json"


@dataclass
class CallStatusResult:
    """Result of asking the carrier about a call (used by boot restore)."""

    is_active: bool = False
    is_terminal: bool = False
    # True when the carrier could not answer (transient error) — keep the
    # call and rely on timers rather than guessing.
    is_unknown: bool = False
    raw_status: Optional[str] = None


class VoiceCallProvider(abc.ABC):
    """Base interface every call provider implements.

    Methods that not all carriers support (``send_dtmf``,
    ``start_listening``/``stop_listening``, ``answer_call``) have safe
    default implementations.
    """

    name: str = ""
    requires_public_webhook: bool = True

    def __init__(self, config: VoiceCallConfig):
        self.config = config
        self.public_url: Optional[str] = None

    def set_public_url(self, url: str) -> None:
        self.public_url = url.rstrip("/")

    @property
    def webhook_url(self) -> Optional[str]:
        if not self.public_url:
            return None
        return f"{self.public_url}{self.config.serve.path}"

    # -- webhook ----------------------------------------------------------

    @abc.abstractmethod
    def verify_webhook(self, ctx: WebhookContext) -> WebhookVerificationResult:
        """Verify the request really came from the carrier."""

    @abc.abstractmethod
    def parse_webhook(self, ctx: WebhookContext) -> WebhookParseResult:
        """Parse a verified request into normalized events + HTTP response."""

    # -- call control ------------------------------------------------------

    @abc.abstractmethod
    async def initiate_call(self, call: CallRecord) -> str:
        """Start an outbound call; return the carrier's call id."""

    async def answer_call(self, call: CallRecord) -> None:
        """Answer an inbound call (carriers like Telnyx require an explicit
        answer command; webhook-response carriers answer implicitly)."""

    @abc.abstractmethod
    async def hangup_call(self, call: CallRecord) -> None:
        """Terminate the call at the carrier."""

    @abc.abstractmethod
    async def speak(self, call: CallRecord, text: str) -> None:
        """Speak ``text`` to the remote party using carrier-native TTS."""

    async def send_dtmf(self, call: CallRecord, digits: str) -> None:
        raise NotImplementedError(f"{self.name} does not support DTMF")

    async def start_listening(self, call: CallRecord) -> None:
        """Start carrier-native transcription of the remote party."""

    async def stop_listening(self, call: CallRecord) -> None:
        """Stop carrier-native transcription."""

    @abc.abstractmethod
    async def get_call_status(self, call: CallRecord) -> CallStatusResult:
        """Ask the carrier whether a call is still alive (boot restore)."""

    # -- realtime media streams (P7) ----------------------------------------

    def streaming_fields(self, stream_url: str, auth_token: str) -> Dict[str, Any]:
        """Provider-specific dial/answer fields that attach a media stream.

        Returns ``{}`` for carriers without media-stream support.
        """
        return {}

    def finalize_response(
        self, ctx: WebhookContext, result: WebhookParseResult
    ) -> WebhookParseResult:
        """Hook called after the webhook's events were processed, letting a
        provider rewrite the HTTP response based on state created during
        processing (Twilio uses this to serve ``<Connect><Stream>`` TwiML
        for freshly-registered realtime calls). Default: unchanged."""
        return result
