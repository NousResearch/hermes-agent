"""Provider registry/factory for the voice_call platform."""

from typing import Callable, Dict

from ..config import VoiceCallConfig
from .base import (
    CallStatusResult,
    VoiceCallProvider,
    WebhookContext,
    WebhookParseResult,
    WebhookVerificationResult,
)


def _mock_factory(config: VoiceCallConfig) -> VoiceCallProvider:
    from .mock import MockProvider

    return MockProvider(config)


def _telnyx_factory(config: VoiceCallConfig) -> VoiceCallProvider:
    from .telnyx import TelnyxProvider

    return TelnyxProvider(config)


def _twilio_factory(config: VoiceCallConfig) -> VoiceCallProvider:
    from .twilio import TwilioProvider

    return TwilioProvider(config)


def _plivo_factory(config: VoiceCallConfig) -> VoiceCallProvider:
    from .plivo import PlivoProvider

    return PlivoProvider(config)


PROVIDER_FACTORIES: Dict[str, Callable[[VoiceCallConfig], VoiceCallProvider]] = {
    "mock": _mock_factory,
    "telnyx": _telnyx_factory,
    "twilio": _twilio_factory,
    "plivo": _plivo_factory,
}


def create_provider(config: VoiceCallConfig) -> VoiceCallProvider:
    """Instantiate the configured provider (imports lazily per provider)."""
    factory = PROVIDER_FACTORIES.get(config.provider)
    if factory is None:
        raise ValueError(f"unknown voice_call provider: {config.provider!r}")
    return factory(config)


__all__ = [
    "CallStatusResult",
    "PROVIDER_FACTORIES",
    "VoiceCallProvider",
    "WebhookContext",
    "WebhookParseResult",
    "WebhookVerificationResult",
    "create_provider",
]
