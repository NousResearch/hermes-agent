"""Shared conformance suite: every provider honors the VoiceCallProvider ABC.

Parametrized over all four providers — anything provider-shaped that the
manager or webhook server relies on is asserted here once.
"""

import inspect

import pytest

from plugins.platforms.voice_call.config import VoiceCallConfig
from plugins.platforms.voice_call.providers import PROVIDER_FACTORIES, create_provider
from plugins.platforms.voice_call.providers.base import (
    VoiceCallProvider,
    WebhookContext,
    WebhookParseResult,
    WebhookVerificationResult,
)

CRED_ENV = {
    "telnyx": {
        "TELNYX_API_KEY": "k", "TELNYX_CONNECTION_ID": "c",
        "TELNYX_PUBLIC_KEY": "cHVibGljLWtleS1ieXRlcy0zMi1sb25nLXBhZGRpbmc=",
    },
    "twilio": {"TWILIO_ACCOUNT_SID": "AC1", "TWILIO_AUTH_TOKEN": "t"},
    "plivo": {"PLIVO_AUTH_ID": "MA1", "PLIVO_AUTH_TOKEN": "t"},
    "mock": {},
}

PROVIDERS = sorted(PROVIDER_FACTORIES)


@pytest.fixture
def make_provider(monkeypatch):
    def _make(name: str) -> VoiceCallProvider:
        for var, value in CRED_ENV[name].items():
            monkeypatch.setenv(var, value)
        cfg = VoiceCallConfig.from_extra(
            {"provider": name, "public_url": "https://hooks.example"}
        )
        provider = create_provider(cfg)
        provider.set_public_url("https://hooks.example")
        return provider

    return _make


@pytest.mark.parametrize("name", PROVIDERS)
def test_provider_shape(name, make_provider):
    provider = make_provider(name)
    assert isinstance(provider, VoiceCallProvider)
    assert provider.name == name
    # Real carriers need a public webhook; the mock doesn't.
    assert provider.requires_public_webhook == (name != "mock")
    assert provider.webhook_url == "https://hooks.example/voice/webhook"
    # Required async methods are coroutine functions.
    for method in ("initiate_call", "answer_call", "hangup_call", "speak",
                   "send_dtmf", "start_listening", "stop_listening",
                   "get_call_status"):
        assert inspect.iscoroutinefunction(getattr(provider, method)), method
    # streaming_fields returns a dict (may be empty for non-streaming carriers).
    assert isinstance(provider.streaming_fields("wss://x/y", "tok"), dict)


@pytest.mark.parametrize("name", PROVIDERS)
def test_verify_webhook_returns_result_not_exception(name, make_provider):
    """Garbage requests produce a clean verification failure (mock: pass)."""
    provider = make_provider(name)
    ctx = WebhookContext(
        method="POST", path="/voice/webhook", body=b"\xff\x00garbage",
        headers={"X-Junk": "1"}, url="https://hooks.example/voice/webhook",
    )
    result = provider.verify_webhook(ctx)
    assert isinstance(result, WebhookVerificationResult)
    assert result.ok is (name == "mock")
    if not result.ok:
        assert result.error


@pytest.mark.parametrize("name", PROVIDERS)
def test_parse_webhook_tolerates_garbage(name, make_provider):
    """Unparseable bodies return a WebhookParseResult, never raise."""
    provider = make_provider(name)
    ctx = WebhookContext(
        method="POST", path="/voice/webhook", body=b"\xff\x00garbage",
        url="https://hooks.example/voice/webhook",
    )
    result = provider.parse_webhook(ctx)
    assert isinstance(result, WebhookParseResult)
    assert result.events == [] or result.response_status >= 400


@pytest.mark.parametrize("name", PROVIDERS)
def test_parse_webhook_empty_body(name, make_provider):
    provider = make_provider(name)
    ctx = WebhookContext(
        method="POST", path="/voice/webhook", body=b"",
        url="https://hooks.example/voice/webhook",
    )
    result = provider.parse_webhook(ctx)
    assert isinstance(result, WebhookParseResult)


def test_factory_rejects_unknown_provider():
    cfg = VoiceCallConfig.from_extra({"provider": "mock"})
    cfg.provider = "smoke-signals"
    with pytest.raises(ValueError, match="unknown voice_call provider"):
        create_provider(cfg)
