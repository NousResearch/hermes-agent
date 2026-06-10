"""Config parsing and validation tests for the voice_call platform."""

import pytest

from plugins.platforms.voice_call.config import (
    DEFAULT_PROVIDER,
    VoiceCallConfig,
    is_e164,
    normalize_e164,
)


TELNYX_ENV = {
    "TELNYX_API_KEY": "test-key",
    "TELNYX_CONNECTION_ID": "test-conn",
    "TELNYX_PUBLIC_KEY": "test-pub",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every voice_call-relevant env var so tests are hermetic."""
    for var in (
        "VOICE_CALL_PROVIDER", "VOICE_CALL_FROM_NUMBER",
        "VOICE_CALL_ALLOWED_NUMBERS", "VOICE_CALL_PUBLIC_URL",
        "NGROK_DOMAIN",
        *TELNYX_ENV,
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
        "PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    cfg = VoiceCallConfig.from_extra({})
    assert cfg.provider == DEFAULT_PROVIDER == "telnyx"
    assert cfg.session_scope == "per-phone"
    assert cfg.inbound_policy == "allowlist"
    assert cfg.serve.bind == "127.0.0.1"
    assert cfg.serve.port == 3334
    assert cfg.serve.path == "/voice/webhook"
    assert cfg.tunnel.provider == "none"
    assert cfg.outbound.default_mode == "notify"
    assert cfg.security.skip_signature_verification is False
    assert cfg.streaming.enabled is False
    assert cfg.realtime.enabled is False


def test_e164_normalization():
    assert normalize_e164("+1 (555) 555-0001") == "+15555550001"
    assert normalize_e164("15555550001") == "+15555550001"
    assert is_e164("+15555550001")
    assert not is_e164("not-a-number")
    assert not is_e164("+0123")


def test_mock_provider_is_valid_without_credentials():
    cfg = VoiceCallConfig.from_extra({"provider": "mock"})
    assert cfg.validate() == []


def test_unknown_provider_rejected():
    cfg = VoiceCallConfig.from_extra({"provider": "doesnotexist"})
    errors = cfg.validate()
    assert len(errors) == 1
    assert "provider" in errors[0]


def test_telnyx_requires_credentials():
    cfg = VoiceCallConfig.from_extra({"provider": "telnyx", "public_url": "https://x.example"})
    errors = cfg.validate()
    assert any("TELNYX_API_KEY" in e for e in errors)
    assert any("TELNYX_PUBLIC_KEY" in e for e in errors)


def test_telnyx_valid_with_env_and_public_url(monkeypatch):
    for key, value in TELNYX_ENV.items():
        monkeypatch.setenv(key, value)
    cfg = VoiceCallConfig.from_extra({"provider": "telnyx", "public_url": "https://x.example"})
    assert cfg.validate() == []


def test_telnyx_without_public_webhook_rejected(monkeypatch):
    for key, value in TELNYX_ENV.items():
        monkeypatch.setenv(key, value)
    cfg = VoiceCallConfig.from_extra({"provider": "telnyx"})
    errors = cfg.validate()
    assert any("publicly reachable webhook" in e for e in errors)
    # A tunnel satisfies the requirement.
    cfg = VoiceCallConfig.from_extra(
        {"provider": "telnyx", "tunnel": {"provider": "ngrok"}}
    )
    assert cfg.validate() == []


def test_telnyx_skip_signature_drops_public_key_requirement(monkeypatch):
    monkeypatch.setenv("TELNYX_API_KEY", "k")
    monkeypatch.setenv("TELNYX_CONNECTION_ID", "c")
    cfg = VoiceCallConfig.from_extra(
        {
            "provider": "telnyx",
            "public_url": "https://x.example",
            "security": {"skip_signature_verification": True},
        }
    )
    assert cfg.validate() == []


def test_provider_credentials_from_extra_block():
    cfg = VoiceCallConfig.from_extra(
        {
            "provider": "telnyx",
            "public_url": "https://x.example",
            "telnyx": {
                "api_key": "k", "connection_id": "c", "public_key": "p",
            },
        }
    )
    assert cfg.validate() == []
    assert cfg.provider_credential("api_key", "TELNYX_API_KEY") == "k"


def test_invalid_numbers_rejected():
    cfg = VoiceCallConfig.from_extra(
        {"provider": "mock", "from_number": "bogus", "allow_from": ["123abc"]}
    )
    errors = cfg.validate()
    assert any("from_number" in e for e in errors)
    assert any("allow_from" in e for e in errors)


def test_allow_from_env_fallback(monkeypatch):
    monkeypatch.setenv("VOICE_CALL_ALLOWED_NUMBERS", "+15555550001, +1 (555) 555-0002")
    cfg = VoiceCallConfig.from_extra({})
    assert cfg.allow_from == ["+15555550001", "+15555550002"]


def test_enum_validation():
    cfg = VoiceCallConfig.from_extra(
        {
            "provider": "mock",
            "session_scope": "weird",
            "inbound_policy": "weird",
            "outbound": {"default_mode": "weird"},
            "tunnel": {"provider": "weird"},
        }
    )
    errors = " | ".join(cfg.validate())
    assert "session_scope" in errors
    assert "inbound_policy" in errors
    assert "default_mode" in errors
    assert "tunnel.provider" in errors


def test_serve_validation():
    cfg = VoiceCallConfig.from_extra(
        {"provider": "mock", "serve": {"port": 99999, "path": "nope"}}
    )
    errors = " | ".join(cfg.validate())
    assert "serve.port" in errors
    assert "serve.path" in errors


def test_realtime_requires_streaming_capable_provider(monkeypatch):
    cfg = VoiceCallConfig.from_extra(
        {"provider": "mock", "realtime": {"enabled": True}}
    )
    errors = " | ".join(cfg.validate())
    assert "realtime" in errors
    # telnyx + realtime is allowed
    for key, value in TELNYX_ENV.items():
        monkeypatch.setenv(key, value)
    cfg = VoiceCallConfig.from_extra(
        {
            "provider": "telnyx",
            "public_url": "https://x.example",
            "realtime": {"enabled": True, "provider": "openai"},
        }
    )
    assert cfg.validate() == []


def test_realtime_provider_enum(monkeypatch):
    for key, value in TELNYX_ENV.items():
        monkeypatch.setenv(key, value)
    cfg = VoiceCallConfig.from_extra(
        {
            "provider": "telnyx",
            "public_url": "https://x.example",
            "realtime": {"enabled": True, "provider": "weird"},
        }
    )
    assert any("realtime.provider" in e for e in cfg.validate())
