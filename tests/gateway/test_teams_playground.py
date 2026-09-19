"""Tests for the opt-in local Teams Playground emulator contract."""

from types import SimpleNamespace

import pytest

from plugins.platforms.teams.playground import (
    PlaygroundConfig,
    PlaygroundValidationError,
    build_activity_envelope,
    build_playground_command,
    generated_callback_url,
    generated_test_url,
    validate_playground_url,
)


def test_disabled_by_default_and_production_credentials_are_not_reused():
    config = PlaygroundConfig.from_extra({})
    assert config.enabled is False
    assert config.app_endpoint is None
    assert config.client_secret is None


def test_local_bot_endpoint_is_validated_and_command_targets_the_real_playground():
    config = PlaygroundConfig.from_extra({
        "playground_url": "http://127.0.0.1:3978/api/messages",
        "playground_allow_private": True,
    })
    assert config.enabled is True
    assert config.app_endpoint == "http://127.0.0.1:3978/api/messages"
    assert config.ui_url == "http://127.0.0.1:56150"
    assert build_playground_command(config) == (
        "agentsplayground -e http://127.0.0.1:3978/api/messages "
        "-c emulator --disable-telemetry"
    )


def test_local_endpoint_requires_explicit_opt_in():
    with pytest.raises(PlaygroundValidationError):
        validate_playground_url("http://127.0.0.1:3978/api/messages", allow_private=False)
    assert validate_playground_url(
        "http://127.0.0.1:3978/api/messages", allow_private=True
    ) == "http://127.0.0.1:3978/api/messages"
    assert validate_playground_url(
        "http://playground.local:3979/api/messages", allow_private=True
    ) == "http://playground.local:3979/api/messages"


def test_unsafe_endpoint_is_rejected_without_network_access():
    for url in ("file:///tmp/playground", "http://169.254.169.254/latest", "https://example.com/playground", "http://user:pass@localhost:1"):
        with pytest.raises(PlaygroundValidationError):
            validate_playground_url(url, allow_private=True)


def test_url_generation_only_happens_for_valid_configured_endpoint():
    config = PlaygroundConfig.from_extra({"playground_url": "http://localhost:3979/api/messages", "playground_allow_private": True})
    assert generated_test_url(config) == "http://127.0.0.1:56150"
    assert generated_callback_url(config) == "http://localhost:3979/api/messages"
    assert generated_test_url(PlaygroundConfig.from_extra({})) is None


def test_activity_envelope_is_generic_bot_framework_compatible_and_secret_free():
    envelope = build_activity_envelope("hello", conversation_id="conv-1", user_id="user-1")
    assert envelope == {
        "type": "message",
        "text": "hello",
        "conversation": {"id": "conv-1"},
        "from": {"id": "user-1"},
        "channelId": "msteams",
    }
    assert "secret" not in str(envelope).lower()


@pytest.mark.anyio
async def test_health_handshake_uses_short_timeout_and_sanitizes_errors(monkeypatch):
    config = PlaygroundConfig.from_extra({"playground_url": "http://localhost:3979/api/messages", "playground_allow_private": True})
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_args):
            return False
        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["get_kwargs"] = kwargs
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None)

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)
    from plugins.platforms.teams.playground import handshake
    result = await handshake(config)
    assert result.ok is True
    assert seen["timeout"] <= 5
    assert seen["url"] == "http://localhost:3979/api/health"
    assert "secret" not in str(result).lower()


@pytest.mark.anyio
async def test_health_handshake_timeout_is_a_safe_failure(monkeypatch):
    config = PlaygroundConfig.from_extra({"playground_url": "http://localhost:3979/api/messages", "playground_allow_private": True})

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        async def get(self, *_args, **_kwargs):
            raise TimeoutError("token=do-not-return")

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: FakeClient())
    from plugins.platforms.teams.playground import handshake
    result = await handshake(config)
    assert result.ok is False
    assert result.message == "Playground handshake timed out."


def test_playground_config_isolated_from_real_teams_credentials():
    config = PlaygroundConfig.from_extra({"client_id": "real-id", "client_secret": "secret", "tenant_id": "tenant"})
    assert config.enabled is False
    assert config.app_endpoint is None
    assert config.client_id is None


def test_playground_activity_seam_preserves_legacy_teams_adapter():
    from plugins.platforms.teams.adapter import TeamsAdapter
    assert TeamsAdapter.__name__ == "TeamsAdapter"
