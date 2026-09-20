from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import api_server as api_module
from gateway.platforms.api_server import APIServerAdapter


def test_configured_native_alias_resolves_to_canonical_session(monkeypatch):
    config = GatewayConfig(
        group_sessions_per_user=True,
        session_key_aliases={
            "phone-main": {
                "platform": "discord",
                "chat_id": "123",
                "chat_type": "group",
                "user_id": "456",
            }
        },
    )
    monkeypatch.setattr(api_module, "load_gateway_config", lambda: config)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    key, source, error = adapter._resolve_api_session_identity("phone-main")
    assert error is None
    assert key == "agent:main:discord:group:123:456"
    assert source is not None
    assert source.platform is Platform.DISCORD
    assert source.chat_id == "123"
    assert source.user_id == "456"
    adapter._response_store.close()


def test_unconfigured_key_preserves_api_identity(monkeypatch):
    monkeypatch.setattr(api_module, "load_gateway_config", lambda: GatewayConfig())
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._resolve_api_session_identity("ordinary-client") == ("ordinary-client", None, None)
    adapter._response_store.close()