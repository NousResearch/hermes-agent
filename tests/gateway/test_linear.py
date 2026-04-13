import hashlib
import hmac
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.linear import LinearAdapter
from toolsets import get_toolset


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _make_adapter(**extra):
    config = PlatformConfig(enabled=True, extra={
        "client_id": "linear-client",
        "client_secret": "linear-secret",
        "webhook_secret": "whsec",
        "public_base_url": "https://jaxmind.xyz",
        **extra,
    })
    return LinearAdapter(config)


def test_apply_env_overrides_configures_linear(monkeypatch):
    config = GatewayConfig()
    monkeypatch.setenv("LINEAR_ENABLED", "true")
    monkeypatch.setenv("LINEAR_CLIENT_ID", "cid")
    monkeypatch.setenv("LINEAR_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("LINEAR_WEBHOOK_SECRET", "whsec")
    monkeypatch.setenv("LINEAR_PUBLIC_BASE_URL", "https://linear.example.com/")
    monkeypatch.setenv("LINEAR_HOST", "0.0.0.0")
    monkeypatch.setenv("LINEAR_PORT", "9001")
    monkeypatch.setenv("LINEAR_SCOPES", "read,comments:create,app:mentionable")

    _apply_env_overrides(config)

    assert Platform.LINEAR in config.platforms
    linear = config.platforms[Platform.LINEAR]
    assert linear.enabled is True
    assert linear.extra["client_id"] == "cid"
    assert linear.extra["client_secret"] == "csecret"
    assert linear.extra["webhook_secret"] == "whsec"
    assert linear.extra["public_base_url"] == "https://linear.example.com"
    assert linear.extra["host"] == "0.0.0.0"
    assert linear.extra["port"] == 9001
    assert linear.extra["scopes"] == ["read", "comments:create", "app:mentionable"]


def test_get_connected_platforms_includes_linear_with_required_credentials():
    config = GatewayConfig(platforms={
        Platform.LINEAR: PlatformConfig(
            enabled=True,
            extra={
                "client_id": "cid",
                "client_secret": "secret",
                "webhook_secret": "whsec",
            },
        )
    })

    assert Platform.LINEAR in config.get_connected_platforms()


def test_linear_authorize_url_uses_actor_app_and_scope_list():
    adapter = _make_adapter(scopes=["read", "comments:create", "app:mentionable"])

    url = adapter._build_authorize_url("state-123")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "linear.app"
    assert parsed.path == "/oauth/authorize"
    assert params["client_id"] == ["linear-client"]
    assert params["redirect_uri"] == ["https://jaxmind.xyz/linear/oauth/callback"]
    assert params["actor"] == ["app"]
    assert params["state"] == ["state-123"]
    assert params["scope"] == ["read,comments:create,app:mentionable"]


def test_validate_signature_accepts_linear_hmac():
    adapter = _make_adapter()
    body = b'{"type":"AgentSessionEvent"}'
    sig = hmac.new(b"whsec", body, hashlib.sha256).hexdigest()

    assert adapter._validate_signature(body, sig) is True
    assert adapter._validate_signature(body, "bad") is False


@pytest.mark.asyncio
async def test_callback_rejects_expired_oauth_state():
    adapter = _make_adapter()
    adapter._save_json(adapter._states_path, {"expired-state": {"created_at": 0}})

    request = SimpleNamespace(query={"code": "code-123", "state": "expired-state", "error": ""})

    response = await adapter._handle_callback(request)

    assert response.status == 400
    assert response.text == "Invalid or expired OAuth state.\n"
    assert adapter._load_json(adapter._states_path) == {}


def test_build_prompt_for_created_event_uses_prompt_context():
    adapter = _make_adapter()
    prompt = adapter._build_prompt({
        "action": "created",
        "promptContext": "<issue>Investigate regression</issue>",
        "guidance": [{"rule": "stay concise"}],
        "agentSession": {
            "url": "https://linear.app/session/123",
            "issue": {"identifier": "PAB-80", "title": "Linear agent"},
        },
    })

    assert "promptContext" in prompt
    assert "PAB-80" in prompt
    assert "Investigate regression" in prompt


@pytest.mark.asyncio
async def test_send_posts_response_activity_via_session_mapping(monkeypatch):
    adapter = _make_adapter()
    adapter._session_info["linear:session-1"] = {
        "agent_session_id": "session-1",
        "app_user_id": "app-user-1",
        "chat_name": "PAB-80",
    }

    captured = {}

    async def _fake_create_activity(**kwargs):
        captured.update(kwargs)
        return {"agentActivityCreate": {"success": True, "agentActivity": {"id": "activity-1"}}}

    monkeypatch.setattr(adapter, "_create_activity", _fake_create_activity)

    result = await adapter.send("linear:session-1", "Done.")

    assert result.success is True
    assert result.message_id == "activity-1"
    assert captured["app_user_id"] == "app-user-1"
    assert captured["agent_session_id"] == "session-1"
    assert captured["activity_type"] == "response"
    assert captured["body"] == "Done."


def test_platform_registries_include_linear():
    from hermes_cli.platforms import PLATFORMS as SHARED_PLATFORMS
    from hermes_cli.gateway import _PLATFORMS as GATEWAY_PLATFORMS
    from hermes_cli.setup import _GATEWAY_PLATFORMS as SETUP_GATEWAY_PLATFORMS

    assert "linear" in SHARED_PLATFORMS
    assert SHARED_PLATFORMS["linear"].default_toolset == "hermes-linear"
    assert get_toolset("hermes-linear") is not None
    assert any(p["key"] == "linear" for p in GATEWAY_PLATFORMS)
    assert any(name == "Linear Agent Sessions" for name, _env, _func in SETUP_GATEWAY_PLATFORMS)
