"""Tests for the second LINE admin channel registration."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")

LineAdapter = _line.LineAdapter
check_requirements = _line.check_requirements
check_requirements_admin = getattr(_line, "check_requirements_admin", lambda: "missing")
interactive_setup_admin = getattr(_line, "interactive_setup_admin", None)
register = _line.register
verify_line_signature = _line.verify_line_signature


def _line_signature(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def test_customer_defaults_remain_unchanged_without_admin_env(monkeypatch):
    from gateway.config import Platform, PlatformConfig

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    for name in (
        "LINE_PORT",
        "LINE_ADMIN_CHANNEL_ACCESS_TOKEN",
        "LINE_ADMIN_CHANNEL_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    adapter = LineAdapter(PlatformConfig(enabled=True))

    assert adapter.channel_access_token == "customer-token"
    assert adapter.channel_secret == "customer-secret"
    assert adapter.platform == Platform("line")
    assert adapter.webhook_port == 8646
    assert adapter.webhook_path == "/line/webhook"
    assert check_requirements() is True
    assert check_requirements_admin() is False


def test_admin_constructor_reads_admin_env_and_defaults(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_PORT", raising=False)
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        adapter = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
        )
    finally:
        platform_registry.unregister("line_admin")

    assert adapter.channel_access_token == "admin-token"
    assert adapter.channel_secret == "admin-secret"
    assert adapter.platform == Platform("line_admin")
    assert adapter.webhook_port == 8647
    assert adapter.webhook_path == "/line-admin/webhook"


def test_register_adds_customer_and_admin_platforms():
    class _FakeCtx:
        def __init__(self):
            self.calls = []

        def register_platform(self, **kwargs):
            self.calls.append(kwargs)

    ctx = _FakeCtx()

    register(ctx)

    by_name = {call["name"]: call for call in ctx.calls}
    assert set(by_name) == {"line", "line_admin"}
    assert by_name["line"]["check_fn"] is check_requirements
    assert by_name["line_admin"]["check_fn"] is check_requirements_admin
    assert by_name["line_admin"]["setup_fn"] is interactive_setup_admin


def test_admin_requirements_gate_needs_token_and_secret(monkeypatch):
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_ADMIN_CHANNEL_SECRET", raising=False)
    assert check_requirements_admin() is False

    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    assert check_requirements_admin() is False

    monkeypatch.delenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    assert check_requirements_admin() is False

    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    assert check_requirements_admin() is True


@pytest.mark.asyncio
async def test_missing_credentials_error_uses_adapter_env_names(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    for name in (
        "LINE_CHANNEL_ACCESS_TOKEN",
        "LINE_CHANNEL_SECRET",
        "LINE_ADMIN_CHANNEL_ACCESS_TOKEN",
        "LINE_ADMIN_CHANNEL_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    assert await customer.connect() is False
    assert customer.fatal_error_message == (
        "LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set"
    )

    assert await admin.connect() is False
    assert admin.fatal_error_message == (
        "LINE_ADMIN_CHANNEL_ACCESS_TOKEN and LINE_ADMIN_CHANNEL_SECRET must be set"
    )


@pytest.mark.asyncio
async def test_lifecycle_lock_namespace_uses_adapter_platform(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    import gateway.status as gateway_status

    class _FakeClient:
        def __init__(self, token):
            self.token = token

        async def get_bot_user_id(self):
            return "bot-user"

    acquire_calls = []
    release_calls = []

    def _acquire(namespace, key):
        acquire_calls.append((namespace, key))
        return True

    def _release(namespace, key):
        release_calls.append((namespace, key))

    monkeypatch.setattr(_line, "_LineClient", _FakeClient)
    monkeypatch.setattr(gateway_status, "acquire_scoped_lock", _acquire)
    monkeypatch.setattr(gateway_status, "release_scoped_lock", _release)
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "channel_access_token": "customer-token",
                    "channel_secret": "customer-secret",
                    "port": 0,
                },
            )
        )
        admin = LineAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "channel_access_token": "admin-token",
                    "channel_secret": "admin-secret",
                    "port": 0,
                },
            ),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    try:
        assert await customer.connect() is True
        assert await admin.connect() is True
    finally:
        await admin.disconnect()
        await customer.disconnect()

    assert acquire_calls[0][0] == "line"
    assert acquire_calls[1][0] == "line_admin"
    assert release_calls[0][0] == "line_admin"
    assert release_calls[1][0] == "line"


@pytest.mark.asyncio
async def test_health_response_reports_adapter_platform(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    customer_response = await customer._handle_health(None)
    admin_response = await admin._handle_health(None)

    assert json.loads(customer_response.text)["platform"] == "line"
    assert json.loads(admin_response.text)["platform"] == "line_admin"


def test_hmac_signatures_are_isolated_by_channel_secret(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    body = b'{"events": []}'
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    customer_sig = _line_signature(body, "customer-secret")
    admin_sig = _line_signature(body, "admin-secret")

    assert verify_line_signature(body, customer_sig, customer.channel_secret)
    assert not verify_line_signature(body, admin_sig, customer.channel_secret)
    assert verify_line_signature(body, admin_sig, admin.channel_secret)
    assert not verify_line_signature(body, customer_sig, admin.channel_secret)


def test_admin_sources_use_distinct_session_namespace(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.session import build_session_key

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    customer_source = customer.build_source(chat_id="U123", chat_type="dm", user_id="U123")
    admin_source = admin.build_source(chat_id="U123", chat_type="dm", user_id="U123")

    assert customer_source.platform.value == "line"
    assert admin_source.platform.value == "line_admin"
    assert build_session_key(customer_source) == "agent:main:line:dm:U123"
    assert build_session_key(admin_source) == "agent:main:line_admin:dm:U123"


def test_admin_media_urls_use_admin_prefix(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_PUBLIC_URL", "https://admin.example.com")
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    assert admin._media_url("tok", "file.png") == "https://admin.example.com/line-admin/media/tok/file.png"


@pytest.mark.asyncio
async def test_verify_code_uses_adapter_platform(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "customer-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "customer-secret")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_ACCESS_TOKEN", "admin-token")
    monkeypatch.setenv("LINE_ADMIN_CHANNEL_SECRET", "admin-secret")
    platform_registry.register(
        PlatformEntry(
            name="line_admin",
            label="LINE (對內)",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
        )
    )

    try:
        customer = LineAdapter(PlatformConfig(enabled=True))
        admin = LineAdapter(
            PlatformConfig(enabled=True),
            platform_value="line_admin",
            env_prefix="LINE_ADMIN_CHANNEL",
            default_port=8647,
            default_webhook_path="/line-admin/webhook",
            default_media_prefix="/line-admin/media",
        )
    finally:
        platform_registry.unregister("line_admin")

    redeem = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(_line, "redeem_verify_code", redeem, raising=False)
    event = {
        "type": "message",
        "replyToken": "reply-token",
        "source": {"type": "user", "userId": "U123"},
        "message": {"type": "text", "text": "verify 123456"},
    }
    for adapter in (customer, admin):
        adapter._client = MagicMock()
        adapter._client.reply = AsyncMock()

    await customer._dispatch_event(event)
    await admin._dispatch_event(event)

    assert redeem.await_args_list[0].kwargs["platform"] == "line"
    assert redeem.await_args_list[1].kwargs["platform"] == "line_admin"


def test_plugin_yaml_declares_admin_env_as_optional():
    plugin_yaml = Path(__file__).resolve().parents[2] / "plugins/platforms/line/plugin.yaml"
    manifest = yaml.safe_load(plugin_yaml.read_text())
    required = {entry["name"] for entry in manifest["requires_env"]}
    optional = {entry["name"] for entry in manifest["optional_env"]}

    admin_names = {
        "LINE_ADMIN_CHANNEL_ACCESS_TOKEN",
        "LINE_ADMIN_CHANNEL_SECRET",
        "LINE_ADMIN_CHANNEL_ALLOWED_USERS",
        "LINE_ADMIN_CHANNEL_PORT",
        "LINE_ADMIN_CHANNEL_HOME",
    }

    assert admin_names <= optional
    assert not (admin_names & required)
