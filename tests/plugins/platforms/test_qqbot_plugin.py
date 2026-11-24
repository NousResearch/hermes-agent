"""QQ Bot is a bundled platform plugin: env seeding and cron delivery, no live gateway."""

import pytest


def _clear_qq(monkeypatch):
    for key in (
        "QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS", "QQ_GROUP_ALLOWED_USERS",
        "QQ_ALLOW_ALL_USERS", "QQBOT_HOME_CHANNEL", "QQBOT_HOME_CHANNEL_NAME",
        "QQBOT_HOME_CHANNEL_THREAD_ID", "QQ_HOME_CHANNEL", "QQ_HOME_CHANNEL_NAME",
        "QQ_HOME_CHANNEL_THREAD_ID", "QQ_MARKDOWN_SUPPORT", "QQ_PORTAL_HOST",
    ):
        monkeypatch.delenv(key, raising=False)


def test_register_shape():
    """The plugin registers the same identity/env/routing the core table carried."""
    from gateway.platform_registry import PlatformEntry
    from plugins.platforms.qqbot import register
    from plugins.platforms.qqbot.adapter import QQAdapter, check_qq_requirements
    from plugins.platforms.qqbot.register import setup_qqbot, standalone_send

    captured = {}

    class _Ctx:
        def register_platform(self, **kwargs):
            captured.update(kwargs)

    register(_Ctx())
    entry = PlatformEntry(**captured)
    assert entry.name == "qqbot" and entry.label == "QQ Bot" and entry.emoji == "🐧"
    assert entry.adapter_factory is QQAdapter and entry.check_fn is check_qq_requirements
    assert entry.required_env == ["QQ_APP_ID", "QQ_CLIENT_SECRET"]
    assert entry.allowed_users_env == "QQ_ALLOWED_USERS" and entry.allow_all_env == "QQ_ALLOW_ALL_USERS"
    assert entry.cron_deliver_env_var == "QQBOT_HOME_CHANNEL"
    assert entry.setup_fn is setup_qqbot and entry.standalone_sender_fn is standalone_send
    # send_message never chunked QQ text before the move; the REST sender truncates itself.
    assert entry.max_message_length == 0


def test_is_connected_needs_both_credentials(monkeypatch):
    _clear_qq(monkeypatch)
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "get_env_value", lambda name: None)
    from gateway.config import PlatformConfig
    from plugins.platforms.qqbot.register import is_connected

    assert is_connected(PlatformConfig(enabled=True, extra={"app_id": "a"})) is False
    assert is_connected(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "s"})) is True
    env = {"QQ_APP_ID": "a", "QQ_CLIENT_SECRET": "s"}
    monkeypatch.setattr(gateway_mod, "get_env_value", env.get)
    assert is_connected(PlatformConfig(enabled=True)) is True


def test_qqbot_platform_hint_comes_from_its_platform_entry():
    from agent.system_prompt import _default_platform_hint
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    hint = platform_registry.get("qqbot").platform_hint
    assert "MEDIA:" in hint
    assert _default_platform_hint("qqbot") == hint


def test_qqbot_is_a_builtin_deliver_target():
    """Webhook and cron treat qqbot as a shipped platform before any plugin load."""
    from cron.scheduler_delivery import (
        _is_known_delivery_platform,
        _resolve_home_env_var,
        _home_env_lookup,
    )
    from gateway.platforms.webhook import _is_known_platform

    assert _is_known_platform("qqbot") is True
    assert _is_known_delivery_platform("qqbot") is True
    assert _resolve_home_env_var("qqbot") == "QQBOT_HOME_CHANNEL"


def test_qq_prefixed_home_env_still_resolves_for_cron(monkeypatch):
    monkeypatch.delenv("QQBOT_HOME_CHANNEL", raising=False)
    monkeypatch.setenv("QQ_HOME_CHANNEL", "renamed-chat")
    from cron.scheduler_delivery import _home_env_lookup

    assert _home_env_lookup("QQBOT_HOME_CHANNEL") == "renamed-chat"


def test_cron_deliver_qqbot_without_a_live_gateway():
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    from cron.scheduler_delivery import _is_known_delivery_platform, _resolve_home_env_var

    assert _is_known_delivery_platform("qqbot") is True
    assert _resolve_home_env_var("qqbot") == "QQBOT_HOME_CHANNEL"


@pytest.mark.asyncio
async def test_standalone_sender_reports_missing_credentials(monkeypatch):
    _clear_qq(monkeypatch)
    from gateway.config import PlatformConfig
    from plugins.platforms.qqbot.register import standalone_send

    result = await standalone_send(PlatformConfig(enabled=True, extra={}), "chat-1", "hi")
    assert "QQ_APP_ID" in result["error"]


@pytest.mark.asyncio
async def test_qq_c2c_attachment_metadata_stays_out_of_info(caplog, monkeypatch):
    """A QQ DM's attachment URL/filename may contain private user data (#46006)."""
    import logging
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.qqbot.adapter import QQAdapter

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={}))
    ingest = AsyncMock()
    monkeypatch.setattr(adapter, "_ingest", ingest)
    attachment = {
        "content_type": "application/pdf",
        "url": "https://files.example.test/attachment?signature=private-sentinel",
        "filename": "Private medical report for Alice.pdf",
    }
    caplog.set_level(logging.DEBUG, logger="plugins.platforms.qqbot.adapter")
    await adapter._handle_c2c_message(
        {"attachments": [attachment]}, "msg-1", "hello", {"user_openid": "qq-user-1"}, "2026-11-12T18:00:00Z",
    )
    ingest.assert_awaited_once()
    messages = [
        r.getMessage() for r in caplog.records
        if r.name == "plugins.platforms.qqbot.adapter" and r.levelno >= logging.INFO
    ]
    assert any("attachment[0]" in line for line in messages), messages
    assert all(
        attachment["filename"] not in line and "private-sentinel" not in line
        for line in messages
    ), messages
    assert any(
        attachment["filename"] in r.getMessage()
        for r in caplog.records
        if r.name == "plugins.platforms.qqbot.adapter" and r.levelno == logging.DEBUG
    )


@pytest.mark.asyncio
async def test_qq_voice_stt_failure_keeps_the_attachment_url_out_of_info(caplog, monkeypatch):
    """The STT-failure warning must not carry the inbound voice URL (#46006)."""
    import logging
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.qqbot.adapter import QQAdapter

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={}))
    monkeypatch.setattr(adapter, "_stt_voice_attachment", AsyncMock(return_value=None))
    url = "https://files.example.test/v?sig=private-sentinel"
    caplog.set_level(logging.DEBUG, logger="plugins.platforms.qqbot.adapter")
    result = await adapter._process_attachments([{"content_type": "voice", "url": url, "filename": "a.silk"}])
    assert result["voice_transcripts"]
    info = [
        r.getMessage() for r in caplog.records
        if r.name == "plugins.platforms.qqbot.adapter" and r.levelno >= logging.INFO
    ]
    assert any("Voice STT failed" in line for line in info), info
    assert all("private-sentinel" not in line and "files.example.test" not in line for line in info), info


@pytest.mark.asyncio
async def test_qq_voice_download_http_error_keeps_the_url_out_of_warning(caplog, monkeypatch):
    """httpx's HTTPStatusError text carries the signed download URL (#46006)."""
    import logging

    import httpx

    from gateway.config import PlatformConfig
    from plugins.platforms.qqbot.adapter import QQAdapter

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(403, request=request)))
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)
    caplog.set_level(logging.DEBUG, logger="plugins.platforms.qqbot.adapter")
    assert await adapter._stt_voice_attachment("https://files.example.test/v?sig=private-sentinel", "voice", "a.silk") is None
    info = [
        r.getMessage() for r in caplog.records
        if r.name == "plugins.platforms.qqbot.adapter" and r.levelno >= logging.INFO
    ]
    assert any("status=403" in line for line in info), info
    assert all("private-sentinel" not in line for line in info), info
