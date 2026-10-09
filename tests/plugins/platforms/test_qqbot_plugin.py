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


def test_unconfigured_plugin_does_not_enable(monkeypatch):
    _clear_qq(monkeypatch)
    from plugins.platforms.qqbot.register import env_enablement

    assert env_enablement() is None


def test_group_allowlist_and_home_seed(monkeypatch):
    _clear_qq(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID", "app")
    monkeypatch.setenv("QQ_CLIENT_SECRET", "sec")
    monkeypatch.setenv("QQ_GROUP_ALLOWED_USERS", "group-1, group-2")
    monkeypatch.setenv("QQBOT_HOME_CHANNEL", "home-chat")
    monkeypatch.setenv("QQBOT_HOME_CHANNEL_NAME", "Lounge")
    monkeypatch.setenv("QQBOT_HOME_CHANNEL_THREAD_ID", "thread-9")
    monkeypatch.setenv("QQ_HOME_CHANNEL", "ignored")
    from plugins.platforms.qqbot.register import env_enablement

    seed = env_enablement()

    assert seed["app_id"] == "app"
    assert seed["group_allow_from"] == "group-1, group-2"
    assert seed["home_channel"] == {
        "chat_id": "home-chat", "name": "Lounge", "thread_id": "thread-9",
    }


def test_qq_prefixed_home_channel_still_seeds(monkeypatch):
    _clear_qq(monkeypatch)
    monkeypatch.setenv("QQ_APP_ID", "app")
    monkeypatch.setenv("QQ_CLIENT_SECRET", "sec")
    monkeypatch.setenv("QQ_HOME_CHANNEL", "renamed-chat")
    monkeypatch.setenv("QQ_HOME_CHANNEL_THREAD_ID", "thread-2")
    from plugins.platforms.qqbot.register import env_enablement

    seed = env_enablement()

    assert seed["home_channel"] == {
        "chat_id": "renamed-chat", "name": "Home", "thread_id": "thread-2",
    }


def test_qqbot_platform_hint_comes_from_the_builtin_table():
    from agent.prompt_builder import PLATFORM_HINTS
    from agent.system_prompt import _default_platform_hint

    assert "MEDIA:" in PLATFORM_HINTS["qqbot"]
    assert _default_platform_hint("qqbot") == PLATFORM_HINTS["qqbot"]


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


def test_sdk_ws_thread_keeps_caller_context():
    """The SDK thread must see the profile context captured at connect()."""
    import contextvars
    import threading

    from qqbot_agent_sdk import WSCallbacks

    from plugins.platforms.qqbot.sdk_bridge import scoped_websocket_class

    profile = contextvars.ContextVar("qq_profile_scope")
    profile.set("profile-a")
    seen = {}
    started = threading.Event()
    cls = scoped_websocket_class()

    class Probe(cls):
        def _run_ws_thread(self, gateway_url):
            seen["profile"] = profile.get(None)
            seen["url"] = gateway_url
            started.set()

    def _noop(*_args, **_kwargs):
        return None

    async def _async_noop(*_args, **_kwargs):
        return None

    ws = Probe(
        callbacks=WSCallbacks(
            on_message_event=_async_noop,
            on_connected=_noop,
            on_disconnected=_noop,
            on_fatal_error=_noop,
            get_token=lambda: "t",
            get_session=lambda: (None, None),
            set_session=_noop,
            set_heartbeat_interval=_noop,
            clear_token=_noop,
            fail_pending=_noop,
            get_gateway_url=lambda: "wss://example",
        ),
        log_tag="t",
    )
    ws.start("wss://example", object())
    assert started.wait(2)
    assert seen == {"profile": "profile-a", "url": "wss://example"}
