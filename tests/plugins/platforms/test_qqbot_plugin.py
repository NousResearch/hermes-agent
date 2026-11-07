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


# ---------------------------------------------------------------------------
# Long cron output must reach QQ whole, not truncated at MAX_PLATFORM_OUTPUT
# ---------------------------------------------------------------------------

class _NonChunkingProbe:
    """Adapter without the flag — legacy behaviour (cron output gets truncated)."""


def test_qq_adapter_declares_native_chunking():
    """send() splits via truncate_message(), so cron delivery must not pre-truncate.

    gateway/delivery.py caps cron output at MAX_PLATFORM_OUTPUT (4000) for
    adapters that do not advertise ``splits_long_messages``.  QQAdapter.send()
    already chunks via truncate_message(MAX_MESSAGE_LENGTH) — without this flag
    the delivery router cut long reports (daily closes run ~10k chars) down to
    4k before the adapter ever saw them.
    """
    from plugins.platforms.qqbot.adapter import QQAdapter

    assert QQAdapter.splits_long_messages is True

    # The flag is only meaningful because send() really does chunk.
    chunks = QQAdapter.truncate_message("行情" * 3000, QQAdapter.MAX_MESSAGE_LENGTH)
    assert len(chunks) > 1
    assert all(len(c) <= QQAdapter.MAX_MESSAGE_LENGTH for c in chunks)


def test_cron_output_not_truncated_for_qq(monkeypatch, tmp_path):
    """A 9000-char cron payload survives delivery for QQ, but not for a legacy adapter."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    from gateway.delivery import MAX_PLATFORM_OUTPUT, DeliveryRouter
    from plugins.platforms.qqbot.adapter import QQAdapter

    # _cap_oversized_output only touches self._save_full_output → no router init needed.
    router = object.__new__(DeliveryRouter)
    payload = "x" * (MAX_PLATFORM_OUTPUT * 2 + 1000)

    assert router._cap_oversized_output(QQAdapter, payload, "job-long") == payload

    visible = router._cap_oversized_output(_NonChunkingProbe(), payload, "job-long")
    assert len(visible) < len(payload)
    assert "full output saved to" in visible

