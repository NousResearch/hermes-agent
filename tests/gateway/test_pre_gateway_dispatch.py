"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0

@pytest.mark.asyncio
async def test_hook_fires_without_session_store_attribute(monkeypatch):
    """A runner missing session_store still delivers the event to plugins.

    Regression: the hook kwargs read ``self.session_store`` directly, so a
    partially-initialized runner raised AttributeError inside the dispatch
    try-block — the hook never fired, and every message logged
    "pre_gateway_dispatch invocation failed: 'GatewayRunner' object has no
    attribute 'session_store'". Plugins must receive the event (with
    session_store=None) instead.
    """
    _clear_auth_env(monkeypatch)

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_store"] = kwargs.get("session_store", "MISSING")
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    del runner.session_store

    result = await runner._handle_message(_make_event("hi"))
    assert result is None
    # Hook actually fired (skip short-circuited before auth) with a None store.
    assert seen == {"session_store": None}
    adapter.send.assert_not_awaited()

@pytest.mark.asyncio
async def test_authorization_is_explicit_and_profile_local(tmp_path, monkeypatch):
    """Real plugin delivery and allowlists remain isolated across A -> B -> A."""
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    _clear_auth_env(monkeypatch)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(plugins, "_plugin_manager", None)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    homes = [tmp_path / "a", tmp_path / "b"]
    verdict = {"action": "authorize"}

    def authorize(**kwargs):
        if verdict["action"] == "error":
            raise RuntimeError("identity lookup failed")
        if verdict["action"] == "authorize":
            kwargs["event"].source.user_id = "resolved-identity"
        return dict(verdict)

    for home in homes:
        home.mkdir()
        (home / ".env").write_text("WHATSAPP_ALLOWED_USERS=someone-else\n")
        with _profile_runtime_scope(home):
            manager = plugins.get_plugin_manager()
            assert isinstance(manager, PluginManager)
            manager._discovered = True
            if home == homes[0]:
                PluginContext(PluginManifest(name="identity", source="user"), manager).register_hook(
                    "pre_gateway_dispatch", authorize,
                )

    runner, adapter = _make_runner(Platform.WHATSAPP)
    for action, admitted in [("authorize", True), ("allow", False), ("rewrite", False),
                             ("skip", False), ("error", False), ("unknown", False)]:
        verdict["action"] = action
        for home in [homes[0], homes[1], homes[0]]:
            event = _make_event()
            with _profile_runtime_scope(home):
                result = await runner._hm_admit_event(event)
            expected = admitted and home == homes[0]
            assert (result is not None) == expected, (action, home.name)
            if expected:
                assert result == (event, event.source, False)
                assert event.source.user_id == "resolved-identity"
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorize_keeps_ingress_guards_and_never_sticks(monkeypatch):
    """A grant cannot admit a rejected route/bot or authorize a later dispatch."""
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "someone-else")
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    seen = []
    verdict = {"action": "authorize"}

    def authorize(**kwargs):
        seen.append(kwargs["event"])
        return dict(verdict)

    PluginContext(PluginManifest(name="identity", source="user"), manager).register_hook(
        "pre_gateway_dispatch", authorize,
    )
    runner, _ = _make_runner(Platform.WHATSAPP)
    internal = _make_event()
    internal.internal = True
    assert (await runner._hm_admit_event(internal))[2] is True
    rejected = _make_event()
    rejected.source.profile_route_rejected = True
    assert await runner._hm_admit_event(rejected) is None
    assert not seen

    runner._admit_bot_message_for_source = lambda source: not source.is_bot
    bot = _make_event()
    bot.source.is_bot = True
    assert await runner._hm_admit_event(bot) is None
    event = _make_event()
    assert await runner._hm_admit_event(event) is not None
    verdict["action"] = "allow"
    assert await runner._hm_admit_event(event) is None
