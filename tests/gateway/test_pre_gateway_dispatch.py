"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
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
async def test_hook_skip_short_circuits_dispatch(monkeypatch):
    """A plugin returning {'action': 'skip'} drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_rewrite_replaces_event_text(monkeypatch):
    """A plugin returning {'action': 'rewrite', 'text': ...} mutates event.text."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen_text = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        seen_text["value"] = event.text
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    assert seen_text.get("value") == "REWRITTEN"


@pytest.mark.asyncio
async def test_hook_allow_falls_through_to_auth(monkeypatch):
    """A plugin returning {'action': 'allow'} continues to normal dispatch."""
    _clear_auth_env(monkeypatch)
    # No allowed users set → auth fails → pairing flow triggers.
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "allow"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    # auth chain ran → pairing code was generated
    assert result is None
    runner.pairing_store.generate_code.assert_called_once()


@pytest.mark.asyncio
async def test_hook_exception_does_not_break_dispatch(monkeypatch):
    """A raising plugin hook does not break the gateway."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = None

    # Should not raise; falls through to auth chain.
    result = await runner._handle_message(_make_event("hi"))
    assert result is None


@pytest.mark.asyncio
async def test_internal_events_run_transport_hook_but_bypass_user_hook(monkeypatch):
    """Internal events run safety hooks while skipping user interception."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = []

    def _fake_hook(name, **kwargs):
        called.append((name, kwargs.get("fail_closed")))
        return [{"action": "skip"}] if name == "pre_gateway_dispatch" else []

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    await runner._handle_message(event)
    assert called == [("pre_gateway_transport", True)]

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
async def test_real_transport_callback_failure_blocks_internal_telegram(monkeypatch):
    """PluginManager callback isolation still surfaces strict transport failure."""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()

    def _boom(**kwargs):
        raise RuntimeError("transport safety failed")

    manager._hooks["pre_gateway_transport"] = [_boom]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = AsyncMock(return_value="must-not-run")
    event = _make_event("internal task", platform=Platform.TELEGRAM)
    event.internal = True

    assert await runner._handle_message(event) is None
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_telegram_event_can_be_blocked_by_transport_safety(monkeypatch):
    """Unsafe internal Telegram agent turns fail closed before dispatch."""
    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_transport":
            return [{"action": "skip", "reason": "unsafe-display"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)
    runner, _adapter = _make_runner(Platform.TELEGRAM)
    runner._handle_message_with_agent = AsyncMock(return_value="must-not-run")
    event = _make_event("internal task", platform=Platform.TELEGRAM)
    event.internal = True

    assert await runner._handle_message(event) is None
    runner._handle_message_with_agent.assert_not_awaited()
