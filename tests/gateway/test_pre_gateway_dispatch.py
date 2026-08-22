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
async def test_hook_receives_agent_busy_when_busy(monkeypatch):
    """When the session holds a running agent, agent_busy is True.

    Regression guard: if ``session_is_busy`` reads ``_running_agents`` with a
    plain attribute access (no getattr fallback), bare-runner tests built via
    ``object.__new__`` — which never set the attribute — raise and the hook
    never fires. Plugins that depend on ``agent_busy`` to throttle or
    reject duplicate sends silently stop working.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
            return [dict(action="allow")]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    _busy_key = runner._session_key_for_source(event.source)
    _state = MagicMock()
    _state.turn.agent = MagicMock()
    runner._running_agents[_busy_key] = MagicMock()  # fake busy agent slot

    # busy path: _handle_message may return None (interrupt/queue short-circuit)
    # since an agent is already running — this test only verifies the hook
    # received the correct agent_busy value, not the dispatch outcome.
    await runner._handle_message(event)
    assert seen.get("agent_busy") is True


@pytest.mark.asyncio
async def test_hook_receives_session_key(monkeypatch):
    """The pre_gateway_dispatch hook receives a session_key kwarg matching
    _session_key_for_source(source), so plugins can correlate the inbound
    source with the slot that session_is_busy() will inspect."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_key"] = kwargs.get("session_key", "MISSING")
            return [dict(action="allow")]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    expected_key = runner._session_key_for_source(event.source)
    await runner._handle_message(event)
    assert seen.get("session_key") == expected_key


@pytest.mark.asyncio
async def test_hook_receives_agent_busy_when_idle(monkeypatch):
    """When the session is idle, agent_busy is False."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["agent_busy"] = kwargs.get("agent_busy", "MISSING")
            return [dict(action="allow")]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("hi"))
    assert result == "ok"
    assert seen.get("agent_busy") is False


@pytest.mark.asyncio
async def test_session_is_busy_public_api(monkeypatch):
    """GatewayRunner.session_is_busy(session_key) is a public predicate that
    mirrors _is_session_running without requiring private-state access."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    key = _make_event().source.chat_id

    # idle -> False
    assert runner.session_is_busy(key) is False

    # busy -> True
    runner._running_agents[key] = object()
    assert runner.session_is_busy(key) is True
