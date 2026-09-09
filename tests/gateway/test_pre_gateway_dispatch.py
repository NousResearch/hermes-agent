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

    async def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook_async", _fake_hook)

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

    async def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_store"] = kwargs.get("session_store", "MISSING")
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook_async", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    del runner.session_store

    result = await runner._handle_message(_make_event("hi"))
    assert result is None
    # Hook actually fired (skip short-circuited before auth) with a None store.
    assert seen == {"session_store": None}
    adapter.send.assert_not_awaited()

@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['skip', 'rewrite', 'allow'])
async def test_discovered_async_gate_preserves_loop_and_observers(tmp_path, monkeypatch, action):
    import asyncio
    from hermes_cli import plugins, observability
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    plugin = tmp_path / 'plugins' / 'async-gate'
    plugin.mkdir(parents=True)
    (tmp_path / 'config.yaml').write_text('plugins:\n  enabled: [async-gate]\n')
    (plugin / 'plugin.yaml').write_text('name: async-gate\nversion: 0.1.0\n')
    (plugin / '__init__.py').write_text('''import asyncio
async def gate(event, gateway):
    assert asyncio.get_running_loop() is gateway.test_loop
    await asyncio.sleep(0)
    gateway.test_calls.append('plugin')
    return {'action': gateway.test_action, 'text': 'rewritten'}
def register(ctx):
    ctx.register_hook('pre_gateway_dispatch', gate)
''')
    plugins._reset_plugin_managers_for_tests()
    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.test_loop = asyncio.get_running_loop()
    runner.test_calls = []
    runner.test_action = action
    monkeypatch.setattr(observability, 'observe_lifecycle', lambda *a, **kw: runner.test_calls.append('observer'))
    event = _make_event()
    try:
        if action == 'skip':
            _clear_auth_env(monkeypatch)
            assert await runner._handle_message(event) is None
            adapter.send.assert_not_awaited()
            runner.pairing_store.generate_code.assert_not_called()
        else:
            result = await runner._hm_pre_gateway_dispatch_hook(event, event.source)
            assert result.text == ('rewritten' if action == 'rewrite' else event.text)
        assert runner.test_calls == ['observer', 'plugin']
    finally:
        plugins._reset_plugin_managers_for_tests()

@pytest.mark.asyncio
async def test_async_gate_failure_isolation_and_cancellation():
    import asyncio
    from hermes_cli.plugins import PluginManager
    manager = PluginManager()
    calls = []
    def sync(event):
        calls.append('sync')
        return event
    async def broken(event):
        calls.append('broken')
        raise ValueError('failed plugin')
    async def good(event):
        calls.append('async')
        return event
    manager._hooks['pre_gateway_dispatch'] = [sync, broken, good]
    assert await manager.invoke_hook_async('pre_gateway_dispatch', event='ok', additive=True) == ['ok', 'ok']
    assert calls == ['sync', 'broken', 'async']
    async def cancelled(event):
        raise asyncio.CancelledError()
    manager._hooks['pre_gateway_dispatch'] = [cancelled, good]
    with pytest.raises(asyncio.CancelledError):
        await manager.invoke_hook_async('pre_gateway_dispatch', event='ok')
