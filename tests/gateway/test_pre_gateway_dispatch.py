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

    async def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen["session_store"] = kwargs.get("session_store", "MISSING")
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    # The inbound path awaits the hook, so the seam is the async entry point.
    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    del runner.session_store

    result = await runner._handle_message(_make_event("hi"))
    assert result is None
    # Hook actually fired (skip short-circuited before auth) with a None store.
    assert seen == {"session_store": None}
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_path_runs_hook_and_honours_skip(monkeypatch):
    """Messages for an already-running session go through the busy handler — and must see the
    plugin hook there too.

    Regression: only the cold path (``_hm_admit_event``) ran ``pre_gateway_dispatch``. A message
    arriving while *its own session* was busy went to ``_handle_active_session_busy_message``,
    which skipped the hook: a plugin's decision was silently lost and the raw message was
    injected into the running turn (an approval mid-run reached a model that acted on it).
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen = []

    async def _hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            seen.append(kwargs["event"].text)
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    result = await runner._handle_active_session_busy_message(_make_event("hi"), "sess-key")

    assert seen == ["hi"]          # fired exactly once, for this message
    assert result is True          # "handled": dropped by the plugin, never dispatched
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_path_still_dispatches_when_hook_allows(monkeypatch):
    """No ``skip`` from the hook ⇒ the busy handler proceeds exactly as before."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = []

    async def _hook(name, **kwargs):
        called.append(name)
        return []

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)

    def _sentinel(source):
        raise RuntimeError("reached-authorization-gate")

    runner._is_user_authorized_for_source = _sentinel  # first check after the hook
    with pytest.raises(RuntimeError, match="reached-authorization-gate"):
        await runner._handle_active_session_busy_message(_make_event("hi"), "sess-key")
    assert called == ["pre_gateway_dispatch"]


@pytest.mark.asyncio
async def test_async_hook_callback_is_awaited_on_the_gateway_loop(monkeypatch):
    """An ``async def`` pre_gateway_dispatch callback is awaited on the gateway's own loop.

    Regression: the inbound path called the sync ``invoke_hook``, which (since #109196) runs an
    async callback on a helper thread while the calling loop blocks in ``done.wait()``. A
    callback that awaits anything scheduled on the gateway loop could never complete, and every
    message stalled the loop for the callback's whole duration. Here the callback waits for a
    sibling task on the same loop to release it; that is only possible if the hook is awaited
    in place.
    """
    import asyncio

    _clear_auth_env(monkeypatch)
    gate = asyncio.Event()

    async def _hook(name, **kwargs):
        assert name == "pre_gateway_dispatch"
        await gate.wait()
        return [{"action": "skip", "reason": "gated"}]

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)

    async def _release():
        await asyncio.sleep(0)
        gate.set()

    runner, adapter = _make_runner(Platform.WHATSAPP)
    asyncio.create_task(_release())
    result = await asyncio.wait_for(runner._handle_message(_make_event("hi")), timeout=5)
    assert result is None
    adapter.send.assert_not_awaited()
