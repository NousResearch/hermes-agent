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
    runner._handle_message_with_agent = _capture

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


@pytest.mark.asyncio
async def test_rewritten_message_keeps_its_callback_provenance_and_is_admitted(monkeypatch):
    """A rewrite builds a new event. Under ``gateway.multiplex_profiles`` an event without its
    adapter callback's provenance is refused ``invalid_params`` (#139045: every message to a profile
    with a rewriting plugin failed), so the rewritten event must carry the callback's provenance and
    be admitted with the rewritten text."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import capture_provenance, native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    async def rewrite(name, **_kwargs):
        return [{"action": "rewrite", "text": "rewritten"}] if name == "pre_gateway_dispatch" else []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", rewrite)
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    # The multiplexed admission check that refused the reporter (thanks @unsupportedpastels).
    runner.config.multiplex_profiles = True
    runner.config._runtime_profile_homes = [("default", get_hermes_home())]
    register_transport_home(runner, None, get_hermes_home())
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    human = MessageEvent(text="original", source=source, message_id="h")

    with native_callback(runner, human, get_hermes_home()):
        expected = capture_provenance(runner, human)
        event = await runner._hm_pre_gateway_dispatch_hook(human, source)
        receipt = await authority.admit_native(event)

    payload = list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)[-1]["payload"]
    assert expected is not None and event is not human
    assert (payload["text"], payload["native_text_v1"].get("provenance")) == ("rewritten", expected)


def test_relay_binding_moves_to_a_text_rewrite_but_never_to_a_new_source():
    """The relay binding gets the same treatment as the native one, under the same guard: it follows
    a text-only rewrite, and is never carried to an event whose source object was swapped."""
    from copy import copy
    from dataclasses import replace

    from gateway.session_ingress_context import _relay_callback, carry_callback, relay_callback

    transport, event = object(), _make_event("a")
    with relay_callback(transport, event):
        rewritten = replace(event, text="b")
        carry_callback(event, rewritten)
        assert _relay_callback.get() == (transport, rewritten)
        carry_callback(rewritten, replace(rewritten, source=copy(rewritten.source), text="c"))
        assert _relay_callback.get()[1] is rewritten
    assert _relay_callback.get() is None
