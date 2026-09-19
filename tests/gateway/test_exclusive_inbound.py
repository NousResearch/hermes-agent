from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.exclusive_inbound import configure_exclusive_inbound
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, Platform, SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class Adapter:
    platform = Platform.WHATSAPP

    def __init__(self, claim):
        self.config = SimpleNamespace(extra={"exclusive_inbound": claim})
        self.handler = None

    def set_exclusive_inbound_handler(self, handler):
        self.handler = handler


class FallthroughAdapter(BasePlatformAdapter):
    async def connect(self): pass
    async def disconnect(self): pass
    async def send(self, *args, **kwargs): pass
    async def get_chat_info(self, *args, **kwargs): pass


def event(chat_id="claimed", user_id="sender"):
    return MessageEvent(text="message", source=SessionSource(
        platform=Platform.WHATSAPP, chat_id=chat_id, chat_type="group", user_id=user_id), message_id="m1")


def claim(**extra):
    return {"chat_id": "claimed", "handler": "capture", **extra}


@pytest.fixture
def scoped(monkeypatch):
    @asynccontextmanager
    async def scope(_home):
        yield
    monkeypatch.setattr("gateway.run._async_profile_runtime_scope", scope)


def runner(authorized=True):
    result = SimpleNamespace()
    result._is_user_authorized_for_source = MagicMock(return_value=authorized)
    return result


@pytest.mark.asyncio
async def test_base_adapter_without_claim_falls_through():
    adapter = FallthroughAdapter(SimpleNamespace(), Platform.WHATSAPP)
    assert await adapter.dispatch_exclusive_inbound(event()) is False


@pytest.mark.asyncio
async def test_unclaimed_event_falls_through(scoped, monkeypatch):
    adapter = Adapter(claim())
    r = runner()
    get_manager = MagicMock()
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", get_manager)
    configure_exclusive_inbound(r, adapter)
    assert await adapter.handler(event("ordinary")) is False
    r._is_user_authorized_for_source.assert_not_called()
    get_manager.assert_not_called()


@pytest.mark.asyncio
async def test_claimed_event_requires_one_awaited_durable_acceptance(scoped, monkeypatch):
    adapter, manager = Adapter(claim()), PluginManager()
    context = PluginContext(PluginManifest(name="capture"), manager)
    accepted = []

    async def accept(inbound):
        accepted.append(inbound.message_id)
        return True

    context.register_exclusive_inbound_handler("capture", accept)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    configure_exclusive_inbound(runner(), adapter)
    assert await adapter.handler(event()) is True
    assert accepted == ["m1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["absent", "collision", "raises", "rejects", "sync"])
async def test_claimed_event_fails_closed_for_plugin_failures(scoped, monkeypatch, mode):
    adapter, manager = Adapter(claim()), PluginManager()
    context = PluginContext(PluginManifest(name="capture"), manager)

    async def accept(_event): return True
    async def raises(_event): raise RuntimeError("boom")
    async def rejects(_event): return False

    if mode == "collision":
        context.register_exclusive_inbound_handler("capture", accept)
        context.register_exclusive_inbound_handler("capture", accept)
    elif mode == "raises": context.register_exclusive_inbound_handler("capture", raises)
    elif mode == "rejects": context.register_exclusive_inbound_handler("capture", rejects)
    elif mode == "sync": context.register_exclusive_inbound_handler("capture", lambda _event: True)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    configure_exclusive_inbound(runner(), adapter)
    assert await adapter.handler(event()) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["authorization", "plugin_discovery"])
async def test_claimed_event_fails_closed_when_admission_setup_raises(scoped, monkeypatch, failure):
    adapter = Adapter(claim())
    r = runner()
    if failure == "authorization":
        r._is_user_authorized_for_source.side_effect = RuntimeError("authorization unavailable")
    else:
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", MagicMock(side_effect=RuntimeError("plugins unavailable")))
    configure_exclusive_inbound(r, adapter)
    assert await adapter.handler(event()) is True


@pytest.mark.asyncio
async def test_authorized_sender_boundary_is_optional_and_additive(scoped, monkeypatch):
    adapter, manager = Adapter(claim(allowed_senders=["sender"])), PluginManager()
    context = PluginContext(PluginManifest(name="capture"), manager)
    accepted = []

    async def accept(_event):
        accepted.append(True)
        return True

    context.register_exclusive_inbound_handler("capture", accept)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    configure_exclusive_inbound(runner(), adapter)
    assert await adapter.handler(event()) is True
    assert await adapter.handler(event(user_id="other")) is True
    assert accepted == [True]


@pytest.mark.parametrize("bad", [[], {}, {"chat_id": "claimed"}, {"handler": "capture"}, claim(allowed_senders=[])])
def test_invalid_claim_fails_at_wiring(bad):
    with pytest.raises(ValueError):
        configure_exclusive_inbound(runner(), Adapter(bad))
