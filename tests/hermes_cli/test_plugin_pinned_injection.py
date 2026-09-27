"""Optional gateway ownership guarantees must never disappear on another host."""
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.parametrize("options", [
    {"expected_session_id": "parent"}, {"on_delivery": lambda accepted: None},
])
def test_cli_rejects_gateway_guarantees_without_queueing(options):
    manager = PluginManager()
    manager._cli_ref = SimpleNamespace(
        _agent_running=True, _pending_input=SimpleQueue(), _interrupt_queue=SimpleQueue())
    ctx = PluginContext(PluginManifest(name="test-plugin", key="test-plugin"), manager)
    assert not ctx.inject_message("checkpoint", session_key="route", **options)
    assert manager._cli_ref._interrupt_queue.empty()
    assert manager._cli_ref._pending_input.empty()


def test_pinned_request_never_falls_through_to_unpinned_tui(monkeypatch):
    manager = PluginManager()
    tui = Mock(return_value=True)
    manager.set_tui_message_injector(object(), tui)
    ctx = PluginContext(PluginManifest(name="test-plugin", key="test-plugin"), manager)
    monkeypatch.setattr(ctx, "_gateway_injection_allowed", lambda: True)
    assert not ctx.inject_message("checkpoint", session_key="route", expected_session_id="parent")
    tui.assert_not_called()


@pytest.mark.parametrize("options", [
    {"expected_session_id": ""}, {"expected_session_id": 12}, {"on_delivery": "not callable"},
])
def test_invalid_guarantees_do_not_schedule_gateway_work(monkeypatch, options):
    manager = PluginManager()
    gateway = Mock(return_value=True)
    manager.set_gateway_message_injector(object(), gateway)
    ctx = PluginContext(PluginManifest(name="test-plugin", key="test-plugin"), manager)
    monkeypatch.setattr(ctx, "_gateway_injection_allowed", lambda: True)
    assert not ctx.inject_message("checkpoint", session_key="route", **options)
    gateway.assert_not_called()
