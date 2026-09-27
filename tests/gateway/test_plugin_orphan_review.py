"""Orphan checkpoint filtering must preserve and drain the parked user turn."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from tests.gateway.test_plugin_message_injection import _RoutingAdapter
from tests.gateway.test_pre_gateway_dispatch import _make_event, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("orphans,action,expected", [
    (["stale"], "skip", ["fresh user input"]),
    (["stale", "stale-2"], "skip", ["fresh user input"]),
    (["stale", "stale-2", "native"], "skip", ["native", "fresh user input"]),
    (["checkpoint"], "allow", ["checkpoint", "fresh user input"]),
    (["checkpoint"], "rewrite", ["rewritten", "fresh user input"]),
    (["native"], "skip", ["native", "fresh user input"]),
])
async def test_rescued_checkpoint_filter_preserves_fifo_drain(
    orphans, action, expected, tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _ = _make_runner(Platform.TELEGRAM)
    adapter = _RoutingAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    incoming = _make_event("fresh user input", Platform.TELEGRAM)
    key = runner._session_key_for_source(incoming.source)
    queued = []
    for text in orphans:
        event = _make_event(text, Platform.TELEGRAM)
        event.internal = True
        event.allow_gateway_control = False
        if text != "native":
            event.metadata = {"hermes_plugin_injection": True, "hermes_plugin_id": "pilot"}
        queued.append(event)
    runner._queued_events = {key: queued}
    runner._is_user_authorized_for_source = lambda source: True
    runner._admit_bot_message_for_source = lambda source: True
    runner._is_telegram_topic_root_lobby = lambda source: False
    runner._external_drain_active = False
    runner._claim_active_session_slot = lambda *args: (None, None)
    runner._persist_active_agents = lambda: None
    runner._run_post_turn_hooks = AsyncMock()
    consumed, seen = [], []

    async def hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            event = kwargs["event"]
            seen.append(event.text)
            if event.internal:
                return [{"action": action, "text": "rewritten"}]
        return []

    async def capture(event, *args):
        consumed.append(event.text)
        return None

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", hook)
    runner._handle_message_with_agent = capture
    adapter.set_message_handler(runner._handle_message)
    # Exercise the real adapter handoff, rather than manually popping parked events.
    adapter._spawn_drain_task(incoming, key)
    for _ in range(10):
        tasks = [task for task in adapter._background_tasks if not task.done()]
        if not tasks:
            break
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    assert consumed == expected
    assert runner._queue_depth(key, adapter=adapter) == 0
    assert key not in adapter._active_sessions
    assert "native" not in seen
    for text in orphans:
        if text != "native":
            assert seen.count(text) == 1
