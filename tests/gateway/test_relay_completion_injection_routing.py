"""Regression: completion injection must resolve the RELAY adapter for
fronted platforms.

Staging incident 2026-08-09 (second occurrence): async delegation batch
completed, watcher drained the event, and delivery silently vanished — no
injection log, no terminal-drop warning, no retry. Root cause:
``_inject_watch_notification`` resolves its adapter with a literal
``p.value == platform_name`` scan of ``self.adapters``. A relay-fronted
gateway registers ONE adapter under ``Platform.RELAY`` fronting N logical
platforms, so the literal scan misses "slack" and the injection returns
``None`` ("no gateway route") — the completion is dropped without a trace.

run.py already documents the trap and ships the alias-aware resolver
(``resolve_delivery_transport``); the handoff path uses it. Contract under
test: the injection path (and by extension every completion delivered on a
relay-plane deployment) must resolve through the shared resolver.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class _RelayAdapter:
    """Stub relay adapter fronting slack."""

    name = "relay"

    def __init__(self):
        self.handled = []
        self.handle_message = AsyncMock(side_effect=self.handled.append)

    def fronts_platform(self, platform):
        return platform == Platform.SLACK


def _runner_with_relay(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.RELAY: adapter}
    runner.config = SimpleNamespace(platforms={})
    return runner


def _slack_async_event():
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_relay_route",
        "session_key": "agent:main:slack:dm:D0BJTDCSR7C:1786298425.877239",
        "platform": "slack",
        "chat_type": "dm",
        "chat_id": "D0BJTDCSR7C",
        "status": "completed",
        "is_batch": True,
        "results": [{"goal": "g1", "status": "completed", "summary": "done"}],
    }


@pytest.mark.asyncio
async def test_injection_resolves_relay_adapter_for_fronted_platform():
    """A gateway whose only adapter is the relay (fronting slack) must
    deliver a slack-routed completion through it — not drop it as
    'no gateway route'."""
    adapter = _RelayAdapter()
    runner = _runner_with_relay(adapter)

    result = await runner._inject_watch_notification(
        "[delegation completed]", _slack_async_event()
    )

    assert result is True, (
        f"injection returned {result!r} on a relay-fronted gateway — the "
        "completion was dropped exactly as in the 2026-08-09 staging "
        "incident (literal adapter scan misses Platform.RELAY)"
    )
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_injection_still_none_when_platform_not_fronted():
    """Control: a platform the relay does NOT front stays undeliverable
    (returns None) — the resolver must not let relay hijack unrelated
    targets."""
    adapter = _RelayAdapter()  # fronts slack only
    runner = _runner_with_relay(adapter)
    evt = _slack_async_event()
    evt["session_key"] = "agent:main:discord:dm:123:456"
    evt["platform"] = "discord"

    result = await runner._inject_watch_notification("[x]", evt)
    assert result is None
    assert adapter.handle_message.await_count == 0
