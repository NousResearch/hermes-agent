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

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


class _RelayAdapter:
    """Stub relay adapter fronting slack."""

    name = "relay"

    def __init__(self):
        self.handled = []
        self.handle_message = AsyncMock(side_effect=self._admit)

    def _admit(self, event):
        self.handled.append(event)
        event._gateway_accepted = True

    def fronts_platform(self, platform):
        return platform == Platform.SLACK


def _runner_with_relay(adapter, tmp_path):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.RELAY: adapter}
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    source = SessionSource(
        platform=Platform.SLACK, chat_type="dm", chat_id="D0BJTDCSR7C",
        thread_id="1786298425.877239",
    )
    return runner, runner.session_store.get_or_create_session(source)


def _slack_async_event(parent_session_id):
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_relay_route",
        "session_key": "agent:main:slack:dm:D0BJTDCSR7C:1786298425.877239",
        "parent_session_id": parent_session_id,
        "platform": "slack",
        "chat_type": "dm",
        "chat_id": "D0BJTDCSR7C",
        "thread_id": "1786298425.877239",
        "status": "completed",
        "is_batch": True,
        "results": [{"goal": "g1", "status": "completed", "summary": "done"}],
    }


@pytest.mark.asyncio
async def test_injection_resolves_relay_adapter_for_fronted_platform(tmp_path):
    """A gateway whose only adapter is the relay (fronting slack) must
    deliver a slack-routed completion through it — not drop it as
    'no gateway route'."""
    adapter = _RelayAdapter()
    runner, entry = _runner_with_relay(adapter, tmp_path)

    result = await runner._inject_watch_notification(
        "[delegation completed]", _slack_async_event(entry.session_id)
    )

    assert result is True, (
        f"injection returned {result!r} on a relay-fronted gateway — the "
        "completion was dropped exactly as in the 2026-08-09 staging "
        "incident (literal adapter scan misses Platform.RELAY)"
    )
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_injection_retries_when_platform_not_fronted(tmp_path):
    """Unavailable transport stays retryable without letting relay hijack unrelated targets."""
    adapter = _RelayAdapter()  # fronts slack only
    runner, entry = _runner_with_relay(adapter, tmp_path)
    evt = _slack_async_event(entry.session_id)
    evt["session_key"] = "agent:main:discord:dm:123:456"
    evt["platform"] = "discord"

    result = await runner._inject_watch_notification("[x]", evt)
    assert result is False
    assert adapter.handle_message.await_count == 0
