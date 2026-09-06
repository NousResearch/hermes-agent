"""Tests for gateway session stall watchdog (#72016 item 2)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.session_activity import ActivityProvenance, build_activity_snapshot
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session_stall import (
    format_session_stall_notification,
    resolve_session_idle_seconds_from_activity,
    should_clear_session_stall_notification,
    should_emit_session_stall_notification,
)


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "content": content, "metadata": metadata}
        )


class _FakeAgent:
    """Exposes the shared #72039 activity snapshot as the sole progress source."""

    def __init__(
        self,
        last_activity_ts: float,
        *,
        description: str = "api call",
        provenance: ActivityProvenance = ActivityProvenance.UNKNOWN,
    ):
        self._last_activity_ts = last_activity_ts
        self._last_activity_desc = description
        self._last_activity_provenance = provenance

    def get_activity_summary(self):
        return build_activity_snapshot(
            last_activity_at=self._last_activity_ts,
            last_activity_description=self._last_activity_desc,
            last_activity_provenance=self._last_activity_provenance,
        )


class _AgentWithoutSummary:
    """Agent with a raw clock but no shared summary consumer API."""

    def __init__(self, last_activity_ts: float):
        self._last_activity_ts = last_activity_ts


def test_should_emit_requires_pending_and_idle():
    assert should_emit_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=400,
        has_pending_inbound=True,
        already_notified=False,
    )
    assert not should_emit_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=400,
        has_pending_inbound=False,
        already_notified=False,
    )
    assert not should_emit_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=100,
        has_pending_inbound=True,
        already_notified=False,
    )
    assert not should_emit_session_stall_notification(
        timeout_seconds=0,
        idle_seconds=9999,
        has_pending_inbound=True,
        already_notified=False,
    )
    assert not should_emit_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=400,
        has_pending_inbound=True,
        already_notified=True,
    )


def test_should_clear_when_pending_gone_or_activity_resumes():
    assert should_clear_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=400,
        has_pending_inbound=False,
    )
    assert should_clear_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=10,
        has_pending_inbound=True,
    )
    assert not should_clear_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=400,
        has_pending_inbound=True,
    )


def test_should_clear_holds_latch_when_idle_unknown():
    assert not should_clear_session_stall_notification(
        timeout_seconds=300,
        idle_seconds=None,
        has_pending_inbound=True,
    )


def test_format_session_stall_notification_minutes():
    msg = format_session_stall_notification(125)
    assert "2 min ago" in msg
    assert "/new" in msg
    assert format_session_stall_notification(30).count("1 min ago") == 1


def test_resolve_idle_uses_shared_activity_snapshot_only():
    now = 1_000_000.0
    snap = build_activity_snapshot(
        last_activity_at=now - 120,
        last_activity_description="tool: terminal",
        last_activity_provenance=ActivityProvenance.UNKNOWN,
        now=now,
    )
    assert resolve_session_idle_seconds_from_activity(snap, now=now) == 120.0
    assert resolve_session_idle_seconds_from_activity(None, now=now) is None
    assert resolve_session_idle_seconds_from_activity({}, now=now) is None


def test_resolve_idle_prefers_seconds_since_activity_field():
    idle = resolve_session_idle_seconds_from_activity(
        {
            "seconds_since_activity": 42.5,
            "last_activity_at": 1.0,  # must be ignored when seconds present
        },
        now=999.0,
    )
    assert idle == 42.5


def _runner_for_stall(adapter: _FakeAdapter) -> GatewayRunner:
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r.adapters = {"fake": adapter}
    r._profile_adapters = {}
    r._running_agents = {}
    r._running_agents_ts = {}
    r._queued_events = {}
    r._session_stall_notified = {}
    r._thread_metadata_for_source = lambda source, *a, **k: {
        "thread_id": getattr(source, "thread_id", None)
    }
    return r


def _pending_event(chat_id: str = "chat-1", thread_id: str | None = None):
    source = SimpleNamespace(chat_id=chat_id, thread_id=thread_id, platform=None)
    return SimpleNamespace(text="follow-up", source=source, timestamp=time.time())


@pytest.mark.asyncio
async def test_check_session_stalls_notifies_once(monkeypatch):
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    monkeypatch.setenv("HERMES_SESSION_STALL_TIMEOUT", "60")
    session_key = "agent:main:telegram:dm:1"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    sent = await runner._check_session_stalls(60)
    assert sent == 1
    assert len(adapter.sent) == 1
    assert "/new" in adapter.sent[0]["content"]
    assert runner._session_stall_notified.get(session_key) is True

    # Second pass must not spam.
    sent2 = await runner._check_session_stalls(60)
    assert sent2 == 0
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_check_session_stalls_skips_fresh_activity():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:2"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 5)

    sent = await runner._check_session_stalls(60)
    assert sent == 0
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_check_session_stalls_skips_without_pending():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:3"
    runner._running_agents[session_key] = _FakeAgent(time.time() - 999)

    sent = await runner._check_session_stalls(60)
    assert sent == 0
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_check_session_stalls_clears_latch_when_pending_drains():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:4"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 200)

    assert await runner._check_session_stalls(60) == 1
    assert session_key in runner._session_stall_notified

    adapter._pending_messages.clear()
    assert await runner._check_session_stalls(60) == 0
    assert session_key not in runner._session_stall_notified


@pytest.mark.asyncio
async def test_check_session_stalls_skips_pending_sentinel_without_activity():
    """Pending construction has no shared activity snapshot — no parallel clocks."""
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:5"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _AGENT_PENDING_SENTINEL
    runner._running_agents_ts[session_key] = time.time() - 90

    sent = await runner._check_session_stalls(60)
    assert sent == 0
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_check_session_stalls_ignores_raw_clock_without_summary():
    """Do not fall back to agent._last_activity_ts outside get_activity_summary()."""
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:raw"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _AgentWithoutSummary(time.time() - 999)
    runner._running_agents_ts[session_key] = time.time() - 999

    sent = await runner._check_session_stalls(60)
    assert sent == 0
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_session_stall_watcher_disabled_when_timeout_zero(monkeypatch):
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    monkeypatch.setenv("HERMES_SESSION_STALL_TIMEOUT", "0")
    session_key = "agent:main:telegram:dm:6"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 999)

    task = asyncio.create_task(runner._session_stall_watcher(interval=1))
    await asyncio.sleep(0.05)
    # Force one check path via timeout read; watcher should no-op when 0.
    assert runner._session_stall_timeout_seconds() == 0.0
    runner._running = False
    await asyncio.wait_for(task, timeout=2)
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_check_session_stalls_queued_events_overflow_notifies():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:overflow"
    event = _pending_event()
    runner._queued_events[session_key] = [event]
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)
    runner._adapter_for_source = lambda source: adapter

    assert await runner._check_session_stalls(60) == 1
    assert adapter.sent and "/new" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_check_session_stalls_scans_profile_adapters():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(_FakeAdapter())  # empty primary path unused
    runner.adapters = {}
    runner._profile_adapters = {"coder": {"fake": adapter}}
    session_key = "agent:coder:telegram:dm:1"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    assert await runner._check_session_stalls(60) == 1
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_check_session_stalls_logs_compression_provenance(caplog):
    """Stale compression stamps still stall, but provenance stays visible."""
    import logging

    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:compress"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(
        time.time() - 120,
        description="compressing context",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )

    with caplog.at_level(logging.WARNING):
        assert await runner._check_session_stalls(60) == 1
    assert any("agent.compression" in r.message for r in caplog.records)
    assert any("compressing context" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_check_session_stalls_skips_active_compression_heartbeat():
    """Fresh agent.compression heartbeats are progress, not a silent stall."""
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:compacting"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(
        time.time() - 5,
        description="context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )

    assert await runner._check_session_stalls(60) == 0
    assert adapter.sent == []
    assert session_key not in runner._session_stall_notified


@pytest.mark.asyncio
async def test_check_session_stalls_does_not_renotify_after_summary_gap():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:gap"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    assert await runner._check_session_stalls(60) == 1

    # Transient observation gap (no summary API).
    runner._running_agents[session_key] = _AgentWithoutSummary(time.time() - 999)
    assert await runner._check_session_stalls(60) == 0
    assert runner._session_stall_notified.get(session_key) is True

    # Stale progress returns — must not spam again in the same episode.
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)
    assert await runner._check_session_stalls(60) == 0
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_check_session_stalls_retries_after_send_failure():
    class _FailThenOk(_FakeAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def send(self, chat_id, content, metadata=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            await super().send(chat_id, content, metadata=metadata)

    adapter = _FailThenOk()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:retry"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    assert await runner._check_session_stalls(60) == 0
    assert session_key not in runner._session_stall_notified
    assert await runner._check_session_stalls(60) == 1
    assert runner._session_stall_notified.get(session_key) is True
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_check_session_stalls_retries_after_soft_send_failure():
    class _SoftFailThenOk(_FakeAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def send(self, chat_id, content, metadata=None):
            from gateway.platforms.base import SendResult

            self.calls += 1
            if self.calls == 1:
                return SendResult(success=False, error="chat not found")
            await super().send(chat_id, content, metadata=metadata)
            return SendResult(success=True)

    adapter = _SoftFailThenOk()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:soft"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    assert await runner._check_session_stalls(60) == 0
    assert session_key not in runner._session_stall_notified
    assert await runner._check_session_stalls(60) == 1
    assert runner._session_stall_notified.get(session_key) is True
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_check_session_stalls_renotifies_after_resume_then_restall():
    adapter = _FakeAdapter()
    runner = _runner_for_stall(adapter)
    session_key = "agent:main:telegram:dm:episode"
    adapter._pending_messages[session_key] = _pending_event()
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)

    assert await runner._check_session_stalls(60) == 1

    # Activity resumes (still pending) — clears latch for a new episode.
    runner._running_agents[session_key] = _FakeAgent(time.time() - 5)
    assert await runner._check_session_stalls(60) == 0
    assert session_key not in runner._session_stall_notified

    # Stall again — second episode may notify once more.
    runner._running_agents[session_key] = _FakeAgent(time.time() - 120)
    assert await runner._check_session_stalls(60) == 1
    assert len(adapter.sent) == 2


def test_resolve_idle_rejects_nonfinite_seconds_since_activity():
    now = 1_000_000.0
    idle = resolve_session_idle_seconds_from_activity(
        {
            "seconds_since_activity": float("nan"),
            "last_activity_at": now - 15,
        },
        now=now,
    )
    assert idle == 15.0


def test_resolve_idle_rejects_boolean_seconds_and_uses_timestamp():
    now = 1_000_000.0
    idle = resolve_session_idle_seconds_from_activity(
        {
            "seconds_since_activity": True,
            "last_activity_ts": now - 3,
        },
        now=now,
    )
    assert idle == 3.0


def test_session_stall_timeout_in_default_config():
    from hermes_cli.config import DEFAULT_CONFIG

    timeout = DEFAULT_CONFIG["agent"]["session_stall_timeout"]
    assert isinstance(timeout, (int, float))
    assert timeout > 0  # enabled by default; 0 would disable the watchdog


class _NeverResolvingAdapter:
    """Adapter whose send() hangs forever (wedged transport)."""

    def __init__(self):
        self._pending_messages = {}
        self.send_attempts = 0

    async def send(self, chat_id, content, metadata=None):
        self.send_attempts += 1
        await asyncio.Event().wait()  # never resolves


@pytest.mark.asyncio
async def test_check_session_stalls_bounds_wedged_send(monkeypatch):
    """Round-2 #2: a never-resolving adapter.send must not wedge the watcher.

    The bounded send times out, does NOT latch (retry next tick), the pass
    completes so the watcher keeps ticking, and a healthy sibling candidate
    still receives its notification in the same pass.
    """
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_STALL_NOTIFY_SEND_TIMEOUT_SECONDS", 0.1
    )
    wedged = _NeverResolvingAdapter()
    healthy = _FakeAdapter()
    runner = _runner_for_stall(wedged)
    runner.adapters = {"wedged": wedged, "healthy": healthy}

    wedged_key = "agent:main:telegram:dm:wedged"
    healthy_key = "agent:main:discord:dm:healthy"
    wedged._pending_messages[wedged_key] = _pending_event(chat_id="chat-w")
    healthy._pending_messages[healthy_key] = _pending_event(chat_id="chat-h")
    runner._running_agents[wedged_key] = _FakeAgent(time.time() - 120)
    runner._running_agents[healthy_key] = _FakeAgent(time.time() - 120)

    # Pass must complete despite the wedged transport (bounded by wait_for).
    sent = await asyncio.wait_for(runner._check_session_stalls(60), timeout=5)

    # Healthy sibling was notified in the SAME pass.
    assert sent == 1
    assert len(healthy.sent) == 1
    assert healthy_key in runner._session_stall_notified
    # Wedged session: send attempted, timed out, NOT latched.
    assert wedged.send_attempts == 1
    assert wedged_key not in runner._session_stall_notified

    # Watcher ticks again: the wedged candidate is retried next pass.
    sent2 = await asyncio.wait_for(runner._check_session_stalls(60), timeout=5)
    assert sent2 == 0  # healthy already latched; wedged timed out again
    assert wedged.send_attempts == 2
    assert wedged_key not in runner._session_stall_notified
