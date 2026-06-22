"""Tests for busy-session acknowledgment when user sends messages during active agent runs.

Verifies that users get an immediate status response instead of total silence
when the agent is working on a task. See PR fix for the @Lonely__MH report.
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys, types

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    Platform,
    SessionSource,
    build_session_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text="hello", chat_id="123", platform_val="telegram"):
    """Build a minimal MessageEvent."""
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    evt = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )
    return evt


def _make_runner():
    """Build a minimal GatewayRunner-like object for testing."""
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    """Build a minimal adapter mock."""
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBusySessionAck:
    """User sends a message while agent is running — should get acknowledgment."""

    @pytest.mark.asyncio
    async def test_handle_message_queue_mode_queues_without_interrupt(self):
        """Runner queue mode must not interrupt an active agent for text follow-ups."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="follow up in queue mode")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._busy_input_mode = "queue"
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_message(runner, event)

        assert result is None
        assert sk in adapter._pending_messages
        assert adapter._pending_messages[sk] is event
        assert sk not in runner._pending_messages
        running_agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_telegram_grace_followups_respect_queue_fifo(self, monkeypatch):
        """Rapid Telegram text follow-ups in queue mode must not merge."""
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        runner._queued_events = {}
        adapter = _make_adapter()

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="user1",
        )
        sk = build_session_key(source)
        runner.adapters[source.platform] = adapter

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "seconds_since_activity": 0.0,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()

        events = [
            MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=f"m-{idx}",
            )
            for idx, text in enumerate(("first", "second", "third"), start=1)
        ]

        for event in events:
            result = await GatewayRunner._handle_message(runner, event)
            assert result is None

        assert adapter._pending_messages[sk].text == "first"
        assert [event.text for event in runner._queued_events[sk]] == [
            "second",
            "third",
        ]
        agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_ack_when_agent_running(self):
        """First message during busy session should get a status ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="Are you working?")
        sk = build_session_key(event.source)

        # Simulate running agent
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 1.0,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min ago
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True  # handled
        # Verify ack was sent
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        if not content and call_kwargs.args:
            # positional args
            content = str(call_kwargs)
        assert "Interrupting" in content or "respond" in content
        assert "/stop" not in content  # no need — we ARE interrupting

        # Verify agent interrupt was called
        agent.interrupt.assert_called_once_with("Are you working?")

    @pytest.mark.asyncio
    async def test_queue_mode_suppresses_interrupt_and_updates_ack(self):
        """When busy_input_mode is 'queue', message is queued WITHOUT interrupt."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="Add this to queue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        # VERIFY: Agent was NOT interrupted
        agent.interrupt.assert_not_called()

        # VERIFY: Ack sent with queue-specific wording
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content
        assert "respond once the current task finishes" in content
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_busy_text_mode_queue_delegates_to_adapter_handle_message(self):
        """busy_text_mode=queue lets the adapter debounce text silently."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        runner._busy_text_mode = "queue"
        adapter = _make_adapter()

        first = _make_event(text="part one")
        second = _make_event(text="part two")
        sk = build_session_key(first.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[first.source.platform] = adapter
        runner.adapters[second.source.platform] = adapter

        result1 = await runner._handle_active_session_busy_message(first, sk)
        result2 = await runner._handle_active_session_busy_message(second, sk)

        assert result1 is False
        assert result2 is False
        assert sk not in adapter._pending_messages
        agent.interrupt.assert_not_called()
        adapter._send_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_mode_calls_agent_steer_no_interrupt_no_queue(self):
        """busy_input_mode='steer' injects via agent.steer() and skips queueing."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="also check the tests")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            await runner._handle_active_session_busy_message(event, sk)

        # VERIFY: Agent was steered, NOT interrupted
        agent.steer.assert_called_once_with("also check the tests")
        agent.interrupt.assert_not_called()

        # VERIFY: No queueing — successful steer must NOT replay as next turn
        mock_merge.assert_not_called()

        # VERIFY: Ack mentions steer wording
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Steered" in content or "steer" in content.lower()
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_falls_back_to_queue_when_agent_rejects(self):
        """If agent.steer() returns False, fall back to queue behavior."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="empty or rejected")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=False)  # rejected
        runner._running_agents[sk] = agent

        await runner._handle_active_session_busy_message(event, sk)

        agent.steer.assert_called_once()
        agent.interrupt.assert_not_called()
        # Fell back to queue semantics: event was stored for the next turn
        # via the FIFO path (each follow-up its own turn — no newline-merge
        # that would mash separate messages together, #43066).
        assert adapter._pending_messages.get(sk) is event

        # Ack uses queue-mode wording (not steer, not interrupt)
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content
        assert "Steered" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_falls_back_to_queue_when_agent_pending(self):
        """If agent is still starting (sentinel), steer mode falls back to queue."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="arrived too early")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Agent is still being set up — sentinel in place
        runner._running_agents[sk] = sentinel

        await runner._handle_active_session_busy_message(event, sk)

        # Event was queued instead of steered (FIFO path, #43066)
        assert adapter._pending_messages.get(sk) is event

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content

    @pytest.mark.asyncio
    async def test_interrupt_mode_text_followups_fifo_not_merged(self):
        """Two TEXT follow-ups during a busy turn (interrupt mode) must each
        get their OWN next-turn slot via FIFO — NOT newline-merged into one
        mashed-together turn (#43066 sub-bug 2). Before the fix the
        interrupt/steer-fallback path called merge_pending_message_event
        with merge_text=True, collapsing 'first' and 'second' into
        'first\\nsecond' and destroying message boundaries."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        runner._queued_events = {}
        adapter = _make_adapter()

        # Both events must share the SAME platform object so they resolve to
        # the same adapter (a fresh MagicMock per event would not).
        shared_platform = Platform.TELEGRAM

        def _evt(text):
            src = SessionSource(
                platform=shared_platform, chat_id="123",
                chat_type="dm", user_id="user1",
            )
            return MessageEvent(text=text, message_type=MessageType.TEXT,
                                source=src, message_id=f"m-{text[:5]}")

        first = _evt("first message")
        second = _evt("second message")
        sk = build_session_key(first.source)
        runner.adapters[shared_platform] = adapter

        agent = MagicMock()
        agent._active_children = []  # real list → not demoted to queue
        runner._running_agents[sk] = agent

        await runner._handle_active_session_busy_message(first, sk)
        runner._busy_ack_ts = {}  # avoid the 30s ack-debounce early return
        await runner._handle_active_session_busy_message(second, sk)

        # First lands in the head slot; second goes to the FIFO overflow —
        # they are NOT merged into a single pending event.
        head = adapter._pending_messages.get(sk)
        assert head is first
        assert head.text == "first message"  # not "first message\nsecond message"
        overflow = runner._queued_events.get(sk, [])
        assert [e.text for e in overflow] == ["second message"]

    @pytest.mark.asyncio
    async def test_debounce_suppresses_rapid_acks(self):
        """Second message within 30s should NOT send another ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event1 = _make_event(text="hello?")
        # Reuse the same source so platform mock matches
        event2 = MessageEvent(
            text="still there?",
            message_type=MessageType.TEXT,
            source=event1.source,
            message_id="msg2",
        )
        sk = build_session_key(event1.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 5,
            "max_iterations": 60,
            "current_tool": None,
            "last_activity_ts": time.time(),
            "last_activity_desc": "api_call",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 60
        runner.adapters[event1.source.platform] = adapter

        # First message — should get ack
        result1 = await runner._handle_active_session_busy_message(event1, sk)
        assert result1 is True
        assert adapter._send_with_retry.call_count == 1

        # Second message within cooldown — should be queued but no ack
        result2 = await runner._handle_active_session_busy_message(event2, sk)
        assert result2 is True
        assert adapter._send_with_retry.call_count == 1  # still 1, no new ack

        # But interrupt should still be called for both (since we are in interrupt mode)
        assert agent.interrupt.call_count == 2

    @pytest.mark.asyncio
    async def test_ack_after_cooldown_expires(self):
        """After 30s cooldown, a new message should send a fresh ack."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hello?")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 10,
            "max_iterations": 60,
            "current_tool": "web_search",
            "last_activity_ts": time.time(),
            "last_activity_desc": "tool",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter

        # First ack
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 1

        # Fake that cooldown expired
        runner._busy_ack_ts[sk] = time.time() - 31

        # Second ack should go through
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_includes_status_detail_when_opted_in(self, monkeypatch):
        """Ack message should include iteration and tool info when available."""
        import gateway.run as _gr

        monkeypatch.setattr(
            _gr,
            "_load_gateway_config",
            lambda: {"display": {"platforms": {"telegram": {"busy_ack_detail": True}}}},
        )
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="yo")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "21/60" in content  # iteration
        assert "terminal" in content  # current tool
        assert "10 min" in content  # elapsed

    @pytest.mark.asyncio
    async def test_telegram_omits_status_detail_by_default(self):
        """Telegram busy acks stay concise unless busy_ack_detail is enabled."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="yo")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Interrupting current task" in content
        assert "21/60" not in content
        assert "terminal" not in content
        assert "10 min" not in content

    @pytest.mark.asyncio
    async def test_draining_still_works(self):
        """Draining case should still produce the drain-specific message."""
        runner, sentinel = _make_runner()
        runner._draining = True
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Mock the drain-specific methods
        runner._queue_during_drain_enabled = lambda: False
        runner._status_action_gerund = lambda: "restarting"

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "restarting" in content

    @pytest.mark.asyncio
    async def test_pending_sentinel_no_interrupt(self):
        """When agent is PENDING_SENTINEL, don't call interrupt (it has no method)."""
        runner, sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="hey")
        sk = build_session_key(event.source)

        runner._running_agents[sk] = sentinel
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True
        # Should still send ack
        adapter._send_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adapter_falls_through(self):
        """If adapter is missing, return False so default path handles it."""
        runner, sentinel = _make_runner()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)

        # No adapter registered
        runner._running_agents[sk] = MagicMock()

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is False  # not handled, let default path try


class TestBusySessionOnboardingHint:
    """First-touch hint appended to the busy-ack the first time it fires."""

    @pytest.mark.asyncio
    async def test_first_busy_ack_appends_interrupt_hint(self, tmp_path, monkeypatch):
        """First busy-while-running message gets an extra hint about /busy."""
        import gateway.run as _gr

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        # mark_seen imports utils.atomic_yaml_write; make sure it resolves
        # against a writable dir by pointing _hermes_home at tmp_path.
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="ping")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 3, "max_iterations": 60,
            "current_tool": None, "last_activity_ts": time.time(),
            "last_activity_desc": "api", "seconds_since_activity": 0.1,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 5
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")

        # Normal ack body
        assert "Interrupting" in content
        # First-touch hint appended
        assert "First-time tip" in content
        assert "/busy queue" in content

        # The flag is now persisted to tmp_path/config.yaml
        import yaml
        cfg = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert cfg["onboarding"]["seen"]["busy_input_prompt"] is True

    @pytest.mark.asyncio
    async def test_second_busy_ack_omits_hint(self, tmp_path, monkeypatch):
        """Once the flag is marked, the hint never appears again."""
        import gateway.run as _gr
        import yaml

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        # Pre-populate the config so is_seen() returns True from the start.
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({
            "onboarding": {"seen": {"busy_input_prompt": True}},
        }))
        monkeypatch.setattr(
            _gr, "_load_gateway_config",
            lambda: yaml.safe_load((tmp_path / "config.yaml").read_text()),
        )

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="ping again")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 3, "max_iterations": 60,
            "current_tool": None, "last_activity_ts": time.time(),
            "last_activity_desc": "api", "seconds_since_activity": 0.1,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 5
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")

        assert "Interrupting" in content
        assert "First-time tip" not in content
        assert "/busy queue" not in content

    @pytest.mark.asyncio
    async def test_queue_mode_hint_points_to_interrupt(self, tmp_path, monkeypatch):
        """In queue mode the hint should suggest /busy interrupt, not /busy queue."""
        import gateway.run as _gr

        monkeypatch.setattr(_gr, "_hermes_home", tmp_path)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="queue me")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Queued for the next turn" in content
        assert "First-time tip" in content
        assert "/busy interrupt" in content
        # Must NOT tell the user to /busy queue when they're already on queue.
        assert "/busy queue" not in content


class TestLongRunningNotificationOwnership:
    """The long-running heartbeat must stop once its run no longer owns the
    session slot or the executor finished — otherwise a stale
    'running: delegate_task' bubble outlives the run that spawned it (#12029).
    """

    def test_notification_stops_after_session_ownership_moves(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._running_agents = {}

        original_agent = MagicMock()
        replacement_agent = MagicMock()
        runner._running_agents["sess"] = replacement_agent

        assert runner._should_emit_long_running_notification(
            "sess", original_agent, executor_task=None
        ) is False

    def test_notification_stops_after_executor_finishes(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        agent = MagicMock()
        runner._running_agents = {"sess": agent}

        done_task = MagicMock()
        done_task.done.return_value = True

        assert runner._should_emit_long_running_notification(
            "sess", agent, executor_task=done_task
        ) is False

    def test_notification_stops_when_agent_is_gone(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._running_agents = {}

        assert runner._should_emit_long_running_notification(
            "sess", None, executor_task=None
        ) is False

    def test_notification_continues_for_live_active_run(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        agent = MagicMock()
        runner._running_agents = {"sess": agent}

        live_task = MagicMock()
        live_task.done.return_value = False

        assert runner._should_emit_long_running_notification(
            "sess", agent, executor_task=live_task
        ) is True


# ---------------------------------------------------------------------------
# Interrupt debounce opt-in (HERMES_INTERRUPT_DEBOUNCE_SECONDS)
# ---------------------------------------------------------------------------

def _make_runner_for_debounce():
    """Extend _make_runner with fields required for debounce tests."""
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._pending_interrupt_tasks = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
    runner._busy_input_mode = "interrupt"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._queued_events = {}
    return runner, _AGENT_PENDING_SENTINEL


class TestInterruptDebounce:
    """Tests for HERMES_INTERRUPT_DEBOUNCE_SECONDS opt-in debounce feature."""

    @pytest.mark.asyncio
    async def test_off_by_default_interrupt_is_immediate(self, monkeypatch):
        """With HERMES_INTERRUPT_DEBOUNCE_SECONDS=0 (default), interrupt fires immediately."""
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.delenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", raising=False)

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="interrupt me now")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
        ):
            await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Interrupt must have been called synchronously (no debounce task created).
        agent.interrupt.assert_called_once()
        assert sk not in runner._pending_interrupt_tasks

    @pytest.mark.asyncio
    async def test_off_explicit_zero_interrupt_is_immediate(self, monkeypatch):
        """With HERMES_INTERRUPT_DEBOUNCE_SECONDS=0 explicitly, behaves same as default."""
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "0")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="still immediate")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
        ):
            await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        agent.interrupt.assert_called_once()
        assert sk not in runner._pending_interrupt_tasks

    @pytest.mark.asyncio
    async def test_debounce_on_single_interrupt_and_single_ack(self, monkeypatch):
        """With debounce>0, a burst of 3 messages produces exactly 1 interrupt and 1 ack."""
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "0.05")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        source = MagicMock()
        source.platform = MagicMock(value="telegram")
        source.chat_id = "123"
        source.chat_type = "private"
        source.user_id = "u1"
        source.thread_id = None

        def _make_msg(text):
            e = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id="m1",
            )
            return e

        sk = build_session_key(source)
        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        agent.interrupt = MagicMock()
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
            patch("gateway.run._reply_anchor_for_event", return_value=None, create=True),
            patch.object(
                GatewayRunner, "_reply_anchor_for_event", return_value=None
            ),
            patch.object(
                GatewayRunner, "_thread_metadata_for_source", return_value={}
            ),
        ):
            for text in ("msg1", "msg2", "msg3"):
                await GatewayRunner._handle_active_session_busy_message(
                    runner, _make_msg(text), sk
                )
                # Let the event loop tick so tasks can be scheduled.
                await asyncio.sleep(0)

            # Wait for the debounce window to expire.
            await asyncio.sleep(0.15)

        # Exactly one interrupt fired.
        agent.interrupt.assert_called_once()
        # Exactly one ack sent.
        assert adapter._send_with_retry.call_count == 1
        # Pending slot must contain all 3 texts merged, no overflow turns.
        pending_text = adapter._pending_messages.get(sk, MagicMock()).text
        assert "msg1" in pending_text
        assert "msg2" in pending_text
        assert "msg3" in pending_text
        # No phantom turns in the FIFO overflow queue.
        assert not runner._queued_events.get(sk)

    @pytest.mark.asyncio
    async def test_debounce_on_agent_gone_before_fire(self, monkeypatch):
        """If agent finishes before debounce window, no interrupt and no ack are sent."""
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "0.05")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="too slow")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
            patch.object(GatewayRunner, "_reply_anchor_for_event", return_value=None),
            patch.object(GatewayRunner, "_thread_metadata_for_source", return_value={}),
        ):
            await GatewayRunner._handle_active_session_busy_message(runner, event, sk)
            await asyncio.sleep(0)

            # Simulate agent finishing before the debounce fires.
            runner._running_agents.pop(sk, None)

            await asyncio.sleep(0.15)

        agent.interrupt.assert_not_called()
        adapter._send_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_pending_interrupt_clears_task(self, monkeypatch):
        """_cancel_pending_interrupt removes and cancels the pending task."""
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "10")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="pending")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
        ):
            await GatewayRunner._handle_active_session_busy_message(runner, event, sk)
            await asyncio.sleep(0)

        assert sk in runner._pending_interrupt_tasks
        task = runner._pending_interrupt_tasks[sk]
        assert not task.done()

        GatewayRunner._cancel_pending_interrupt(runner, sk)

        assert sk not in runner._pending_interrupt_tasks
        # After cancel() the task enters "cancelling" state; it becomes done
        # only after the event loop processes the CancelledError. Allow one tick.
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_debounce_finally_does_not_evict_newer_task(self, monkeypatch):
        """The finally block in _debounced_interrupt must not evict a newer task.

        Simulates the race condition by directly creating two tasks: after the first
        completes, the second must still be present in _pending_interrupt_tasks.
        """
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "0.03")
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="first")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
            patch.object(GatewayRunner, "_reply_anchor_for_event", return_value=None),
            patch.object(GatewayRunner, "_thread_metadata_for_source", return_value={}),
        ):
            # Directly create first debounce task and register it.
            first_task = asyncio.create_task(
                GatewayRunner._debounced_interrupt(runner, sk, event, 0.03)
            )
            runner._pending_interrupt_tasks[sk] = first_task
            await asyncio.sleep(0)

            # Before the first task fires, replace it with a second (simulates new message).
            first_task.cancel()
            event2 = _make_event(text="second")
            second_task = asyncio.create_task(
                GatewayRunner._debounced_interrupt(runner, sk, event2, 0.03)
            )
            runner._pending_interrupt_tasks[sk] = second_task
            await asyncio.sleep(0)

            assert runner._pending_interrupt_tasks.get(sk) is second_task

            # Wait for the second task to complete.
            await asyncio.sleep(0.15)

        # Main invariant: agent interrupted exactly once (by the second task).
        agent.interrupt.assert_called_once()
        # The second task should have cleaned itself up or left sk absent in the dict;
        # critically it must NOT be the (cancelled) first_task.
        remaining = runner._pending_interrupt_tasks.get(sk)
        assert remaining is not first_task

    @pytest.mark.asyncio
    async def test_debounce_not_activated_when_demoted_to_queue(self, monkeypatch):
        """When interrupt mode is demoted to queue (active subagents), debounce must not fire.

        Even with HERMES_INTERRUPT_DEBOUNCE_SECONDS > 0, the demotion sets
        effective_mode='queue' before the debounce gate runs, so no debounce task
        is created and no interrupt fires. The message is queued via the normal
        FIFO path instead.
        """
        import asyncio
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_INTERRUPT_DEBOUNCE_SECONDS", "0.05")

        runner, _sentinel = _make_runner_for_debounce()
        adapter = _make_adapter()
        adapter._send_with_retry = AsyncMock()

        event = _make_event(text="subagent active, should queue not debounce")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {}
        agent.interrupt = MagicMock()
        # Simulate active subagents so the runner demotes interrupt -> queue (#30170).
        agent._active_children = ["child1"]
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        with (
            patch("gateway.run._load_gateway_config", return_value={}),
            patch("gateway.run._platform_config_key", return_value="telegram"),
            patch("gateway.display_config.resolve_display_setting", return_value=False),
            # _agent_has_active_subagents must return True for the demotion to happen.
            patch.object(GatewayRunner, "_agent_has_active_subagents", return_value=True),
        ):
            result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)
            await asyncio.sleep(0)

        # No debounce task should have been scheduled.
        assert sk not in runner._pending_interrupt_tasks
        # No interrupt should have been called (queue mode, not interrupt mode).
        agent.interrupt.assert_not_called()
        # Message was stored via FIFO (head pending slot on the adapter).
        assert sk in adapter._pending_messages
