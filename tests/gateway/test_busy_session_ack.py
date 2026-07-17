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

    @pytest.fixture(autouse=True)
    def _no_gateway_config(self, monkeypatch):
        """Mock _load_gateway_config to return empty dict so tests control
        busy_input_mode purely via runner._busy_input_mode."""
        import gateway.run as _gr
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})

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
    async def test_steer_mode_calls_agent_steer_no_interrupt_no_queue(self, monkeypatch):
        """busy_input_mode='steer' injects via agent.steer() and skips queueing."""
        import gateway.run as _gr

        monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})
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
    async def test_steer_mode_can_suppress_visible_ack_without_disabling_steer(self, monkeypatch):
        """busy_steer_ack_enabled=false keeps steering but drops the echo bubble."""
        import gateway.run as _gr

        monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
        monkeypatch.setattr(
            _gr,
            "_load_gateway_config",
            lambda: {"display": {"platforms": {"telegram": {"busy_steer_ack_enabled": False}}}},
        )

        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="also check the tests")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        await runner._handle_active_session_busy_message(event, sk)

        agent.steer.assert_called_once_with("also check the tests")
        agent.interrupt.assert_not_called()
        adapter._send_with_retry.assert_not_called()
        assert sk not in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_steer_ack_env_override_can_suppress_visible_ack(self, monkeypatch):
        """Env override supports process-level suppression for gateway services."""
        import gateway.run as _gr

        monkeypatch.setenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", "false")
        monkeypatch.setattr(
            _gr,
            "_load_gateway_config",
            lambda: {"display": {"platforms": {"telegram": {"busy_steer_ack_enabled": True}}}},
        )

        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="steer silently")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        await runner._handle_active_session_busy_message(event, sk)

        agent.steer.assert_called_once_with("steer silently")
        adapter._send_with_retry.assert_not_called()
        assert sk not in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_busy_ack_debounce_skips_steer_ack_config_load(self, monkeypatch):
        """Rapid follow-ups should not reload display config when ack is debounced."""
        import gateway.run as _gr

        def _boom():
            raise AssertionError("config should not be loaded inside ack cooldown")

        monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
        monkeypatch.setattr(_gr, "_load_gateway_config", _boom)

        runner, sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()

        event = _make_event(text="rapid steer")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent
        runner._busy_ack_ts[sk] = time.time()

        result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True
        agent.steer.assert_called_once_with("rapid steer")
        adapter._send_with_retry.assert_not_called()

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
# Per-platform busy_input_mode override
# ---------------------------------------------------------------------------

class TestPerPlatformBusyInputMode:
    """busy_input_mode can be overridden per platform via display.platforms.<platform>.

    Overrides are loaded once at startup into runner._busy_input_mode_by_platform
    (see GatewayRunner._load_busy_input_mode_by_platform), so these tests set the
    map directly; the loader itself is covered by TestBusyInputModeByPlatformLoader.
    """

    @pytest.mark.asyncio
    async def test_platform_steer_overrides_global_interrupt(self):
        """sendblue configured as steer, global is interrupt: sendblue gets steer."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"  # global default
        runner._busy_input_mode_by_platform = {"sendblue": "steer"}
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="steer this", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event") as mock_merge:
            await runner._handle_active_session_busy_message(event, sk)

        # Steer was used; the agent was NOT interrupted.
        agent.steer.assert_called_once_with("steer this")
        agent.interrupt.assert_not_called()
        mock_merge.assert_not_called()

    @pytest.mark.asyncio
    async def test_platform_interrupt_overrides_global_steer(self):
        """sendblue configured as interrupt, global is steer: sendblue gets interrupt."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "steer"  # global default
        runner._busy_input_mode_by_platform = {"sendblue": "interrupt"}
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="interrupt this", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        # Interrupt was used; the agent was NOT steered.
        agent.interrupt.assert_called_once_with("interrupt this")
        if hasattr(agent, "steer"):
            agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_priority_path_honors_per_platform_queue_override(self):
        """The PRIORITY running-agent path (_handle_message) must honor a
        display.platforms.<platform>.busy_input_mode override, not only the
        normal busy handler. Global is interrupt; sendblue overrides to queue,
        so the priority path must queue (not interrupt) the follow-up."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"  # global default
        runner._busy_input_mode_by_platform = {"sendblue": "queue"}
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="queue me", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        running_agent = MagicMock()
        # A real activity summary so the stale-eviction path (if reached) and
        # the interrupt/ack path in RED don't trip on MagicMock comparisons.
        running_agent.get_activity_summary.return_value = {
            "seconds_since_activity": 1,
            "last_activity_desc": "running",
            "api_call_count": 1,
            "max_iterations": 10,
        }
        runner._running_agents[sk] = running_agent
        runner._queue_or_replace_pending_event = MagicMock()

        result = await GatewayRunner._handle_message(runner, event)

        # Per-platform queue override honored on the priority path.
        runner._queue_or_replace_pending_event.assert_called_once_with(sk, event)
        running_agent.interrupt.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    async def test_unconfigured_platform_uses_global(self):
        """telegram has no platform override: falls back to global busy_input_mode."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        runner._busy_input_mode_by_platform = {"sendblue": "steer"}
        runner._busy_text_mode = "queue"  # realistic production default
        adapter = _make_adapter(platform_val="telegram")

        event = _make_event(text="queue me", platform_val="telegram")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        result = await runner._handle_active_session_busy_message(event, sk)

        # With busy_text_mode="queue" + effective_mode="queue", the handler
        # returns False (early exit for text in queue mode). The actual queueing
        # is handled by the caller via the adapter's _pending_messages.
        assert result is False
        agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_override_map_falls_back_to_global(self):
        """A runner without the startup map attribute (e.g. constructed by
        older tests via object.__new__) must still resolve the global mode."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        if hasattr(runner, "_busy_input_mode_by_platform"):
            del runner._busy_input_mode_by_platform

        event = _make_event(text="hi", platform_val="telegram")
        assert runner._effective_busy_input_mode(event.source.platform) == "queue"

    @pytest.mark.asyncio
    async def test_per_platform_interrupt_demoted_with_subagents(self):
        """Per-platform interrupt override also gets demoted to queue when
        subagents are active (#30170 interaction)."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "queue"
        runner._busy_input_mode_by_platform = {"sendblue": "interrupt"}
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="don't kill my subagents", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Parent with active subagents
        parent = MagicMock()
        parent._active_children = [MagicMock()]
        runner._running_agents[sk] = parent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        # Per-platform interrupt demoted to queue; parent.interrupt NOT called.
        parent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_platform_steer_fallback_to_queue_when_agent_pending(self):
        """Per-platform steer falls back to queue when agent is still starting."""
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        runner._busy_input_mode_by_platform = {"sendblue": "steer"}
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="arrived too early", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Agent is still being set up; sentinel in place.
        runner._running_agents[sk] = _sentinel
        runner._queue_or_replace_pending_event = MagicMock()

        await runner._handle_active_session_busy_message(event, sk)

        # Steer can't work with the sentinel, so the event is queued for the
        # next turn (via the FIFO path, #43066) and the ack says so.
        runner._queue_or_replace_pending_event.assert_called_once_with(sk, event)
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        assert "Queued for the next turn" in content


class TestBusyInputModeByPlatformLoader:
    """GatewayRunner._load_busy_input_mode_by_platform() builds the startup map."""

    def test_valid_overrides_loaded_invalid_dropped(self, monkeypatch):
        """Valid per-platform values land in the map; a typo is dropped at load
        time so runtime resolution falls back to the global mode."""
        import gateway.run as _gr
        from gateway.run import GatewayRunner

        monkeypatch.setattr(
            _gr,
            "_load_gateway_runtime_config",
            lambda: {
                "display": {
                    "busy_input_mode": "interrupt",
                    "platforms": {
                        "sendblue": {"busy_input_mode": "Steer"},  # normalised
                        "telegram": {"busy_input_mode": "steerr"},  # typo: dropped
                        "discord": {"busy_ack_detail": False},  # unrelated key
                        "weird": "not-a-dict",  # tolerated
                    },
                }
            },
        )

        assert GatewayRunner._load_busy_input_mode_by_platform() == {
            "sendblue": "steer"
        }

    def test_no_platforms_section_yields_empty_map(self, monkeypatch):
        import gateway.run as _gr
        from gateway.run import GatewayRunner

        monkeypatch.setattr(_gr, "_load_gateway_runtime_config", lambda: {})
        assert GatewayRunner._load_busy_input_mode_by_platform() == {}

    @pytest.mark.asyncio
    async def test_dropped_invalid_value_reaches_global_through_handler(self, monkeypatch):
        """End to end: a typo'd override is dropped by the loader and the
        handler then uses the global mode (steer), not interrupt."""
        import gateway.run as _gr
        from gateway.run import GatewayRunner

        monkeypatch.setattr(
            _gr,
            "_load_gateway_runtime_config",
            lambda: {
                "display": {
                    "platforms": {
                        "sendblue": {"busy_input_mode": "steerr"},  # typo
                    },
                }
            },
        )

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "steer"  # global
        runner._busy_input_mode_by_platform = (
            GatewayRunner._load_busy_input_mode_by_platform()
        )
        adapter = _make_adapter(platform_val="sendblue")

        event = _make_event(text="should steer not interrupt", platform_val="sendblue")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        # Typo in platform override was dropped at load; global "steer" used.
        agent.steer.assert_called_once_with("should steer not interrupt")
        agent.interrupt.assert_not_called()
