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
    async def test_followup_grace_applies_to_non_telegram_platforms(self, monkeypatch):
        """A rapid text follow-up must not interrupt on ANY platform, not just Telegram.

        Some transports deliver ONE human message as TWO events. iMessage does this
        whenever the message carries a URL preview or an attachment: the user types
        "subscribe to this calendar?" plus a link, presses send once, and the adapter
        receives two TEXT events a second or two apart.

        Without a follow-up grace window the second event lands mid-turn and is treated
        as the user interrupting themselves — the run is aborted and the user is told
        they interrupted a message they sent as one piece.

        The grace window already solves this. It was simply gated to Telegram, so every
        other adapter re-encounters the same bug.
        """
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "3.0")

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        runner._queued_events = {}
        adapter = _make_adapter()

        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="any;-;+15550001111",
            chat_type="dm",
            user_id="+15550001111",
        )
        sk = build_session_key(source)
        runner.adapters[source.platform] = adapter

        agent = MagicMock()
        agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()

        # The two halves of a single iMessage send.
        halves = [
            "Can you subscribe to this calendar?",
            "https://example.com/calendars/team.ics",
        ]
        for idx, text in enumerate(halves, start=1):
            result = await GatewayRunner._handle_message(
                runner,
                MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=f"bb-{idx}",
                ),
            )
            assert result is None

        # The run must survive: the user did not interrupt anything.
        agent.interrupt.assert_not_called()
        # And the second half must still be waiting to be handled, not dropped.
        assert sk in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_control_command_is_not_swallowed_by_the_grace_window(self, monkeypatch):
        """`/stop` inside the grace window must still reach command handling.

        The window exists to absorb the trailing half of a split send. A control
        command is never that — it is the user deliberately intervening, and it is
        exactly the moment they most need to be heard. If `/stop` lands in the
        pending slot instead, the fix trades "user interrupted themselves" for
        "user cannot stop a runaway run", which is a strictly worse bug.
        """
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "3.0")

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
        runner._queued_events = {}
        adapter = _make_adapter()

        source = SessionSource(
            platform=Platform.BLUEBUBBLES,
            chat_id="any;-;+15550001111",
            chat_type="dm",
            user_id="+15550001111",
        )
        sk = build_session_key(source)
        runner.adapters[source.platform] = adapter

        agent = MagicMock()
        agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()   # run started just now

        stop_event = MessageEvent(
            text="/stop",
            message_type=MessageType.TEXT,
            source=source,
            message_id="stop-1",
        )
        assert stop_event.is_command(), "precondition: /stop must parse as a command"

        await GatewayRunner._handle_message(runner, stop_event)

        # The command must NOT have been parked in the pending slot.
        parked = adapter._pending_messages.get(sk)
        assert parked is None or parked.text != "/stop", (
            "/stop was swallowed by the follow-up grace window"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "banana"])
    async def test_non_finite_grace_values_fall_back(self, monkeypatch, bad):
        """`float()` accepts nan/inf; neither is a window a user could have meant.

        NaN fails every comparison and silently disables the feature; inf makes the
        window unbounded so nothing ever interrupts. Both fail QUIETLY, which is the
        worst way for a misconfiguration to behave — hence an explicit guard.
        """
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", bad)
        monkeypatch.delenv("HERMES_BLUEBUBBLES_FOLLOWUP_GRACE_SECONDS", raising=False)

        assert GatewayRunner._followup_grace_seconds(Platform.BLUEBUBBLES) == 3.0

    @pytest.mark.asyncio
    async def test_per_platform_override_beats_the_gateway_wide_value(self, monkeypatch):
        """Per-platform wins; other platforms still read the gateway-wide value."""
        from gateway.run import GatewayRunner

        monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "1.0")
        monkeypatch.setenv("HERMES_BLUEBUBBLES_FOLLOWUP_GRACE_SECONDS", "5.0")

        assert GatewayRunner._followup_grace_seconds(Platform.BLUEBUBBLES) == 5.0
        assert GatewayRunner._followup_grace_seconds(Platform.MATRIX) == 1.0

    @pytest.mark.asyncio
    async def test_telegram_env_var_still_honoured_after_generalisation(self, monkeypatch):
        """The platform-specific env var must keep working — it is documented and in use."""
        from gateway.run import GatewayRunner

        # Only the LEGACY Telegram name is set; the generic one is explicitly off.
        monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
        monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "0")

        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "interrupt"
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
        agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()

        result = await GatewayRunner._handle_message(
            runner,
            MessageEvent(
                text="follow-up",
                message_type=MessageType.TEXT,
                source=source,
                message_id="tg-1",
            ),
        )
        assert result is None
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
    async def test_steer_mode_transcribes_voice_before_injection(self, monkeypatch):
        """A busy voice follow-up is transcribed and steered, never queued."""
        import gateway.run as _gr

        monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
        monkeypatch.setattr(_gr, "_load_gateway_config", lambda: {})
        runner, _sentinel = _make_runner()
        runner._busy_input_mode = "steer"
        runner._should_echo_stt_transcripts = MagicMock(return_value=False)
        runner._enrich_message_with_transcription = AsyncMock(
            return_value=('"yönü teknik mimariye çevir"', ["yönü teknik mimariye çevir"])
        )
        adapter = _make_adapter()

        event = _make_event(text="")
        event.message_type = MessageType.VOICE
        event.media_urls = ["/tmp/follow-up.ogg"]
        event.media_types = ["audio/ogg"]
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent

        await runner._handle_active_session_busy_message(event, sk)

        runner._enrich_message_with_transcription.assert_awaited_once_with(
            "", ["/tmp/follow-up.ogg"]
        )
        agent.steer.assert_called_once_with('"yönü teknik mimariye çevir"')
        agent.interrupt.assert_not_called()
        assert sk not in adapter._pending_messages
        content = adapter._send_with_retry.call_args.kwargs["content"]
        assert "Steered" in content
        assert "Queued" not in content


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


