"""Runner integration tests for busy-session controls.

Covers:
- Halt-phrase pre-flight (multilingual stop intent triggers immediate
  interrupt + 🙊 reaction without falling through to mode dispatch).
- Button-tap dispatch (steer / interrupt / stop) and the contract that
  one tap acts on ALL pending follow-ups.
- Multi follow-up text concatenation in the order received.
- Reaction lifecycle (👍 / ⚡ / 🙊 emitted on each follow-up).
- Control-bubble fallback when no tool bubble exists.

Adapter is mocked; we exercise GatewayRunner methods directly to keep
the surface tight.  These complement the platform-neutral wire-format
tests in ``tests/gateway/test_busy_session_buttons.py``.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub telegram so gateway.run imports cleanly on a bare CI runner.
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

from gateway.busy_session_buttons import (
    PRIMITIVE_INTERRUPT,
    PRIMITIVE_STEER,
    PRIMITIVE_STOP,
    REACTION_INTERRUPT,
    REACTION_STEER,
    REACTION_STOP,
)
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)


def _make_event(text="hello", chat_id="123", platform_val="telegram", message_id="msg1"):
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


def _make_runner():
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._busy_ack_tool_bubble_defer_seconds = 0.0
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "queue"

    # New-state fields exercised by busy-session-buttons.
    runner._tool_bubble_msg_ids = {}
    runner._busy_control_bubble_ids = {}
    runner._pending_followups = {}
    runner._session_run_generation = {}

    # _interrupt_and_clear_session needs these.
    runner._invalidate_session_run_generation = MagicMock()
    runner._release_running_agent_state = MagicMock(return_value=True)
    runner._is_session_run_current = MagicMock(return_value=True)

    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter():
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.set_busy_reaction = AsyncMock(return_value=True)
    adapter.attach_busy_session_buttons = AsyncMock(return_value=True)
    adapter.clear_busy_session_buttons = AsyncMock(return_value=True)
    adapter.send_or_update_busy_control_bubble = AsyncMock(return_value="ctrl_msg_id")
    adapter.delete_busy_control_bubble = AsyncMock(return_value=True)
    # get_pending_message has consume-and-discard semantics on real adapters.
    adapter.get_pending_message = MagicMock(
        side_effect=lambda sk: adapter._pending_messages.pop(sk, None)
    )
    adapter.interrupt_session_activity = AsyncMock()
    adapter.edit_message = AsyncMock(return_value=MagicMock(success=True))
    return adapter


# ---------------------------------------------------------------------------
# Halt-phrase pre-flight
# ---------------------------------------------------------------------------


class TestHaltPhrasePreflight:
    @pytest.mark.asyncio
    async def test_english_stop_word_triggers_immediate_halt(self):
        from gateway.run import GatewayRunner

        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="stop")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent
        adapter._pending_messages[sk] = "leftover"

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        assert result is True
        # Halt phrase routes through the full _interrupt_and_clear_session
        # path so the chat unlocks even when the agent is wedged inside a
        # tool — same behavior as /stop and the [Stop] button.
        agent.interrupt.assert_called_once_with("Stop requested")
        adapter.set_busy_reaction.assert_awaited_with(event, REACTION_STOP)
        # Pending slot cleared — halt does NOT replay text as next turn.
        assert sk not in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_japanese_halt_phrase_matches(self):
        from gateway.run import GatewayRunner

        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="止まれ")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        assert result is True
        adapter.set_busy_reaction.assert_awaited_with(event, REACTION_STOP)

    @pytest.mark.asyncio
    async def test_long_message_with_stop_word_does_not_halt(self):
        """The conservative length cap is the main false-positive defense."""
        from gateway.run import GatewayRunner

        runner, _ = _make_runner()
        adapter = _make_adapter()
        long_text = "we should stop including Bob in the email today"
        event = _make_event(text=long_text)
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent

        await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Mode is "queue" — agent is NOT interrupted.
        agent.interrupt.assert_not_called()


# ---------------------------------------------------------------------------
# Button-tap dispatch
# ---------------------------------------------------------------------------


class TestButtonTap:
    @pytest.mark.asyncio
    async def test_steer_tap_routes_to_running_agent_steer(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="also include vector DBs")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent
        runner._pending_followups[sk] = [event]
        adapter._pending_messages[sk] = event  # would normally be replayed

        toast = await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_STEER, event.source
        )

        agent.steer.assert_called_once_with("also include vector DBs")
        # Steer landed inside the run, so the queued copy must be cleared
        # to avoid double-processing it as next turn.
        assert sk not in adapter._pending_messages
        adapter.set_busy_reaction.assert_awaited_with(event, REACTION_STEER)
        assert "Steered" in toast or REACTION_STEER in toast

    @pytest.mark.asyncio
    async def test_interrupt_tap_calls_running_agent_interrupt_with_text(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="completely different task: refactor module X")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner._pending_followups[sk] = [event]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_INTERRUPT, event.source
        )

        agent.interrupt.assert_called_once_with(
            "completely different task: refactor module X"
        )
        adapter.set_busy_reaction.assert_awaited_with(event, REACTION_INTERRUPT)

    @pytest.mark.asyncio
    async def test_interrupt_tap_replaces_pending_slot_with_joined_text(self):
        """The post-run drain promotes ``adapter._pending_messages[sk]`` as
        the next-turn prompt.  Because that slot is single-slot text-
        replacing, an interrupt tap with multiple follow-ups must
        rewrite its text to the joined version, otherwise the next turn
        only sees the last follow-up."""
        runner, _ = _make_runner()
        adapter = _make_adapter()
        e1 = _make_event(text="first", message_id="m1")
        e2 = _make_event(text="second", message_id="m2")
        sk = build_session_key(e1.source)
        runner.adapters[e1.source.platform] = adapter

        # Simulate the upstream busy-handler outcome: only the latest
        # follow-up is in adapter._pending_messages, but both are tracked
        # in runner._pending_followups for the button-tap path.
        adapter._pending_messages[sk] = e2
        runner._running_agents[sk] = MagicMock()
        runner._pending_followups[sk] = [e1, e2]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_INTERRUPT, e1.source
        )

        # The pending event's text now carries BOTH messages so the next
        # turn replay sees the full conversation, not just "second".
        assert "first" in adapter._pending_messages[sk].text
        assert "second" in adapter._pending_messages[sk].text

    @pytest.mark.asyncio
    async def test_stop_tap_runs_full_clear_session_path(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="please halt")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner._pending_followups[sk] = [event]
        adapter._pending_messages[sk] = "should be cleared"

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_STOP, event.source
        )

        agent.interrupt.assert_called_once()
        # _interrupt_and_clear_session pops the pending slot.
        assert sk not in adapter._pending_messages
        adapter.set_busy_reaction.assert_awaited_with(event, REACTION_STOP)

    @pytest.mark.asyncio
    async def test_unknown_primitive_returns_unknown_action(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event()
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        toast = await runner._handle_busy_session_button_tap(
            sk, "fortify", event.source
        )

        assert "Unknown" in toast

    @pytest.mark.asyncio
    async def test_cross_user_tap_in_shared_chat_is_rejected(self):
        """Per-user session keys: another authorized user in the chat
        must not be able to control someone else's run via a visible
        button.  Reject if the tapping user's session_key doesn't match
        the target session_key."""
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="x")
        sk_owner = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk_owner] = MagicMock()
        runner._pending_followups[sk_owner] = [event]

        # A different authorized user clicks the button.  Their source
        # has a different user_id and therefore a different session_key.
        from gateway.platforms.base import SessionSource as _SS
        other = _SS(
            platform=event.source.platform,
            chat_id="123",
            chat_type="private",
            user_id="user2",  # different from event.source.user_id == "user1"
        )

        toast = await runner._handle_busy_session_button_tap(
            sk_owner, PRIMITIVE_INTERRUPT, other
        )

        assert "isn't your session" in toast.lower() or "not your session" in toast.lower()
        # And the agent was not interrupted on behalf of the wrong user.
        runner._running_agents[sk_owner].interrupt.assert_not_called()


class TestMultipleFollowUps:
    @pytest.mark.asyncio
    async def test_two_followups_get_concatenated_and_each_reacted(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        e1 = _make_event(text="first follow up", message_id="m1")
        e2 = _make_event(text="second follow up", message_id="m2")
        sk = build_session_key(e1.source)
        runner.adapters[e1.source.platform] = adapter

        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent
        runner._pending_followups[sk] = [e1, e2]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_STEER, e1.source
        )

        agent.steer.assert_called_once()
        passed = agent.steer.call_args.args[0]
        assert "first follow up" in passed
        assert "second follow up" in passed
        # First in list, first in joined text — preserves arrival order.
        assert passed.index("first follow up") < passed.index("second follow up")

        # Each follow-up gets its own reaction.
        reaction_targets = [
            call.args[0] for call in adapter.set_busy_reaction.await_args_list
        ]
        assert e1 in reaction_targets
        assert e2 in reaction_targets

    @pytest.mark.asyncio
    async def test_pending_followups_cleared_after_tap(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        e1 = _make_event(text="x")
        sk = build_session_key(e1.source)
        runner.adapters[e1.source.platform] = adapter
        runner._pending_followups[sk] = [e1]
        runner._running_agents[sk] = MagicMock()

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_INTERRUPT, e1.source
        )
        assert sk not in runner._pending_followups


class TestAckAnchorAndToolBubbleAnchor:
    @pytest.mark.asyncio
    async def test_no_tool_bubble_does_not_send_standalone_message(self):
        """No tool bubble → _ensure_busy_session_controls is a no-op.

        The upstream busy-ack message becomes the keyboard anchor via
        ``_anchor_busy_session_buttons_to_ack`` (called separately from
        the busy handler after the ack send).  This avoids the duplicate
        "/queue'd ..." + standalone-control message pair that the
        previous design produced.
        """
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="hello before any tool")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        await runner._ensure_busy_session_controls(sk, event)

        adapter.attach_busy_session_buttons.assert_not_called()
        adapter.send_or_update_busy_control_bubble.assert_not_called()

    @pytest.mark.asyncio
    async def test_deferred_queue_ack_anchors_to_tool_bubble_and_suppresses_ack(self):
        """If the first tool bubble appears during the short defer window,
        controls move to that bubble instead of sending a separate queue
        notice above it.
        """
        from gateway.run import GatewayRunner

        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="follow-up just before first tool")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()
        runner._busy_ack_tool_bubble_defer_seconds = 0.01

        task = asyncio.create_task(
            GatewayRunner._handle_active_session_busy_message(runner, event, sk)
        )
        await asyncio.sleep(0)
        runner._tool_bubble_msg_ids[sk] = "tool_bubble_99"

        assert await task is True
        adapter._send_with_retry.assert_not_called()
        adapter.attach_busy_session_buttons.assert_awaited_with(sk, "tool_bubble_99")

    @pytest.mark.asyncio
    async def test_tool_bubble_present_attaches_keyboard_directly(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="mid-tool follow up")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._tool_bubble_msg_ids[sk] = "bubble_42"
        runner._busy_control_bubble_ids[sk] = ["ack_old"]

        await runner._ensure_busy_session_controls(sk, event)

        adapter.attach_busy_session_buttons.assert_awaited_once_with(sk, "bubble_42")
        adapter.clear_busy_session_buttons.assert_awaited_with(sk, "ack_old")
        assert sk not in runner._busy_control_bubble_ids
        adapter.send_or_update_busy_control_bubble.assert_not_called()

    @pytest.mark.asyncio
    async def test_anchor_to_ack_attaches_keyboard_to_ack_message(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event()
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        await runner._anchor_busy_session_buttons_to_ack(sk, event, "ack_msg_77")

        adapter.attach_busy_session_buttons.assert_awaited_once_with(sk, "ack_msg_77")
        assert runner._busy_control_bubble_ids[sk] == ["ack_msg_77"]

    @pytest.mark.asyncio
    async def test_anchor_to_ack_replaces_previous_ack_anchor(self):
        """Long turns crossing the 30s ack-cooldown may produce another
        ack, but only the newest ack may keep a live keyboard.
        """
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event()
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        await runner._anchor_busy_session_buttons_to_ack(sk, event, "ack_1")
        await runner._anchor_busy_session_buttons_to_ack(sk, event, "ack_2")
        await runner._anchor_busy_session_buttons_to_ack(sk, event, "ack_3")

        assert runner._busy_control_bubble_ids[sk] == ["ack_3"]
        adapter.clear_busy_session_buttons.assert_any_await(sk, "ack_1")
        adapter.clear_busy_session_buttons.assert_any_await(sk, "ack_2")

    @pytest.mark.asyncio
    async def test_anchor_to_ack_uses_tool_bubble_if_one_exists(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event()
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._tool_bubble_msg_ids[sk] = "bubble_9"

        await runner._anchor_busy_session_buttons_to_ack(sk, event, "ack_should_not_get_buttons")

        adapter.attach_busy_session_buttons.assert_awaited_once_with(sk, "bubble_9")
        assert sk not in runner._busy_control_bubble_ids


class TestAckTextUpdateAfterTap:
    """A button tap must rewrite the ack body so it doesn't keep saying
    'Queued for the next turn...' after the user picked Steer / Interrupt
    / Stop."""

    @pytest.mark.asyncio
    async def test_steer_tap_rewrites_ack_text(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="hello")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        agent = MagicMock()
        agent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = agent
        runner._pending_followups[sk] = [event]
        runner._busy_control_bubble_ids[sk] = ["ack_77"]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_STEER, event.source
        )

        edit_calls = [
            c.kwargs for c in adapter.edit_message.await_args_list
        ]
        assert any("Steered" in (kw.get("content") or "") for kw in edit_calls)
        # Keyboard cleared before edit so the keyboard re-attach in
        # edit_message() doesn't immediately re-add it.
        adapter.clear_busy_session_buttons.assert_any_await(sk, "ack_77")

    @pytest.mark.asyncio
    async def test_interrupt_tap_rewrites_ack_text(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="redirect")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()
        runner._pending_followups[sk] = [event]
        runner._busy_control_bubble_ids[sk] = ["ack_91"]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_INTERRUPT, event.source
        )

        edit_calls = [
            c.kwargs for c in adapter.edit_message.await_args_list
        ]
        assert any("Interrupted" in (kw.get("content") or "") for kw in edit_calls)

    @pytest.mark.asyncio
    async def test_stop_tap_rewrites_ack_text(self):
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="halt")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._running_agents[sk] = MagicMock()
        runner._pending_followups[sk] = [event]
        runner._busy_control_bubble_ids[sk] = ["ack_55"]

        await runner._handle_busy_session_button_tap(
            sk, PRIMITIVE_STOP, event.source
        )

        edit_calls = [
            c.kwargs for c in adapter.edit_message.await_args_list
        ]
        assert any("Stopped" in (kw.get("content") or "") for kw in edit_calls)


class TestCleanup:
    @pytest.mark.asyncio
    async def test_clear_busy_session_controls_tears_everything_down(self):
        """Cleanup detaches the keyboard from BOTH anchors and forgets state.

        The control-bubble entry is the ack message (real chat history we
        don't own), so cleanup detaches the keyboard but does NOT delete
        the message itself.
        """
        runner, _ = _make_runner()
        adapter = _make_adapter()
        event = _make_event()
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter
        runner._tool_bubble_msg_ids[sk] = "bubble_1"
        runner._busy_control_bubble_ids[sk] = ["ack_1"]
        runner._pending_followups[sk] = [event]

        await runner._clear_busy_session_controls(sk, event.source)

        clear_calls = [c.args for c in adapter.clear_busy_session_buttons.await_args_list]
        assert (sk, "bubble_1") in clear_calls
        assert (sk, "ack_1") in clear_calls
        # Cleanup must NOT delete the ack message — it's part of chat history.
        adapter.delete_busy_control_bubble.assert_not_called()
        assert sk not in runner._tool_bubble_msg_ids
        assert sk not in runner._busy_control_bubble_ids
        assert sk not in runner._pending_followups
