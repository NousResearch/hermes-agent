"""One-time progress acknowledgement is a SEPARATE quantity from the repeat interval (#125887).

Before the split, one knob (``agent.gateway_notify_interval``) drove both the first status message
and every repeat. A user who wanted a 3-second "got it" receipt therefore also bought ~20 send/edit
attempts per minute of longer work -- enough to hit the Telegram flood envelope the same code
already throttles elsewhere (``display.tool_progress: new`` edits at 1.5s).

Two independent clocks:

* ``agent.gateway_notify_ack_interval`` (opt-in) -- seconds before the ONE-TIME acknowledgement.
* ``agent.gateway_notify_interval`` -- the REPEAT interval, except that a NEGATIVE value now means
  "acknowledge once, never repeat". 0 and positive values mean exactly what they always did.

The first test pins all four cells of that contract. The second is the backward-compatibility
guard: a home that never wrote the new key must see nothing at all before the repeat interval
elapses, so splitting the knob cannot silently change any existing deployment.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


_ACK = 0.01       # the one-time acknowledgement, due almost immediately
_REPEAT = 1.0     # the repeat interval, far enough out that "shortened" is unambiguous


def _make_turn(ack_interval, repeat_interval, monkeypatch, *, already_showed=False):
    """A GatewayRunner with one live turn + a double adapter recording every outbound call."""
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    if ack_interval is None:
        monkeypatch.delenv("HERMES_AGENT_NOTIFY_ACK_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_ACK_INTERVAL", str(ack_interval))
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", str(repeat_interval))

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False

    calls: list[tuple[str, str]] = []

    async def _send(chat_id, text, **kwargs):
        calls.append(("send", str(text)))
        return SimpleNamespace(success=True, message_id="status-1")

    async def _edit(chat_id, message_id, text, **kwargs):
        calls.append(("edit", str(text)))
        return SimpleNamespace(success=True, message_id=message_id)

    adapter = MagicMock()
    adapter.send = AsyncMock(side_effect=_send)
    adapter.edit_message = AsyncMock(side_effect=_edit)
    runner._delivery_adapter_for = lambda source: adapter
    runner._agent_activity_summary = staticmethod(lambda agent: None)

    agent = MagicMock()
    runner._running_agents["sess"] = agent

    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False
    disp._generic_status_phrase.return_value = "ACK"

    ctx = TurnContext(source=SimpleNamespace(chat_id="c", platform="telegram"), session_key="sess")
    ctx.agent_holder[0] = agent
    if already_showed:
        # A turn that already put content in front of the user: commentary delivered, no
        # turn-final yet. ``showed_user_content`` is the consumer's own view of that
        # (gateway/stream_consumer.py), the same one the gateway reads.
        consumer = MagicMock()
        consumer.showed_user_content = True
        ctx.stream_consumer_holder[0] = consumer
    return runner, disp, ctx, calls


async def _wait_for_calls(calls, count, timeout):
    """Poll until ``count`` outbound calls have landed; returns how many are there."""
    deadline = time.monotonic() + timeout
    while len(calls) < count and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    return len(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_interval", ["-1", _REPEAT])
@pytest.mark.parametrize("already_showed", [False, True])
async def test_ack_is_one_time_and_independent_of_the_repeat_interval(
    repeat_interval, already_showed, monkeypatch,
):
    """All four cells of the split, on one contract.

    ==========================  ================  =================================================
    repeat interval             turn showed       expected
                                  content
    ==========================  ================  =================================================
    positive                    no                ack lands on the ack clock; the next call is an
                                                   EDIT of that same bubble, and nothing arrives
                                                   until the repeat interval is due
    positive                    yes               ack suppressed (content already speaks for the
                                                   turn); the repeat's own send still arrives
    negative ("never repeat")   no                exactly one send, for the rest of the turn
    negative ("never repeat")   yes               nothing at all, ever
    ==========================  ================  =================================================
    """
    repeats = repeat_interval != "-1"
    runner, disp, ctx, calls = _make_turn(
        _ACK, repeat_interval, monkeypatch, already_showed=already_showed,
    )

    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    try:
        if already_showed:
            # The ack is suppressed, so the first outbound call cannot precede the repeat.
            await asyncio.sleep(_ACK * 20)
            assert calls == [], (
                f"an acknowledgement must stay silent once the turn already showed content; "
                f"got {calls!r}"
            )
            if not repeats:
                await asyncio.sleep(1.0)
                assert calls == [], (
                    f"ack suppressed + 'never repeat' means nothing is ever sent; got {calls!r}"
                )
                return
            assert await _wait_for_calls(calls, 1, timeout=5) == 1
            assert calls[0][0] == "send", (
                f"the first call must be the repeat's own send, not an edit of an ack that "
                f"was never sent; calls: {calls!r}"
            )
            return

        # The ack is due on its own short clock, not the repeat interval: the unfixed baseline
        # sleeps _NOTIFY_INTERVAL before its FIRST send, so nothing is here yet.
        assert await _wait_for_calls(calls, 1, timeout=_REPEAT / 2) == 1, (
            f"the one-time acknowledgement must not wait for the repeat interval; calls: {calls!r}"
        )
        assert calls[0][0] == "send"

        if not repeats:
            # A negative repeat interval means never; wait well past the ack's own clock.
            await asyncio.sleep(1.0)
            assert len(calls) == 1, (
                f"'never repeat' must acknowledge exactly once; got {len(calls)} calls: {calls!r}"
            )
            return

        # The defect: a short ack must not have bought a short REPEAT.
        await asyncio.sleep(_REPEAT / 2)
        assert len(calls) == 1, (
            f"a {_ACK}s acknowledgement must not shorten the {_REPEAT}s repeat interval; "
            f"got {len(calls)} outbound calls: {calls!r}"
        )
        assert await _wait_for_calls(calls, 2, timeout=5) == 2, (
            f"the repeat heartbeat must still fire on its own clock; calls: {calls!r}"
        )
        assert calls[1][0] == "edit", (
            f"the repeat must edit the acknowledgement's bubble, not add another one; {calls!r}"
        )
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_absent_ack_key_keeps_first_send_on_the_repeat_interval(monkeypatch):
    """A home that never wrote the new key keeps today's behaviour: nothing before the repeat.

    The whole backward-compatibility argument for the split. Splitting one knob into two is only
    safe because the new one defaults to off; this fails the moment it defaults to anything else.
    """
    runner, disp, ctx, calls = _make_turn(None, _REPEAT, monkeypatch)

    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    try:
        await asyncio.sleep(_ACK * 20)
        assert calls == [], (
            "writing only agent.gateway_notify_interval must not gain an early "
            f"acknowledgement; got {calls!r}"
        )
        assert await _wait_for_calls(calls, 1, timeout=5) == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
