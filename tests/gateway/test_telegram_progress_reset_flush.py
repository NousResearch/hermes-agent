"""Regression test for the __reset__ handler in send_progress_messages.

Issue: #29371

When a tool event arrives during the throttle window (1.5s) and is the last
event of the turn, the consumer appends it to ``progress_lines`` and sleeps
out the throttle. The next thing dequeued is the ``__reset__`` marker queued
by the gateway when the final content bubble lands. Without an explicit
flush in the __reset__ handler, ``progress_lines`` is cleared before any
edit fires — the user sees N-1 lines in the progress bubble.

The cancellation drain path at the bottom of send_progress_messages already
flushes pending progress before clearing on __reset__; the hot-loop path
must mirror that behavior.

Mirrors the simulation pattern used in test_telegram_progress_edit_transient.py.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional
from unittest.mock import AsyncMock

import pytest

EditFn = Callable[[str, str], Awaitable[object]]


async def _simulate_reset_handler(
    *,
    can_edit: bool,
    progress_lines: list[str],
    progress_msg_id: Optional[str],
    edit_fn: EditFn,
) -> tuple[list[str], Optional[str]]:
    """Mirror the hot-loop __reset__ handler from send_progress_messages.

    Returns the (progress_lines, progress_msg_id) state after the handler
    runs, so callers can assert state was cleared regardless of flush path.
    """
    if can_edit and progress_lines and progress_msg_id:
        try:
            await edit_fn(progress_msg_id, "\n".join(progress_lines))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
    return [], None


@pytest.mark.asyncio
async def test_reset_flushes_pending_progress_before_clearing():
    """The load-bearing case: pending tail line MUST be flushed before reset."""
    edit_fn = AsyncMock()
    progress_lines = ["tool A done", "tool B done", "tool C done (the tail)"]

    new_lines, new_msg_id = await _simulate_reset_handler(
        can_edit=True,
        progress_lines=progress_lines,
        progress_msg_id="msg-42",
        edit_fn=edit_fn,
    )

    edit_fn.assert_awaited_once_with(
        "msg-42", "tool A done\ntool B done\ntool C done (the tail)"
    )
    assert new_lines == []
    assert new_msg_id is None


@pytest.mark.asyncio
async def test_reset_no_flush_when_progress_lines_empty():
    """No edit if there's nothing to flush — avoid editing an empty bubble."""
    edit_fn = AsyncMock()

    new_lines, new_msg_id = await _simulate_reset_handler(
        can_edit=True,
        progress_lines=[],
        progress_msg_id="msg-42",
        edit_fn=edit_fn,
    )

    edit_fn.assert_not_awaited()
    assert new_lines == []
    assert new_msg_id is None


@pytest.mark.asyncio
async def test_reset_no_flush_when_no_progress_msg_id():
    """No edit before the first progress message has been sent."""
    edit_fn = AsyncMock()

    new_lines, new_msg_id = await _simulate_reset_handler(
        can_edit=True,
        progress_lines=["tool A done"],
        progress_msg_id=None,
        edit_fn=edit_fn,
    )

    edit_fn.assert_not_awaited()
    assert new_lines == []
    assert new_msg_id is None


@pytest.mark.asyncio
async def test_reset_no_flush_when_edit_disabled():
    """Edit-disabled platforms (e.g. iMessage) skip the flush."""
    edit_fn = AsyncMock()

    new_lines, new_msg_id = await _simulate_reset_handler(
        can_edit=False,
        progress_lines=["tool A done"],
        progress_msg_id="msg-42",
        edit_fn=edit_fn,
    )

    edit_fn.assert_not_awaited()
    assert new_lines == []
    assert new_msg_id is None


@pytest.mark.asyncio
async def test_reset_swallows_edit_failures_and_still_clears():
    """An edit failure during flush must not block state from being cleared."""
    edit_fn = AsyncMock(side_effect=RuntimeError("edit failed"))

    new_lines, new_msg_id = await _simulate_reset_handler(
        can_edit=True,
        progress_lines=["tool A done"],
        progress_msg_id="msg-42",
        edit_fn=edit_fn,
    )

    edit_fn.assert_awaited_once()
    assert new_lines == []
    assert new_msg_id is None
