"""Regression tests for the hot-loop ``__reset__`` handler in
``send_progress_messages`` (gateway/run.py).

Issue: #29371

When the last tool event of a turn arrives inside the edit-throttle window
(``_PROGRESS_EDIT_INTERVAL`` = 1.5s) and no further event follows, that line
sits in ``progress_lines`` having been appended but not yet edited onto the
platform. The gateway then enqueues ``__reset__`` once the final content
bubble lands. The hot-loop ``__reset__`` handler used to clear
``progress_lines`` *without* flushing first, so the pending tail line was
silently dropped — the user saw N-1 tool lines even though every tool ran.

The drain-path ``__reset__`` handler at the bottom of the consumer already
flushes pending progress before clearing; the hot loop must mirror it.

``send_progress_messages`` is a local closure inside ``run_conversation`` and
cannot be imported and driven directly, so the behavioural tests mirror the
handler logic — the same approach
``tests/gateway/test_telegram_progress_edit_transient.py`` uses for #27828.
``test_both_reset_handlers_flush_before_clear_source_invariant`` additionally
asserts the fix against the real source, so removing the flush is caught.
"""

from __future__ import annotations

import inspect
import re
from typing import Awaitable, Callable, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Behavioural mirror of the hot-loop __reset__ handler
# (gateway/run.py send_progress_messages). Returns the (progress_lines,
# progress_msg_id) state after the handler runs.
# ---------------------------------------------------------------------------
async def _run_reset_handler(
    *,
    can_edit: bool,
    progress_lines: List[str],
    progress_msg_id: Optional[str],
    edit_fn: Callable[[str, str], Awaitable[object]],
    progress_text: Callable[[List[str]], str] = lambda lines: "\n".join(lines),
) -> Tuple[List[str], Optional[str]]:
    if can_edit and progress_lines and progress_msg_id:
        _pending_text = progress_text(progress_lines)
        try:
            await edit_fn(progress_msg_id, _pending_text)
        except Exception:
            pass
    return [], None


@pytest.mark.asyncio
async def test_reset_flushes_pending_tail_before_clearing():
    """The load-bearing case: a tail line MUST be flushed before reset clears."""
    edit_fn = AsyncMock()
    lines, msg_id = await _run_reset_handler(
        can_edit=True,
        progress_lines=["tool A", "tool B", "tool C (arrived in throttle window)"],
        progress_msg_id="msg-1",
        edit_fn=edit_fn,
    )
    edit_fn.assert_awaited_once_with(
        "msg-1", "tool A\ntool B\ntool C (arrived in throttle window)"
    )
    assert lines == []
    assert msg_id is None


@pytest.mark.asyncio
async def test_reset_no_flush_when_no_pending_lines():
    """Nothing pending → don't edit an empty bubble."""
    edit_fn = AsyncMock()
    lines, msg_id = await _run_reset_handler(
        can_edit=True, progress_lines=[], progress_msg_id="msg-1", edit_fn=edit_fn
    )
    edit_fn.assert_not_awaited()
    assert lines == []
    assert msg_id is None


@pytest.mark.asyncio
async def test_reset_no_flush_when_editing_disabled():
    """Platform doesn't support edits (can_edit False) → no flush attempt."""
    edit_fn = AsyncMock()
    await _run_reset_handler(
        can_edit=False, progress_lines=["tool A"], progress_msg_id="msg-1", edit_fn=edit_fn
    )
    edit_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_no_flush_without_message_id():
    """No bubble created yet (msg_id None) → nothing to edit."""
    edit_fn = AsyncMock()
    await _run_reset_handler(
        can_edit=True, progress_lines=["tool A"], progress_msg_id=None, edit_fn=edit_fn
    )
    edit_fn.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_flush_swallows_edit_errors_and_still_clears():
    """A transient edit failure during flush must not break the reset path."""
    edit_fn = AsyncMock(side_effect=RuntimeError("transient network error"))
    lines, msg_id = await _run_reset_handler(
        can_edit=True, progress_lines=["tool A"], progress_msg_id="msg-1", edit_fn=edit_fn
    )
    edit_fn.assert_awaited_once()
    assert lines == []
    assert msg_id is None


def test_both_reset_handlers_flush_before_clear_source_invariant():
    """Real-source guard against regressing #29371.

    Every ``__reset__`` handler in ``send_progress_messages`` must flush
    (call ``_edit_progress_message``) before it clears ``progress_lines``.
    This fails if the hot-loop flush is removed, which the behavioural mirrors
    above cannot catch (the closure can't be driven directly).
    """
    import gateway.run as run_module

    src = inspect.getsource(run_module)
    markers = [m.start() for m in re.finditer(r'raw\[0\] == "__reset__"', src)]
    assert len(markers) >= 2, (
        f"expected >=2 __reset__ handlers in gateway/run.py, found {len(markers)}"
    )
    for start in markers:
        clear_at = src.find("progress_lines = []", start)
        assert clear_at != -1, "no progress_lines clear found after a __reset__ handler"
        block = src[start:clear_at]
        assert "_edit_progress_message" in block, (
            "a __reset__ handler clears progress_lines without flushing the "
            "pending tail first — regression of #29371"
        )
