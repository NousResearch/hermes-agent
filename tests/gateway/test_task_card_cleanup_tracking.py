"""The native "Hermes is working" task card joins cleanup_progress tracking.

Main's cleanup_progress deletes tracked progress bubbles only after the final reply is
delivered (a failed run keeps them). The native card was never tracked, so it stayed in
the thread forever. These tests drive the real TurnRunner._task_card_publish caller.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

from gateway.platforms.base import SendResult


class _CardAdapter:
    def __init__(self, result: SendResult) -> None:
        self._result = result
        self.calls: List[dict] = []

    async def send_native_task_card_progress(self, **kwargs: Any) -> SendResult:
        self.calls.append(kwargs)
        return self._result


def _runner(adapter: _CardAdapter, *, cleanup: bool):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1", platform="slack"),
        _progress_reply_to=None, _progress_metadata={},
        _cleanup_progress=cleanup, _cleanup_msg_ids=[],
    )

    async def _fallback(_st):
        raise AssertionError("text fallback must not run for a working card")

    runner._task_card_send_or_edit_fallback = _fallback
    return runner


def _state(adapter: _CardAdapter):
    return SimpleNamespace(
        tasks=[{"text": "step"}], native_failed=False, publication_suppressed=False,
        visible_tasks=lambda: [{"text": "step"}], fallback_text=lambda: "step", adapter=adapter,
        title=lambda: "Working",
    )


def test_card_id_is_tracked_once_for_cleanup():
    adapter = _CardAdapter(SendResult(success=True, message_id="CARD1"))
    runner = _runner(adapter, cleanup=True)
    st = _state(adapter)

    for _ in range(3):  # every update of one card returns the same stream ts
        asyncio.run(runner._task_card_publish(st))

    assert len(adapter.calls) == 3
    assert runner._ctx._cleanup_msg_ids == ["CARD1"]


def test_card_not_tracked_when_cleanup_progress_is_off():
    adapter = _CardAdapter(SendResult(success=True, message_id="CARD1"))
    runner = _runner(adapter, cleanup=False)

    asyncio.run(runner._task_card_publish(_state(adapter)))

    assert runner._ctx._cleanup_msg_ids == []
