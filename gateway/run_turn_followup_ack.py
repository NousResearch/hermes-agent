"""Processing-lifecycle hooks for runner-drained queued follow-ups.

A message that arrives mid-turn is parked in the adapter's pending slot and drained in-band by
``TurnRunner._run_agent_queued_followup``, never by ``BasePlatformAdapter._process_message_background``
— the only other call site for ``on_processing_start`` / ``on_processing_complete``. Without firing
them here every adapter that renders a read-receipt reaction from the hooks silently skips queued,
interrupting and steer-demoted messages (#72502, salvage #72503).
"""

from __future__ import annotations

import asyncio

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome


def _followup_processing_hooks_apply(adapter, event: MessageEvent | None) -> bool:
    """Both conditions are necessary: a real inbound platform message to acknowledge (adapters key their
    marker off ``message_id`` — Slack/Telegram/Feishu/Matrix/Photon — or off ``raw_message`` — Signal,
    Discord; synthetic drains such as ``/goal`` continuations, wake-ups and startup auto-resume carry
    neither and must stay silent), and an adapter that overrides ``on_processing_start``. We bracket, so
    both halves must belong to us: a complete-only adapter (Google Chat reaps its typing card there,
    webhook ends its per-delivery session) would otherwise be handed a completion for a turn whose reply
    is delivered only after the drain chain unwinds back into ``_process_message_background``."""
    if adapter is None or event is None:
        return False
    if not (getattr(event, "message_id", None) or getattr(event, "raw_message", None)):
        return False
    start_hook = getattr(type(adapter), "on_processing_start", None)
    return start_hook is not None and start_hook is not BasePlatformAdapter.on_processing_start


def _followup_cancel_outcome(adapter) -> ProcessingOutcome:
    """Classify a cancelled follow-up exactly as ``_process_message_background`` does: only cancels the
    adapter itself routed (``/stop``, ``/new``, ``/reset``, cleanup) are CANCELLED, anything else is a
    failure. Signal and Matrix leave the in-progress marker in place on CANCELLED, so reporting an
    unexpected cancellation as CANCELLED would strand it."""
    expected = getattr(adapter, "_expected_cancelled_tasks", None)
    if expected is None:
        return ProcessingOutcome.FAILURE
    try:
        current = asyncio.current_task()
    except RuntimeError:
        current = None
    if current is None:
        return ProcessingOutcome.FAILURE
    try:
        return ProcessingOutcome.CANCELLED if current in expected else ProcessingOutcome.FAILURE
    except TypeError:
        return ProcessingOutcome.FAILURE


async def _run_followup_processing_hook(adapter, event: MessageEvent | None, hook_name: str, *args) -> None:
    """Fire one lifecycle hook for a runner-drained follow-up; no-op per ``_followup_processing_hooks_apply``."""
    if not _followup_processing_hooks_apply(adapter, event):
        return
    run_hook = getattr(adapter, "_run_processing_hook", None)
    if not callable(run_hook):
        return
    await run_hook(hook_name, event, *args)
