"""The adapter processing lifecycle of a producer-admitted turn (relay, webhook: ``admit_producer``).

A chat message an adapter hands to ``handle_message`` is bracketed by
``BasePlatformAdapter._process_message_background``: the typing refresh, ``on_processing_start``,
and ``on_processing_complete`` once its reply was sent. A producer delivery is only committed at
ingress (off the transport reader, acknowledged on the commit) and the authority's drain runs it, so
no adapter task brackets it. The executor does, for those admissions only (a messaging admission's
own adapter task already brackets it):

- opened when the admission starts executing, not when it commits: a delivery queued behind a
  running turn is acknowledged when it runs, as the in-band queue's are (``run_turn_followup_ack``);
- typing lives as long as execution, so it stops before the reply is sent (#117300 pauses it there);
- the outcome is reported once: an interrupted turn as CANCELLED and a failed one as FAILURE when
  execution ends; a completed one after its settlement, by the reply's send (``deliver_settled``),
  or as FAILURE when settlement dropped the reply (``drop_producer_turn``).
"""
import asyncio
from contextlib import asynccontextmanager

from gateway.platforms.event import ProcessingOutcome


def _turns(authority):
    """admission id -> None (committed by a producer, not started) or its open ``_Bracket``."""
    turns = getattr(authority, 'producer_turns', None)
    if turns is None:
        turns = authority.producer_turns = {}
    return turns


def mark_producer_admission(authority, event, row):
    """``SessionAuthority._admitted``, in the commit's own tracked task: a producer's freshly queued
    row is the executor's to bracket. A re-send of a row already running or settled is not marked."""
    if getattr(event, '_producer_delivery', False) and row['status'] == 'queued':
        _turns(authority).setdefault(row['admission_id'], None)


class _Bracket:
    def __init__(self, adapter, event, hooks):
        from gateway.platforms.base import _thread_metadata_for_event
        self.adapter, self.event, self.hooks = adapter, event, hooks
        self.metadata = _thread_metadata_for_event(event)
        self.stop = asyncio.Event()
        self.typing = None
        self.closed = False

    async def stop_typing(self):
        task, self.typing = self.typing, None
        if task is not None:
            self.stop.set()
            await self.adapter._stop_typing_refresh(self.event.source.chat_id, task, metadata=self.metadata)

    async def close(self, outcome):
        if self.closed:
            return
        self.closed = True
        await self.stop_typing()
        if self.hooks:
            await self.adapter._run_processing_hook('on_processing_complete', self.event, outcome)


async def _open(authority, row, event):
    """The bracket of a producer-admitted row starting to execute, else None: an adapter that
    renders neither typing nor the processing hooks (a webhook) is left alone."""
    turns = _turns(authority)
    if row['admission_id'] not in turns:
        return None
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.run_turn_followup_ack import _followup_processing_hooks_apply
    adapter = authority.runner._adapter_for_source(event.source)
    typing = adapter is not None and type(adapter).send_typing is not BasePlatformAdapter.send_typing
    hooks = _followup_processing_hooks_apply(adapter, event)
    if not (typing or hooks):
        turns.pop(row['admission_id'], None)
        return None
    bracket = turns[row['admission_id']] = _Bracket(adapter, event, hooks)
    if typing:
        # None for a heartbeat or ``typing_indicator: false``, as on the adapter path.
        bracket.typing = adapter._start_typing_refresh(event, bracket.stop, bracket.metadata)
    if hooks:
        try:
            await adapter._run_processing_hook('on_processing_start', event)
        except BaseException:
            # Cancelled before ``producer_turn``'s finally exists: release what was opened here.
            bracket.stop.set()
            if bracket.typing is not None:
                bracket.typing.cancel()
            turns.pop(row['admission_id'], None)
            raise
    return bracket


def _executed_outcome(captured):
    """The outcome settlement will commit (``session_results.prepare_result``), as a hook outcome."""
    result = captured.get('terminal_result', captured.get('result')) or {}
    if result.get('interrupted'):
        return ProcessingOutcome.CANCELLED
    if result.get('failed') or result.get('error'):
        return ProcessingOutcome.FAILURE
    return ProcessingOutcome.SUCCESS


@asynccontextmanager
async def producer_turn(authority, row, event, captured):
    """Around ``execute_admission``'s run of *row*: ``captured`` is its ``execution_result``."""
    bracket = await _open(authority, row, event)
    if bracket is None:
        yield
        return
    outcome = ProcessingOutcome.FAILURE
    try:
        yield
        outcome = _executed_outcome(captured)
    finally:
        await bracket.stop_typing()
        if outcome is not ProcessingOutcome.SUCCESS:
            await close_producer_turn(authority, row['admission_id'], outcome)


async def close_producer_turn(authority, admission_id, outcome):
    """Report a settled producer turn's outcome (once; a no-op for any other admission)."""
    bracket = _turns(authority).pop(admission_id, None)
    if bracket is not None:
        await bracket.close(outcome)


def drop_producer_turn(authority, admission_id):
    """The admission settled without its reply (unknown, discarded) or was cancelled before it ran:
    a started turn reports FAILURE, off the caller's path."""
    bracket = _turns(authority).pop(admission_id, None)
    if bracket is None or bracket.closed:
        return
    task = asyncio.get_running_loop().create_task(bracket.close(ProcessingOutcome.FAILURE))
    bracket.adapter._background_tasks.add(task)
    task.add_done_callback(bracket.adapter._background_tasks.discard)
