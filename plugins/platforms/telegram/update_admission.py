"""Telegram admission before every PTB handler group, including native plugins.

Claims belong to the receiving adapter across Application rebuilds. Completed
history is bounded and has no TTL; dispatch and its PTB tasks pin active claims.
No disk receipt, cross-process coordination or exactly-once effects are promised.
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from telegram import Update
from telegram.ext import Application, ApplicationHandlerStop, ConversationHandler

from gateway.platforms.helpers import bounded_put


_DEFAULT_BLOCK = object()


@dataclass
class _Claim:
    key: str
    dispatch_task: asyncio.Task | None
    accepted: bool = False
    completed: bool = False
    failed: bool = False
    owners: int = 1


class _ErrorCallback:
    """Admit callback entry while keeping PTB's original registration/removal key."""

    def __init__(self, callback):
        self.__wrapped__ = callback

    def __hash__(self):
        return hash(self.__wrapped__)

    def __eq__(self, other):
        if isinstance(other, _ErrorCallback):
            other = other.__wrapped__
        return self.__wrapped__ == other

    async def __call__(self, update, context):
        # Registration is a native extension boundary, unlike PTB's internal logger.
        # Resolve the receiving app so a shared callback never accepts another bot's claim.
        admission = getattr(context.application, "_current_claim", None)
        claim = admission.get() if admission is not None else None
        if claim is not None and claim.owners:
            claim.accepted = True
        return await self.__wrapped__(update, context)


class TelegramApplication(Application):
    __slots__ = ("adapter", "_current_claim")

    def __init__(self, *, adapter, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter
        if adapter._update_admission is None:
            adapter._update_admission = ContextVar[_Claim | None]("telegram_update_claim", default=None)
        self._current_claim: ContextVar[_Claim | None] = adapter._update_admission

    def add_handler(self, handler, group=0):
        super().add_handler(handler, group)
        self._admit_callbacks(handler)

    def _admit_callbacks(self, handler):
        # ConversationHandler has no callback. Keep its identity (including persistence,
        # state transitions and block resolution) and instrument its selected children.
        if isinstance(handler, ConversationHandler):
            for child in [*handler.entry_points, *handler.fallbacks,
                          *(child for handlers in handler.states.values() for child in handlers)]:
                self._admit_callbacks(child)
            return
        callback = getattr(handler, "callback", None)
        if callback is None or getattr(callback, "_telegram_admitted", False):
            return

        @wraps(callback)
        async def admitted_callback(update, context):
            # Native plugins own arbitrary effects. Entry is the handoff; retrying a partly
            # completed plugin would be unsafe. Core preparation has explicit handoff markers.
            # A plugin may reuse the same handler across applications/reconnects. Resolve
            # the receiving claim at invocation, rather than closing over its first owner.
            application = context.application
            admission = getattr(application, "_current_claim", None)
            claim = admission.get() if admission is not None else None
            if claim is not None and getattr(callback, "__self__", None) is not application.adapter:
                claim.accepted = True
            result = await callback(update, context)
            if claim is not None:
                if claim.failed and not claim.accepted:
                    # Stop subsequent groups only in blocking dispatch. PTB cannot stop
                    # groups already scheduled via block=False; their outcomes settle below.
                    if asyncio.current_task() is claim.dispatch_task:
                        raise ApplicationHandlerStop
                else:
                    claim.completed = True
            return result

        admitted_callback._telegram_admitted = True
        handler.callback = admitted_callback

    def add_error_handler(self, callback, block=_DEFAULT_BLOCK):
        if not isinstance(callback, _ErrorCallback):
            callback = _ErrorCallback(callback)
        # Omitted block must remain PTB's default, not explicit True (bot defaults differ).
        if block is _DEFAULT_BLOCK:
            super().add_error_handler(callback)
        else:
            super().add_error_handler(callback, block=block)

    def _Application__create_task(self, coroutine, update=None, is_error_handler=False, name=None):
        # PTB's process_error bypasses public create_task. Delegate at the shared task seam
        # without copying dispatch or losing its is_error_handler recursion protection.
        task = super()._Application__create_task(
            coroutine, update=update, is_error_handler=is_error_handler, name=name)
        claim = self._current_claim.get()
        if claim is not None and claim.owners:
            # Pin scheduled work before callback entry, including cancellation before entry.
            claim.owners += 1

            def finished(done):
                # PTB's task wrapper may never start, leaving its callback unawaited.
                if done.cancelled() and asyncio.iscoroutine(coroutine):
                    coroutine.close()
                self._release_claim(claim, done)

            task.add_done_callback(finished)
        return task

    def _release_claim(self, claim, task=None):
        if task is not None and task.cancelled():
            claim.failed = True
        claim.owners -= 1
        if claim.owners:
            return
        del self.adapter._inflight_update_ids[claim.key]
        if claim.accepted or (claim.completed and not claim.failed):
            bounded_put(self.adapter._seen_update_ids, claim.key, None, 4096)

    async def process_error(self, update, error, job=None, coroutine=None):
        claim = self._current_claim.get()
        if claim is not None:
            claim.failed = True
        stopped = await super().process_error(update, error, job, coroutine)
        # PTB catches callback errors. Do not let a later observer accept a failed preparation.
        return stopped or (claim is not None and not claim.accepted)

    async def process_update(self, update):
        if not isinstance(update, Update):
            return await super().process_update(update)
        key = f"{self.bot.id}:{update.update_id}"
        # Dispatch happened even when preparation fails before the group-99 observer.
        self.adapter._updates_dispatched_total += 1
        seen = self.adapter._seen_update_ids
        pending = self.adapter._inflight_update_ids
        if key in seen or key in pending:
            return
        claim = _Claim(key, asyncio.current_task())
        # Atomic on PTB's event loop: no await between lookup and claim. Dispatch and its
        # PTB tasks share ownership; completed-history pressure cannot evict active work.
        pending[key] = claim
        token = self._current_claim.set(claim)
        try:
            await super().process_update(update)
        except BaseException:
            claim.failed = True
            raise
        finally:
            self._current_claim.reset(token)
            self._release_claim(claim)
