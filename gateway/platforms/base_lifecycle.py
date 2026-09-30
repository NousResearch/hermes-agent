"""Adapter processing hooks and outcome reactions."""

from __future__ import annotations

import logging
from typing import Any

from gateway.platforms.event import MessageEvent, ProcessingOutcome

logger = logging.getLogger("gateway.platforms.base")

class BaseLifecycleMixin:
    name: str
    _OK_EMOJI: str | None
    _FAIL_EMOJI: str | None

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Hook called when background processing begins."""


    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Hook called when background processing completes. Default: opt-in reaction ack — with
        ``_OK_EMOJI``/``_FAIL_EMOJI`` set and ``_add_reaction``/``_remove_reaction`` present, swap
        the in-progress reaction for the outcome one. Remove-then-add is deterministic whether the
        platform replaces or stacks a sender's reactions. CANCELLED leaves it unreacted."""
        if self._OK_EMOJI is None and self._FAIL_EMOJI is None:
            return
        add: Any = getattr(self, "_add_reaction", None)
        remove: Any = getattr(self, "_remove_reaction", None)
        enabled = getattr(self, "_reactions_enabled", None)
        chat_id = getattr(event.source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if (not callable(add) or not callable(remove) or (callable(enabled) and not enabled())
                or not chat_id or not message_id):
            return
        await remove(chat_id, message_id)
        emoji = {ProcessingOutcome.SUCCESS: self._OK_EMOJI,
                 ProcessingOutcome.FAILURE: self._FAIL_EMOJI}.get(outcome)
        if emoji:
            await add(chat_id, message_id, emoji)

    async def on_inline_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Acknowledge an inline command without changing the active turn's lifecycle."""

    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        event = args[0] if args else None
        if isinstance(event, MessageEvent):
            if hook_name == "on_processing_complete":
                completion = event._processing_state.pending_completion
                if completion is not None:
                    event._processing_state.pending_completion = None
                    await completion.adapter._run_processing_hook(
                        hook_name, completion.event, *args[1:], **kwargs)
                for absorbed in event._processing_state.take_absorbed():
                    await absorbed.adapter._run_processing_hook(
                        hook_name, absorbed.event, *args[1:], **kwargs)
            if hook_name == "on_processing_start":
                if not event._processing_state.start():
                    return
            elif hook_name == "on_processing_complete":
                if not event._processing_state.complete():
                    return
            elif hook_name == "on_inline_processing_complete":
                if not event._processing_state.complete_inline():
                    return
        try:
            await hook(*args, **kwargs)
        except Exception as e:
            logger.warning("[%s] %s hook failed: %s", self.name, hook_name, e)

